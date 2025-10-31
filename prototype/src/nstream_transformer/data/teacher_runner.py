"""LLM-backed teacher runner that materialises notes and stride snapshots on demand."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from .extraction.llm_client import LLMClient, LLMMessage, create_llm_client
from .extraction.planner import TeacherPlanner


@dataclass(slots=True)
class TeacherSnapshotText:
    """Textual snapshot produced by the teacher runner."""

    version: int
    stride: int
    role_notes: Dict[str, list[str]]
    coverage: Dict[str, float]


@dataclass(slots=True)
class TeacherRunResult:
    """Structured teacher supervision containing notes and stride snapshots."""

    example_id: str
    role_notes: Dict[str, list[str]]
    snapshots: list[TeacherSnapshotText]
    teacher_plan: Dict[str, Any]


@dataclass(slots=True)
class TeacherRunnerConfig:
    """Configuration payload for the teacher runner."""

    provider: str
    model: str
    api_key: Optional[str] = None
    planner_provider: str = "ollama"
    planner_model: str = "llama3.1:70b-instruct-q5_K_M"
    planner_api_key: Optional[str] = None
    cache_dir: Optional[str] = None
    id_field: str = "example_id"
    document_field: str = "document_text"
    paragraph_field: str = "document_paragraphs"
    plan_field: str = "teacher_plan"
    roles: Sequence[str] = ("intro", "core", "wrap")
    max_snapshots: int = 4
    stride_window: int = 128
    refresh_cache: bool = False
    notes_per_snapshot: int = 1
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("teacher_runner.provider must be provided.")
        if not self.model:
            raise ValueError("teacher_runner.model must be provided.")
        if self.max_snapshots <= 0:
            raise ValueError("max_snapshots must be positive.")
        if self.stride_window <= 0:
            raise ValueError("stride_window must be positive.")
        if self.notes_per_snapshot <= 0:
            raise ValueError("notes_per_snapshot must be positive.")


class TeacherRunner:
    """Generates authoritative notes and stride snapshots using an LLM teacher."""

    def __init__(
        self,
        config: TeacherRunnerConfig,
        *,
        llm_client: Optional[LLMClient] = None,
        planner: Optional[TeacherPlanner] = None,
    ) -> None:
        self.config = config
        self.roles = [role.lower() for role in config.roles]
        self.llm = llm_client or create_llm_client(
            provider=config.provider,
            api_key=config.api_key,
            model=config.model,
        )
        self.planner = planner or TeacherPlanner(
            provider=config.planner_provider,
            model=config.planner_model,
            api_key=config.planner_api_key,
        )
        self.cache_dir: Optional[Path] = Path(config.cache_dir) if config.cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run(self, example: Mapping[str, Any]) -> TeacherRunResult:
        """Materialise teacher supervision for an example."""
        example_id = self._resolve_example_id(example)
        cached = None if self.config.refresh_cache else self._load_from_cache(example_id)
        if cached is not None:
            return cached

        metadata = self._extract_metadata(example)
        document_text = metadata[self.config.document_field]
        paragraphs = self._resolve_paragraphs(metadata, document_text)
        teacher_plan = self._resolve_plan(metadata, example, paragraphs)
        role_notes = self._generate_role_notes(example_id, document_text, teacher_plan)
        snapshots = self._build_snapshots(role_notes, teacher_plan)

        result = TeacherRunResult(
            example_id=example_id,
            role_notes=role_notes,
            snapshots=snapshots,
            teacher_plan=teacher_plan,
        )
        self._store_to_cache(result)
        return result

    # ---------------------------------------------------------------------
    # Plan + note extraction
    # ---------------------------------------------------------------------

    def _resolve_example_id(self, example: Mapping[str, Any]) -> str:
        candidate = example.get(self.config.id_field)
        if candidate is None:
            metadata = example.get("metadata") or {}
            candidate = metadata.get(self.config.id_field) or metadata.get("id")
        if candidate is None:
            raise ValueError(f"TeacherRunner requires '{self.config.id_field}' on each example.")
        return str(candidate)

    def _extract_metadata(self, example: Mapping[str, Any]) -> Mapping[str, Any]:
        metadata = example.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("Examples must include a 'metadata' mapping for the teacher runner.")
        if self.config.document_field not in metadata:
            raise KeyError(
                f"TeacherRunner metadata missing required document field {self.config.document_field!r}."
            )
        return metadata

    def _resolve_paragraphs(
        self,
        metadata: Mapping[str, Any],
        document_text: str,
    ) -> Sequence[str]:
        paragraphs = metadata.get(self.config.paragraph_field)
        if isinstance(paragraphs, Sequence):
            return [str(paragraph) for paragraph in paragraphs]
        return self._split_paragraphs(document_text)

    def _resolve_plan(
        self,
        metadata: Mapping[str, Any],
        example: Mapping[str, Any],
        paragraphs: Sequence[str],
    ) -> Dict[str, Any]:
        plan_payload = metadata.get(self.config.plan_field)
        if isinstance(plan_payload, Mapping):
            return dict(plan_payload)
        title = (
            metadata.get("title")
            or example.get("title")
            or example.get("prompt")
            or metadata.get("document_title")
            or "document"
        )
        plan = self.planner.build_plan(str(title), paragraphs)
        return plan.raw

    def _generate_role_notes(
        self,
        example_id: str,
        document_text: str,
        teacher_plan: Mapping[str, Any],
    ) -> Dict[str, list[str]]:
        plan_items = [
            f"{item.get('role', '').lower()}: {item.get('summary', '')}"
            for item in teacher_plan.get("plan", [])
        ]
        plan_notes = {
            str(item.get("role", "")).lower(): item.get("notes", [])
            for item in teacher_plan.get("plan", [])
        }
        segments = teacher_plan.get("segments", [])
        plan_text = "\n".join(plan_items) or "intro: outline context\ncore: develop key ideas\nwrap: summarise outcome"

        prompt = (
            f"You are supervising a multi-role writing task. Each role must emit factual, non-redundant notes.\n"
            f"Document ID: {example_id}\n\n"
            f"Role plan:\n{plan_text}\n\n"
            "Produce concise bullet notes for each role that cover the plan requirements.\n"
            "Each note must be grounded in the document and maximum ~160 characters.\n"
            "Return JSON with this exact structure:\n"
            '{\n'
            '  "notes": {\n'
            '    "intro": ["..."],\n'
            '    "core": ["..."],\n'
            '    "wrap": ["..."]\n'
            '  }\n'
            '}\n'
            "If a role already has canonical notes in the plan, refine them rather than copying verbatim."
        )

        excerpt = document_text[:12000]
        messages = [
            LLMMessage(
                role="system",
                content=(
                    "You generate authoritative supervision notes for planner-aligned multi-stream training. "
                    "Always reply with valid JSON matching the requested schema. "
                    "Do not include prose outside the JSON payload."
                ),
            ),
            LLMMessage(
                role="user",
                content=f"{prompt}\n\nDocument excerpt (first 12k chars):\n{excerpt}",
            ),
        ]

        response = self.llm.complete(messages, temperature=0.0, max_tokens=2000)
        data = self.llm.extract_json(response)
        notes_payload = data.get("notes")
        if not isinstance(notes_payload, Mapping):
            raise ValueError("Teacher runner received malformed JSON: 'notes' field missing.")

        role_notes: Dict[str, list[str]] = {}
        for role in self.roles:
            items = notes_payload.get(role) or notes_payload.get(role.capitalize())
            if items is None and role in plan_notes:
                items = plan_notes[role]
            if items is None:
                raise ValueError(f"Teacher runner did not return notes for role '{role}'.")
            texts = [self._clean_note(str(entry)) for entry in items if str(entry).strip()]
            if not texts:
                raise ValueError(f"Teacher runner produced empty notes for role '{role}'.")
            role_notes[role] = texts
        return role_notes

    def _build_snapshots(
        self,
        role_notes: Mapping[str, Sequence[str]],
        teacher_plan: Mapping[str, Any],
    ) -> list[TeacherSnapshotText]:
        max_notes = max(len(notes) for notes in role_notes.values())
        snapshot_count = min(self.config.max_snapshots, max_notes)
        snapshots: list[TeacherSnapshotText] = []
        for version in range(snapshot_count):
            snapshot_notes: Dict[str, list[str]] = {}
            coverage: Dict[str, float] = {}
            for role in self.roles:
                notes = list(role_notes[role])
                target = min(len(notes), (version + 1) * self.config.notes_per_snapshot)
                target = max(1, target)
                snapshot_notes[role] = notes[:target]
                coverage[role] = target / float(len(notes))
            stride = self._resolve_stride(version, teacher_plan)
            snapshots.append(
                TeacherSnapshotText(
                    version=version,
                    stride=stride,
                    role_notes=snapshot_notes,
                    coverage=coverage,
                )
            )
        return snapshots

    def _resolve_stride(self, version: int, teacher_plan: Mapping[str, Any]) -> int:
        segments = teacher_plan.get("segments") or []
        if not segments:
            return (version + 1) * self.config.stride_window
        span = 0
        for segment in segments:
            start = int(segment.get("paragraph_start", 0))
            end = int(segment.get("paragraph_end", start))
            span = max(span, end - start)
        span = max(1, span)
        stride = min(
            (version + 1) * self.config.stride_window,
            (version + 1) * span,
        )
        return stride

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _split_paragraphs(self, text: str) -> list[str]:
        chunks = [block.strip() for block in text.replace("\r", "").split("\n\n") if block.strip()]
        return chunks or [text.strip()]

    def _clean_note(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("- "):
            cleaned = cleaned[2:].strip()
        return cleaned

    # ---------------------------------------------------------------------
    # Caching
    # ---------------------------------------------------------------------

    def _cache_path(self, example_id: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(example_id.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _load_from_cache(self, example_id: str) -> Optional[TeacherRunResult]:
        path = self._cache_path(example_id)
        if path is None or not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        snapshots = [
            TeacherSnapshotText(
                version=int(snapshot["version"]),
                stride=int(snapshot["stride"]),
                role_notes={role: list(notes) for role, notes in snapshot["role_notes"].items()},
                coverage={role: float(value) for role, value in snapshot["coverage"].items()},
            )
            for snapshot in data["snapshots"]
        ]
        return TeacherRunResult(
            example_id=example_id,
            role_notes={role: list(notes) for role, notes in data["role_notes"].items()},
            snapshots=snapshots,
            teacher_plan=data["teacher_plan"],
        )

    def _store_to_cache(self, payload: TeacherRunResult) -> None:
        path = self._cache_path(payload.example_id)
        if path is None:
            return
        serializable = {
            "role_notes": payload.role_notes,
            "snapshots": [
                {
                    "version": snapshot.version,
                    "stride": snapshot.stride,
                    "role_notes": snapshot.role_notes,
                    "coverage": snapshot.coverage,
                }
                for snapshot in payload.snapshots
            ],
            "teacher_plan": payload.teacher_plan,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, ensure_ascii=True)


__all__ = [
    "TeacherRunner",
    "TeacherRunnerConfig",
    "TeacherRunResult",
    "TeacherSnapshotText",
]
