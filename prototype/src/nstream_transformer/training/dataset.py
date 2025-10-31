"""JSONL dataset feeding the knowledge distillation collator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import Dataset

from ..data.snapshots import SnapshotFeatures


@dataclass(slots=True)
class KDRecord:
    """Decoded JSONL record ready for collation."""

    student_ids: torch.Tensor
    student_labels: torch.Tensor
    planner_ids: torch.Tensor
    notes_student: torch.Tensor
    notes_teacher: Optional[torch.Tensor]
    role: str
    teacher_snapshots: List[SnapshotFeatures] = field(default_factory=list)
    student_snapshots: List[SnapshotFeatures] = field(default_factory=list)
    stride_ids: torch.Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.long))
    commit_points: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    agreement_labels: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    metadata: Dict[str, Any] = field(default_factory=dict)
    example_id: Optional[str] = None
    plan_items: List[str] = field(default_factory=list)
    plan_catalog: List[str] = field(default_factory=list)
    plan_catalog_roles: List[str] = field(default_factory=list)
    coverage_targets: List[int] = field(default_factory=list)
    notes_text: str = ""
    raw_teacher_notes: Dict[str, List[str]] = field(default_factory=dict)


class KDJsonlDataset(Dataset[Dict[str, object]]):
    """Loads KD training examples from a JSONL manifest.

    Each line must encode keys: ``student_ids``, ``student_labels``, ``planner_ids``,
    ``notes_student``, ``notes_teacher``, and ``role``. Additional optional keys are
    supported to express per-stride snapshots, agreement labels, and metadata.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        if not path.exists():
            raise FileNotFoundError(f"KDJsonlDataset expects dataset at {path} to exist.")
        self._records: List[KDRecord] = [self._decode(line) for line in self._iter_lines(path)]

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, object]:  # type: ignore[override]
        record = self._records[index]
        return {
            "student_ids": record.student_ids,
            "student_labels": record.student_labels,
            "planner_ids": record.planner_ids,
            "notes_student": record.notes_student,
            "notes_teacher": record.notes_teacher,
            "role": record.role,
            "teacher_snapshots": record.teacher_snapshots,
            "student_snapshots": record.student_snapshots,
            "stride_ids": record.stride_ids,
            "commit_points": record.commit_points,
            "agreement_labels": record.agreement_labels,
            "metadata": record.metadata,
            "example_id": record.example_id,
            "plan_items": record.plan_items,
            "plan_catalog": record.plan_catalog,
            "plan_catalog_roles": record.plan_catalog_roles,
            "coverage_targets": record.coverage_targets,
            "notes_text": record.notes_text,
            "raw_teacher_notes": record.raw_teacher_notes,
        }

    def _decode(self, payload: str) -> KDRecord:
        data = json.loads(payload)
        student_ids = self._tensor(data["student_ids"], torch.long)
        student_labels = self._tensor(data.get("student_labels", data["student_ids"]), torch.long)
        planner_ids = self._tensor(data["planner_ids"], torch.long)
        notes_student = self._tensor(data["notes_student"], torch.float32)
        teacher_notes_raw = data.get("notes_teacher")
        notes_teacher = (
            self._tensor(teacher_notes_raw, torch.float32)
            if teacher_notes_raw is not None
            else None
        )
        metadata = dict(data.get("metadata", {}))
        if "document_text" not in metadata or "document_paragraphs" not in metadata:
            raise KeyError(
                "KDJsonlDataset metadata must include 'document_text' and 'document_paragraphs'."
            )
        metadata["document_text"] = str(metadata["document_text"])
        metadata["document_paragraphs"] = [
            str(paragraph) for paragraph in metadata.get("document_paragraphs", [])
        ]
        teacher_notes_text = {
            role: [str(note) for note in notes]
            for role, notes in (metadata.get("teacher_notes") or {}).items()
        }
        teacher_snapshots = self._build_snapshots(
            self._snapshot_payload(data, "teacher"), notes_teacher, source="teacher"
        )
        student_snapshots = self._build_snapshots(
            self._snapshot_payload(data, "student"), notes_student, source="student"
        )
        stride_ids = self._tensor(data.get("stride_ids", [0]), torch.long)
        commit_points = self._tensor(data.get("commit_points", []), torch.long)
        agreement_labels = self._tensor(data.get("agreement_labels", []), torch.long)
        example_id = self._resolve_example_id(data, metadata)
        (
            plan_items,
            coverage_targets,
            notes_text,
            plan_catalog,
            plan_catalog_roles,
        ) = self._extract_plan_items(
            role=data.get("role", "core"),
            plan_tokens=data.get("plan_tokens", []),
            notes_tokens=data.get("notes_tokens", []),
            metadata=metadata,
        )
        return KDRecord(
            student_ids=student_ids,
            student_labels=student_labels,
            planner_ids=planner_ids,
            notes_student=notes_student,
            notes_teacher=notes_teacher,
            role=data.get("role", "core"),
            teacher_snapshots=teacher_snapshots,
            student_snapshots=student_snapshots,
            stride_ids=stride_ids,
            commit_points=commit_points,
            agreement_labels=agreement_labels,
            metadata=metadata,
            example_id=example_id,
            plan_items=plan_items,
            plan_catalog=plan_catalog,
            plan_catalog_roles=plan_catalog_roles,
            coverage_targets=coverage_targets,
            notes_text=notes_text,
            raw_teacher_notes=teacher_notes_text,
        )

    def _tensor(self, values: Sequence, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            return values.to(dtype=dtype)
        tensor = torch.tensor(values, dtype=dtype)
        return tensor

    def _build_snapshots(
        self,
        payload: Sequence[Mapping[str, Any]],
        default_notes: Optional[torch.Tensor],
        *,
        source: str,
    ) -> List[SnapshotFeatures]:
        if not payload:
            if default_notes is None:
                return []
            return [SnapshotFeatures(notes=default_notes.clone(), source=source)]
        snapshots: List[SnapshotFeatures] = []
        for index, item in enumerate(payload):
            if "notes" in item:
                notes = self._tensor(item["notes"], torch.float32)
            elif default_notes is not None:
                notes = default_notes.clone()
            else:
                raise KeyError("Snapshot payload missing notes and no default provided.")
            stride = int(item.get("stride", item.get("stride_id", 0)))
            version = int(item.get("version", item.get("snapshot_id", index)))
            role = item.get("role")
            coverage = item.get("coverage_flags")
            coverage_tensor = None
            if coverage is not None:
                coverage_tensor = self._tensor(coverage, torch.float32)
            snapshots.append(
                SnapshotFeatures(
                    notes=notes,
                    stride=stride,
                    version=version,
                    role=role,
                    coverage=coverage_tensor,
                    source=source,
                )
            )
        return snapshots

    def _snapshot_payload(
        self, data: Mapping[str, Any], prefix: str
    ) -> Sequence[Mapping[str, Any]]:
        if prefix == "teacher":
            return data.get("teacher_snapshots") or data.get("snapshots") or []
        if prefix == "student":
            return data.get("student_snapshots") or data.get("speculative_snapshots") or []
        return []

    def _resolve_example_id(
        self, data: Mapping[str, Any], metadata: Mapping[str, Any]
    ) -> Optional[str]:
        candidate = data.get("example_id") or data.get("id") or metadata.get("id")
        if candidate is None:
            return None
        return str(candidate)

    def _extract_plan_items(
        self,
        *,
        role: str,
        plan_tokens: Sequence[str],
        notes_tokens: Sequence[str],
        metadata: Mapping[str, Any],
    ) -> tuple[List[str], List[int], str, List[str], List[str]]:
        role_lower = role.lower()
        teacher_plan = metadata.get("teacher_plan", {}) or {}
        teacher_notes = metadata.get("teacher_notes", {}) or {}
        notes_text = " ".join(str(token) for token in notes_tokens)

        catalog: List[tuple[str, str]] = []

        def _append_item(entry_role: str, text: Any) -> None:
            item_text = str(text).strip()
            if not item_text:
                return
            catalog.append((entry_role.lower(), item_text))

        for entry in teacher_plan.get("plan", []):
            entry_role = str(entry.get("role", "")).strip().lower()
            if not entry_role:
                continue
            for note in entry.get("notes") or []:
                _append_item(entry_role, note)
            summary = entry.get("summary")
            if summary is not None:
                _append_item(entry_role, summary)

        if not catalog and plan_tokens:
            for token in plan_tokens:
                if not isinstance(token, str):
                    continue
                token = token.strip()
                if not token:
                    continue
                if "::" in token:
                    prefix, body = token.split("::", 1)
                    entry_role = prefix.strip().lower() or role_lower
                    item_text = body
                else:
                    entry_role = role_lower
                    item_text = token
                _append_item(entry_role, item_text)

        if teacher_notes:
            role_notes = teacher_notes.get(role_lower) or teacher_notes.get(role)
            if role_notes:
                notes_text = " ".join(str(note) for note in role_notes)

        plan_catalog_roles = [role_name for role_name, _ in catalog]
        plan_catalog = [text for _, text in catalog]
        plan_items_for_role = [text for role_name, text in catalog if role_name == role_lower]
        coverage_targets = [1 if role_name == role_lower else 0 for role_name, _ in catalog]

        if not plan_catalog and plan_tokens:
            plan_items_for_role = []
            coverage_targets = []

        return plan_items_for_role, coverage_targets, notes_text, plan_catalog, plan_catalog_roles

    def _iter_lines(self, path: Path) -> Iterator[str]:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    yield stripped


__all__ = ["KDJsonlDataset", "KDRecord"]
