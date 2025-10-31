"""Teacher notes providers and on-demand generation utilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

from .snapshots import SnapshotFeatures
from .teacher_runner import TeacherRunner, TeacherRunnerConfig, TeacherRunResult, TeacherSnapshotText
from ..utils import resolve_device

try:  # pragma: no cover - optional heavy dependency
    from transformers import AutoModel, AutoTokenizer  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency guard
    AutoModel = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    _TRANSFORMERS_IMPORT_ERROR = exc
else:  # pragma: no cover - dependency guard
    _TRANSFORMERS_IMPORT_ERROR = None


@dataclass(slots=True)
class TeacherNotes:
    """Container returned by teacher providers containing dense notes and snapshots."""

    notes: torch.Tensor
    snapshots: list[SnapshotFeatures] = field(default_factory=list)
    raw_notes: Dict[str, list[str]] = field(default_factory=dict)

    def clone(self) -> "TeacherNotes":
        """Deep copy to avoid mutating cached payloads."""
        return TeacherNotes(
            notes=self.notes.clone(),
            snapshots=[_clone_snapshot(snapshot) for snapshot in self.snapshots],
            raw_notes={role: list(values) for role, values in self.raw_notes.items()},
        )


class TeacherNotesProviderBase:
    """Base interface for teacher-note providers."""

    def fetch(self, example: Mapping[str, Any]) -> TeacherNotes:
        """Return teacher notes for the provided example."""
        raise NotImplementedError


@dataclass(slots=True)
class NoOpTeacherNotesProvider(TeacherNotesProviderBase):
    """Provider that trusts the dataset payload verbatim."""

    field_name: str = "notes_teacher"
    snapshots_field: str = "teacher_snapshots"
    raw_field: str = "raw_teacher_notes"

    def fetch(self, example: Mapping[str, Any]) -> TeacherNotes:
        value = example.get(self.field_name)
        if value is None:
            raise KeyError(f"Teacher notes missing from example; looked for {self.field_name!r}.")
        if isinstance(value, torch.Tensor):
            notes = value.clone()
        else:
            notes = torch.tensor(value, dtype=torch.float32)
        raw_notes = {
            role: list(strings)
            for role, strings in (example.get(self.raw_field) or {}).items()
        }
        raw_snapshots = example.get(self.snapshots_field) or []
        snapshots: list[SnapshotFeatures] = []
        for item in raw_snapshots:
            if isinstance(item, SnapshotFeatures):
                snapshots.append(_clone_snapshot(item))
            elif isinstance(item, Mapping):
                notes_tensor = _coerce_tensor(item.get("notes"))
                coverage_tensor = (
                    _coerce_tensor(item.get("coverage")) if item.get("coverage") is not None else None
                )
                snapshots.append(
                    SnapshotFeatures(
                        notes=notes_tensor,
                        stride=int(item.get("stride", 0)),
                        version=int(item.get("version", 0)),
                        role=item.get("role"),
                        coverage=coverage_tensor,
                        source=str(item.get("source", "teacher")),
                    )
                )
        return TeacherNotes(notes=notes, snapshots=snapshots, raw_notes=raw_notes)


@dataclass(slots=True)
class CachedTeacherNotesProvider(TeacherNotesProviderBase):
    """Provider wrapper that caches teacher outputs keyed by a stable identifier."""

    backend: TeacherNotesProviderBase
    cache: Dict[str, TeacherNotes] = field(default_factory=dict)
    id_field: str = "example_id"
    cache_dir: Optional[Path] = None
    refresh: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, example: Mapping[str, Any]) -> TeacherNotes:
        key = self._resolve_key(example)
        if key is not None and not self.refresh:
            cached = self.cache.get(key)
            if cached is not None:
                return cached.clone()
            disk_cached = self._load_from_disk(key)
            if disk_cached is not None:
                self.cache[key] = disk_cached.clone()
                return disk_cached.clone()

        payload = self.backend.fetch(example)
        clone = payload.clone()
        if key is not None:
            self.cache[key] = clone
            self._store_to_disk(key, payload)
        return payload.clone()

    def _resolve_key(self, example: Mapping[str, Any]) -> Optional[str]:
        candidate = example.get(self.id_field)
        if candidate is None:
            metadata = example.get("metadata") or {}
            candidate = metadata.get(self.id_field) or metadata.get("id")
        if candidate is None:
            return None
        return str(candidate)

    def _cache_path(self, key: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        safe = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{safe}.pt"

    def _store_to_disk(self, key: str, payload: TeacherNotes) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        serializable = {
            "notes": payload.notes.cpu(),
            "snapshots": [
                {
                    "notes": snapshot.notes.cpu(),
                    "stride": snapshot.stride,
                    "version": snapshot.version,
                    "role": snapshot.role,
                    "coverage": snapshot.coverage.cpu() if snapshot.coverage is not None else None,
                    "source": snapshot.source,
                }
                for snapshot in payload.snapshots
            ],
            "raw_notes": payload.raw_notes,
        }
        torch.save(serializable, path)

    def _load_from_disk(self, key: str) -> Optional[TeacherNotes]:
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None
        data: Dict[str, Any] = torch.load(path, map_location="cpu")
        notes_tensor = _coerce_tensor(data.get("notes", torch.zeros(1)))
        snapshots: list[SnapshotFeatures] = []
        for entry in data.get("snapshots", []):
            coverage_tensor = None
            if entry.get("coverage") is not None:
                coverage_tensor = _coerce_tensor(entry["coverage"])
            snapshots.append(
                SnapshotFeatures(
                    notes=_coerce_tensor(entry["notes"]),
                    stride=int(entry.get("stride", 0)),
                    version=int(entry.get("version", 0)),
                    role=entry.get("role"),
                    coverage=coverage_tensor,
                    source=str(entry.get("source", "teacher")),
                )
            )
        raw_notes = {
            role: list(entries) for role, entries in (data.get("raw_notes") or {}).items()
        }
        return TeacherNotes(notes=notes_tensor, snapshots=snapshots, raw_notes=raw_notes)


class TeacherNotesProvider(TeacherNotesProviderBase):
    """Generates teacher notes and stride snapshots via the LLM-backed runner."""

    def __init__(
        self,
        config: TeacherRunnerConfig,
        *,
        notes_dim: int,
        role_to_id: Mapping[str, int],
        embedder: Optional["_BaseNoteEmbedder"] = None,
        runner: Optional[TeacherRunner] = None,
    ) -> None:
        if not role_to_id:
            raise ValueError("role_to_id mapping must not be empty.")
        self.config = config
        self.notes_dim = notes_dim
        self._role_items: list[Tuple[str, int]] = sorted(
            ((role.lower(), int(idx)) for role, idx in role_to_id.items()),
            key=lambda item: item[1],
        )
        self._role_names = [role for role, _ in self._role_items]
        self.runner = runner or TeacherRunner(config)
        self.embedder = embedder or _TransformerNoteEmbedder(config.embedder_model)

    def fetch(self, example: Mapping[str, Any]) -> TeacherNotes:
        run = self.runner.run(example)
        notes_tensor = self._embed_role_notes(run.role_notes)
        snapshots = [self._embed_snapshot(snapshot) for snapshot in run.snapshots]
        return TeacherNotes(
            notes=notes_tensor,
            snapshots=snapshots,
            raw_notes={role: list(values) for role, values in run.role_notes.items()},
        )

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed_role_notes(self, role_notes: Mapping[str, Sequence[str]]) -> torch.Tensor:
        matrix = torch.zeros((len(self._role_items), self.notes_dim), dtype=torch.float32)
        for role, index in self._role_items:
            texts = role_notes.get(role)
            if not texts:
                raise ValueError(f"Teacher runner did not produce notes for role '{role}'.")
            matrix[index] = self.embedder.aggregate(texts, self.notes_dim)
        return matrix

    def _embed_snapshot(self, snapshot: TeacherSnapshotText) -> SnapshotFeatures:
        matrix = self._embed_role_notes(snapshot.role_notes)
        coverage_values = torch.tensor(
            [float(snapshot.coverage.get(role, 0.0)) for role, _ in self._role_items],
            dtype=torch.float32,
        )
        return SnapshotFeatures(
            notes=matrix,
            stride=snapshot.stride,
            version=snapshot.version,
            role=None,
            coverage=coverage_values,
            source="teacher",
        )


class _BaseNoteEmbedder:
    """Abstract embedder used to project note strings into fixed-width vectors."""

    def aggregate(self, texts: Sequence[str], target_dim: int) -> torch.Tensor:
        raise NotImplementedError


class _TransformerNoteEmbedder(_BaseNoteEmbedder):
    """Transformer-backed embedder using mean pooled hidden states."""

    def __init__(self, model_name: str) -> None:
        if AutoTokenizer is None or AutoModel is None:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "The optional 'transformers' dependency is required for on-demand teacher notes."
            ) from _TRANSFORMERS_IMPORT_ERROR
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore[operator]
        self.model = AutoModel.from_pretrained(model_name)  # type: ignore[operator]
        self.model.eval()
        preferred = resolve_device()
        if preferred == "cuda" and not torch.cuda.is_available():
            preferred = "cpu"
        if preferred == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                preferred = "cpu"
        self.device = torch.device(preferred)
        self.model.to(self.device)

    def aggregate(self, texts: Sequence[str], target_dim: int) -> torch.Tensor:
        if not texts:
            raise ValueError("Cannot aggregate empty note text sequence.")
        inputs = self.tokenizer(  # type: ignore[attr-defined]
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts
        embedding = torch.nn.functional.normalize(pooled.mean(dim=0), dim=0).cpu()
        return _project_embedding(embedding, target_dim)


def _project_embedding(embedding: torch.Tensor, target_dim: int) -> torch.Tensor:
    current_dim = int(embedding.numel())
    if current_dim == target_dim:
        return embedding.clone().to(torch.float32)
    if current_dim > target_dim:
        return embedding[:target_dim].clone().to(torch.float32)
    repeats = target_dim // current_dim
    remainder = target_dim % current_dim
    tiled = embedding.repeat(repeats)
    if remainder:
        tiled = torch.cat([tiled, embedding[:remainder]])
    return torch.nn.functional.normalize(tiled, dim=0).to(torch.float32)


def _coerce_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.clone().to(torch.float32)
    return torch.tensor(value, dtype=torch.float32)


def _clone_snapshot(snapshot: SnapshotFeatures) -> SnapshotFeatures:
    coverage = snapshot.coverage.clone() if isinstance(snapshot.coverage, torch.Tensor) else snapshot.coverage
    return SnapshotFeatures(
        notes=snapshot.notes.clone(),
        stride=snapshot.stride,
        version=snapshot.version,
        role=snapshot.role,
        coverage=coverage,
        source=snapshot.source,
    )


__all__ = [
    "TeacherNotes",
    "TeacherNotesProviderBase",
    "NoOpTeacherNotesProvider",
    "CachedTeacherNotesProvider",
    "TeacherNotesProvider",
]
