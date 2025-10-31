"""Two-branch collator wiring student and teacher tensors for KD training."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from .snapshots import SnapshotFeatures
from .teacher_provider import NoOpTeacherNotesProvider, TeacherNotesProviderBase


@dataclass(slots=True)
class TwoBranchKDCollatorConfig:
    pad_token_id: int
    label_pad_id: int = -100
    notes_dim: int = 2048
    max_length: int = 2048
    max_snapshots: int = 4
    commit_horizon: int = 0
    role_to_id: Mapping[str, int] = field(
        default_factory=lambda: {"intro": 0, "core": 1, "wrap": 2}
    )
    plan_hash_buckets: int = 65536
    dtype: str = "bfloat16"


class TwoBranchKnowledgeDistillationCollator:
    """Assembles student and teacher tensors in a single batch dictionary."""

    def __init__(
        self,
        config: TwoBranchKDCollatorConfig,
        *,
        teacher_provider: Optional[TeacherNotesProviderBase] = None,
    ) -> None:
        self.config = config
        self._dtype = _resolve_dtype(config.dtype)
        self.teacher_provider = teacher_provider or NoOpTeacherNotesProvider()

    def __call__(self, batch: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        examples = list(batch)
        if not examples:
            raise ValueError("TwoBranchKnowledgeDistillationCollator received an empty batch.")

        teacher_payloads = []
        for example in examples:
            payload = self.teacher_provider.fetch(example)
            teacher_payloads.append(payload)
            example["notes_teacher"] = payload.notes
            example["teacher_snapshots"] = payload.snapshots

        student_ids = self._pad_sequence(
            [self._ensure_list(example["student_ids"]) for example in examples],
            self.config.pad_token_id,
        )
        student_labels = self._pad_sequence(
            [
                self._ensure_list(example.get("student_labels", example["student_ids"]))
                for example in examples
            ],
            self.config.label_pad_id,
        )
        planner_ids = self._pad_sequence(
            [self._ensure_list(example["planner_ids"]) for example in examples],
            self.config.pad_token_id,
        )

        roles = torch.tensor(
            [self._role_id(example.get("role", "core")) for example in examples],
            dtype=torch.long,
        )

        notes_student = torch.stack(
            [self._ensure_tensor(example["notes_student"]).to(dtype=self._dtype) for example in examples]
        )
        notes_teacher = torch.stack(
            [payload.notes.to(dtype=self._dtype) for payload in teacher_payloads]
        )
        if notes_student.size() != notes_teacher.size():
            raise ValueError("Student and teacher notes must have identical shapes.")
        if notes_teacher.size(-1) != self.config.notes_dim:
            raise ValueError(
                f"Notes dim mismatch. Expected {self.config.notes_dim}, got {notes_teacher.size(-1)}."
            )

        attention_mask = (student_ids != self.config.pad_token_id).long()
        planner_mask = (planner_ids != self.config.pad_token_id).long()
        commit_mask = self._build_commit_mask(attention_mask)

        teacher_snapshots = self._collate_snapshot_block(
            examples,
            snapshots_key="teacher_snapshots",
            fallback_key="notes_teacher",
            source="teacher",
        )
        student_snapshots = self._collate_snapshot_block(
            examples,
            snapshots_key="student_snapshots",
            fallback_key="notes_student",
            source="student",
        )

        plan_texts = [list(example.get("plan_items", [])) for example in examples]
        notes_text = [example.get("notes_text", "") for example in examples]
        plan_catalogs = [
            list(example.get("plan_catalog", example.get("plan_items", []))) for example in examples
        ]
        plan_catalog_roles = []
        for index, catalog in enumerate(plan_catalogs):
            raw_roles = examples[index].get("plan_catalog_roles") or []
            role_list = list(raw_roles)
            if role_list and len(role_list) != len(catalog):
                if len(role_list) < len(catalog):
                    role_list.extend([examples[index].get("role")] * (len(catalog) - len(role_list)))
                else:
                    role_list = role_list[: len(catalog)]
            if not role_list:
                role_list = [examples[index].get("role")] * len(catalog)
            plan_catalog_roles.append(role_list)
        plan_item_ids, plan_item_mask, plan_item_roles = self._encode_plan_items(
            plan_catalogs,
            plan_catalog_roles,
        )
        coverage_targets, coverage_mask = self._pad_float_sequences(
            [example.get("coverage_targets", []) for example in examples],
            plan_item_ids.size(1),
        )

        agreement_labels, agreement_mask = self._pad_vector(
            [self._ensure_tensor(example.get("agreement_labels", []), dtype=torch.long) for example in examples],
            teacher_snapshots.mask.size(1),
            pad_value=0,
        )
        metadata = [example.get("metadata", {}) for example in examples]
        example_ids = [example.get("example_id") for example in examples]

        return {
            "input_ids": student_ids,
            "attention_mask": attention_mask,
            "commit_mask": commit_mask,
            "labels": student_labels,
            "planner_ids": planner_ids,
            "planner_mask": planner_mask,
            "notes_student": notes_student,
            "notes_teacher": notes_teacher,
            "role_ids": roles,
            "teacher_notes_bus": teacher_snapshots.notes,
            "teacher_bus_mask": teacher_snapshots.mask,
            "teacher_bus_stride": teacher_snapshots.stride,
            "teacher_bus_version": teacher_snapshots.version,
            "teacher_bus_roles": teacher_snapshots.role_ids,
            "teacher_bus_coverage": teacher_snapshots.coverage,
            "student_notes_bus": student_snapshots.notes,
            "student_bus_mask": student_snapshots.mask,
            "student_bus_stride": student_snapshots.stride,
            "student_bus_version": student_snapshots.version,
            "student_bus_roles": student_snapshots.role_ids,
            "student_bus_coverage": student_snapshots.coverage,
            "agreement_labels": agreement_labels,
            "agreement_mask": agreement_mask,
            "plan_item_ids": plan_item_ids,
            "plan_item_mask": plan_item_mask,
            "plan_item_roles": plan_item_roles,
            "coverage_targets": coverage_targets,
            "coverage_mask": coverage_mask,
            "plan_text": plan_texts,
            "notes_text": notes_text,
            "metadata": metadata,
            "example_ids": example_ids,
        }

    def _pad_sequence(self, sequences: List[List[int]], pad_value: int) -> torch.Tensor:
        target_length = min(self.config.max_length, max(len(seq) for seq in sequences))
        batch = len(sequences)
        padded = torch.full((batch, target_length), pad_value, dtype=torch.long)
        for index, seq in enumerate(sequences):
            truncated = seq[:target_length]
            padded[index, : len(truncated)] = torch.tensor(truncated, dtype=torch.long)
        return padded

    def _role_id(self, role: Optional[str]) -> int:
        if role is None:
            return 0
        try:
            return self.config.role_to_id[role]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"Unknown role token provided to collator: {role!r}") from exc

    def _build_commit_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.config.commit_horizon <= 0:
            return torch.zeros_like(attention_mask, dtype=torch.bool)
        horizon = self.config.commit_horizon
        commit_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        for index, mask_row in enumerate(attention_mask):
            active_tokens = int(mask_row.sum().item())
            if active_tokens == 0:
                continue
            start = max(0, active_tokens - horizon)
            commit_mask[index, start:active_tokens] = True
        return commit_mask

    def _collate_snapshot_block(
        self,
        examples: List[Mapping[str, Any]],
        *,
        snapshots_key: str,
        fallback_key: str,
        source: str,
    ) -> "SnapshotBatch":
        fallback_shapes: List[torch.Tensor] = []
        for example in examples:
            fallback = example.get(fallback_key)
            if fallback is None:
                raise KeyError(f"Expected key {fallback_key!r} in example for snapshot fallback.")
            fallback_shapes.append(self._ensure_tensor(fallback).to(dtype=self._dtype))
        roles = fallback_shapes[0].shape[0] if fallback_shapes[0].dim() > 1 else 1
        notes_dim = fallback_shapes[0].shape[-1]
        if notes_dim != self.config.notes_dim:
            raise ValueError(
                f"Snapshot notes dim mismatch. Expected {self.config.notes_dim}, got {notes_dim}."
            )
        normalized: List[List[SnapshotFeatures]] = []
        for example, fallback in zip(examples, fallback_shapes):
            raw_snapshots = example.get(snapshots_key) or []
            normalized.append(
                self._normalize_snapshots(
                    raw_snapshots,
                    fallback=fallback,
                    roles=roles,
                    source=source,
                )
            )
        return self._stack_snapshots(normalized, roles=roles, notes_dim=notes_dim)

    def _normalize_snapshots(
        self,
        snapshots: Sequence[Any],
        *,
        fallback: torch.Tensor,
        roles: int,
        source: str,
    ) -> List[SnapshotFeatures]:
        if not snapshots:
            snapshots = [SnapshotFeatures(notes=fallback.clone(), source=source)]
        normalized: List[SnapshotFeatures] = []
        for index, snapshot in enumerate(snapshots):
            features = self._coerce_snapshot(snapshot, fallback=fallback, roles=roles, source=source, index=index)
            normalized.append(features)
        return normalized[: self.config.max_snapshots]

    def _coerce_snapshot(
        self,
        snapshot: Any,
        *,
        fallback: torch.Tensor,
        roles: int,
        source: str,
        index: int,
    ) -> SnapshotFeatures:
        if isinstance(snapshot, SnapshotFeatures):
            features = snapshot.to(dtype=self._dtype)
        elif isinstance(snapshot, Mapping):
            notes = self._ensure_tensor(snapshot.get("notes", fallback))
            stride = int(snapshot.get("stride", snapshot.get("stride_id", 0)))
            version = int(snapshot.get("version", snapshot.get("snapshot_id", index)))
            role = snapshot.get("role")
            coverage_payload = snapshot.get("coverage_flags")
            coverage_tensor = None
            if coverage_payload is not None:
                coverage_tensor = self._ensure_tensor(coverage_payload).to(dtype=self._dtype)
            features = SnapshotFeatures(
                notes=notes.to(dtype=self._dtype),
                stride=stride,
                version=version,
                role=role,
                coverage=coverage_tensor,
                source=source,
            )
        else:
            tensor = self._ensure_tensor(snapshot).to(dtype=self._dtype)
            features = SnapshotFeatures(notes=tensor, source=source, stride=0, version=index)
        padded_notes = self._pad_notes(features.notes, roles=roles, notes_dim=self.config.notes_dim)
        padded_coverage = self._pad_coverage(features.coverage, roles=roles)
        role_id = features.role
        return SnapshotFeatures(
            notes=padded_notes,
            stride=features.stride,
            version=features.version,
            role=role_id,
            coverage=padded_coverage,
            source=features.source,
        )

    def _pad_notes(self, notes: torch.Tensor, *, roles: int, notes_dim: int) -> torch.Tensor:
        if notes.dim() == 1:
            notes = notes.unsqueeze(0)
        padded = torch.zeros((roles, notes_dim), dtype=notes.dtype)
        rows = min(roles, notes.size(0))
        cols = min(notes_dim, notes.size(-1))
        padded[:rows, :cols] = notes[:rows, :cols]
        return padded

    def _pad_coverage(
        self,
        coverage: Optional[torch.Tensor],
        *,
        roles: int,
    ) -> Optional[torch.Tensor]:
        if coverage is None:
            return None
        flat = coverage.view(-1)
        padded = torch.zeros(roles, dtype=coverage.dtype)
        length = min(roles, flat.numel())
        padded[:length] = flat[:length]
        return padded

    def _stack_snapshots(
        self,
        snapshots: Sequence[Sequence[SnapshotFeatures]],
        *,
        roles: int,
        notes_dim: int,
    ) -> "SnapshotBatch":
        batch_size = len(snapshots)
        max_snapshots = max(1, self.config.max_snapshots)
        notes_tensor = torch.zeros(
            (batch_size, max_snapshots, roles, notes_dim),
            dtype=self._dtype,
        )
        stride_tensor = torch.zeros((batch_size, max_snapshots), dtype=torch.long)
        version_tensor = torch.zeros((batch_size, max_snapshots), dtype=torch.long)
        coverage_tensor = torch.zeros((batch_size, max_snapshots, roles), dtype=self._dtype)
        mask_tensor = torch.zeros((batch_size, max_snapshots), dtype=torch.bool)
        role_tensor = torch.full((batch_size, max_snapshots), fill_value=-1, dtype=torch.long)
        for batch_index, snapshot_list in enumerate(snapshots):
            truncated = list(snapshot_list)[:max_snapshots]
            for snapshot_index, snapshot in enumerate(truncated):
                mask_tensor[batch_index, snapshot_index] = True
                notes_tensor[batch_index, snapshot_index] = snapshot.notes
                stride_tensor[batch_index, snapshot_index] = snapshot.stride
                version_tensor[batch_index, snapshot_index] = snapshot.version
                if snapshot.coverage is not None:
                    coverage_tensor[batch_index, snapshot_index] = snapshot.coverage
                if snapshot.role is not None:
                    try:
                        role_tensor[batch_index, snapshot_index] = self._role_id(snapshot.role)
                    except ValueError:
                        role_tensor[batch_index, snapshot_index] = -1
        return SnapshotBatch(
            notes=notes_tensor,
            stride=stride_tensor,
            version=version_tensor,
            coverage=coverage_tensor,
            mask=mask_tensor,
            role_ids=role_tensor,
        )

    def _pad_vector(
        self,
        tensors: Sequence[torch.Tensor],
        target_length: int,
        *,
        pad_value: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(tensors)
        padded = torch.full((batch_size, target_length), pad_value, dtype=torch.long)
        mask = torch.zeros((batch_size, target_length), dtype=torch.bool)
        for index, tensor in enumerate(tensors):
            vector = tensor.view(-1)
            length = min(target_length, vector.numel())
            if length == 0:
                continue
            padded[index, :length] = vector[:length]
            mask[index, :length] = True
        return padded, mask

    def _encode_plan_items(
        self,
        plan_texts: Sequence[Sequence[str]],
        plan_roles: Sequence[Sequence[Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(plan_texts)
        max_items = max((len(items) for items in plan_texts), default=0)
        if max_items == 0:
            return (
                torch.zeros((batch_size, 1), dtype=torch.long),
                torch.zeros((batch_size, 1), dtype=torch.bool),
                torch.full((batch_size, 1), fill_value=-1, dtype=torch.long),
            )
        plan_ids = torch.zeros((batch_size, max_items), dtype=torch.long)
        mask = torch.zeros((batch_size, max_items), dtype=torch.bool)
        role_ids = torch.full((batch_size, max_items), fill_value=-1, dtype=torch.long)
        for batch_index, items in enumerate(plan_texts):
            for item_index, text in enumerate(items[:max_items]):
                mask[batch_index, item_index] = True
                plan_ids[batch_index, item_index] = self._hash_plan_text(text)
                role_value = None
                if plan_roles and batch_index < len(plan_roles):
                    roles_for_batch = plan_roles[batch_index]
                    if item_index < len(roles_for_batch):
                        role_value = roles_for_batch[item_index]
                role_ids[batch_index, item_index] = self._plan_role_to_id(role_value)
        return plan_ids, mask, role_ids

    def _hash_plan_text(self, text: str) -> int:
        digest = hashlib.sha256(text.lower().encode("utf-8")).hexdigest()
        hashed = int(digest, 16) % self.config.plan_hash_buckets
        return hashed if hashed != 0 else 1

    def _pad_float_sequences(
        self,
        sequences: Sequence[Sequence[float]],
        target_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if target_length <= 0:
            target_length = 1
        batch_size = len(sequences)
        tensor = torch.zeros((batch_size, target_length), dtype=torch.float32)
        mask = torch.zeros((batch_size, target_length), dtype=torch.bool)
        for index, seq in enumerate(sequences):
            if not seq:
                continue
            length = min(target_length, len(seq))
            tensor[index, :length] = torch.tensor(seq[:length], dtype=torch.float32)
            mask[index, :length] = True
        return tensor, mask

    def _plan_role_to_id(self, role_value: Any) -> int:
        if role_value is None:
            return -1
        if isinstance(role_value, int):
            return role_value
        if isinstance(role_value, str):
            role_normalized = role_value.strip().lower()
            if not role_normalized:
                return -1
            try:
                return self._role_id(role_normalized)
            except ValueError:
                return -1
        return -1

    def _ensure_list(self, tensor_like: Any) -> List[int]:
        if isinstance(tensor_like, torch.Tensor):
            return tensor_like.view(-1).tolist()
        if isinstance(tensor_like, list):
            return tensor_like
        if isinstance(tensor_like, (tuple, range)):
            return list(tensor_like)
        raise TypeError(f"Unsupported sequence type: {type(tensor_like)!r}")

    def _ensure_tensor(self, value: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype) if dtype else value
        tensor = torch.tensor(value, dtype=dtype or torch.float32)
        return tensor


@dataclass(slots=True)
class SnapshotBatch:
    """Packed snapshot tensors for either teacher or student branches."""

    notes: torch.Tensor
    stride: torch.Tensor
    version: torch.Tensor
    coverage: torch.Tensor
    mask: torch.Tensor
    role_ids: torch.Tensor


def _resolve_dtype(alias: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[alias]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported dtype alias: {alias!r}") from exc


__all__ = [
    "TwoBranchKnowledgeDistillationCollator",
    "TwoBranchKDCollatorConfig",
    "SnapshotBatch",
]
