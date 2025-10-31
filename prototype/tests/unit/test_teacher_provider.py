from __future__ import annotations

import hashlib
from typing import Mapping

import torch

from nstream_transformer.data.teacher_provider import (
    CachedTeacherNotesProvider,
    TeacherNotesProvider,
)
from nstream_transformer.data.teacher_runner import (
    TeacherRunnerConfig,
    TeacherRunResult,
    TeacherSnapshotText,
)


class DummyEmbedder:
    """Deterministic embedder for testing without heavy dependencies."""

    def __init__(self) -> None:
        self.calls = 0

    def aggregate(self, texts: list[str], target_dim: int) -> torch.Tensor:
        self.calls += 1
        vector = torch.zeros(target_dim, dtype=torch.float32)
        if not texts:
            raise ValueError("DummyEmbedder received empty texts.")
        for offset, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            for index, byte in enumerate(digest):
                position = (index + offset) % target_dim
                value = 1.0 if byte % 2 == 0 else -1.0
                vector[position] += value
        return torch.nn.functional.normalize(vector, dim=0)


def _build_example() -> dict[str, object]:
    paragraphs = [
        "Intro context paragraph that sets the scene.",
        "Key definitions and framing.",
        "Core argument A describing primary evidence.",
        "Core argument B expanding with detail.",
        "Supporting evidence with citations.",
        "Wrap up paragraph that summarises implications.",
    ]
    metadata = {
        "document_text": "\n\n".join(paragraphs),
        "document_paragraphs": paragraphs,
        "teacher_plan": {
            "plan": [
                {
                    "role": "intro",
                    "summary": "Set context and definitions.",
                    "notes": ["Intro context", "Key definitions"],
                },
                {
                    "role": "core",
                    "summary": "Present core evidence.",
                    "notes": ["Core argument A", "Core argument B", "Supporting evidence"],
                },
                {
                    "role": "wrap",
                    "summary": "Summarise and call to action.",
                    "notes": ["Summarise implications", "Call to action"],
                },
            ],
            "segments": [
                {"role": "intro", "paragraph_start": 0, "paragraph_end": 2},
                {"role": "core", "paragraph_start": 2, "paragraph_end": 5},
                {"role": "wrap", "paragraph_start": 5, "paragraph_end": 6},
            ],
        },
        "title": "Sample Article",
    }
    return {
        "example_id": "example-123",
        "student_ids": torch.arange(24),
        "notes_student": torch.zeros((3, 6)),
        "role": "core",
        "metadata": metadata,
        "stride_ids": torch.tensor([6, 12, 18]),
        "plan_items": ["Core argument A", "Summarise implications"],
        "coverage_targets": [1, 0],
    }


def _build_teacher_run(example_id: str = "example-123") -> TeacherRunResult:
    role_notes = {
        "intro": ["Intro context", "Key definitions"],
        "core": ["Core argument A", "Core argument B", "Supporting evidence"],
        "wrap": ["Summarise implications", "Call to action"],
    }
    snapshots = [
        TeacherSnapshotText(
            version=0,
            stride=6,
            role_notes={role: notes[:1] for role, notes in role_notes.items()},
            coverage={role: 1 / len(notes) for role, notes in role_notes.items()},
        ),
        TeacherSnapshotText(
            version=1,
            stride=12,
            role_notes={
                "intro": role_notes["intro"],
                "core": role_notes["core"][:2],
                "wrap": role_notes["wrap"],
            },
            coverage={
                "intro": 1.0,
                "core": 2 / len(role_notes["core"]),
                "wrap": 1.0,
            },
        ),
        TeacherSnapshotText(
            version=2,
            stride=18,
            role_notes=role_notes,
            coverage={role: 1.0 for role in role_notes},
        ),
    ]
    teacher_plan = _build_example()["metadata"]["teacher_plan"]
    return TeacherRunResult(
        example_id=example_id,
        role_notes=role_notes,
        snapshots=snapshots,
        teacher_plan=teacher_plan,
    )


class StaticTeacherRunner:
    def __init__(self, result: TeacherRunResult) -> None:
        self.result = result
        self.calls = 0

    def run(self, example: Mapping[str, Any]) -> TeacherRunResult:
        self.calls += 1
        return self.result


def test_on_demand_teacher_provider_generates_snapshots() -> None:
    config = TeacherRunnerConfig(provider="mock", model="mock-teacher", cache_dir=None, max_snapshots=3)
    embedder = DummyEmbedder()
    provider = TeacherNotesProvider(
        config,
        notes_dim=6,
        role_to_id={"intro": 0, "core": 1, "wrap": 2},
        embedder=embedder,
        runner=StaticTeacherRunner(_build_teacher_run()),
    )
    example = _build_example()

    payload = provider.fetch(example)

    assert payload.notes.shape == torch.Size([3, 6])
    assert len(payload.snapshots) == 3
    assert all(snapshot.notes.shape == torch.Size([3, 6]) for snapshot in payload.snapshots)
    assert [snapshot.stride for snapshot in payload.snapshots] == [6, 12, 18]
    assert embedder.calls >= len(payload.snapshots) + 1  # base matrix + snapshots
    # Coverage should monotonically increase for the core role (index 1).
    coverage_values = [snapshot.coverage[1].item() if snapshot.coverage is not None else 0.0 for snapshot in payload.snapshots]
    assert coverage_values[0] <= coverage_values[-1]


def test_cached_teacher_provider_reuses_embedded_results(tmp_path) -> None:
    config = TeacherRunnerConfig(
        provider="mock",
        model="mock-teacher",
        cache_dir=str(tmp_path),
        max_snapshots=2,
        id_field="example_id",
    )
    embedder = DummyEmbedder()
    backend = TeacherNotesProvider(
        config,
        notes_dim=8,
        role_to_id={"intro": 0, "core": 1, "wrap": 2},
        embedder=embedder,
        runner=StaticTeacherRunner(_build_teacher_run()),
    )
    cached = CachedTeacherNotesProvider(
        backend=backend,
        cache_dir=tmp_path,
        id_field="example_id",
    )
    example = _build_example()

    first_payload = cached.fetch(example)
    calls_after_first = embedder.calls
    second_payload = cached.fetch(example)

    assert embedder.calls == calls_after_first
    assert torch.allclose(first_payload.notes, second_payload.notes)
    assert len(first_payload.snapshots) == len(second_payload.snapshots)
