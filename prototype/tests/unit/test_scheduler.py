"""Deterministic scheduling tests for multi-role decoding."""

from __future__ import annotations

from nstream_transformer.inference.scheduler import TriangularScheduler


def test_triangular_scheduler_cycles_roles_per_stride() -> None:
    scheduler = TriangularScheduler(["intro", "core", "wrap"], stride=2)

    role_sequence: list[str] = []
    token_indices: dict[str, list[int]] = {"intro": [], "core": [], "wrap": []}
    stride_markers: list[int] = []

    for _ in range(6):
        tick = scheduler.tick()
        role_sequence.append(tick.role)
        token_indices[tick.role].append(tick.token_index)
        stride_markers.append(tick.stride_index)
        scheduler.advance()

    assert role_sequence == ["intro", "intro", "core", "core", "wrap", "wrap"]
    for role, indices in token_indices.items():
        expected = [0, 1]
        assert indices == expected
    assert stride_markers[0] == stride_markers[1] == 0
    assert stride_markers[2] == stride_markers[3] == 0
    assert stride_markers[4] == stride_markers[5] == 0

    # Next tick should advance to stride 1 and restart cycle.
    tick = scheduler.tick()
    assert tick.stride_index == 1
    assert tick.role == "intro"
    assert tick.token_index == 0
