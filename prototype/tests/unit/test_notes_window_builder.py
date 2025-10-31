"""Tests covering Dynamic Notes Bus window construction logic."""

from __future__ import annotations

import torch

from nstream_transformer.inference import (
    DynamicNotesBus,
    DynamicNotesBusConfig,
    InferenceConfig,
    NotesWindowBuilder,
)
from nstream_transformer.inference.state import RoleState


def _make_state(role: str) -> RoleState:
    input_ids = torch.tensor([[101, 102]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return RoleState(
        role=role,
        input_ids=input_ids,
        attention_mask=attention_mask,
        commit_stride=2,
        commit_horizon=4,
    )


def _make_bus(snapshot_dim: int = 4) -> DynamicNotesBus:
    return DynamicNotesBus(
        DynamicNotesBusConfig(
            snapshot_dim=snapshot_dim,
            max_snapshots=8,
            lag=0,
            dtype="float32",
        )
    )


def _push_snapshot(bus: DynamicNotesBus, value: float, *, stride: int = 1) -> None:
    notes = torch.full((snapshot_dim(bus),), value, dtype=torch.float32)
    bus.push(notes, stride=stride)


def snapshot_dim(bus: DynamicNotesBus) -> int:
    return bus.config.snapshot_dim


def test_notes_window_builder_applies_lag_and_monotonicity() -> None:
    config = InferenceConfig(
        roles=("intro", "core"),
        stride_B=3,
        commit_L=12,
        read_lag_delta=1,
        max_snapshots_K=2,
        emission_cadence_M_by_role={"intro": 3, "core": 3},
    )
    builder = NotesWindowBuilder.from_config(
        config,
        notes_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    bus_intro = _make_bus()
    bus_core = _make_bus()
    for idx, value in enumerate((1.0, 2.0, 3.0), start=1):
        _push_snapshot(bus_intro, value, stride=idx)

    state = _make_state("core")
    window = builder.build(state, {"intro": bus_intro, "core": bus_core})

    assert window.notes.shape == (1, 2, 4)
    assert window.mask.shape == (1, 2)
    assert window.mask.dtype == torch.bool
    assert window.producers == ("intro", "intro")
    assert window.versions.tolist() == [1, 2]
    assert torch.allclose(window.notes[0, 0], torch.full((4,), 1.0))
    assert torch.allclose(window.notes[0, 1], torch.full((4,), 2.0))

    for version in window.versions.tolist():
        state.update_last_seen_version("intro", version)

    _push_snapshot(bus_intro, 4.0, stride=4)
    window_after_update = builder.build(state, {"intro": bus_intro, "core": bus_core})

    assert window_after_update.notes.shape == (1, 1, 4)
    assert window_after_update.producers == ("intro",)
    assert window_after_update.versions.tolist() == [3]
    assert torch.allclose(window_after_update.notes[0, 0], torch.full((4,), 3.0))


def test_notes_window_builder_hierarchical_topology() -> None:
    config = InferenceConfig(
        roles=("intro", "core", "wrap"),
        stride_B=2,
        commit_L=8,
        read_lag_delta=0,
        max_snapshots_K=3,
        topology="hierarchical",
        hierarchy_levels=(("intro",), ("core",), ("wrap",)),
        emission_cadence_M_by_role={"intro": 2, "core": 2, "wrap": 2},
    )
    builder = NotesWindowBuilder.from_config(
        config,
        notes_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    bus_map = {role: _make_bus() for role in config.roles}
    _push_snapshot(bus_map["intro"], 10.0, stride=1)
    _push_snapshot(bus_map["core"], 20.0, stride=1)

    state = _make_state("wrap")
    window = builder.build(state, bus_map)

    assert window.notes.shape == (1, 2, 4)
    assert window.producers == ("intro", "core")
    assert torch.allclose(window.notes[0, 0], torch.full((4,), 10.0))
    assert torch.allclose(window.notes[0, 1], torch.full((4,), 20.0))
    assert window.versions.tolist() == [1, 1]
