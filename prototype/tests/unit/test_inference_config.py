"""Tests for inference configuration builders."""

from __future__ import annotations

from nstream_transformer.inference.config import (
    CadencePolicyConfig,
    DecodeConfig,
    GateAnnealingConfig,
    InferenceConfig,
    build_inference_config,
)


class _StubCurriculum:
    def __init__(self, *, B: int, L: int, delta: int, rho_by_role: dict[str, float]) -> None:
        self.B = B
        self.L = L
        self.delta = delta
        self.rho_by_role = dict(rho_by_role)


class _StubTrainingConfig:
    def __init__(self, *, curriculum: _StubCurriculum, agreement_threshold: float) -> None:
        self.curriculum = curriculum
        self.agreement_threshold = agreement_threshold


def test_build_inference_config_honours_overrides() -> None:
    training_cfg = _StubTrainingConfig(
        curriculum=_StubCurriculum(B=3, L=12, delta=2, rho_by_role={"intro": 2.0}),
        agreement_threshold=0.2,
    )
    decode_cfg = DecodeConfig()

    config = build_inference_config(
        training_cfg,
        role_to_id={"intro": 0, "core": 1},
        decode_config=decode_cfg,
        emission_cadence={"core": 5},
        gate_g=0.75,
        max_snapshots=6,
        rng_seed=123,
        logit_blend_alpha=0.6,
        gate_annealing={"enabled": True, "min_value": 0.25, "decay": 0.5},
        cadence_policy={"mode": "stochastic", "min_probability": 0.2, "max_interval": 4},
        hierarchy_levels=(("intro",), ("core",)),
    )

    assert isinstance(config, InferenceConfig)
    assert config.roles == ("intro", "core")
    assert config.stride_B == 3
    assert config.commit_L == 12
    assert config.read_lag_delta == 2
    assert config.max_snapshots_K == 6
    assert config.topology == "hierarchical"
    assert config.hierarchy_levels == (("intro",), ("core",))
    assert config.cadence_for("intro") == 2
    assert config.cadence_for("core") == 5
    assert config.gate_g == 0.75
    assert config.agreement_threshold_tau == 0.2
    assert config.logit_blend_alpha == 0.6
    assert config.decode is decode_cfg
    assert config.decode.seed == 123
    assert config.rng_seed == 123
    assert isinstance(config.gate_annealing, GateAnnealingConfig)
    assert config.gate_annealing.min_value == 0.25
    assert isinstance(config.cadence_policy, CadencePolicyConfig)
    assert config.cadence_policy.mode == "stochastic"
    assert config.cadence_policy.min_probability == 0.2
    assert config.cadence_policy.max_interval == 4
