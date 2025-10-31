"""Inference runtime configuration for the N-Stream Transformer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple, Literal, Any, Sequence

# Avoid importing training during inference module import to prevent cycles.
# Use a structural type at call sites; TrainingConfig is only needed for
# attribute access (curriculum, agreement_threshold).
from typing import Any as TrainingConfig  # type: ignore


@dataclass(slots=True)
class DecodeConfig:
    """Sampling controls used during inference-time decoding."""

    temperature: float = 0.7
    top_k: int = 0
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    max_new_tokens: int = 512
    do_sample: bool = True
    seed: Optional[int] = None

    def as_sampling_kwargs(self) -> Dict[str, object]:
        """Return kwargs compatible with Hugging Face generation utilities."""

        payload: Dict[str, object] = {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        return payload


@dataclass(slots=True)
class GateAnnealingConfig:
    """Runtime annealing policy for the SNC gate."""

    enabled: bool = True
    decay: float = 0.6  # multiplicative drop when volatility detected
    min_value: float = 0.1  # lower bound for the gate scaling factor
    recovery: float = 0.05  # additive recovery per stable emission
    stability_margin: float = 0.05  # margin above tau treated as stable
    cooldown: int = 1  # emissions to wait before recovering


@dataclass(slots=True)
class CadencePolicyConfig:
    """Controls how note emission cadence is decided at inference time."""

    mode: Literal["deterministic", "stochastic", "adaptive"] = "deterministic"
    min_probability: float = 1e-4
    max_interval: int = 0
    multiplier_min: float = 0.5
    multiplier_max: float = 2.0
    agreement_low: float = 0.25
    agreement_high: float = 0.6
    age_boost: float = 0.0


@dataclass(slots=True)
class InferenceConfig:
    """Configuration encapsulating runtime behaviour for multi-role decoding."""

    roles: Tuple[str, ...]
    stride_B: int
    commit_L: int
    read_lag_delta: int
    max_snapshots_K: int
    topology: Literal["triangular", "hierarchical"] = "triangular"
    gate_g: float = 1.0
    agreement_threshold_tau: float = 0.15
    emission_cadence_M_by_role: Dict[str, int] = field(default_factory=dict)
    logit_blend_alpha: float = 1.0
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    rng_seed: Optional[int] = None
    gate_annealing: GateAnnealingConfig = field(default_factory=GateAnnealingConfig)
    cadence_policy: CadencePolicyConfig = field(default_factory=CadencePolicyConfig)
    topology_hierarchy: Optional[Dict[str, Tuple[str, ...]]] = None
    hierarchy_levels: Optional[Tuple[Tuple[str, ...], ...]] = None

    def __post_init__(self) -> None:
        if not self.roles:
            raise ValueError("InferenceConfig.roles must contain at least one role.")
        self.roles = tuple(role.lower() for role in self.roles)
        self._validate_integer("stride_B", self.stride_B, minimum=1)
        self._validate_integer("commit_L", self.commit_L, minimum=1)
        self._validate_integer("read_lag_delta", self.read_lag_delta, minimum=0)
        self._validate_integer("max_snapshots_K", self.max_snapshots_K, minimum=1)
        if self.topology not in {"triangular", "hierarchical"}:
            raise ValueError("InferenceConfig.topology must be one of {'triangular', 'hierarchical'}.")
        if not 0.0 <= self.gate_g <= 1.0:
            raise ValueError("InferenceConfig.gate_g must be within [0, 1].")
        if self.agreement_threshold_tau <= 0.0 or self.agreement_threshold_tau >= 1.0:
            raise ValueError("InferenceConfig.agreement_threshold_tau must lie within (0, 1).")
        if not 0.0 <= self.logit_blend_alpha <= 1.0:
            raise ValueError("InferenceConfig.logit_blend_alpha must lie within [0, 1].")
        # Normalise cadence dictionary to contain every role.
        cadence: Dict[str, int] = {}
        for role in self.roles:
            cadence_value = self.emission_cadence_M_by_role.get(role, self.stride_B)
            cadence_value = int(round(cadence_value))
            if cadence_value <= 0:
                cadence_value = self.stride_B
            cadence[role] = cadence_value
        self.emission_cadence_M_by_role = cadence
        self._validate_gate_policy()
        self._validate_cadence_policy()
        self._normalise_hierarchy()

    def cadence_for(self, role: str) -> int:
        """Return the note emission cadence for a role."""

        try:
            return self.emission_cadence_M_by_role[role.lower()]
        except KeyError as exc:  # pragma: no cover - defensive path
            raise ValueError(f"Unknown role: {role!r}") from exc

    @staticmethod
    def _validate_integer(name: str, value: int, *, minimum: int) -> None:
        if value < minimum:
            raise ValueError(f"InferenceConfig.{name} must be >= {minimum}, received {value}.")

    def _validate_gate_policy(self) -> None:
        policy = self.gate_annealing
        if not policy.enabled:
            return
        if not 0.0 <= policy.min_value <= 1.0:
            raise ValueError("Gate annealing min_value must lie within [0,1].")
        if not 0.0 < policy.decay <= 1.0:
            raise ValueError("Gate annealing decay must lie within (0,1].")
        if not 0.0 <= policy.recovery <= 1.0:
            raise ValueError("Gate annealing recovery must lie within [0,1].")
        if policy.cooldown < 0:
            raise ValueError("Gate annealing cooldown must be non-negative.")
        if not 0.0 <= policy.stability_margin < 1.0:
            raise ValueError("Gate annealing stability_margin must lie within [0,1).")
        if policy.min_value > self.gate_g:
            raise ValueError("Gate annealing min_value cannot exceed the base gate_g.")

    def _validate_cadence_policy(self) -> None:
        policy = self.cadence_policy
        if policy.mode not in {"deterministic", "stochastic", "adaptive"}:
            raise ValueError(
                "Cadence policy mode must be one of {'deterministic','stochastic','adaptive'}."
            )
        if policy.min_probability <= 0.0 or policy.min_probability > 1.0:
            raise ValueError("Cadence min_probability must lie within (0,1].")
        if policy.max_interval < 0:
            raise ValueError("Cadence max_interval must be non-negative.")
        if policy.multiplier_min <= 0.0:
            raise ValueError("Cadence multiplier_min must be positive.")
        if policy.multiplier_max < policy.multiplier_min:
            raise ValueError("Cadence multiplier_max must be >= multiplier_min.")
        if not 0.0 <= policy.agreement_low <= 1.0:
            raise ValueError("Cadence agreement_low must lie within [0,1].")
        if not 0.0 <= policy.agreement_high <= 1.0:
            raise ValueError("Cadence agreement_high must lie within [0,1].")
        if policy.agreement_high <= policy.agreement_low:
            raise ValueError("Cadence agreement_high must exceed agreement_low.")
        if policy.age_boost < 0.0:
            raise ValueError("Cadence age_boost must be non-negative.")

    def _normalise_hierarchy(self) -> None:
        if self.topology != "hierarchical":
            self.topology_hierarchy = None
            self.hierarchy_levels = None
            return

        if self.hierarchy_levels is not None:
            self.hierarchy_levels = self._normalise_levels(self.hierarchy_levels)
            derived = self._derive_hierarchy_from_levels(self.hierarchy_levels)
            if self.topology_hierarchy is None:
                self.topology_hierarchy = derived
            else:
                normalised_map = self._normalise_hierarchy_map(self.topology_hierarchy)
                if normalised_map != derived:
                    raise ValueError("Provided topology_hierarchy does not match hierarchy_levels.")
                self.topology_hierarchy = normalised_map
        elif self.topology_hierarchy is not None:
            self.topology_hierarchy = self._normalise_hierarchy_map(self.topology_hierarchy)
            self.hierarchy_levels = self._derive_levels_from_hierarchy(self.topology_hierarchy)
        else:
            raise ValueError("Hierarchical topology requires hierarchy_levels or topology_hierarchy.")

    def _normalise_levels(self, levels: Tuple[Tuple[str, ...], ...]) -> Tuple[Tuple[str, ...], ...]:
        if not levels:
            raise ValueError("hierarchy_levels must not be empty for hierarchical topology.")
        seen: set[str] = set()
        normalised: list[Tuple[str, ...]] = []
        for level in levels:
            if not level:
                raise ValueError("hierarchy_levels entries must not be empty.")
            normalised_level = tuple(role.lower() for role in level)
            for role in normalised_level:
                if role not in self.roles:
                    raise ValueError(f"Unknown role {role!r} referenced in hierarchy_levels.")
                if role in seen:
                    raise ValueError(f"Role {role!r} appears multiple times in hierarchy_levels.")
                seen.add(role)
            normalised.append(normalised_level)
        if seen != set(self.roles):
            missing = set(self.roles) - seen
            raise ValueError(f"hierarchy_levels missing roles: {sorted(missing)}.")
        return tuple(normalised)

    def _normalise_hierarchy_map(
        self, mapping: Mapping[str, Sequence[str]]
    ) -> Dict[str, Tuple[str, ...]]:
        normalised: Dict[str, Tuple[str, ...]] = {}
        for consumer, producers in mapping.items():
            consumer_norm = consumer.lower()
            if consumer_norm not in self.roles:
                raise ValueError(f"Unknown consumer role {consumer!r} in topology_hierarchy.")
            producer_list: list[str] = []
            seen = set()
            for producer in producers:
                producer_norm = producer.lower()
                if producer_norm not in self.roles:
                    raise ValueError(f"Unknown producer role {producer!r} in topology_hierarchy.")
                if producer_norm == consumer_norm:
                    raise ValueError("A role cannot list itself as a producer in topology_hierarchy.")
                if producer_norm not in seen:
                    producer_list.append(producer_norm)
                    seen.add(producer_norm)
            normalised[consumer_norm] = tuple(producer_list)
        return normalised

    def _derive_hierarchy_from_levels(
        self, levels: Tuple[Tuple[str, ...], ...]
    ) -> Dict[str, Tuple[str, ...]]:
        producers: Dict[str, Tuple[str, ...]] = {}
        seen: list[str] = []
        for level in levels:
            for role in level:
                producers[role] = tuple(seen)
            seen.extend(level)
        return producers

    def _derive_levels_from_hierarchy(
        self, mapping: Mapping[str, Tuple[str, ...]]
    ) -> Tuple[Tuple[str, ...], ...]:
        # Simple topological layering: start with roles that have no producers,
        # then peel layers where all producers already assigned.
        remaining = {role: set(producers) for role, producers in mapping.items()}
        for role in self.roles:
            remaining.setdefault(role, set())
        assigned: set[str] = set()
        levels: list[Tuple[str, ...]] = []
        while remaining:
            current_level = tuple(sorted(role for role, deps in remaining.items() if deps <= assigned))
            if not current_level:
                cycle = ", ".join(sorted(remaining.keys()))
                raise ValueError(f"Cannot derive hierarchy levels; cycle detected among: {cycle}.")
            levels.append(current_level)
            for role in current_level:
                assigned.add(role)
                remaining.pop(role, None)
        return tuple(levels)


def build_inference_config(
    training_config: TrainingConfig,
    *,
    role_to_id: Mapping[str, int],
    topology: Optional[Literal["triangular", "hierarchical"]] = None,
    decode_config: Optional[DecodeConfig] = None,
    emission_cadence: Optional[Mapping[str, float]] = None,
    gate_g: Optional[float] = None,
    max_snapshots: Optional[int] = None,
    rng_seed: Optional[int] = None,
    logit_blend_alpha: Optional[float] = None,
    gate_annealing: Optional[Any] = None,
    cadence_policy: Optional[Any] = None,
    hierarchy_levels: Optional[Sequence[Sequence[str]]] = None,
    topology_hierarchy: Optional[Mapping[str, Sequence[str]]] = None,
    read_lag_delta: Optional[int] = None,
) -> InferenceConfig:
    """Derive an :class:`InferenceConfig` aligned with the training curriculum."""

    if not role_to_id:
        raise ValueError("role_to_id mapping must not be empty.")
    ordered_roles = tuple(role.lower() for role, _ in sorted(role_to_id.items(), key=lambda item: item[1]))

    curriculum = training_config.curriculum
    stride_B = int(curriculum.B)
    commit_L = int(curriculum.L)
    read_lag = int(read_lag_delta) if read_lag_delta is not None else int(curriculum.delta)
    max_snapshots_K = int(max_snapshots) if max_snapshots is not None else max(stride_B, 4)

    cadence_payload: Dict[str, float] = {
        role.lower(): float(value) for role, value in curriculum.rho_by_role.items()
    }
    if emission_cadence is not None:
        cadence_payload.update({role.lower(): float(value) for role, value in emission_cadence.items()})
    cadence_int: Dict[str, int] = {
        role: _coerce_positive_int(cadence_payload.get(role, stride_B), fallback=stride_B)
        for role in ordered_roles
    }

    hierarchy_levels_payload: Optional[Tuple[Tuple[str, ...], ...]] = None
    if hierarchy_levels is not None:
        hierarchy_levels_payload = tuple(
            tuple(str(role).lower() for role in level)
            for level in hierarchy_levels
        )

    topology_hierarchy_payload: Optional[Dict[str, Tuple[str, ...]]] = None
    if topology_hierarchy is not None:
        topology_hierarchy_payload = {
            str(consumer).lower(): tuple(str(producer).lower() for producer in producers)
            for consumer, producers in topology_hierarchy.items()
        }

    if topology is not None:
        topology_value: Literal["triangular", "hierarchical"] = topology
    elif hierarchy_levels_payload is not None or topology_hierarchy_payload is not None:
        topology_value = "hierarchical"
    else:
        topology_value = "triangular"

    if isinstance(gate_annealing, Mapping):
        gate_policy = GateAnnealingConfig(**gate_annealing)  # type: ignore[arg-type]
    elif isinstance(gate_annealing, GateAnnealingConfig):
        gate_policy = gate_annealing
    elif gate_annealing is None:
        gate_policy = GateAnnealingConfig()
    else:  # pragma: no cover - defensive guard
        raise TypeError("gate_annealing must be a Mapping, GateAnnealingConfig, or None.")

    if isinstance(cadence_policy, Mapping):
        cadence_policy_cfg = CadencePolicyConfig(**cadence_policy)  # type: ignore[arg-type]
    elif isinstance(cadence_policy, CadencePolicyConfig):
        cadence_policy_cfg = cadence_policy
    elif cadence_policy is None:
        cadence_policy_cfg = CadencePolicyConfig()
    else:  # pragma: no cover - defensive guard
        raise TypeError("cadence_policy must be a Mapping, CadencePolicyConfig, or None.")

    if decode_config is None:
        decode_cfg = DecodeConfig(seed=rng_seed)
    else:
        decode_cfg = decode_config
        if rng_seed is not None and decode_cfg.seed is None:
            decode_cfg.seed = rng_seed

    config = InferenceConfig(
        roles=ordered_roles,
        stride_B=stride_B,
        commit_L=commit_L,
        read_lag_delta=read_lag,
        max_snapshots_K=max_snapshots_K,
        topology=topology_value,
        gate_g=gate_g if gate_g is not None else 1.0,
        agreement_threshold_tau=float(training_config.agreement_threshold),
        emission_cadence_M_by_role=cadence_int,
        logit_blend_alpha=logit_blend_alpha if logit_blend_alpha is not None else 1.0,
        decode=decode_cfg,
        rng_seed=rng_seed,
        gate_annealing=gate_policy,
        cadence_policy=cadence_policy_cfg,
        hierarchy_levels=hierarchy_levels_payload,
        topology_hierarchy=topology_hierarchy_payload,
    )
    return config


def _coerce_positive_int(value: float, *, fallback: int) -> int:
    result = int(round(value))
    if result <= 0:
        return int(fallback)
    return result


__all__ = [
    "CadencePolicyConfig",
    "DecodeConfig",
    "GateAnnealingConfig",
    "InferenceConfig",
    "build_inference_config",
]
