"""Inference orchestrator wiring Dynamic Notes Bus and SNC for multi-role decoding."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

try:  # pragma: no cover - optional dependency for typing only
    from transformers.tokenization_utils import PreTrainedTokenizerBase
except ImportError:  # pragma: no cover - allows running without transformers during tests
    PreTrainedTokenizerBase = object  # type: ignore[misc,assignment]

from .config import DecodeConfig, InferenceConfig
from .dnb_bus import DynamicNotesBus, DynamicNotesBusConfig
from .scheduler import TriangularScheduler
from .state import PastKeyValues, RoleState
from .window import NotesWindowBuilder, TopologyMask


@dataclass(frozen=True, slots=True)
class AgreementResult:
    """Outcome of evaluating an agreement gate."""

    score: float
    triggered: bool


class AgreementGate:
    """Applies an agreement threshold on attended hidden states."""

    def __init__(self, threshold: float) -> None:
        if threshold <= 0.0 or threshold >= 1.0:
            raise ValueError("AgreementGate threshold must lie inside (0, 1).")
        self.threshold = threshold

    def evaluate(self, agreement_tensor: torch.Tensor) -> AgreementResult:
        if agreement_tensor.numel() == 0:
            raise ValueError("Agreement head returned an empty tensor.")
        score = float(agreement_tensor.detach().mean().item())
        return AgreementResult(score=score, triggered=score < self.threshold)


@dataclass(slots=True)
class StepOutcome:
    """Telemetry describing a single decode step."""

    role: str
    token_id: int
    token_text: str
    stride_index: int
    stride_completed: bool
    role_completed: bool
    agreement: float
    notes_emitted: bool
    rollback_performed: bool
    cadence_mode: Optional[str] = None
    cadence_probability: Optional[float] = None
    cadence_multiplier: Optional[float] = None
    cadence_forced: bool = False
    coverage_logits: Optional[List[float]] = None


class MultiStreamOrchestrator:
    """Drives synchronous multi-role decoding with Dynamic Notes Bus + SNC."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        config: InferenceConfig,
        *,
        topology_mask: Optional[TopologyMask] = None,
        decode_config: Optional[DecodeConfig] = None,
        logit_blend_alpha: Optional[float] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.decode_config = decode_config or config.decode
        self._validate_decode_config(self.decode_config)
        alpha_source = config.logit_blend_alpha if logit_blend_alpha is None else logit_blend_alpha
        self.alpha = float(max(0.0, min(1.0, alpha_source)))

        self.device = self._resolve_device()
        self.dtype = self._resolve_dtype()

        notes_dim = self._resolve_notes_dim()
        bus_dtype = self._resolve_bus_dtype()

        self.window_builder = NotesWindowBuilder.from_config(
            config,
            notes_dim,
            topology_mask=topology_mask,
            device=self.device,
            dtype=self.dtype,
        )
        self.scheduler = TriangularScheduler(
            config.roles,
            stride=config.stride_B,
            levels=config.hierarchy_levels,
        )
        self.agreement_gate = AgreementGate(config.agreement_threshold_tau)

        self.bus_by_role: Dict[str, DynamicNotesBus] = {
            role: DynamicNotesBus(
                DynamicNotesBusConfig(
                    snapshot_dim=notes_dim,
                    max_snapshots=config.max_snapshots_K,
                    lag=config.read_lag_delta,
                    dtype=bus_dtype,
                    device=str(self.device),
                )
            )
            for role in config.roles
        }

        self.states: Dict[str, RoleState] = {}
        self._base_hidden: Dict[str, torch.Tensor] = {}
        self._attended_history: Dict[str, List[torch.Tensor]] = {}
        self._rng = self._build_generator(config.rng_seed)

        self._active = False
        self._step_count = 0
        self._completed_roles: set[str] = set()
        self._rollback_events: List[Dict[str, Any]] = []
        self._timings: Dict[str, float] = {}
        self._start_time: Optional[float] = None
        self._last_stride_start: Optional[float] = None
        self._plan_token_ids: Optional[torch.Tensor] = None
        self._plan_mask: Optional[torch.Tensor] = None
        self._plan_logits: Optional[torch.Tensor] = None
        self._plan_source: str = "none"
        self._gate_values: Dict[str, float] = {role: config.gate_g for role in config.roles}
        self._gate_cooldown: Dict[str, int] = {role: 0 for role in config.roles}
        self._cadence_events: List[Dict[str, Any]] = []
        self._coverage_history: Dict[str, List[List[float]]] = {role: [] for role in config.roles}
        self._coverage_manifest: Dict[str, List[Dict[str, Any]]] = {
            role: [] for role in config.roles
        }
        self._plan_embeddings: Optional[torch.Tensor] = None
        self._plan_mask_bool: Optional[torch.Tensor] = None
        self._plan_ids_list: Optional[List[int]] = None
        self._plan_mask_list: Optional[List[int]] = None

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def start(
        self,
        prompt: str,
        planner_notes: Optional[Any] = None,
        *,
        prefix_by_role: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialise decoding state from a prompt."""

        if not prompt:
            raise ValueError("Prompt must be a non-empty string.")
        self._reset_runtime_state()
        self._start_time = time.time()
        self._last_stride_start = self._start_time
        if hasattr(self.model, "eval"):
            self.model.eval()

        # Prepare a default tokenization once when no per-role prefix is provided.
        default_encoded = None
        if not prefix_by_role:
            default_encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

        with torch.inference_mode():
            for index, role in enumerate(self.config.roles):
                if prefix_by_role and role in prefix_by_role:
                    role_prompt = f"{prefix_by_role[role]}{prompt}"
                    encoded = self.tokenizer(role_prompt, return_tensors="pt", add_special_tokens=True)
                else:
                    encoded = default_encoded or self.tokenizer(
                        prompt, return_tensors="pt", add_special_tokens=True
                    )
                input_ids = encoded["input_ids"].to(self.device).clone()
                attention_mask = encoded.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, device=self.device)
                else:
                    attention_mask = attention_mask.to(self.device).clone()

                state = RoleState(
                    role=role,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    commit_stride=self.config.stride_B,
                    commit_horizon=self.config.commit_L,
                )
                outputs = self._run_trunk(
                    input_ids=state.input_ids,
                    attention_mask=state.attention_mask,
                    past_key_values=None,
                )
                if index == 0:
                    if planner_notes is not None:
                        payload = self._normalise_planner_payload(planner_notes, attention_mask)
                        self._plan_token_ids = payload["plan_token_ids"]
                        self._plan_mask = self._derive_plan_mask(payload["plan_mask"])
                        self._plan_logits = payload.get("plan_logits")
                        self._plan_source = payload.get("source", "external")
                    else:
                        planner_logits = self.model.planner_head(outputs.hidden_states[-1])
                        self._plan_logits = planner_logits.detach()
                        self._plan_token_ids = torch.argmax(planner_logits, dim=-1).long()
                        self._plan_mask = self._derive_plan_mask(attention_mask.clone())
                        self._plan_source = "model"
                state.past_key_values = outputs.past_key_values
                base_hidden = outputs.hidden_states[-1][:, -1:, :].to(device=self.device)
                self._base_hidden[role] = base_hidden
                adapted = self._apply_role_adapter(role, base_hidden)
                speculative = self.model.speculation_head(adapted)
                snapshot = self.bus_by_role[role].push(speculative.detach(), stride=0)
                state.mark_snapshot_version(snapshot.version)
                self.states[role] = state
                self._attended_history[role] = []

        for role, state in self.states.items():
            window = self.window_builder.build(state, self.bus_by_role)
            state.update_notes_window(window.notes, window.mask)
            self._mark_versions_consumed(state, window.producers, window.versions)

        if self._plan_token_ids is not None and self._plan_mask is not None:
            plan_ids = self._plan_token_ids.to(device=self.device, dtype=torch.long)
            self._plan_embeddings = self.model.plan_embedding(plan_ids).detach()
            self._plan_mask_bool = self._plan_mask.to(dtype=torch.bool, device=self.device)
            self._plan_ids_list = [int(value) for value in plan_ids.view(-1).tolist()]
            self._plan_mask_list = [int(value) for value in self._plan_mask.view(-1).tolist()]
        else:
            self._plan_embeddings = None
            self._plan_mask_bool = None
            self._plan_ids_list = None
            self._plan_mask_list = None

        self._active = True
        self._timings["bootstrap"] = time.time() - self._start_time
        self._step_count = 0

    def step(self) -> Optional[StepOutcome]:
        """Advance the orchestrator by one token emission."""

        if not self._active:
            return None

        tick = self.scheduler.tick()
        role = tick.role
        state = self.states[role]

        if self._role_completed(state):
            self._completed_roles.add(role)
            outcome = self.scheduler.advance()
            if outcome.stride_completed:
                self._on_stride_complete()
            if len(self._completed_roles) == len(self.states):
                self._active = False
            return None

        window = self.window_builder.build(state, self.bus_by_role)
        state.update_notes_window(window.notes, window.mask)
        self._mark_versions_consumed(state, window.producers, window.versions)

        base_hidden = self._base_hidden[role]
        adapted = self._apply_role_adapter(role, base_hidden)
        attended = self._apply_cross_attention(role, base_hidden, adapted, window.notes, window.mask)

        attended_logits = self._lm_head(attended)
        base_logits = self._lm_head(base_hidden) if self.alpha < 0.999 else attended_logits
        logits = self._blend_logits(attended_logits, base_logits)

        agreement_tensor = self.model.agreement_head(attended)
        agreement = float(agreement_tensor.detach().squeeze().item())

        token_id = self._sample_token(logits, state)
        token_text = self._decode_token(token_id)

        prev_kv = state.past_key_values
        state.append_token(token_id, past_key_values=None, token_text=token_text)

        outputs = self._run_trunk(
            input_ids=state.input_ids[:, -1:],
            attention_mask=state.attention_mask,
            past_key_values=prev_kv,
        )
        state.past_key_values = outputs.past_key_values
        self._base_hidden[role] = outputs.hidden_states[-1][:, -1:, :]

        self._attended_history[role].append(attended.detach())
        self._track_coverage(role, attended)
        coverage_list = self._coverage_current(role)

        state.register_commit()

        cadence = self.config.cadence_for(role)
        notes_emitted = False
        rollback_performed = False
        emit, cadence_meta = self._should_emit_notes(role, state, cadence, agreement)
        if cadence_meta is not None:
            cadence_meta.update(
                {
                    "role": role,
                    "stride_index": tick.stride_index,
                    "token_index": tick.token_index,
                }
            )
            self._cadence_events.append(cadence_meta)
        if emit:
            notes_emitted = True
            history = self._stack_attended_history(role)
            note_summary = self.model.notes_head(history).mean(dim=1, keepdim=True)
            stride = max(1, state.tokens_since_snapshot)
            snapshot = self.bus_by_role[role].push(note_summary.detach(), stride=stride)
            state.mark_snapshot_version(snapshot.version)
            state.reset_snapshot_counter()
            self._attended_history[role] = []

            result = self.agreement_gate.evaluate(agreement_tensor)
            agreement = result.score
            if result.triggered:
                rollback_performed = self._perform_rollback(role, state)
            coverage_list = self._finalise_coverage(role, tick.stride_index, tick.token_index)
            self._update_gate_on_emission(role, result, agreement)
        else:
            self._update_gate_on_stable_step(role, agreement)
            coverage_list = self._coverage_current(role)

        outcome = self.scheduler.advance()
        if outcome.role_completed:
            self._completed_roles.add(role)
        if outcome.stride_completed:
            self._on_stride_complete()

        if len(self._completed_roles) == len(self.states):
            self._active = False

        self._step_count += 1
        coverage_payload = list(coverage_list) if coverage_list is not None else None

        return StepOutcome(
            role=role,
            token_id=token_id,
            token_text=token_text,
            stride_index=tick.stride_index,
            stride_completed=outcome.stride_completed,
            role_completed=outcome.role_completed,
            agreement=agreement,
            notes_emitted=notes_emitted,
            rollback_performed=rollback_performed,
            cadence_mode=cadence_meta["mode"] if cadence_meta is not None else None,
            cadence_probability=cadence_meta.get("final_probability") if cadence_meta else None,
            cadence_multiplier=cadence_meta.get("multiplier") if cadence_meta else None,
            cadence_forced=bool(cadence_meta.get("forced")) if cadence_meta else False,
            coverage_logits=coverage_payload,
        )

    def stream(self) -> Dict[str, Dict[str, Any]]:
        """Return current per-role text and token statistics."""

        payload: Dict[str, Dict[str, Any]] = {}
        for role, state in self.states.items():
            payload[role] = {
                "text": state.generated_text,
                "token_count": state.generated_count,
                "latest_version": state.latest_snapshot_version,
                "gate": self._gate_values.get(role, self.config.gate_g),
                "cadence_mode": self.config.cadence_policy.mode,
                "coverage": self._coverage_snapshot(role, None),
            }
        return payload

    def finalize(self) -> Dict[str, Any]:
        """Return a manifest summarising the inference run."""

        end_time = time.time()
        total_duration = float(end_time - (self._start_time or end_time))
        self._timings["total"] = total_duration
        manifest = {
            "timings": dict(self._timings),
            "config": {
                "stride_B": self.config.stride_B,
                "commit_L": self.config.commit_L,
                "read_lag_delta": self.config.read_lag_delta,
                "max_snapshots_K": self.config.max_snapshots_K,
                "topology": self.config.topology,
                "gate_g": self.config.gate_g,
                "tau": self.config.agreement_threshold_tau,
                "M_by_role": dict(self.config.emission_cadence_M_by_role),
                "alpha": self.alpha,
                "gate_annealing": {
                    "enabled": self.config.gate_annealing.enabled,
                    "decay": self.config.gate_annealing.decay,
                    "min_value": self.config.gate_annealing.min_value,
                    "recovery": self.config.gate_annealing.recovery,
                    "stability_margin": self.config.gate_annealing.stability_margin,
                    "cooldown": self.config.gate_annealing.cooldown,
                },
                "cadence_policy": {
                    "mode": self.config.cadence_policy.mode,
                    "min_probability": self.config.cadence_policy.min_probability,
                    "max_interval": self.config.cadence_policy.max_interval,
                    "multiplier_min": self.config.cadence_policy.multiplier_min,
                    "multiplier_max": self.config.cadence_policy.multiplier_max,
                    "agreement_low": self.config.cadence_policy.agreement_low,
                    "agreement_high": self.config.cadence_policy.agreement_high,
                    "age_boost": self.config.cadence_policy.age_boost,
                },
                "rng_seed": self.config.rng_seed,
            },
            "roles": {
                role: {
                    "text": state.generated_text,
                    "token_ids": list(state.generated_tokens),
                    "latest_version": state.latest_snapshot_version,
                    "rollback_buffer": list(state.rollback_buffer),
                    "gate": self._gate_values.get(role, self.config.gate_g),
                    "coverage": self._coverage_snapshot(role, None),
                }
                for role, state in self.states.items()
            },
            "rollbacks": self._rollback_events,
            "steps": self._step_count,
        }
        if self._plan_token_ids is not None and self._plan_mask is not None:
            manifest["plan"] = {
                "source": self._plan_source,
                "token_ids": self._tensor_to_int_list(self._plan_token_ids),
                "mask": self._tensor_to_int_list(self._plan_mask),
            }
            if self._plan_logits is not None:
                manifest["plan"]["logits_shape"] = list(self._plan_logits.shape)
        if self._cadence_events:
            manifest["cadence_events"] = list(self._cadence_events)
        return manifest

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _reset_runtime_state(self) -> None:
        self.states.clear()
        self._base_hidden.clear()
        self._attended_history.clear()
        self._completed_roles.clear()
        self._rollback_events.clear()
        self._timings.clear()
        for role, bus in list(self.bus_by_role.items()):
            self.bus_by_role[role] = DynamicNotesBus(bus.config)
        self.scheduler = TriangularScheduler(
            self.config.roles,
            stride=self.config.stride_B,
            levels=self.config.hierarchy_levels,
        )
        self._active = False
        self._plan_token_ids = None
        self._plan_mask = None
        self._plan_logits = None
        self._plan_source = "none"
        self._gate_values = {role: self.config.gate_g for role in self.config.roles}
        self._gate_cooldown = {role: 0 for role in self.config.roles}
        self._cadence_events = []
        self._coverage_history = {role: [] for role in self.config.roles}
        self._coverage_manifest = {role: [] for role in self.config.roles}
        self._plan_embeddings = None
        self._plan_mask_bool = None
        self._plan_ids_list = None
        self._plan_mask_list = None

    def _role_completed(self, state: RoleState) -> bool:
        max_tokens = self.decode_config.max_new_tokens
        if max_tokens <= 0:
            return False
        return state.generated_count >= max_tokens

    def _run_trunk(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[PastKeyValues],
    ):
        trunk = self.model.trunk_adapter.model
        return trunk(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

    def _apply_role_adapter(self, role: str, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.role_adapters(role, hidden_states)

    def _apply_cross_attention(
        self,
        role: str,
        base_hidden: torch.Tensor,
        adapted: torch.Tensor,
        notes: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attended = self.model.cross_attention(adapted, notes, notes_mask=mask)
        gate_value = self._gate_values.get(role, self.config.gate_g)
        if abs(gate_value - 1.0) < 1e-6:
            return attended
        residual = attended - base_hidden
        return base_hidden + residual * gate_value

    def _lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        head = self.model.trunk_adapter.model.lm_head
        return head(hidden)

    def _blend_logits(self, attended: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        if self.alpha >= 0.999:
            return attended
        return self.alpha * attended + (1.0 - self.alpha) * base

    def _sample_token(self, logits: torch.Tensor, state: RoleState) -> int:
        scores = logits[:, -1, :]
        scores = scores.squeeze(0)
        if self.decode_config.temperature and self.decode_config.temperature != 1.0:
            scores = scores / self.decode_config.temperature
        scores = self._apply_repetition_penalty(scores, state.generated_tokens)
        if self.decode_config.do_sample:
            filtered = self._top_k_top_p_filter(scores)
            if torch.isinf(filtered).all():
                filtered = scores
            probs = torch.softmax(filtered, dim=-1)
            sample_kwargs: Dict[str, Any] = {}
            if self._rng is not None:
                sample_kwargs["generator"] = self._rng
            token_id = torch.multinomial(probs, num_samples=1, **sample_kwargs)
            return int(token_id.item())
        return int(torch.argmax(scores).item())

    def _apply_repetition_penalty(self, logits: torch.Tensor, history: Sequence[int]) -> torch.Tensor:
        penalty = self.decode_config.repetition_penalty
        if penalty <= 1.0 or not history:
            return logits
        unique_tokens = set(history)
        adjusted = logits.clone()
        for token in unique_tokens:
            score = adjusted[token]
            adjusted[token] = torch.sign(score) * (torch.abs(score) / penalty)
        return adjusted

    def _top_k_top_p_filter(self, logits: torch.Tensor) -> torch.Tensor:
        top_k = self.decode_config.top_k
        top_p = self.decode_config.top_p
        filtered = logits.clone()
        if top_k > 0 and top_k < filtered.numel():
            threshold = torch.topk(filtered, top_k).values[-1]
            filtered[filtered < threshold] = float("-inf")
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
            cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            cutoff = cumulative > top_p
            cutoff[..., 0] = False
            filtered_indices = sorted_indices[cutoff]
            filtered[filtered_indices] = float("-inf")
        return filtered

    def _decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id], skip_special_tokens=False)

    def _stack_attended_history(self, role: str) -> torch.Tensor:
        history = self._attended_history.get(role, [])
        if not history:
            return self._base_hidden[role]
        return torch.cat(history, dim=1)

    def _perform_rollback(self, role: str, state: RoleState) -> bool:
        removed, restored_kv = state.rollback()
        if not removed:
            return False
        refresh = self._run_trunk(
            input_ids=state.input_ids,
            attention_mask=state.attention_mask,
            past_key_values=None,
        )
        state.past_key_values = refresh.past_key_values if refresh.past_key_values is not None else restored_kv
        self._base_hidden[role] = refresh.hidden_states[-1][:, -1:, :]
        self._attended_history[role] = []
        self._rollback_events.append(
            {
                "role": role,
                "tokens_removed": removed,
                "remaining_tokens": state.generated_tokens.copy(),
            }
        )
        corrected = self.model.notes_head(self._base_hidden[role])
        snapshot = self.bus_by_role[role].push(corrected.detach(), stride=len(removed))
        state.mark_snapshot_version(snapshot.version)
        return True

    def _on_stride_complete(self) -> None:
        now = time.time()
        if self._last_stride_start is not None:
            self._timings.setdefault("stride_durations", []).append(now - self._last_stride_start)
        self._last_stride_start = now

    def _mark_versions_consumed(
        self,
        state: RoleState,
        producers: Sequence[str],
        versions: torch.Tensor,
    ) -> None:
        if len(producers) != len(versions):
            return
        for producer, version in zip(producers, versions.tolist()):
            state.update_last_seen_version(producer, int(version))

    def _track_coverage(self, role: str, attended: torch.Tensor) -> None:
        if self._plan_embeddings is None or self._plan_mask_bool is None:
            return
        logits = self.model.coverage_head(attended, self._plan_embeddings, self._plan_mask_bool)
        logits_list = [float(value) for value in logits.squeeze(0).detach().to("cpu", torch.float32).tolist()]
        self._coverage_history[role].append(logits_list)

    def _coverage_current(self, role: str) -> Optional[List[float]]:
        history = self._coverage_history.get(role)
        if not history:
            return None
        return history[-1]

    def _finalise_coverage(self, role: str, stride_index: int, token_index: int) -> Optional[List[float]]:
        latest = self._coverage_current(role)
        if latest is None:
            return None
        self._coverage_manifest[role].append(
            {
                "stride_index": stride_index,
                "token_index": token_index,
                "logits": list(latest),
            }
        )
        return list(latest)

    def _coverage_snapshot(self, role: str, override: Optional[List[float]]) -> Optional[Dict[str, Any]]:
        if self._plan_embeddings is None:
            return None
        latest = override or self._coverage_current(role)
        if latest is None:
            return None
        latest_copy = list(latest)
        payload: Dict[str, Any] = {
            "latest_logits": latest_copy,
        }
        if self._plan_ids_list is not None:
            payload["plan_ids"] = self._plan_ids_list
        if self._plan_mask_list is not None:
            payload["plan_mask"] = self._plan_mask_list
        manifest = self._coverage_manifest.get(role)
        if manifest:
            payload["emissions"] = [dict(record) for record in manifest]
        return payload

    def _should_emit_notes(
        self,
        role: str,
        state: RoleState,
        cadence: int,
        agreement_score: float,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        policy = self.config.cadence_policy
        if policy.mode == "deterministic":
            return state.cadence_reached(cadence), None

        tokens_since = state.tokens_since_snapshot
        if policy.max_interval > 0 and tokens_since >= policy.max_interval:
            metadata = {
                "mode": policy.mode,
                "forced": True,
                "base_probability": 1.0,
                "multiplier": 1.0,
                "final_probability": 1.0,
                "sample": None,
                "tokens_since": tokens_since,
            }
            return True, metadata

        base_prob = min(1.0, max(policy.min_probability, 1.0 / max(1, cadence)))
        multiplier = 1.0
        reason = "base"

        if policy.mode == "adaptive":
            clamped = max(0.0, min(1.0, agreement_score))
            if clamped <= policy.agreement_low:
                multiplier = policy.multiplier_max
                reason = "agreement_low"
            elif clamped >= policy.agreement_high:
                multiplier = policy.multiplier_min
                reason = "agreement_high"
            else:
                span = policy.agreement_high - policy.agreement_low
                ratio = (policy.agreement_high - clamped) / span
                multiplier = policy.multiplier_min + (
                    (policy.multiplier_max - policy.multiplier_min) * ratio
                )
                reason = "agreement_interp"
            if policy.age_boost > 0.0:
                age_ratio = max(0.0, tokens_since / max(1, cadence) - 1.0)
                if age_ratio > 0.0:
                    age_multiplier = 1.0 + policy.age_boost * age_ratio
                    multiplier *= age_multiplier
                    reason = "agreement_age"
            multiplier = max(policy.multiplier_min, min(policy.multiplier_max, multiplier))

        final_prob = min(1.0, max(policy.min_probability, base_prob * multiplier))

        if self._rng is not None:
            sample = torch.rand(1, generator=self._rng).item()
        else:
            sample = torch.rand(1).item()
        emit = sample < final_prob

        metadata = {
            "mode": policy.mode,
            "base_probability": base_prob,
            "multiplier": multiplier,
            "final_probability": final_prob,
            "sample": sample,
            "tokens_since": tokens_since,
            "forced": False,
            "emitted": emit,
        }
        if policy.mode == "adaptive":
            metadata["reason"] = reason
        return emit, metadata

    def _update_gate_on_emission(
        self,
        role: str,
        agreement_result: AgreementResult,
        agreement_score: float,
    ) -> None:
        policy = self.config.gate_annealing
        if not policy.enabled:
            return
        current = self._gate_values.get(role, self.config.gate_g)
        margin = self.config.agreement_threshold_tau + policy.stability_margin
        volatile = agreement_result.triggered or (agreement_score < margin)
        if volatile:
            updated = max(policy.min_value, current * policy.decay)
            self._gate_values[role] = updated
            self._gate_cooldown[role] = policy.cooldown
        else:
            self._recover_gate(role)

    def _update_gate_on_stable_step(self, role: str, agreement_score: float) -> None:
        policy = self.config.gate_annealing
        if not policy.enabled:
            return
        margin = self.config.agreement_threshold_tau + policy.stability_margin
        if agreement_score < margin:
            return
        self._recover_gate(role)

    def _recover_gate(self, role: str) -> None:
        policy = self.config.gate_annealing
        if not policy.enabled:
            return
        cooldown = self._gate_cooldown.get(role, 0)
        if cooldown > 0:
            self._gate_cooldown[role] = cooldown - 1
            return
        current = self._gate_values.get(role, self.config.gate_g)
        if current >= self.config.gate_g:
            self._gate_values[role] = self.config.gate_g
            return
        updated = min(self.config.gate_g, current + policy.recovery)
        self._gate_values[role] = updated

    def _normalise_planner_payload(
        self,
        payload: Any,
        attention_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        plan_logits: Optional[torch.Tensor] = None
        source = "external"
        if isinstance(payload, dict):
            if "plan_token_ids" not in payload:
                raise ValueError("planner_notes dict must include 'plan_token_ids'.")
            plan_ids = self._coerce_plan_ids(payload["plan_token_ids"], attention_mask)
            plan_mask = self._coerce_plan_mask(
                payload.get("plan_mask", attention_mask.clone()),
                attention_mask,
            )
            raw_logits = payload.get("plan_logits")
            if raw_logits is not None:
                plan_logits = raw_logits.to(device=self.device)
        else:
            plan_ids = self._coerce_plan_ids(payload, attention_mask)
            plan_mask = attention_mask.clone()
        return {
            "plan_token_ids": plan_ids,
            "plan_mask": plan_mask,
            "plan_logits": plan_logits,
            "source": source,
        }

    def _coerce_plan_ids(self, value: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=self.device, dtype=torch.long)
        elif isinstance(value, (list, tuple)):
            tensor = torch.tensor(value, dtype=torch.long, device=self.device)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
        else:
            raise ValueError("Unsupported plan_token_ids payload type.")
        if tensor.dim() != 2 or tensor.size(0) != 1:
            raise ValueError("plan_token_ids must be shaped [1, seq_len].")
        target_len = attention_mask.size(1)
        if tensor.size(1) != target_len:
            raise ValueError(
                f"plan_token_ids length ({tensor.size(1)}) must match prompt length ({target_len})."
            )
        return tensor

    def _coerce_plan_mask(self, value: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=self.device, dtype=attention_mask.dtype)
        elif isinstance(value, (list, tuple)):
            tensor = torch.tensor(value, dtype=attention_mask.dtype, device=self.device)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
        else:
            raise ValueError("Unsupported plan_mask payload type.")
        if tensor.dim() != 2 or tensor.size(0) != 1:
            raise ValueError("plan_mask must be shaped [1, seq_len].")
        target_len = attention_mask.size(1)
        if tensor.size(1) != target_len:
            raise ValueError(
                f"plan_mask length ({tensor.size(1)}) must match prompt length ({target_len})."
            )
        return tensor

    def _tensor_to_int_list(self, tensor: torch.Tensor) -> List[List[int]]:
        payload = tensor.detach().to(device="cpu", dtype=torch.long)
        return [[int(value) for value in row] for row in payload.tolist()]

    def _derive_plan_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Derive a plan mask gated by the first `</plan>` sentinel if present."""

        if self._plan_token_ids is None:
            return attention_mask
        end_token_id = self._resolve_plan_end_token_id()
        if end_token_id is None:
            return attention_mask
        plan_row = self._plan_token_ids[0]
        try:
            end_index = plan_row.tolist().index(end_token_id)
        except ValueError:
            return attention_mask
        mask = attention_mask.clone()
        mask[:, end_index + 1 :] = 0
        return mask

    def _resolve_plan_end_token_id(self) -> Optional[int]:
        token = "</plan>"
        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if convert is None:
            return None
        try:
            token_id = convert(token)
        except Exception:
            return None
        if isinstance(token_id, int) and token_id >= 0:
            return token_id
        return None

    def _resolve_notes_dim(self) -> int:
        config = getattr(self.model, "config", None)
        if config is None:
            raise RuntimeError("Model configuration missing; notes_dim cannot be resolved.")
        notes_dim = getattr(config, "notes_dim", None)
        if notes_dim is None:
            head_cfg = getattr(config, "notes_head", None)
            if head_cfg is None or getattr(head_cfg, "notes_dim", None) is None:
                raise RuntimeError("notes_dim not available on model configuration.")
            notes_dim = head_cfg.notes_dim
        return int(notes_dim)

    def _resolve_bus_dtype(self) -> str:
        config = getattr(self.model, "config", None)
        if config is None or getattr(config, "notes_bus", None) is None:
            return "bfloat16"
        bus_cfg = config.notes_bus
        return str(getattr(bus_cfg, "dtype", "bfloat16"))

    def _resolve_device(self) -> torch.device:
        if hasattr(self.model.trunk_adapter.model, "device"):
            device = getattr(self.model.trunk_adapter.model, "device")
            if isinstance(device, torch.device):
                return device
        first_param = next(self.model.trunk_adapter.model.parameters())
        return first_param.device

    def _resolve_dtype(self) -> torch.dtype:
        first_param = next(self.model.trunk_adapter.model.parameters())
        return first_param.dtype

    def _build_generator(self, seed: Optional[int]) -> Optional[torch.Generator]:
        if seed is None:
            return None
        random.seed(seed)
        try:  # pragma: no branch - optional dependency
            import numpy as np

            np.random.seed(seed)
        except Exception:
            pass
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        torch.manual_seed(seed)
        return generator

    @staticmethod
    def _validate_decode_config(config: DecodeConfig) -> None:
        if config.top_k < 0:
            raise ValueError("DecodeConfig.top_k must be non-negative.")
        if config.top_p < 0 or config.top_p > 1:
            raise ValueError("DecodeConfig.top_p must lie within [0, 1].")


__all__ = [
    "AgreementGate",
    "AgreementResult",
    "MultiStreamOrchestrator",
    "StepOutcome",
]
