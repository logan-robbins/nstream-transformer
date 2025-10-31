"""Minimal fine-tuning loop for the GPT-OSS backed N-Stream transformer."""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from ..data.collator_kd import (
    TwoBranchKDCollatorConfig,
    TwoBranchKnowledgeDistillationCollator,
)
from ..data.teacher_provider import (
    CachedTeacherNotesProvider,
    TeacherNotesProvider,
    TeacherNotesProviderBase,
)
from ..data.teacher_runner import TeacherRunnerConfig
from ..models import NStreamTransformer
from ..utils import resolve_device, seed_everything
from ..utils.nli import NliScorer, NliScorerConfig


@dataclass(slots=True)
class TeacherBranchConfig:
    enabled: bool = True
    type: str = "stop_grad"  # {stop_grad, ema}
    ema_decay: float = 0.99


@dataclass(slots=True)
class CurriculumConfig:
    B: int = 1
    L: int = 32
    delta: int = 1
    rho_by_role: Dict[str, float] = field(default_factory=dict)
    steps_per_stage: int = 0
    stage_schedule: Tuple[int, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class LossWeights:
    kd: float = 1.0
    stab: float = 0.1
    use: float = 0.0
    cov: float = 0.2
    nli: float = 0.05
    red: float = 0.0
    spec_kl: float = 0.1
    role: float = 0.0
    agree: float = 0.0


@dataclass(slots=True)
class NotesNoiseConfig:
    drop_p: float = 0.0
    paraphrase_p: float = 0.0


@dataclass(slots=True)
class StagePolicyConfig:
    name: str = ""
    freeze: Tuple[str, ...] = ()
    unfreeze: Tuple[str, ...] = ()
    bus_mix_prob: Optional[float] = None
    role_dropout_prob: Optional[float] = None
    notes_noise: Optional[NotesNoiseConfig] = None


@dataclass(slots=True)
class MetricsConfig:
    mask_ablation_every: int = 0
    stability_every: int = 0


@dataclass(slots=True)
class NegativeSamplingConfig:
    enabled: bool = False
    start_stage: int = 3
    contradiction_ratio: float = 0.0
    max_contradictions: int = 4
    noise_ratio: float = 0.0
    noise_std: float = 0.02


@dataclass(slots=True)
class GradNormConfig:
    enabled: bool = False
    target_ratio: float = 1.0
    alpha: float = 0.05
    min_scale: float = 0.1
    max_scale: float = 5.0


@dataclass(slots=True)
class TrainingConfig:
    dataset_path: Optional[str] = None
    eval_dataset_path: Optional[str] = None
    telemetry_dir: Optional[str] = None
    batch_size: int = 1
    seed: Optional[int] = None
    grad_accumulation: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_steps: int = 1000
    warmup_steps: int = 0
    log_interval: int = 10
    eval_interval: int = 200
    device: Optional[str] = None
    teacher: TeacherBranchConfig = field(default_factory=TeacherBranchConfig)
    teacher_runner: Optional[TeacherRunnerConfig] = None
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    coverage_threshold: float = 0.5
    bus_mix_prob: float = 0.0
    role_dropout_prob: float = 0.0
    parallel_micro_steps: int = 0
    notes_noise: NotesNoiseConfig = field(default_factory=NotesNoiseConfig)
    nli_scorer: Optional[str] = None
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    negative_sampling: NegativeSamplingConfig = field(default_factory=NegativeSamplingConfig)
    gradnorm: GradNormConfig = field(default_factory=GradNormConfig)
    stage_policies: Dict[int, StagePolicyConfig] = field(default_factory=dict)
    nli_margin: float = 0.1
    spec_kl_temperature: float = 1.0
    agreement_threshold: float = 0.15


@dataclass(slots=True)
class TrainerState:
    global_step: int = 0
    epoch: int = 0
    best_eval_loss: float = float("inf")
    stage_index: int = 0
    stage_history: list[Dict[str, float]] = field(default_factory=list)


class Trainer:
    """Orchestrates a lean PEFT-style training loop."""

    def __init__(
        self,
        model: NStreamTransformer,
        config: TrainingConfig,
        *,
        collator_config: TwoBranchKDCollatorConfig,
        dataset: Optional[Dataset[Dict[str, object]]] = None,
        eval_dataset: Optional[Dataset[Dict[str, object]]] = None,
    ) -> None:
        self.model = model
        self.config = config
        seed_everything(self.config.seed)
        self.state = TrainerState()
        self.logger = logging.getLogger("nstream.trainer")
        preferred = config.device or resolve_device()
        if preferred == "cuda" and not torch.cuda.is_available():
            preferred = "cpu"
        if preferred == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                preferred = "cpu"
        self.device = torch.device(preferred)
        self.model.to(self.device)
        self.metric_history: Dict[str, List[Dict[str, float]]] = {"train": [], "eval": []}
        self._stage_transitions: List[Dict[str, Any]] = []
        self._stage_start_step: int = 0
        self._stage_start_time: float = time.time()
        self._stage_history_finalized: bool = False
        schedule = tuple(self.config.curriculum.stage_schedule)
        if schedule:
            if list(schedule) != sorted(schedule):
                raise ValueError("curriculum.stage_schedule must be non-decreasing.")
            if schedule[0] != 0:
                raise ValueError("curriculum.stage_schedule must start at step 0.")
            if any(step < 0 for step in schedule):
                raise ValueError("curriculum.stage_schedule cannot contain negative steps.")
        self.teacher_model: Optional[NStreamTransformer] = None
        if config.teacher.enabled and config.teacher.type == "ema":
            self.teacher_model = copy.deepcopy(model)
            self.teacher_model.to(self.device)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad_(False)
        elif config.teacher.enabled:
            self.teacher_model = model
        self.collator_config = collator_config
        if self.collator_config.commit_horizon <= 0:
            self.collator_config.commit_horizon = config.curriculum.L
        self.collator_config.max_snapshots = max(self.collator_config.max_snapshots, config.curriculum.B)
        self.teacher_provider = self._build_teacher_provider()
        self.collator = TwoBranchKnowledgeDistillationCollator(
            self.collator_config,
            teacher_provider=self.teacher_provider,
        )
        self.role_lookup = {index: role for role, index in collator_config.role_to_id.items()}
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.optimizer = AdamW(self._trainable_parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self._lr_lambda)
        self.kd_scale = 1.0
        self.plan_hash_buckets = collator_config.plan_hash_buckets
        self.nli_scorer: Optional[NliScorer] = None
        if config.nli_scorer:
            scorer_config = NliScorerConfig(model_name=config.nli_scorer)
            self.nli_scorer = NliScorer(scorer_config, device=self.device)

    def _trainable_parameters(self) -> Iterable[nn.Parameter]:
        for param in self.model.iter_trainable_parameters():
            if param.requires_grad:
                yield param

    def _build_teacher_provider(self) -> TeacherNotesProviderBase:
        runner_cfg = self.config.teacher_runner
        if runner_cfg is None:
            raise ValueError("TrainingConfig.teacher_runner must be provided to build teacher notes.")
        backend = TeacherNotesProvider(
            runner_cfg,
            notes_dim=self.collator_config.notes_dim,
            role_to_id=self.collator_config.role_to_id,
        )
        cache_dir = Path(runner_cfg.cache_dir) if runner_cfg.cache_dir else None
        return CachedTeacherNotesProvider(
            backend=backend,
            cache_dir=cache_dir,
            id_field=runner_cfg.id_field,
            refresh=runner_cfg.refresh_cache,
        )

    def _lr_lambda(self, step: int) -> float:
        if self.config.warmup_steps <= 0:
            return 1.0
        return min(1.0, step / float(self.config.warmup_steps))

    def _determine_stage(self) -> int:
        schedule = tuple(self.config.curriculum.stage_schedule)
        steps_per_stage = self.config.curriculum.steps_per_stage
        if schedule:
            computed_stage = 0
            for index, threshold in enumerate(schedule):
                if self.state.global_step >= threshold:
                    computed_stage = index
            computed_stage = min(computed_stage, 4)
        elif steps_per_stage and steps_per_stage > 0:
            computed_stage = min(4, self.state.global_step // steps_per_stage)
        else:
            computed_stage = self.state.stage_index
        previous_stage = self.state.stage_index
        first_transition = not self.state.stage_history
        if first_transition or computed_stage != previous_stage:
            transition_from = previous_stage if not first_transition else -1
            self.state.stage_index = computed_stage
            self.state.stage_history.append({"step": self.state.global_step, "stage": computed_stage})
            self._on_stage_transition(transition_from, computed_stage)
        return self.state.stage_index

    def _on_stage_transition(self, previous_stage: int, new_stage: int) -> None:
        now = datetime.now(timezone.utc)
        if previous_stage >= 0 and self._stage_transitions:
            self._finalize_stage_record(self._stage_transitions[-1], now)
        policy = self.config.stage_policies.get(new_stage)
        stage_name = policy.name if policy and policy.name else f"stage_{new_stage}"
        record = {
            "stage_index": int(new_stage),
            "stage_name": stage_name,
            "start_step": int(self.state.global_step),
            "timestamp": now.isoformat(),
            "actions": {},
        }
        self._stage_transitions.append(record)
        self._stage_start_step = self.state.global_step
        self._stage_start_time = time.time()
        if policy is not None:
            self._apply_stage_policy(policy, record["actions"])
        if not record["actions"]:
            record.pop("actions", None)
        if self.state.stage_history:
            self.state.stage_history[-1]["stage_name"] = stage_name
        origin = previous_stage if previous_stage >= 0 else "init"
        self.logger.info(
            "stage_transition | from=%s | to=%d | step=%d | name=%s",
            origin,
            new_stage,
            self.state.global_step,
            stage_name,
        )

    def _apply_stage_policy(self, policy: StagePolicyConfig, actions: Dict[str, Any]) -> None:
        if policy.bus_mix_prob is not None:
            self.config.bus_mix_prob = float(policy.bus_mix_prob)
            actions["bus_mix_prob"] = self.config.bus_mix_prob
        if policy.role_dropout_prob is not None:
            self.config.role_dropout_prob = float(policy.role_dropout_prob)
            actions["role_dropout_prob"] = self.config.role_dropout_prob
        if policy.notes_noise is not None:
            notes_cfg = NotesNoiseConfig(
                drop_p=policy.notes_noise.drop_p,
                paraphrase_p=policy.notes_noise.paraphrase_p,
            )
            self.config.notes_noise = notes_cfg
            actions["notes_noise"] = {
                "drop_p": notes_cfg.drop_p,
                "paraphrase_p": notes_cfg.paraphrase_p,
            }
        if policy.freeze:
            frozen = self._update_trainable(policy.freeze, trainable=False)
            if frozen:
                actions["freeze"] = frozen
        if policy.unfreeze:
            unfrozen = self._update_trainable(policy.unfreeze, trainable=True)
            if unfrozen:
                actions["unfreeze"] = unfrozen

    def _update_trainable(self, identifiers: Tuple[str, ...], *, trainable: bool) -> List[str]:
        applied: List[str] = []
        for resolved_name, module in self._resolve_policy_modules(identifiers):
            if module is None:
                self.logger.warning(
                    "stage_policy_missing_module | stage=%d | module=%s",
                    self.state.stage_index,
                    resolved_name,
                )
                continue
            for param in module.parameters():
                param.requires_grad_(trainable)
            applied.append(resolved_name)
        return applied

    def _resolve_policy_modules(self, identifiers: Iterable[str]) -> List[Tuple[str, Optional[nn.Module]]]:
        modules: List[Tuple[str, Optional[nn.Module]]] = []
        for identifier in identifiers:
            key = identifier.strip()
            if not key:
                continue
            lower = key.lower()
            if lower == "trunk":
                modules.append(("trunk", getattr(self.model.trunk_adapter, "model", None)))
                continue
            if lower in {
                "role_adapters",
                "cross_attention",
                "notes_bus",
                "planner_head",
                "notes_head",
                "speculation_head",
                "agreement_head",
                "coverage_head",
                "role_classifier",
                "plan_embedding",
            }:
                modules.append((lower, getattr(self.model, lower, None)))
                continue
            if lower in {"heads", "all_heads"}:
                modules.extend(
                    [
                        ("planner_head", getattr(self.model, "planner_head", None)),
                        ("notes_head", getattr(self.model, "notes_head", None)),
                        ("speculation_head", getattr(self.model, "speculation_head", None)),
                        ("agreement_head", getattr(self.model, "agreement_head", None)),
                        ("coverage_head", getattr(self.model, "coverage_head", None)),
                        ("role_classifier", getattr(self.model, "role_classifier", None)),
                    ]
                )
                continue
            modules.append((lower, getattr(self.model, lower, None)))
        return modules

    def _finalize_stage_record(self, record: Dict[str, Any], timestamp: datetime) -> None:
        if record.get("steps") is not None:
            return
        elapsed_steps = max(0, self.state.global_step - self._stage_start_step)
        duration = max(0.0, time.time() - self._stage_start_time)
        record["end_step"] = int(self.state.global_step)
        record["steps"] = int(elapsed_steps)
        record["duration"] = float(duration)
        record["completed_at"] = timestamp.isoformat()

    def _finalize_stage_history(self) -> None:
        if self._stage_history_finalized:
            return
        if self._stage_transitions:
            self._finalize_stage_record(self._stage_transitions[-1], datetime.now(timezone.utc))
        self._stage_history_finalized = True

    def write_stage_history(self, telemetry_dir: Path | str | None) -> Optional[Path]:
        if telemetry_dir is None:
            return None
        self._finalize_stage_history()
        if not self._stage_transitions:
            return None
        target = Path(telemetry_dir)
        target.mkdir(parents=True, exist_ok=True)
        path = target / "train_run_stages.json"
        path.write_text(json.dumps(self._stage_transitions, indent=2), encoding="utf-8")
        self.logger.info("stage_history_written | path=%s", path)
        return path

    def _active_loss_weights(self, stage: int) -> LossWeights:
        base = self.config.loss_weights
        return LossWeights(
            kd=(base.kd * self.kd_scale) if stage >= 2 else 0.0,
            stab=base.stab if stage >= 4 else 0.0,
            use=base.use if stage >= 3 else 0.0,
            cov=base.cov if stage >= 4 else 0.0,
            nli=base.nli if stage >= 4 else 0.0,
            red=base.red if stage >= 4 else 0.0,
            spec_kl=base.spec_kl if stage >= 4 else 0.0,
            role=base.role if stage >= 4 else 0.0,
            agree=base.agree if stage >= 4 else 0.0,
        )

    def fit(self) -> None:
        if self.dataset is None:
            raise RuntimeError("Trainer.fit requires a training dataset instance.")
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collator,
        )
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        while self.state.global_step < self.config.max_steps:
            for batch in dataloader:
                loss, metrics = self._training_step(batch)
                loss.backward()
                if (self.state.global_step + 1) % self.config.grad_accumulation == 0:
                    self.optimizer.step()
                    self._update_teacher_ema()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                self.state.global_step += 1
                if self.state.global_step % self.config.log_interval == 0:
                    self._log_metrics("train", metrics)
                if self.state.global_step % self.config.eval_interval == 0:
                    self.evaluate()
                if self.state.global_step >= self.config.max_steps:
                    break
            self.state.epoch += 1
        self._finalize_stage_history()

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, float]]:
        stage = self._determine_stage()
        self._maybe_apply_negative_sampling(batch, stage)
        batch = {
            key: value.to(self.device) if torch.is_tensor(value) else value for key, value in batch.items()
        }
        plan_item_ids = batch.get("plan_item_ids")
        plan_item_mask = batch.get("plan_item_mask")
        if plan_item_ids is not None:
            plan_item_ids = plan_item_ids.to(self.device)
        if plan_item_mask is not None:
            plan_item_mask = plan_item_mask.to(self.device)
        hidden_states = self.model.encode(batch["input_ids"], attention_mask=batch["attention_mask"])
        active_weights = self._active_loss_weights(stage)
        teacher_branch = self._prepare_branch_inputs(batch, branch="teacher", stage=stage)
        student_branch = self._prepare_branch_inputs(
            batch,
            branch="student",
            stage=stage,
            teacher_branch=teacher_branch,
        )
        teacher_branch["notes"] = teacher_branch["notes"].to(hidden_states.dtype)
        if "pre_notes" in student_branch:
            student_branch["pre_notes"] = student_branch["pre_notes"].to(hidden_states.dtype)

        if self._uses_separate_teacher():
            teacher_hidden = self._teacher_encode(batch)
        else:
            teacher_hidden = hidden_states.detach()

        student_outputs = self._run_student_pass(
            hidden_states,
            batch,
            student_branch,
            stage=stage,
            plan_item_ids=plan_item_ids,
            plan_item_mask=plan_item_mask,
        )
        teacher_outputs = self._teacher_forward(
            teacher_hidden,
            role=batch["role_ids"],
            notes=teacher_branch["notes"],
            notes_mask=teacher_branch["notes_mask"],
        )
        stability_logging_due = (
            self.config.metrics.stability_every <= 0
            or (
                self.config.metrics.stability_every > 0
                and (self.state.global_step % self.config.metrics.stability_every == 0)
            )
        )
        need_pre_logits = (
            active_weights.stab > 0.0
            or stability_logging_due
        )
        pre_update_logits: Optional[torch.Tensor] = None
        if need_pre_logits and "pre_notes" in student_branch and "pre_notes_mask" in student_branch:
            with torch.no_grad():
                pre_outputs = self.model(
                    hidden_states,
                    role=batch["role_ids"],
                    notes=student_branch["pre_notes"],
                    notes_mask=student_branch["pre_notes_mask"],
                    plan_item_ids=plan_item_ids,
                    plan_item_mask=plan_item_mask,
                )
            pre_update_logits = pre_outputs["planner_logits"]

        total_loss, metrics = self._compute_losses(
            batch,
            student_outputs,
            teacher_outputs,
            student_branch=student_branch,
            teacher_branch=teacher_branch,
            hidden_states=hidden_states,
            stage=stage,
            weights=active_weights,
            step=self.state.global_step,
            pre_update_logits=pre_update_logits,
            stability_logging_due=stability_logging_due,
        )
        metrics["loss"] = float(total_loss.detach().cpu())
        metrics["stage"] = float(stage)
        return total_loss, metrics

    def generate_training_report(self) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        train_history = self.metric_history.get("train", [])
        eval_history = self.metric_history.get("eval", [])
        summary_keys = [
            "mask_ablation",
            "kd_ce_ratio",
            "agreement_precision",
            "rollback_kl",
            "stability_kl",
            "repair_error_rate",
            "stability_error_rate",
            "repair_margin",
            "stability_margin",
            "usage_loss",
            "coverage_precision",
            "coverage_recall",
            "coverage_f1",
            "coverage_cross_role_fp_rate",
            "coverage_same_role_recall",
        ]

        def _aggregate(key: str) -> Optional[Dict[str, float]]:
            values = [
                float(entry[key])
                for entry in train_history
                if key in entry and isinstance(entry[key], (int, float)) and not math.isnan(float(entry[key]))
            ]
            if not values:
                return None
            return {
                "last": values[-1],
                "mean": float(sum(values) / len(values)),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }

        train_summary = {}
        for key in summary_keys:
            aggregated = _aggregate(key)
            if aggregated is not None:
                train_summary[key] = aggregated

        if train_history:
            loss_summary = _aggregate("loss") or _aggregate("planner_loss")
            if loss_summary is not None:
                train_summary.setdefault("loss", loss_summary)
            last_stage_value = train_history[-1].get("stage")
            if not isinstance(last_stage_value, (int, float)):
                last_stage_value = self.state.stage_index
            train_summary.setdefault("stage", {"last": float(last_stage_value)})

        eval_summary: Dict[str, float] = {}
        if eval_history:
            last_eval = dict(eval_history[-1])
            last_eval.pop("timestamp", None)
            last_eval.pop("step", None)
            eval_summary = last_eval

        report = {
            "generated_at": now,
            "global_step": self.state.global_step,
            "best_eval_loss": self.state.best_eval_loss,
            "stage": self.state.stage_index,
            "train_history_length": len(train_history),
            "eval_history_length": len(eval_history),
            "train_metrics": train_summary,
            "eval_metrics": eval_summary,
        }
        return report

    def _maybe_apply_negative_sampling(self, batch: Dict[str, Any], stage: int) -> None:
        cfg = self.config.negative_sampling
        if not cfg.enabled or stage < cfg.start_stage:
            return
        if cfg.contradiction_ratio > 0.0 and cfg.max_contradictions > 0:
            self._inject_negative_plan_items(batch, cfg)
        if cfg.noise_ratio > 0.0 and cfg.noise_std > 0.0:
            self._inject_negative_noise(batch, cfg)

    def _inject_negative_plan_items(self, batch: Dict[str, Any], cfg: NegativeSamplingConfig) -> None:
        plan_item_ids = batch.get("plan_item_ids")
        plan_item_mask = batch.get("plan_item_mask")
        if plan_item_ids is None or plan_item_mask is None:
            return
        if plan_item_ids.numel() == 0:
            return
        plan_item_mask = plan_item_mask.to(dtype=torch.bool)
        batch_size, width = plan_item_ids.shape
        neg_counts: List[int] = []
        max_negatives = 0
        for index in range(batch_size):
            positive = int(plan_item_mask[index].sum().item())
            if positive == 0:
                neg_counts.append(0)
                continue
            desired = max(1, math.ceil(positive * cfg.contradiction_ratio))
            desired = min(desired, cfg.max_contradictions)
            if desired <= 0:
                neg_counts.append(0)
                continue
            neg_counts.append(desired)
            max_negatives = max(max_negatives, desired)
        if max_negatives == 0:
            return
        device = plan_item_ids.device
        dtype = plan_item_ids.dtype
        new_width = width + max_negatives
        new_plan_ids = torch.zeros((batch_size, new_width), dtype=dtype, device=device)
        new_plan_ids[:, :width] = plan_item_ids
        new_plan_mask = torch.zeros((batch_size, new_width), dtype=torch.bool, device=device)
        new_plan_mask[:, :width] = plan_item_mask

        coverage_targets = batch.get("coverage_targets")
        coverage_mask = batch.get("coverage_mask")
        if coverage_targets is None:
            base_targets = torch.zeros((batch_size, width), dtype=torch.float32, device=device)
        else:
            base_targets = coverage_targets.to(device=device, dtype=torch.float32)
        if coverage_mask is None:
            base_mask = torch.zeros((batch_size, width), dtype=torch.bool, device=device)
        else:
            base_mask = coverage_mask.to(device=device, dtype=torch.bool)

        new_coverage_targets = torch.zeros((batch_size, new_width), dtype=torch.float32, device=device)
        new_coverage_targets[:, :width] = base_targets
        new_coverage_mask = torch.zeros((batch_size, new_width), dtype=torch.bool, device=device)
        new_coverage_mask[:, :width] = base_mask

        plan_text = batch.get("plan_text")
        for index, neg_count in enumerate(neg_counts):
            if neg_count <= 0:
                continue
            existing_ids = set(plan_item_ids[index, plan_item_mask[index]].tolist())
            generated: List[int] = []
            attempts = 0
            while len(generated) < neg_count:
                candidate = int(torch.randint(1, self.plan_hash_buckets, (1,), device=device).item())
                if candidate == 0:
                    continue
                if candidate in existing_ids or candidate in generated:
                    attempts += 1
                    if attempts > neg_count * 8:
                        candidate = (candidate + attempts + len(generated) + 1) % self.plan_hash_buckets
                        candidate = candidate or 1
                if candidate not in existing_ids and candidate not in generated:
                    generated.append(candidate)
            start = width
            end = width + neg_count
            new_plan_ids[index, start:end] = torch.tensor(generated, dtype=dtype, device=device)
            new_plan_mask[index, start:end] = True
            new_coverage_targets[index, start:end] = 0.0
            new_coverage_mask[index, start:end] = True
            if isinstance(plan_text, list) and index < len(plan_text):
                plan_text[index].extend([f"[negative-{value}]" for value in generated])

        batch["plan_item_ids"] = new_plan_ids
        batch["plan_item_mask"] = new_plan_mask
        batch["coverage_targets"] = new_coverage_targets
        batch["coverage_mask"] = new_coverage_mask

    def _inject_negative_noise(self, batch: Dict[str, Any], cfg: NegativeSamplingConfig) -> None:
        notes_student = batch.get("notes_student")
        if notes_student is None or notes_student.numel() == 0:
            return
        device = notes_student.device
        sample_mask = torch.rand((notes_student.size(0),), device=device) < cfg.noise_ratio
        if not sample_mask.any():
            return
        for index, selected in enumerate(sample_mask.tolist()):
            if not selected:
                continue
            original = notes_student[index].to(dtype=torch.float32)
            noise = torch.randn_like(original) * cfg.noise_std
            perturbed = original + noise
            notes_student[index] = perturbed.to(dtype=notes_student.dtype)
    def _prepare_branch_inputs(
        self,
        batch: Dict[str, Any],
        *,
        branch: str,
        stage: int,
        teacher_branch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if branch not in {"teacher", "student"}:
            raise ValueError(f"Unknown branch specifier: {branch}")
        prefix = "teacher" if branch == "teacher" else "student"
        lag_current = max(0, self.config.curriculum.delta)
        notes, coverage, snapshot_roles, snapshot_stride, snapshot_version = self._extract_notes_from_bus(
            batch,
            notes_bus_key=f"{prefix}_notes_bus",
            mask_key=f"{prefix}_bus_mask",
            fallback_key="notes_teacher" if branch == "teacher" else "notes_student",
            coverage_key=f"{prefix}_bus_coverage",
            roles_key=f"{prefix}_bus_roles",
            stride_key=f"{prefix}_bus_stride",
            version_key=f"{prefix}_bus_version",
            lag_override=lag_current,
        )
        pre_notes: Optional[torch.Tensor] = None
        pre_snapshot_version: Optional[torch.Tensor] = None
        if branch == "student":
            (
                pre_notes,
                _,
                _,
                _,
                pre_snapshot_version,
            ) = self._extract_notes_from_bus(
                batch,
                notes_bus_key=f"{prefix}_notes_bus",
                mask_key=f"{prefix}_bus_mask",
                fallback_key="notes_student",
                coverage_key=f"{prefix}_bus_coverage",
                roles_key=f"{prefix}_bus_roles",
                stride_key=f"{prefix}_bus_stride",
                version_key=f"{prefix}_bus_version",
                lag_override=lag_current + 1,
            )
        if branch == "student":
            if stage == 0 and teacher_branch is not None:
                notes = teacher_branch["notes"].clone()
            elif self.model.training:
                bus_mix_prob = self.config.bus_mix_prob if stage >= 3 else 0.0
                role_dropout = self.config.role_dropout_prob if stage >= 3 else 0.0
                noise_cfg = self.config.notes_noise if stage >= 3 else NotesNoiseConfig()
                if bus_mix_prob > 0.0 and teacher_branch is not None:
                    teacher_notes = teacher_branch["notes"]
                    mix_mask = (
                        torch.rand((notes.size(0), 1, 1), device=notes.device) < bus_mix_prob
                    )
                    notes = torch.where(mix_mask, teacher_notes, notes)
                if role_dropout > 0.0:
                    drop_mask = (
                        torch.rand((notes.size(0), notes.size(1)), device=notes.device)
                        < role_dropout
                    )
                    notes = notes.masked_fill(drop_mask.unsqueeze(-1), 0.0)
                if noise_cfg.drop_p > 0.0:
                    noise_drop = (
                        torch.rand((notes.size(0), notes.size(1)), device=notes.device)
                        < noise_cfg.drop_p
                    )
                    notes = notes.masked_fill(noise_drop.unsqueeze(-1), 0.0)
                if noise_cfg.paraphrase_p > 0.0:
                    noise_mask = (
                        torch.rand((notes.size(0), notes.size(1)), device=notes.device)
                        < noise_cfg.paraphrase_p
                    )
                    if noise_mask.any():
                        gaussian = torch.randn_like(notes)
                        notes = notes + gaussian * noise_mask.unsqueeze(-1) * 0.05
        notes_mask = (notes.abs().sum(dim=-1) > 0).long()
        if notes_mask.sum() == 0:
            notes_mask = torch.ones_like(notes_mask)
        branch_payload: Dict[str, torch.Tensor] = {
            "notes": notes,
            "notes_mask": notes_mask,
            "lag": torch.tensor(lag_current, device=notes.device, dtype=torch.long),
        }
        if pre_notes is not None:
            pre_mask = (pre_notes.abs().sum(dim=-1) > 0).long()
            if pre_mask.sum() == 0:
                pre_mask = torch.ones_like(pre_mask)
            branch_payload["pre_notes"] = pre_notes
            branch_payload["pre_notes_mask"] = pre_mask
            if pre_snapshot_version is not None:
                branch_payload["pre_snapshot_version"] = pre_snapshot_version
        if coverage is not None:
            branch_payload["coverage"] = coverage.to(device=notes.device, dtype=torch.float32)
        if snapshot_roles is not None:
            branch_payload["snapshot_roles"] = snapshot_roles
        if snapshot_stride is not None:
            branch_payload["snapshot_stride"] = snapshot_stride
        if snapshot_version is not None:
            branch_payload["snapshot_version"] = snapshot_version
        return branch_payload

    def _extract_notes_from_bus(
        self,
        batch: Dict[str, Any],
        *,
        notes_bus_key: str,
        mask_key: str,
        fallback_key: str,
        coverage_key: str,
        roles_key: str,
        stride_key: str,
        version_key: str,
        lag_override: Optional[int] = None,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        fallback = batch[fallback_key].to(self.device)
        notes_bus = batch.get(notes_bus_key)
        mask = batch.get(mask_key)
        if notes_bus is None or mask is None:
            return fallback, None, None, None, None
        notes_bus = notes_bus.to(self.device)
        mask = mask.to(self.device)
        if mask.long().sum().item() == 0:
            return fallback, None, None, None, None

        batch_indices = torch.arange(notes_bus.size(0), device=notes_bus.device)
        lag_value = self.config.curriculum.delta if lag_override is None else lag_override
        lag = max(0, lag_value)
        snapshot_counts = mask.long().sum(dim=1).clamp(min=1)
        snapshot_indices = snapshot_counts - 1 - lag
        snapshot_indices = torch.clamp(snapshot_indices, min=0, max=notes_bus.size(1) - 1)

        gathered_notes = notes_bus[batch_indices, snapshot_indices]

        coverage_bus = batch.get(coverage_key)
        gathered_coverage: Optional[torch.Tensor]
        if coverage_bus is None:
            gathered_coverage = None
        else:
            coverage_bus = coverage_bus.to(self.device)
            gathered_coverage = coverage_bus[batch_indices, snapshot_indices]

        roles_bus = batch.get(roles_key)
        gathered_roles: Optional[torch.Tensor]
        if roles_bus is None:
            gathered_roles = None
        else:
            gathered_roles = roles_bus.to(self.device)[batch_indices, snapshot_indices]

        stride_bus = batch.get(stride_key)
        gathered_stride: Optional[torch.Tensor]
        if stride_bus is None:
            gathered_stride = None
        else:
            gathered_stride = stride_bus.to(self.device)[batch_indices, snapshot_indices]

        version_bus = batch.get(version_key)
        gathered_version: Optional[torch.Tensor]
        if version_bus is None:
            gathered_version = None
        else:
            gathered_version = version_bus.to(self.device)[batch_indices, snapshot_indices]

        return (
            gathered_notes,
            gathered_coverage,
            gathered_roles,
            gathered_stride,
            gathered_version,
        )

    def _run_student_pass(
        self,
        hidden_states: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        student_branch: Dict[str, torch.Tensor],
        *,
        stage: int,
        plan_item_ids: Optional[torch.Tensor],
        plan_item_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        student_branch["notes"] = student_branch["notes"].to(hidden_states.dtype)
        notes_mask = student_branch["notes_mask"]
        if (
            self.config.parallel_micro_steps > 0
            and stage >= 2
            and self.model.training
        ):
            micro_notes = student_branch["notes"]
            last_outputs: Optional[Dict[str, torch.Tensor]] = None
            for micro_step in range(self.config.parallel_micro_steps):
                last_outputs = self.model(
                    hidden_states,
                    role=batch["role_ids"],
                    notes=micro_notes,
                    notes_mask=notes_mask,
                    plan_item_ids=plan_item_ids,
                    plan_item_mask=plan_item_mask,
                )
                new_notes = last_outputs["speculative_notes"].detach()
                coverage_tensor = student_branch.get("coverage")
                roles_tensor = student_branch.get("snapshot_roles")
                meta = self._update_student_bus(
                    batch,
                    new_notes,
                    snapshot_roles=roles_tensor,
                    coverage=coverage_tensor,
                )
                if "version" in meta:
                    student_branch["snapshot_version"] = meta["version"]
                if "stride" in meta:
                    student_branch["snapshot_stride"] = meta["stride"]
                if micro_step < self.config.parallel_micro_steps - 1:
                    micro_notes = new_notes
                self._advance_commit_mask(batch)
            if last_outputs is None:
                raise RuntimeError("Parallel micro-steps requested but no outputs produced.")
            student_branch["notes"] = last_outputs["speculative_notes"]
            student_branch["notes_mask"] = (
                student_branch["notes"].abs().sum(dim=-1) > 0
            ).long()
            self._refresh_pre_notes(batch, student_branch)
            return last_outputs
        return self.model(
            hidden_states,
            role=batch["role_ids"],
            notes=student_branch["notes"],
            notes_mask=notes_mask,
            plan_item_ids=plan_item_ids,
            plan_item_mask=plan_item_mask,
        )

    def _refresh_pre_notes(
        self,
        batch: Dict[str, torch.Tensor],
        student_branch: Dict[str, torch.Tensor],
    ) -> None:
        lag_tensor = student_branch.get("lag")
        lag_value = int(lag_tensor.item()) if isinstance(lag_tensor, torch.Tensor) else max(0, self.config.curriculum.delta)
        pre_notes, pre_coverage, pre_roles, pre_stride, pre_version = self._extract_notes_from_bus(
            batch,
            notes_bus_key="student_notes_bus",
            mask_key="student_bus_mask",
            fallback_key="notes_student",
            coverage_key="student_bus_coverage",
            roles_key="student_bus_roles",
            stride_key="student_bus_stride",
            version_key="student_bus_version",
            lag_override=lag_value + 1,
        )
        student_branch["pre_notes"] = pre_notes.to(device=self.device, dtype=student_branch["notes"].dtype)
        pre_mask = (pre_notes.abs().sum(dim=-1) > 0).long()
        if pre_mask.sum() == 0:
            pre_mask = torch.ones_like(pre_mask)
        student_branch["pre_notes_mask"] = pre_mask
        if pre_version is not None:
            student_branch["pre_snapshot_version"] = pre_version
        if pre_stride is not None:
            student_branch["pre_snapshot_stride"] = pre_stride
        if pre_roles is not None:
            student_branch["pre_snapshot_roles"] = pre_roles
        if pre_coverage is not None:
            student_branch["pre_snapshot_coverage"] = pre_coverage

    def _update_student_bus(
        self,
        batch: Dict[str, torch.Tensor],
        new_notes: torch.Tensor,
        *,
        snapshot_roles: Optional[torch.Tensor],
        coverage: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        bus = batch.get("student_notes_bus")
        mask = batch.get("student_bus_mask")
        if bus is None or mask is None:
            return {}
        coverage_bus = batch.get("student_bus_coverage")
        roles_bus = batch.get("student_bus_roles")
        stride_bus = batch.get("student_bus_stride")
        version_bus = batch.get("student_bus_version")
        bus = bus.to(self.device)
        mask = mask.to(self.device)
        batch_size, max_snapshots, _, _ = bus.shape
        new_notes = new_notes.to(bus.device)
        meta: Dict[str, torch.Tensor] = {}
        if version_bus is not None:
            meta["version"] = torch.zeros(batch_size, dtype=version_bus.dtype, device=version_bus.device)
        if stride_bus is not None:
            meta["stride"] = torch.zeros(batch_size, dtype=stride_bus.dtype, device=stride_bus.device)
        stride_meta = meta.get("stride")
        version_meta = meta.get("version")
        for index in range(batch_size):
            active_count = int(mask[index].sum().item())
            current_max_version = 0
            if version_bus is not None and active_count > 0:
                current_mask = mask[index].clone()
                current_max_version = int(version_bus[index][current_mask].max().item())
            if active_count >= max_snapshots:
                bus[index, :-1] = bus[index, 1:]
                mask[index, :-1] = mask[index, 1:]
                mask[index, -1] = False
                if coverage_bus is not None:
                    coverage_bus[index, :-1] = coverage_bus[index, 1:]
                    coverage_bus[index, -1].zero_()
                if roles_bus is not None:
                    roles_bus[index, :-1] = roles_bus[index, 1:]
                    roles_bus[index, -1] = -1
                if stride_bus is not None:
                    stride_bus[index, :-1] = stride_bus[index, 1:]
                    stride_bus[index, -1] = 0
                if version_bus is not None:
                    version_bus[index, :-1] = version_bus[index, 1:]
                    version_bus[index, -1] = 0
                active_count = max_snapshots - 1
            insert_idx = active_count
            bus[index, insert_idx] = new_notes[index]
            mask[index, insert_idx] = True
            if coverage_bus is not None:
                if coverage is not None:
                    coverage_bus[index, insert_idx] = coverage[index].to(
                        device=coverage_bus.device,
                        dtype=coverage_bus[index, insert_idx].dtype,
                    )
                else:
                    coverage_bus[index, insert_idx].zero_()
            if roles_bus is not None:
                if snapshot_roles is not None:
                    roles_bus[index, insert_idx] = snapshot_roles[index].to(
                        device=roles_bus.device,
                        dtype=roles_bus.dtype,
                    )
                else:
                    roles_bus[index, insert_idx] = -1
            if stride_bus is not None:
                stride_value = max(1, self.config.curriculum.B)
                stride_bus[index, insert_idx] = stride_value
                if stride_meta is not None:
                    stride_meta[index] = stride_value
            if version_bus is not None:
                new_version = current_max_version + 1
                version_bus[index, insert_idx] = new_version
                if version_meta is not None:
                    version_meta[index] = new_version
        batch["student_notes_bus"] = bus
        batch["student_bus_mask"] = mask
        if coverage_bus is not None:
            batch["student_bus_coverage"] = coverage_bus
        if roles_bus is not None:
            batch["student_bus_roles"] = roles_bus
        if stride_bus is not None:
            batch["student_bus_stride"] = stride_bus
        if version_bus is not None:
            batch["student_bus_version"] = version_bus
        return meta

    def _advance_commit_mask(self, batch: Dict[str, torch.Tensor]) -> None:
        commit_mask = batch.get("commit_mask")
        if commit_mask is None:
            return
        stride = max(1, self.config.curriculum.B)
        commit_mask = commit_mask.to(device=self.device, dtype=torch.bool)
        for index in range(commit_mask.size(0)):
            active_positions = commit_mask[index].nonzero(as_tuple=False).flatten()
            count = active_positions.numel()
            if count <= 1:
                continue
            max_shift = max(count - 1, 0)
            if max_shift == 0:
                continue
            shift = min(stride, max_shift)
            if shift <= 0:
                continue
            release_indices = active_positions[:shift]
            commit_mask[index, release_indices] = False
        batch["commit_mask"] = commit_mask

    def _uses_separate_teacher(self) -> bool:
        return bool(
            self.config.teacher.enabled
            and self.teacher_model is not None
            and self.teacher_model is not self.model
        )

    def _teacher_encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.teacher_model is None:
            raise RuntimeError("Teacher model requested but not initialised.")
        with torch.no_grad():
            return self.teacher_model.encode(batch["input_ids"], attention_mask=batch["attention_mask"])

    def _teacher_forward(
        self,
        hidden_states: torch.Tensor,
        *,
        role: torch.Tensor,
        notes: torch.Tensor,
        notes_mask: torch.Tensor,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not self.config.teacher.enabled and self.teacher_model is None:
            return None
        model = self.teacher_model if self.teacher_model is not None else self.model
        with torch.no_grad():
            return model(hidden_states, role=role, notes=notes, notes_mask=notes_mask)

    def _compute_losses(
        self,
        batch: Dict[str, torch.Tensor],
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Optional[Dict[str, torch.Tensor]],
        *,
        student_branch: Dict[str, torch.Tensor],
        teacher_branch: Dict[str, torch.Tensor],
        hidden_states: torch.Tensor,
        stage: int,
        weights: LossWeights,
        step: int,
        pre_update_logits: Optional[torch.Tensor] = None,
        stability_logging_due: bool = False,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        label_pad = self.collator_config.label_pad_id
        planner_logits_student = student_outputs["planner_logits"]
        vocab = planner_logits_student.size(-1)
        planner_ids = batch["planner_ids"]
        planner_loss = F.cross_entropy(
            planner_logits_student.view(-1, vocab),
            planner_ids.view(-1),
            ignore_index=label_pad,
        )

        teacher_notes = teacher_branch["notes"].to(student_outputs["notes_logits"].dtype)
        student_notes_pred = student_outputs["notes_logits"]
        if student_notes_pred.dim() != 3 or teacher_notes.dim() != 3:
            raise RuntimeError("Expected rank-3 tensors for student and teacher notes.")
        if student_notes_pred.size(2) != teacher_notes.size(2):
            raise RuntimeError(
                f"Notes dim mismatch: student {student_notes_pred.size()} vs teacher {teacher_notes.size()}"
            )
        if student_notes_pred.size(1) != teacher_notes.size(1):
            pooled = student_notes_pred.mean(dim=1, keepdim=True)
            student_notes_pred = pooled.expand(-1, teacher_notes.size(1), -1)
        notes_loss = F.mse_loss(student_notes_pred, teacher_notes)
        spec_pred = student_outputs["speculative_notes"]
        if spec_pred.dim() != 3:
            raise RuntimeError(f"Expected rank-3 speculative_notes, got {spec_pred.shape}.")
        if spec_pred.size(2) != teacher_notes.size(2):
            raise RuntimeError(
                f"Spec notes dim mismatch: student {spec_pred.size()} vs teacher {teacher_notes.size()}"
            )
        if spec_pred.size(1) != teacher_notes.size(1):
            spec_pred = spec_pred.mean(dim=1, keepdim=True).expand(-1, teacher_notes.size(1), -1)
        spec_loss = F.mse_loss(spec_pred, teacher_notes)

        spec_kl_loss = torch.tensor(0.0, device=self.device)
        if weights.spec_kl > 0.0:
            spec_predictions = student_outputs["speculative_notes"].to(self.device)
            notes_mask = student_branch["notes_mask"].to(self.device)
            coverage = student_branch.get("coverage")
            spec_kl_loss = self._interhead_spec_kl(
                spec_predictions,
                notes_mask,
                temperature=self.config.spec_kl_temperature,
                coverage=coverage.to(self.device) if isinstance(coverage, torch.Tensor) else None,
            )

        kd_loss = torch.tensor(0.0, device=self.device)
        stability_loss = torch.tensor(0.0, device=self.device)
        planner_mask = batch["planner_mask"].bool()
        commit_mask_tensor = batch.get("commit_mask")
        if commit_mask_tensor is not None:
            commit_mask_tensor = commit_mask_tensor.bool()
        commit_mask = (
            commit_mask_tensor
            if commit_mask_tensor is not None
            else torch.zeros_like(planner_mask, dtype=torch.bool)
        )
        rollback_mask = planner_mask & commit_mask
        stability_mask = planner_mask & (~commit_mask)

        teacher_logits: Optional[torch.Tensor] = None
        if teacher_outputs is not None:
            teacher_logits = teacher_outputs["planner_logits"].detach()
        if teacher_logits is not None and weights.kd > 0.0:
            kd_loss = self._masked_kl(planner_logits_student, teacher_logits, planner_mask)
        if pre_update_logits is not None and weights.stab > 0.0:
            stability_loss = self._masked_kl(
                planner_logits_student,
                pre_update_logits.detach(),
                stability_mask,
            )

        usage_loss = torch.tensor(0.0, device=self.device)
        mask_ablation_delta: Optional[float] = None
        log_usage_metric = (
            self.model.training
            and self.config.metrics.mask_ablation_every > 0
            and (step % self.config.metrics.mask_ablation_every == 0)
        )
        should_compute_usage = weights.use > 0.0 or log_usage_metric
        if should_compute_usage:
            masked_outputs = self.model(
                hidden_states,
                role=batch["role_ids"],
                notes=torch.zeros_like(student_branch["notes"]),
                notes_mask=torch.zeros_like(student_branch["notes_mask"]),
            )
            masked_logits = masked_outputs["planner_logits"]
            masked_loss = F.cross_entropy(
                masked_logits.view(-1, vocab),
                planner_ids.view(-1),
                ignore_index=label_pad,
            )
            mask_ablation_delta = float((masked_loss - planner_loss).detach().cpu())
            if weights.use > 0.0:
                usage_loss = torch.relu(masked_loss - planner_loss)

        coverage_loss = torch.tensor(0.0, device=self.device)
        nli_loss = torch.tensor(0.0, device=self.device)
        redundancy_loss = torch.tensor(0.0, device=self.device)
        role_loss = torch.tensor(0.0, device=self.device)
        agreement_loss = torch.tensor(0.0, device=self.device)
        agreement_precision = None
        coverage_metrics: Dict[str, float] = {}

        auto_agreement_labels = False
        if (
            stage >= 4
            and pre_update_logits is not None
        ):
            existing_labels = batch.get("agreement_labels")
            existing_mask = batch.get("agreement_mask")
            has_existing = bool(
                existing_labels is not None
                and existing_mask is not None
                and existing_mask.to(device=self.device).any()
            )
            if not has_existing:
                derived = self._derive_agreement_targets(
                    pre_update_logits.detach(),
                    planner_logits_student.detach(),
                    commit_mask,
                )
                if derived is not None:
                    labels_tensor, mask_tensor = derived
                    batch["agreement_labels"] = labels_tensor
                    batch["agreement_mask"] = mask_tensor
                    auto_agreement_labels = True

        coverage_logits = student_outputs.get("coverage_logits")
        coverage_mask = batch.get("coverage_mask")
        coverage_targets = batch.get("coverage_targets")
        if (
            coverage_logits is not None
            and coverage_targets is not None
            and coverage_mask is not None
        ):
            coverage_logits = coverage_logits.to(self.device)
            coverage_targets = coverage_targets.to(self.device)
            coverage_mask = coverage_mask.to(self.device).bool()
            if coverage_mask.any():
                if weights.cov > 0.0:
                    coverage_loss = F.binary_cross_entropy_with_logits(
                        coverage_logits[coverage_mask],
                        coverage_targets[coverage_mask],
                    )
                coverage_metrics = self._coverage_metrics(
                    coverage_logits,
                    coverage_targets,
                    coverage_mask,
                    plan_item_roles=batch.get("plan_item_roles"),
                    role_ids=batch["role_ids"],
                )
        if weights.nli > 0.0 and self.nli_scorer is not None:
            plan_mask = coverage_mask if coverage_mask is not None else batch.get("plan_item_mask")
            if plan_mask is not None:
                plan_mask = plan_mask.to(self.device).bool()
            nli_loss = self._nli_loss(batch, plan_mask)
        if weights.red > 0.0:
            redundancy_loss = self._redundancy_loss(batch)
        if weights.role > 0.0:
            role_logits = student_outputs.get("role_logits")
            if role_logits is not None:
                role_loss = F.cross_entropy(role_logits, batch["role_ids"])
        if weights.agree > 0.0 and stage >= 4:
            agreement_loss, agreement_precision = self._agreement_loss(
                batch,
                student_outputs,
            )

        should_log_stability = stability_logging_due
        rollback_kl_value = torch.tensor(0.0, device=self.device)
        stability_kl_value = torch.tensor(0.0, device=self.device)
        repair_error_rate: Optional[float] = None
        stability_error_rate: Optional[float] = None
        repair_margin: Optional[float] = None
        stability_margin: Optional[float] = None
        rollback_ratio: Optional[float] = None
        stability_ratio: Optional[float] = None
        if should_log_stability:
            total_tokens = float(planner_mask.sum().item())
            rollback_tokens = float(rollback_mask.sum().item())
            stability_tokens = float(stability_mask.sum().item())
            denom = max(total_tokens, 1.0)
            rollback_ratio = rollback_tokens / denom
            stability_ratio = stability_tokens / denom
            if teacher_logits is not None and rollback_tokens > 0.0:
                rollback_kl_value = self._masked_kl(planner_logits_student, teacher_logits, rollback_mask)
            if pre_update_logits is not None and stability_tokens > 0.0:
                post_detached = planner_logits_student.detach()
                pre_detached = pre_update_logits.detach()
                post_vs_pre = self._masked_kl(post_detached, pre_detached, stability_mask)
                pre_vs_post = self._masked_kl(pre_detached, post_detached, stability_mask)
                stability_kl_value = 0.5 * (post_vs_pre + pre_vs_post)

            student_argmax = planner_logits_student.argmax(dim=-1)
            if teacher_logits is not None:
                teacher_topk = torch.topk(teacher_logits, k=2, dim=-1) if teacher_logits.size(-1) > 1 else None
            else:
                teacher_topk = None
            if teacher_logits is not None and rollback_tokens > 0.0:
                teacher_argmax = teacher_logits.argmax(dim=-1)
                rollback_diff = (student_argmax != teacher_argmax) & rollback_mask
                repair_error_rate = float(rollback_diff.sum().item() / rollback_tokens)
                if teacher_topk is not None and rollback_mask.any():
                    margins = teacher_topk.values[..., 0] - teacher_topk.values[..., 1]
                    repair_margin = float(margins[rollback_mask].mean().item())
                else:
                    repair_margin = 0.0
            if pre_update_logits is not None and stability_tokens > 0.0:
                pre_argmax = pre_update_logits.argmax(dim=-1)
                stability_diff = (student_argmax != pre_argmax) & stability_mask
                stability_error_rate = float(stability_diff.sum().item() / stability_tokens)
                if stability_mask.any() and pre_update_logits.size(-1) > 1:
                    pre_topk = torch.topk(pre_update_logits, k=2, dim=-1)
                    margins = pre_topk.values[..., 0] - pre_topk.values[..., 1]
                    stability_margin = float(margins[stability_mask].mean().item())
                else:
                    stability_margin = 0.0

        total_loss = (
            planner_loss
            + notes_loss
            + 0.5 * spec_loss
            + weights.kd * kd_loss
            + weights.stab * stability_loss
            + weights.use * usage_loss
            + weights.cov * coverage_loss
            + weights.nli * nli_loss
            + weights.red * redundancy_loss
            + weights.spec_kl * spec_kl_loss
            + weights.role * role_loss
            + weights.agree * agreement_loss
        )
        metrics = {
            "planner_loss": float(planner_loss.detach().cpu()),
            "notes_loss": float(notes_loss.detach().cpu()),
            "spec_loss": float(spec_loss.detach().cpu()),
            "spec_kl_loss": float(spec_kl_loss.detach().cpu()),
            "kd_loss": float(kd_loss.detach().cpu()),
            "stability_loss": float(stability_loss.detach().cpu()),
            "usage_loss": float(usage_loss.detach().cpu()),
            "coverage_loss": float(coverage_loss.detach().cpu()),
            "nli_loss": float(nli_loss.detach().cpu()),
            "redundancy_loss": float(redundancy_loss.detach().cpu()),
            "role_loss": float(role_loss.detach().cpu()),
            "agreement_loss": float(agreement_loss.detach().cpu()),
        }
        if mask_ablation_delta is not None:
            metrics["mask_ablation"] = mask_ablation_delta
        if should_log_stability:
            metrics["rollback_kl"] = float(rollback_kl_value.detach().cpu())
            metrics["stability_kl"] = float(stability_kl_value.detach().cpu())
            metrics["rollback_ratio"] = float(rollback_ratio or 0.0)
            metrics["stability_ratio"] = float(stability_ratio or 0.0)
            metrics["rollback_tokens"] = float(rollback_mask.sum().item())
            metrics["stability_tokens"] = float(stability_mask.sum().item())
            if repair_error_rate is not None:
                metrics["repair_error_rate"] = repair_error_rate
            if stability_error_rate is not None:
                metrics["stability_error_rate"] = stability_error_rate
            if repair_margin is not None:
                metrics["repair_margin"] = repair_margin
            if stability_margin is not None:
                metrics["stability_margin"] = stability_margin
        planner_value = metrics["planner_loss"]
        if planner_value > 0:
            metrics["kd_ce_ratio"] = metrics["kd_loss"] / max(planner_value, 1e-6)
        if agreement_precision is not None:
            metrics["agreement_precision"] = agreement_precision
        if auto_agreement_labels:
            metrics["agreement_auto"] = 1.0
        if self.model.training:
            self._maybe_adjust_gradnorm(kd_loss.detach(), planner_loss.detach(), stage)
            metrics["kd_scale"] = float(self.kd_scale)
        metrics.update(coverage_metrics)
        return total_loss, metrics

    def _coverage_metrics(
        self,
        coverage_logits: torch.Tensor,
        coverage_targets: torch.Tensor,
        coverage_mask: torch.Tensor,
        *,
        plan_item_roles: Optional[torch.Tensor],
        role_ids: torch.Tensor,
    ) -> Dict[str, float]:
        mask = coverage_mask.to(device=self.device, dtype=torch.bool)
        if coverage_logits.shape != coverage_targets.shape:
            raise ValueError(
                "coverage_logits and coverage_targets must share shape "
                f"(got {coverage_logits.shape} vs {coverage_targets.shape})."
            )
        if mask.shape != coverage_targets.shape:
            raise ValueError(
                "coverage_mask must match coverage_targets shape "
                f"(got {mask.shape} vs {coverage_targets.shape})."
            )
        if mask.numel() == 0 or not bool(mask.any()):
            return {}
        probs = torch.sigmoid(coverage_logits)
        predictions = probs >= self.config.coverage_threshold
        targets = coverage_targets >= 0.5

        tp = float((predictions & targets & mask).sum().item())
        fp = float((predictions & (~targets) & mask).sum().item())
        fn = float((~predictions & targets & mask).sum().item())
        tn = float((~predictions & (~targets) & mask).sum().item())

        predicted_positive = tp + fp
        positive_support = tp + fn
        negative_support = fp + tn

        precision = tp / predicted_positive if predicted_positive > 0 else 0.0
        recall = tp / positive_support if positive_support > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0.0
            else 0.0
        )

        metrics: Dict[str, float] = {
            "coverage_precision": precision,
            "coverage_recall": recall,
            "coverage_f1": f1,
            "coverage_tp": tp,
            "coverage_fp": fp,
            "coverage_fn": fn,
            "coverage_support_pos": positive_support,
            "coverage_support_neg": negative_support,
            "coverage_threshold": float(self.config.coverage_threshold),
        }

        if plan_item_roles is not None and plan_item_roles.shape == coverage_targets.shape:
            roles_tensor = plan_item_roles.to(device=self.device, dtype=torch.long)
            role_ids_tensor = role_ids.to(device=self.device, dtype=torch.long).view(-1, 1)
            role_ids_tensor = role_ids_tensor.expand_as(roles_tensor)
            valid_roles = roles_tensor >= 0
            same_role_mask = mask & valid_roles & (roles_tensor == role_ids_tensor)
            cross_role_mask = mask & valid_roles & (roles_tensor != role_ids_tensor)

            same_role_tp = float((predictions & targets & same_role_mask).sum().item())
            same_role_positive = float((targets & same_role_mask).sum().item())
            cross_role_fp = float((predictions & (~targets) & cross_role_mask).sum().item())
            cross_role_total = float(cross_role_mask.sum().item())

            metrics["coverage_same_role_recall"] = (
                same_role_tp / same_role_positive if same_role_positive > 0 else 0.0
            )
            metrics["coverage_same_role_support"] = same_role_positive
            metrics["coverage_cross_role_fp_rate"] = (
                cross_role_fp / cross_role_total if cross_role_total > 0 else 0.0
            )
            metrics["coverage_cross_role_support"] = cross_role_total

        return metrics

    def _masked_kl(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="none") * (temperature**2)
        kl = kl.sum(dim=-1)
        mask = mask.to(dtype=kl.dtype, device=kl.device)
        denom = mask.sum().clamp_min(1.0)
        return (kl * mask).sum() / denom

    def _interhead_spec_kl(
        self,
        speculative_notes: torch.Tensor,
        notes_mask: torch.Tensor,
        *,
        temperature: float,
        coverage: Optional[torch.Tensor] = None,
        min_overlap: float = 1e-5,
    ) -> torch.Tensor:
        if temperature <= 0.0:
            raise ValueError("spec_kl_temperature must be positive.")
        if speculative_notes.dim() != 3:
            raise ValueError(
                f"Expected speculative_notes to have shape [batch, notes, dim], got {speculative_notes.shape}."
            )
        if notes_mask.dim() != 2:
            raise ValueError(
                f"Expected notes_mask to have shape [batch, notes], got {notes_mask.shape}."
            )
        batch_size, _, feature_dim = speculative_notes.shape
        if feature_dim == 0:
            return torch.tensor(0.0, device=speculative_notes.device)
        coverage_tensor: Optional[torch.Tensor] = None
        if coverage is not None:
            if coverage.dim() != 2:
                raise ValueError(
                    f"Expected coverage to have shape [batch, notes], got {coverage.shape}."
                )
            coverage_tensor = coverage.to(device=speculative_notes.device, dtype=torch.float32)

        total = torch.tensor(0.0, device=speculative_notes.device)
        weight_total = torch.tensor(0.0, device=speculative_notes.device)
        notes_mask = notes_mask.to(device=speculative_notes.device, dtype=torch.bool)

        for batch_index in range(batch_size):
            active_indices = notes_mask[batch_index].nonzero(as_tuple=False).flatten()
            if active_indices.numel() < 2:
                continue
            sample_notes = speculative_notes[batch_index, active_indices]
            log_probs = F.log_softmax(sample_notes / temperature, dim=-1)
            probs = log_probs.exp()
            log_ratio = log_probs.unsqueeze(1) - log_probs.unsqueeze(0)
            kl_matrix = torch.sum(probs.unsqueeze(1) * log_ratio, dim=-1)
            sym_kl = kl_matrix + kl_matrix.transpose(0, 1)
            indices = torch.triu_indices(sym_kl.size(0), sym_kl.size(1), offset=1, device=sym_kl.device)
            if indices.numel() == 0:
                continue
            pair_values = sym_kl[indices[0], indices[1]]

            if coverage_tensor is not None:
                sample_cov = coverage_tensor[batch_index, active_indices]
                weights = torch.minimum(sample_cov[indices[0]], sample_cov[indices[1]])
                weights = weights.clamp_min(0.0)
                if min_overlap > 0.0:
                    active = weights > min_overlap
                    if not torch.any(active):
                        continue
                    pair_values = pair_values[active]
                    weights = weights[active]
            else:
                weights = torch.ones_like(pair_values, device=pair_values.device)

            total = total + torch.sum(pair_values * weights)
            weight_total = weight_total + torch.sum(weights)

        if weight_total.item() == 0.0:
            return torch.tensor(0.0, device=speculative_notes.device)
        return total / weight_total

    def _update_teacher_ema(self) -> None:
        if not self.config.teacher.enabled or self.config.teacher.type != "ema":
            return
        if self.teacher_model is None:
            return
        decay = self.config.teacher.ema_decay
        with torch.no_grad():
            for student_param, teacher_param in zip(self.model.parameters(), self.teacher_model.parameters()):
                teacher_param.mul_(decay).add_(student_param * (1.0 - decay))

    def _nli_loss(
        self,
        batch: Dict[str, torch.Tensor],
        plan_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.nli_scorer is None:
            return torch.tensor(0.0, device=self.device)
        notes_text = batch.get("notes_text")
        plan_text = batch.get("plan_text")
        if notes_text is None or plan_text is None:
            return torch.tensor(0.0, device=self.device)
        pairs: List[Tuple[str, str]] = []
        for batch_index, (note, items) in enumerate(zip(notes_text, plan_text)):
            if not note or not items:
                continue
            item_mask = None
            if plan_mask is not None and plan_mask.size(0) > batch_index:
                item_mask = plan_mask[batch_index]
            for item_index, item in enumerate(items):
                if item_mask is not None and item_mask.size(0) > item_index:
                    if not bool(item_mask[item_index]):
                        continue
                pairs.append((note, item))
        if not pairs:
            return torch.tensor(0.0, device=self.device)
        probs = self.nli_scorer.score(pairs)
        if probs.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        contra_idx = self.nli_scorer.label_index.get("contradiction", 0)
        neutral_idx = self.nli_scorer.label_index.get("neutral", 1)
        contradiction = probs[:, contra_idx]
        neutral = probs[:, neutral_idx]
        margin = self.config.nli_margin
        penalties = torch.relu(contradiction - neutral - margin)
        return penalties.mean()

    def _redundancy_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        student_bus = batch.get("student_notes_bus")
        mask = batch.get("student_bus_mask")
        if student_bus is None or mask is None:
            return torch.tensor(0.0, device=self.device)
        notes = student_bus.to(self.device, dtype=torch.float32)
        mask = mask.to(self.device)
        roles = notes.size(-2)
        if roles < 2 or mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        total = torch.tensor(0.0, device=self.device)
        count = torch.tensor(0.0, device=self.device)
        margin = 0.7
        for batch_index in range(notes.size(0)):
            for snapshot_index in range(notes.size(1)):
                if not mask[batch_index, snapshot_index]:
                    continue
                snapshot = notes[batch_index, snapshot_index]
                for i in range(roles):
                    for j in range(i + 1, roles):
                        sim = F.cosine_similarity(
                            snapshot[i].unsqueeze(0), snapshot[j].unsqueeze(0), dim=-1
                        )
                        total = total + torch.relu(sim - margin)
                        count = count + 1
        if count.item() == 0:
            return torch.tensor(0.0, device=self.device)
        return total / count

    def _derive_agreement_targets(
        self,
        pre_logits: torch.Tensor,
        post_logits: torch.Tensor,
        commit_mask: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        commit_mask = commit_mask.to(device=post_logits.device, dtype=torch.bool)
        if commit_mask.sum().item() == 0:
            return None
        pre_logits = pre_logits.to(device=post_logits.device)
        pre_log_probs = F.log_softmax(pre_logits, dim=-1)
        post_log_probs = F.log_softmax(post_logits, dim=-1)
        pre_probs = pre_log_probs.exp()
        post_probs = post_log_probs.exp()
        kl_pre_post = F.kl_div(pre_log_probs, post_probs, reduction="none").sum(dim=-1)
        kl_post_pre = F.kl_div(post_log_probs, pre_probs, reduction="none").sum(dim=-1)
        sym_kl = 0.5 * (kl_pre_post + kl_post_pre)
        same_argmax = pre_logits.argmax(dim=-1) == post_logits.argmax(dim=-1)
        stable = same_argmax & (sym_kl <= self.config.agreement_threshold)
        batch_size, sequence_length = stable.shape
        commit_counts = commit_mask.sum(dim=1)
        max_commit = int(commit_counts.max().item())
        if max_commit == 0:
            return None
        labels = torch.zeros(
            (batch_size, max_commit),
            dtype=torch.float32,
            device=post_logits.device,
        )
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for batch_index in range(batch_size):
            count = int(commit_counts[batch_index].item())
            if count == 0:
                continue
            slot_start = max_commit - count
            token_indices = commit_mask[batch_index].nonzero(as_tuple=False).flatten()
            values = stable[batch_index, token_indices]
            labels[batch_index, slot_start:] = values[-count:].float()
            mask[batch_index, slot_start:] = True
        return labels, mask

    def _agreement_loss(
        self,
        batch: Dict[str, torch.Tensor],
        student_outputs: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[float]]:
        agreement_labels = batch.get("agreement_labels")
        agreement_mask = batch.get("agreement_mask")
        if agreement_labels is None or agreement_mask is None:
            return torch.tensor(0.0, device=self.device), None
        labels = agreement_labels.to(self.device)
        mask = agreement_mask.to(self.device)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device), None
        scores = student_outputs["agreement"].squeeze(-1)
        pooled = torch.zeros_like(labels, dtype=scores.dtype)
        sequence_length = scores.size(1)
        slots = labels.size(1)
        for batch_index in range(scores.size(0)):
            start = max(0, sequence_length - slots)
            selected = scores[batch_index, start:sequence_length]
            pooled[batch_index, -selected.size(0) :] = selected
        flat_mask = mask.view(-1)
        valid_indices = flat_mask.nonzero(as_tuple=False).view(-1)
        if valid_indices.numel() == 0:
            return torch.tensor(0.0, device=self.device), None
        pooled_flat = pooled.view(-1)
        labels_flat = labels.view(-1).float()
        logits = pooled_flat[valid_indices]
        targets = labels_flat[valid_indices]
        loss = F.binary_cross_entropy(logits, targets)
        preds = (logits > 0.5).float()
        predicted_positive = preds.sum()
        if predicted_positive.item() == 0:
            precision = 1.0
        else:
            correct_positive = (((preds == targets) & (preds == 1)).float()).sum()
            precision = float((correct_positive / predicted_positive).detach().cpu())
        return loss, precision

    def _maybe_adjust_gradnorm(
        self,
        kd_loss: torch.Tensor,
        planner_loss: torch.Tensor,
        stage: int,
    ) -> None:
        cfg = self.config.gradnorm
        if not cfg.enabled or stage < 2:
            return
        if kd_loss.isnan() or planner_loss.isnan():
            return
        planner_value = planner_loss.detach().abs().clamp_min(1e-6)
        ratio = float((kd_loss.detach().abs() / planner_value).cpu())
        target = cfg.target_ratio
        error = ratio - target
        new_scale = self.kd_scale * (1.0 - cfg.alpha * error)
        new_scale = max(cfg.min_scale, min(cfg.max_scale, new_scale))
        self.kd_scale = new_scale

    def evaluate(self) -> Dict[str, float]:
        if self.eval_dataset is None:
            return {}
        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )
        self.model.eval()
        losses: list[float] = []
        stage = self._determine_stage()
        active_weights = self._active_loss_weights(stage)
        with torch.no_grad():
            for batch in dataloader:
                batch = {
                    key: value.to(self.device) if torch.is_tensor(value) else value for key, value in batch.items()
                }
                plan_item_ids = batch.get("plan_item_ids")
                plan_item_mask = batch.get("plan_item_mask")
                if plan_item_ids is not None:
                    plan_item_ids = plan_item_ids.to(self.device)
                if plan_item_mask is not None:
                    plan_item_mask = plan_item_mask.to(self.device)
                hidden_states = self.model.encode(batch["input_ids"], attention_mask=batch["attention_mask"])
                teacher_branch = self._prepare_branch_inputs(batch, branch="teacher", stage=stage)
                student_branch = self._prepare_branch_inputs(
                    batch,
                    branch="student",
                    stage=stage,
                    teacher_branch=teacher_branch,
                )
                teacher_branch["notes"] = teacher_branch["notes"].to(hidden_states.dtype)
                student_branch["notes"] = student_branch["notes"].to(hidden_states.dtype)
                if "pre_notes" in student_branch:
                    student_branch["pre_notes"] = student_branch["pre_notes"].to(hidden_states.dtype)
                if self._uses_separate_teacher():
                    teacher_hidden = self._teacher_encode(batch)
                else:
                    teacher_hidden = hidden_states.detach()
                student_outputs = self._run_student_pass(
                    hidden_states,
                    batch,
                    student_branch,
                    stage=stage,
                    plan_item_ids=plan_item_ids,
                    plan_item_mask=plan_item_mask,
                )
                teacher_outputs = self._teacher_forward(
                    teacher_hidden,
                    role=batch["role_ids"],
                    notes=teacher_branch["notes"],
                    notes_mask=teacher_branch["notes_mask"],
                )
                pre_update_logits: Optional[torch.Tensor] = None
                if active_weights.stab > 0.0 and "pre_notes" in student_branch and "pre_notes_mask" in student_branch:
                    pre_outputs = self.model(
                        hidden_states,
                        role=batch["role_ids"],
                        notes=student_branch["pre_notes"],
                        notes_mask=student_branch["pre_notes_mask"],
                        plan_item_ids=plan_item_ids,
                        plan_item_mask=plan_item_mask,
                    )
                    pre_update_logits = pre_outputs["planner_logits"]
                total_loss, _ = self._compute_losses(
                    batch,
                    student_outputs,
                    teacher_outputs,
                    student_branch=student_branch,
                    teacher_branch=teacher_branch,
                    hidden_states=hidden_states,
                    stage=stage,
                    weights=active_weights,
                    step=self.state.global_step,
                    pre_update_logits=pre_update_logits,
                    stability_logging_due=False,
                )
                losses.append(float(total_loss.detach().cpu()))
        avg_loss = float(sum(losses) / max(1, len(losses)))
        self.model.train()
        if self.teacher_model is not None and self.teacher_model is not self.model:
            self.teacher_model.eval()
        metrics = {"eval_loss": avg_loss}
        if avg_loss < self.state.best_eval_loss:
            self.state.best_eval_loss = avg_loss
        self._log_metrics("eval", metrics)
        return metrics

    def _log_metrics(self, prefix: str, metrics: Dict[str, float]) -> None:
        message = " | ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        print(f"{prefix}: step={self.state.global_step} | {message}")
        record = dict(metrics)
        record["step"] = self.state.global_step
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        history = self.metric_history.setdefault(prefix, [])
        history.append(record)
