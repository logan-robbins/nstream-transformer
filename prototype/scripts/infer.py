"""Inference CLI for the GPT-OSS backed N-Stream Transformer."""

from __future__ import annotations

# Ensure local src/ is on sys.path when running from the repo without installation
import os as _os, sys as _sys
_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_SRC_PATH = _os.path.join(_REPO_ROOT, "src")
if _SRC_PATH not in _sys.path and _os.path.isdir(_SRC_PATH):
    _sys.path.insert(0, _SRC_PATH)

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from nstream_transformer.config import ModelConfig, TrainingConfig
from nstream_transformer.inference import (
    DecodeConfig,
    MultiStreamOrchestrator,
    StepOutcome,
    build_inference_config,
)
from nstream_transformer.models import NStreamTransformer
from nstream_transformer.integration.gpt_oss import TrunkAdapterConfig
from nstream_transformer.models.role_adapters import RoleAdapterConfig
from nstream_transformer.models.heads import (
    PlannerHeadConfig,
    NotesHeadConfig,
    SpeculationHeadConfig,
    AgreementHeadConfig,
)
from nstream_transformer.inference.dnb_bus import DynamicNotesBusConfig
from nstream_transformer.inference.snc_cross_attn import SharedNotesCrossAttentionConfig
from nstream_transformer.training.trainer import (
    CurriculumConfig,
    GradNormConfig,
    LossWeights,
    MetricsConfig,
    NegativeSamplingConfig,
    NotesNoiseConfig,
    StagePolicyConfig,
    TeacherBranchConfig,
)
from nstream_transformer.data.collator_kd import TwoBranchKDCollatorConfig
from nstream_transformer.data.teacher_provider import TeacherRunnerConfig
from nstream_transformer.data.tokenizer import TokenizerConfig, resolve_tokenizer
from nstream_transformer.utils import configure_logging, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run N-Stream inference with GPT-OSS backbone.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gpt_oss_transfer.yaml"),
        help="Path to YAML configuration file used for training.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text to decode. Mutually exclusive with --prompt-file.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional text file containing the prompt. Overrides --prompt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/infer/manifest.json"),
        help="Where to write the inference manifest JSON.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional adapter checkpoint file or directory containing adapters.pt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed overriding the inference config.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override for maximum tokens generated per role.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override sampling temperature.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override sampling top-k.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Override sampling top-p.",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Disable sampling and use greedy decoding.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default=None,
        help=(
            "Override the preferred runtime device (defaults to NSTREAM_DEVICE env or auto-detect)."
        ),
    )
    parser.add_argument(
        "--hf-verbosity",
        choices=["critical", "error", "warning", "info", "debug"],
        default=None,
        help="Optional override for Hugging Face `TRANSFORMERS_VERBOSITY`.",
    )
    parser.add_argument(
        "--topology",
        choices=["triangular", "hierarchical"],
        default=None,
        help="Override the notes topology used during inference.",
    )
    parser.add_argument(
        "--hierarchy-level",
        action="append",
        default=None,
        help=(
            "Comma-separated list of roles for a hierarchy level (earliest to latest). "
            "Repeat the flag to add additional levels."
        ),
    )
    parser.add_argument(
        "--cadence-mode",
        choices=["deterministic", "stochastic", "adaptive"],
        default=None,
        help="Override the cadence policy mode used during inference.",
    )
    parser.add_argument(
        "--cadence-max-interval",
        type=int,
        default=None,
        help="Force an emission after this many tokens without notes when using stochastic/adaptive cadence.",
    )
    parser.add_argument(
        "--cadence-min-prob",
        type=float,
        default=None,
        help="Minimum emission probability clamp for stochastic/adaptive cadence.",
    )
    parser.add_argument(
        "--cadence-m-min",
        type=float,
        default=None,
        help="Lower multiplier bound for adaptive cadence modulation.",
    )
    parser.add_argument(
        "--cadence-m-max",
        type=float,
        default=None,
        help="Upper multiplier bound for adaptive cadence modulation.",
    )
    parser.add_argument(
        "--cadence-agreement-low",
        type=float,
        default=None,
        help="Agreement threshold below which adaptive cadence applies maximum multiplier.",
    )
    parser.add_argument(
        "--cadence-agreement-high",
        type=float,
        default=None,
        help="Agreement threshold above which adaptive cadence applies minimum multiplier.",
    )
    parser.add_argument(
        "--cadence-age-boost",
        type=float,
        default=None,
        help="Additional multiplier growth per cadence interval when notes age exceeds the base cadence.",
    )
    parser.add_argument(
        "--gate-g",
        type=float,
        default=None,
        help="Override SNC gate scalar in [0,1] (set 0 for no cross-lane influence).",
    )
    parser.add_argument(
        "--role-prefix-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping { role: prefix } to prepend per-role before the shared prompt. "
            "Useful for POC demos to steer each lane to a distinct part."
        ),
    )
    parser.add_argument(
        "--role",
        action="append",
        default=None,
        help=(
            "Override role names (repeat for each lane, e.g. --role stream_1 --role stream_2 --role stream_3). "
            "Names are normalised to lowercase."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream per-token outputs to stdout.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        text = args.prompt_file.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Prompt file {args.prompt_file} is empty.")
        return text
    if args.prompt is None or not args.prompt.strip():
        raise ValueError("Provide either --prompt or --prompt-file.")
    return args.prompt


def _coerce_model_config(payload: Dict[str, Any]) -> ModelConfig:
    data = dict(payload)
    if isinstance(data.get("trunk"), dict):
        data["trunk"] = TrunkAdapterConfig(**data["trunk"])
    if isinstance(data.get("role_adapters"), dict):
        data["role_adapters"] = RoleAdapterConfig(**data["role_adapters"])
    if isinstance(data.get("notes_bus"), dict):
        data["notes_bus"] = DynamicNotesBusConfig(**data["notes_bus"])
    if isinstance(data.get("cross_attention"), dict):
        data["cross_attention"] = SharedNotesCrossAttentionConfig(**data["cross_attention"])
    if isinstance(data.get("planner_head"), dict):
        data["planner_head"] = PlannerHeadConfig(**data["planner_head"])
    if isinstance(data.get("notes_head"), dict):
        data["notes_head"] = NotesHeadConfig(**data["notes_head"])
    if isinstance(data.get("speculation_head"), dict):
        data["speculation_head"] = SpeculationHeadConfig(**data["speculation_head"])
    if isinstance(data.get("agreement_head"), dict):
        data["agreement_head"] = AgreementHeadConfig(**data["agreement_head"])
    if isinstance(data.get("collator"), dict):
        data["collator"] = TwoBranchKDCollatorConfig(**data["collator"])
    return ModelConfig(**data)


def _coerce_training_config(payload: Dict[str, Any]) -> TrainingConfig:
    data = dict(payload)
    if isinstance(data.get("curriculum"), dict):
        data["curriculum"] = CurriculumConfig(**data["curriculum"])
    if isinstance(data.get("teacher"), dict):
        data["teacher"] = TeacherBranchConfig(**data["teacher"])
    if isinstance(data.get("teacher_runner"), dict):
        data["teacher_runner"] = TeacherRunnerConfig(**data["teacher_runner"])
    if isinstance(data.get("loss_weights"), dict):
        data["loss_weights"] = LossWeights(**data["loss_weights"])
    if isinstance(data.get("notes_noise"), dict):
        data["notes_noise"] = NotesNoiseConfig(**data["notes_noise"])
    if isinstance(data.get("metrics"), dict):
        data["metrics"] = MetricsConfig(**data["metrics"])
    if isinstance(data.get("negative_sampling"), dict):
        data["negative_sampling"] = NegativeSamplingConfig(**data["negative_sampling"])
    if isinstance(data.get("gradnorm"), dict):
        data["gradnorm"] = GradNormConfig(**data["gradnorm"])
    if isinstance(data.get("stage_policies"), dict):
        policies: Dict[int, StagePolicyConfig] = {}
        for key, policy_payload in data["stage_policies"].items():
            try:
                index = int(key)
            except (TypeError, ValueError) as err:
                raise ValueError(f"Stage policy keys must be integers, received {key!r}.") from err
            payload_copy = dict(policy_payload or {})
            if isinstance(payload_copy.get("notes_noise"), dict):
                payload_copy["notes_noise"] = NotesNoiseConfig(**payload_copy["notes_noise"])
            policies[index] = StagePolicyConfig(**payload_copy)
        data["stage_policies"] = policies
    return TrainingConfig(**data)


def override_decode_config(config: DecodeConfig, args: argparse.Namespace) -> None:
    if args.max_new_tokens is not None:
        config.max_new_tokens = int(args.max_new_tokens)
    if args.temperature is not None:
        config.temperature = float(args.temperature)
    if args.top_k is not None:
        config.top_k = int(args.top_k)
    if args.top_p is not None:
        config.top_p = float(args.top_p)
    if args.no_sample:
        config.do_sample = False


def format_event(event: StepOutcome) -> str:
    token_repr = event.token_text
    if "\n" in token_repr:
        token_repr = token_repr.replace("\n", "\\n")
    return (
        f"[role={event.role} stride={event.stride_index}] token={event.token_id} "
        f"text='{token_repr}' agree={event.agreement:.3f} "
        f"notes={'Y' if event.notes_emitted else 'N'} rollback={'Y' if event.rollback_performed else 'N'}"
    )


def main() -> None:
    args = parse_args()
    if args.hf_verbosity is not None:
        _os.environ["TRANSFORMERS_VERBOSITY"] = args.hf_verbosity
    logger = configure_logging(
        name="nstream.cli.infer",
        extra_loggers=[
            "nstream.gpt_oss.trunk",
            "nstream.inference",
            "nstream.training",
            "transformers",
        ],
    )
    if args.device is not None:
        _os.environ["NSTREAM_DEVICE"] = args.device
    prompt = resolve_prompt(args)
    raw_cfg = load_config(args.config)
    model_cfg = _coerce_model_config(raw_cfg.get("model", {}))
    training_cfg = _coerce_training_config(raw_cfg.get("training", {}))

    if args.role:
        role_list = [role.strip().lower() for role in args.role if role and role.strip()]
        if not role_list:
            raise ValueError("At least one --role value must be provided when overriding roles.")
        collator_cfg = model_cfg.collator
        model_cfg.collator = replace(
            collator_cfg,
            role_to_id={name: index for index, name in enumerate(role_list)},
        )
        adapter_cfg = model_cfg.role_adapters
        model_cfg.role_adapters = replace(adapter_cfg, roles=tuple(role_list))
        if getattr(training_cfg, "teacher_runner", None) is not None:
            training_cfg.teacher_runner = replace(training_cfg.teacher_runner, roles=tuple(role_list))

    if args.device is not None:
        trunk_cfg = model_cfg.trunk
        dtype_alias = trunk_cfg.torch_dtype
        if args.device == "mps" and isinstance(dtype_alias, str) and dtype_alias.lower() in {"bfloat16", "bf16"}:
            dtype_alias = "float16"
        device_map_override = trunk_cfg.device_map
        if args.device != "cuda":
            device_map_override = None
        model_cfg.trunk = replace(trunk_cfg, torch_dtype=dtype_alias, device_map=device_map_override)
        if hasattr(training_cfg, "device"):
            training_cfg.device = args.device

    hierarchy_levels: Optional[Tuple[Tuple[str, ...], ...]] = None
    if args.hierarchy_level:
        parsed_levels: list[Tuple[str, ...]] = []
        for index, level_spec in enumerate(args.hierarchy_level, start=1):
            entries = [item.strip() for item in level_spec.split(",") if item.strip()]
            if not entries:
                raise ValueError(f"Hierarchy level {index} is empty.")
            parsed_levels.append(tuple(entries))
        hierarchy_levels = tuple(parsed_levels)

    topology_override = args.topology
    if hierarchy_levels is not None:
        if topology_override is None:
            topology_override = "hierarchical"
        elif topology_override != "hierarchical":
            raise ValueError("Hierarchy levels provided but topology is not set to 'hierarchical'.")

    base_model_ref = model_cfg.trunk.base_model
    tokenizer_override: Optional[Path] = None
    try:
        base_model_path = Path(base_model_ref)
        if base_model_path.exists():
            candidate = base_model_path.parent / "tokenizer"
            if candidate.is_dir():
                tokenizer_override = candidate
    except (TypeError, ValueError):  # pragma: no cover - defensive path for remote refs
        tokenizer_override = None

    tokenizer_cfg = TokenizerConfig(
        pretrained_name=base_model_ref,
        custom_path=tokenizer_override,
    )
    tokenizer, tokenizer_manifest = resolve_tokenizer(tokenizer_cfg)

    seed_override = args.seed if args.seed is not None else getattr(training_cfg, "seed", None)
    seed_everything(seed_override)

    logger.info(
        "infer_start | config=%s | prompt_chars=%d | base_model=%s",
        str(args.config),
        len(prompt),
        model_cfg.trunk.base_model,
    )

    model = NStreamTransformer(model_cfg)
    model.trunk_adapter.load_model()
    logger.info(
        "trunk_loaded | base_model=%s | device=%s",
        model_cfg.trunk.base_model,
        model.trunk_adapter.model.device if hasattr(model.trunk_adapter.model, "device") else "unknown",
    )
    if args.checkpoint is not None:
        import torch
        state = None
        if args.checkpoint.is_dir():
            for candidate in (
                args.checkpoint / "adapters.pt",
                args.checkpoint / "adapter_state.pt",
                args.checkpoint / "state_dict.pt",
            ):
                if candidate.exists():
                    state = torch.load(candidate, map_location="cpu")
                    break
        else:
            state = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(state, dict):
            model.load_adapters(state, strict=False)

    inference_cfg = build_inference_config(
        training_cfg,
        role_to_id=dict(model_cfg.collator.role_to_id),
        rng_seed=seed_override,
        topology=topology_override,
        hierarchy_levels=hierarchy_levels,
    )
    if seed_override is not None:
        inference_cfg.rng_seed = seed_override
        inference_cfg.decode.seed = seed_override
    override_decode_config(inference_cfg.decode, args)

    cadence_policy = inference_cfg.cadence_policy
    if args.cadence_mode is not None:
        cadence_policy.mode = args.cadence_mode
    if args.cadence_max_interval is not None:
        cadence_policy.max_interval = int(args.cadence_max_interval)
    if args.cadence_min_prob is not None:
        cadence_policy.min_probability = float(args.cadence_min_prob)
    if args.cadence_m_min is not None:
        cadence_policy.multiplier_min = float(args.cadence_m_min)
    if args.cadence_m_max is not None:
        cadence_policy.multiplier_max = float(args.cadence_m_max)
    if args.cadence_agreement_low is not None:
        cadence_policy.agreement_low = float(args.cadence_agreement_low)
    if args.cadence_agreement_high is not None:
        cadence_policy.agreement_high = float(args.cadence_agreement_high)
    if args.cadence_age_boost is not None:
        cadence_policy.age_boost = float(args.cadence_age_boost)
    inference_cfg._validate_cadence_policy()

    if args.gate_g is not None:
        try:
            gate_val = float(args.gate_g)
        except Exception as err:  # pragma: no cover - CLI guard
            raise ValueError("--gate-g must be a float in [0,1].") from err
        if not (0.0 <= gate_val <= 1.0):
            raise ValueError("--gate-g must lie within [0,1].")
        inference_cfg.gate_g = gate_val

    orchestrator = MultiStreamOrchestrator(
        model,
        tokenizer,
        inference_cfg,
    )
    role_prefix_map = None
    if args.role_prefix_file is not None:
        if not args.role_prefix_file.exists():
            raise FileNotFoundError(f"role prefix file not found: {args.role_prefix_file}")
        try:
            with args.role_prefix_file.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            if not isinstance(raw, dict):
                raise ValueError("--role-prefix-file must contain a JSON object mapping role -> prefix string.")
            role_prefix_map = {}
            for key, value in raw.items():
                if not isinstance(value, str):
                    raise ValueError("role prefix values must be strings.")
                role_key = str(key).lower()
                if role_key not in inference_cfg.roles:
                    raise ValueError(f"role '{role_key}' in role-prefix mapping not present in inference roles {inference_cfg.roles}.")
                role_prefix_map[role_key] = value
        except Exception:
            raise

    orchestrator.start(prompt, prefix_by_role=role_prefix_map)

    events: list[StepOutcome] = []
    while True:
        outcome = orchestrator.step()
        if outcome is None:
            break
        events.append(outcome)
        if args.verbose:
            logger.info(format_event(outcome))

    manifest = orchestrator.finalize()
    manifest["prompt"] = prompt
    manifest["tokenizer"] = tokenizer_manifest.to_dict()
    manifest["events"] = [
        {
            "role": event.role,
            "token_id": event.token_id,
            "token_text": event.token_text,
            "stride_index": event.stride_index,
            "stride_completed": event.stride_completed,
            "role_completed": event.role_completed,
            "agreement": event.agreement,
            "notes_emitted": event.notes_emitted,
            "rollback_performed": event.rollback_performed,
            "cadence_mode": event.cadence_mode,
            "cadence_probability": event.cadence_probability,
            "cadence_multiplier": event.cadence_multiplier,
            "cadence_forced": event.cadence_forced,
        }
        for event in events
    ]
    manifest_path = args.output
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("infer_complete | manifest=%s | roles=%s", str(manifest_path), ", ".join(inference_cfg.roles))


if __name__ == "__main__":
    main()
