"""Training entry point for the GPT-OSS powered N-Stream Transformer."""

from __future__ import annotations

# Ensure local src/ is on sys.path when running from the repo without installation
import os as _os, sys as _sys
_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_SRC_PATH = _os.path.join(_REPO_ROOT, "src")
if _SRC_PATH not in _sys.path and _os.path.isdir(_SRC_PATH):
    _sys.path.insert(0, _SRC_PATH)

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml
import logging

from nstream_transformer.config import ModelConfig, TrainingConfig
from nstream_transformer.models import NStreamTransformer
from nstream_transformer.models.role_adapters import RoleAdapterConfig
from nstream_transformer.models.heads import (
    PlannerHeadConfig,
    NotesHeadConfig,
    SpeculationHeadConfig,
    AgreementHeadConfig,
)
from nstream_transformer.inference.dnb_bus import DynamicNotesBusConfig
from nstream_transformer.inference.snc_cross_attn import SharedNotesCrossAttentionConfig
from nstream_transformer.integration.gpt_oss import TrunkAdapterConfig
from nstream_transformer.training import KDJsonlDataset
from nstream_transformer.training.trainer import (
    Trainer,
    CurriculumConfig,
    LossWeights,
    MetricsConfig,
    NotesNoiseConfig,
    NegativeSamplingConfig,
    GradNormConfig,
    StagePolicyConfig,
    TeacherBranchConfig,
)
from nstream_transformer.data.teacher_provider import TeacherRunnerConfig
from nstream_transformer.utils import configure_logging, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune GPT-OSS with N-Stream adapters.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gpt_oss_transfer.yaml"),
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def maybe_create_dataset(path: str | None) -> KDJsonlDataset | None:
    if not path:
        return None
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    return KDJsonlDataset(dataset_path)


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
    return ModelConfig(**data)


def _coerce_training_config(payload: Dict[str, Any]) -> TrainingConfig:
    data = dict(payload)
    if isinstance(data.get("teacher"), dict):
        data["teacher"] = TeacherBranchConfig(**data["teacher"])
    if isinstance(data.get("teacher_runner"), dict):
        data["teacher_runner"] = TeacherRunnerConfig(**data["teacher_runner"])
    if isinstance(data.get("curriculum"), dict):
        data["curriculum"] = CurriculumConfig(**data["curriculum"])
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
            payload = dict(policy_payload or {})
            if isinstance(payload.get("notes_noise"), dict):
                payload["notes_noise"] = NotesNoiseConfig(**payload["notes_noise"])
            policies[index] = StagePolicyConfig(**payload)
        data["stage_policies"] = policies
    return TrainingConfig(**data)


def main() -> None:
    args = parse_args()
    logger = configure_logging(logging.INFO, name="nstream.cli.train")
    raw_cfg = load_config(args.config)
    model_cfg = _coerce_model_config(raw_cfg.get("model", {}))
    training_cfg = _coerce_training_config(raw_cfg.get("training", {}))

    seed_everything(training_cfg.seed)
    if training_cfg.seed is not None:
        logger.info("seed_configured | seed=%d", int(training_cfg.seed))

    logger.info(
        "train_start | config=%s | dataset=%s | eval=%s",
        str(args.config),
        training_cfg.dataset_path,
        training_cfg.eval_dataset_path,
    )

    dataset = maybe_create_dataset(training_cfg.dataset_path)
    eval_dataset = maybe_create_dataset(training_cfg.eval_dataset_path)

    model = NStreamTransformer(model_cfg)
    model.trunk_adapter.load_model()
    trainer = Trainer(
        model,
        training_cfg,
        collator_config=model_cfg.collator,
        dataset=dataset,
        eval_dataset=eval_dataset,
    )
    trainer.fit()
    trainer.evaluate()

    telemetry_dir = Path(
        training_cfg.telemetry_dir or "experiments/gpt_oss"
    )
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    stages_path = trainer.write_stage_history(telemetry_dir)

    manifest = {
        "config_path": str(args.config.resolve()),
        "dataset": training_cfg.dataset_path,
        "eval_dataset": training_cfg.eval_dataset_path,
        "global_step": trainer.state.global_step,
        "best_eval_loss": trainer.state.best_eval_loss,
    }
    if stages_path is not None:
        if stages_path.parent == telemetry_dir:
            manifest["stages_file"] = stages_path.name
        else:
            manifest["stages_file"] = str(stages_path)
    manifest_path = telemetry_dir / "train_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("train_complete | manifest=%s", str(manifest_path))

    report = trainer.generate_training_report()
    report_path = telemetry_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("train_report_written | %s", str(report_path))

    # Save lightweight adapter checkpoint for evaluation/inference reuse
    try:
        adapter_state = model.adapter_state_dict()
        ckpt_path = telemetry_dir / "adapters.pt"
        import torch
        torch.save(adapter_state, ckpt_path)
        logger.info("adapters_checkpoint_written | %s", str(ckpt_path))
    except Exception as err:  # pragma: no cover - defensive path
        logger.warning("adapters_checkpoint_write_failed | err=%s", str(err))


if __name__ == "__main__":
    main()
