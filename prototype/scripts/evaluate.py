"""Evaluation entry point for the GPT-OSS powered N-Stream Transformer."""

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
from nstream_transformer.training import KDJsonlDataset, Trainer
from nstream_transformer.utils import configure_logging, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GPT-OSS with N-Stream adapters.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to model checkpoint")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gpt_oss_transfer.yaml"),
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--eval-dataset",
        type=Path,
        default=None,
        help="Optional override for evaluation dataset path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/evaluation"),
        help="Directory to write evaluation manifest",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def maybe_create_dataset(path: Path | None) -> KDJsonlDataset | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset path {path} does not exist.")
    return KDJsonlDataset(path)


def main() -> None:
    args = parse_args()
    logger = configure_logging(logging.INFO, name="nstream.cli.evaluate")
    raw_cfg = load_config(args.config)
    model_cfg = ModelConfig(**raw_cfg.get("model", {}))
    training_cfg = TrainingConfig(**raw_cfg.get("training", {}))

    seed_everything(training_cfg.seed)
    if training_cfg.seed is not None:
        logger.info("seed_configured | seed=%d", int(training_cfg.seed))

    eval_path = args.eval_dataset or (
        Path(training_cfg.eval_dataset_path) if training_cfg.eval_dataset_path else None
    )
    eval_dataset = maybe_create_dataset(eval_path)

    model = NStreamTransformer(model_cfg)
    # Load trunk weights
    model.trunk_adapter.load_model()
    # Load adapter checkpoint if provided
    if args.checkpoint is not None:
        import torch
        ckpt_path = args.checkpoint
        state: Dict[str, Any] | None = None
        if ckpt_path.is_dir():
            # Prefer a standard adapters filename inside the directory
            candidate_files = [
                ckpt_path / "adapters.pt",
                ckpt_path / "adapter_state.pt",
                ckpt_path / "state_dict.pt",
            ]
            for path in candidate_files:
                if path.exists():
                    state = torch.load(path, map_location="cpu")
                    break
            if state is None:
                logger.warning("no_adapter_checkpoint_found_in_dir | dir=%s", str(ckpt_path))
        else:
            state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict):
            try:
                # Heuristic: if top-level keys look like adapter names, use load_adapters
                sample_keys = list(state.keys())[:5]
                if any(key.startswith("role_adapters.") for key in sample_keys):
                    model.load_adapters(state, strict=False)
                    logger.info("adapters_loaded | source=%s", str(ckpt_path))
                else:
                    missing, unexpected = model.load_state_dict(state, strict=False)
                    if missing or unexpected:
                        logger.info(
                            "state_dict_loaded_with_mismatch | missing=%d unexpected=%d",
                            len(missing),
                            len(unexpected),
                        )
                    logger.info("state_dict_loaded | source=%s", str(ckpt_path))
            except Exception as err:
                logger.error("checkpoint_load_failed | %s", str(err))

    trainer = Trainer(
        model,
        training_cfg,
        collator_config=model_cfg.collator,
        dataset=None,
        eval_dataset=eval_dataset,
    )
    metrics = trainer.evaluate()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config_path": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "eval_dataset": str(eval_path) if eval_path else None,
        "metrics": metrics,
    }
    manifest_path = output_dir / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("evaluation_complete | manifest=%s", str(manifest_path))


if __name__ == "__main__":
    main()
