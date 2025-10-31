"""Summarise training and evaluation outputs for a given run directory."""

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


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def summarise_run(run_dir: Path) -> Dict[str, Any]:
    telemetry_dir = run_dir / "telemetry"
    evaluation_dir = run_dir / "evaluation"
    config_path = run_dir / "config.yaml"

    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "telemetry": {},
        "evaluation": {},
        "config": {},
    }

    if config_path.exists():
        summary["config"] = load_yaml(config_path)

    manifest_path = telemetry_dir / "train_run_manifest.json"
    if manifest_path.exists():
        summary["telemetry"]["manifest"] = load_json(manifest_path)
    metrics_path = telemetry_dir / "train_run_steps.jsonl"
    if metrics_path.exists():
        summary["telemetry"]["steps_file"] = str(metrics_path)
    stages_path = telemetry_dir / "train_run_stages.json"
    if stages_path.exists():
        summary["telemetry"]["stages_file"] = str(stages_path)

    eval_manifest = evaluation_dir / "eval_manifest.json"
    if eval_manifest.exists():
        summary["evaluation"]["manifest"] = load_json(eval_manifest)
    metrics_json = evaluation_dir / "metrics.json"
    if metrics_json.exists():
        summary["evaluation"]["metrics"] = load_json(metrics_json)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing telemetry/evaluation outputs")
    args = parser.parse_args()

    summary = summarise_run(args.run_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
