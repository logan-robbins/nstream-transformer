#!/usr/bin/env python3
"""Utility for advancing the staged training curriculum."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the training YAML configuration to update.",
    )
    parser.add_argument(
        "--start-stage",
        type=int,
        required=True,
        help="Stage index to treat as the new starting point (0-based).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the updated configuration. Defaults to in-place update.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the adjusted schedule without writing any files.",
    )
    return parser.parse_args(argv)


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def remap_stage_schedule(schedule: List[int], start_stage: int) -> Tuple[List[int], int]:
    if not schedule:
        raise ValueError("Stage schedule is empty; nothing to advance.")
    if start_stage < 0 or start_stage >= len(schedule):
        raise ValueError(
            f"start_stage {start_stage} is out of range for schedule of length {len(schedule)}."
        )
    base = int(schedule[start_stage])
    adjusted = [max(0, int(threshold) - base) for threshold in schedule[start_stage:]]
    adjusted[0] = 0
    return adjusted, base


def remap_stage_policies(policies: Dict, start_stage: int) -> Dict[str, Dict]:
    if not policies:
        return {}
    remapped: Dict[str, Dict] = {}
    for key, value in policies.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            # Preserve unexpected keys verbatim.
            remapped[str(key)] = value
            continue
        if index < start_stage:
            continue
        remapped[str(index - start_stage)] = value
    return remapped


def advance_curriculum(config_path: Path, start_stage: int, output_path: Path | None, dry_run: bool) -> None:
    payload = load_yaml(config_path)
    training = payload.get("training")
    if training is None:
        raise ValueError("The provided configuration does not contain a 'training' section.")
    curriculum = training.get("curriculum")
    if curriculum is None or "stage_schedule" not in curriculum:
        raise ValueError("'training.curriculum.stage_schedule' not found in configuration.")
    original_schedule = list(curriculum["stage_schedule"])
    new_schedule, offset = remap_stage_schedule(original_schedule, start_stage)
    curriculum["stage_schedule"] = new_schedule

    stage_policies = remap_stage_policies(training.get("stage_policies", {}), start_stage)
    if stage_policies:
        training["stage_policies"] = stage_policies
    elif "stage_policies" in training:
        training.pop("stage_policies")

    summary = {
        "original_schedule": original_schedule,
        "new_schedule": new_schedule,
        "offset": offset,
        "start_stage": start_stage,
        "policies_kept": sorted(stage_policies.keys()),
    }

    if dry_run:
        yaml.safe_dump(summary, sys.stdout, sort_keys=False)
        return

    target = output_path or config_path
    dump_yaml(target, payload)
    print(yaml.safe_dump(summary, sort_keys=False), end="")  # noqa: T201
    print(f"Updated curriculum written to {target}")  # noqa: T201


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        advance_curriculum(args.config, args.start_stage, args.output, args.dry_run)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
