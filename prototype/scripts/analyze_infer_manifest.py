"""Compute runtime proxies (alpha, beta, S) from an inference manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze N-Stream inference manifest telemetry.")
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to the manifest JSON emitted by scripts/infer.py.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print metrics as JSON instead of a human-readable table.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_metrics(manifest: Dict[str, Any]) -> Dict[str, Any]:
    timings = manifest.get("timings", {})
    stride_durations: List[float] = list(timings.get("stride_durations", []))
    total_time = float(timings.get("total", 0.0) or 0.0)

    roles = manifest.get("roles", {})
    tokens_total = sum(len(data.get("token_ids", [])) for data in roles.values())
    rollbacks = manifest.get("rollbacks", [])
    rollback_events = len(rollbacks)
    rollback_tokens = sum(len(entry.get("tokens_removed", [])) for entry in rollbacks)
    strides = len(stride_durations)

    alpha = mean(stride_durations) if stride_durations else 0.0
    beta = (rollback_tokens / tokens_total) if tokens_total else 0.0
    speed = (tokens_total / total_time) if total_time > 0.0 else 0.0

    return {
        "alpha_s_per_stride": alpha,
        "beta_rollback_token_fraction": beta,
        "S_tokens_per_second": speed,
        "total_time_s": total_time,
        "stride_count": strides,
        "rollback_events": rollback_events,
        "rollback_tokens": rollback_tokens,
        "token_count": tokens_total,
    }


def print_table(metrics: Dict[str, Any]) -> None:
    rows = [
        ("alpha (s/stride)", metrics["alpha_s_per_stride"]),
        ("beta (rollback token frac)", metrics["beta_rollback_token_fraction"]),
        ("S (tokens/s)", metrics["S_tokens_per_second"]),
        ("stride count", metrics["stride_count"]),
        ("rollback events", metrics["rollback_events"]),
        ("rollback tokens", metrics["rollback_tokens"]),
        ("token count", metrics["token_count"]),
        ("total time (s)", metrics["total_time_s"]),
    ]
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        if isinstance(value, float):
            text = f"{value:.4f}"
        else:
            text = str(value)
        print(f"{name.ljust(width)} : {text}")


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    metrics = compute_metrics(manifest)
    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        print_table(metrics)


if __name__ == "__main__":
    main()
