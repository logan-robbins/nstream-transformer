"""CLI for transforming raw Wikipedia shards into planner and multi-stream datasets."""

from __future__ import annotations

# Ensure local src/ is on sys.path when running from the repo without installation
import os as _os, sys as _sys
_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_SRC_PATH = _os.path.join(_REPO_ROOT, "src")
if _SRC_PATH not in _sys.path and _os.path.isdir(_SRC_PATH):
    _sys.path.insert(0, _SRC_PATH)

import argparse
from pathlib import Path

from nstream_transformer.data.pipelines import (
    WikipediaPipelineConfig,
    run_wikipedia_pipeline,
)
from nstream_transformer.utils import configure_logging
import logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        required=True,
        help="Directory containing JSONL shards produced by the ingestion step",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Destination root for processed artefacts (default: data/processed)",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Identifier used to create data/processed/<run_id>/ outputs",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional limit on the number of records to process",
    )
    parser.add_argument(
        "--include-references",
        action="store_true",
        help="Include reference section bodies in metadata output",
    )
    parser.add_argument(
        "--disable-teacher-planner",
        action="store_true",
        help="Skip external LLM planner calls (use heuristic fallback)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = configure_logging(logging.INFO, name="nstream.cli.prepare")
    logger.info(
        "prepare_cli_begin | raw_dir=%s | out=%s | run_id=%s | max_records=%s | include_refs=%s",
        args.raw_dir, args.output_dir, args.run_id, args.max_records, args.include_references,
    )

    config = WikipediaPipelineConfig(
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
        max_records=args.max_records,
        include_references=args.include_references,
        use_teacher_planner=not args.disable_teacher_planner,
    )

    result = run_wikipedia_pipeline(config)

    logger.info(
        "processed | articles=%d | planner=%s | multistream=%s",
        result.processed,
        result.planner_path,
        result.multistream_path,
    )
    logger.info("Stats: %s", result.stats_path)
    logger.info("Schema report: %s", result.schema_report_path)


if __name__ == "__main__":
    main()
