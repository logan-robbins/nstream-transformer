"""CLI for validating Wikipedia schema metrics on processed datasets with progress logs."""

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
import logging

from nstream_transformer.data.validation import WikipediaSchemaMetricsAggregator
from nstream_transformer.utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Path to multistream JSONL produced by the preprocessing pipeline",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for the schema report (default: alongside the input file)",
    )
    parser.add_argument(
        "--print",
        dest="print_report",
        action="store_true",
        help="Print the computed schema report to stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = configure_logging(logging.INFO, name="nstream.cli.validate")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist")

    aggregator = WikipediaSchemaMetricsAggregator()
    processed = 0
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            aggregator.update(record)
            processed += 1
            if processed % 100 == 0:
                logger.info("validate_progress | processed=%d", processed)

    report = aggregator.finalize()

    output_path = (
        Path(args.output)
        if args.output
        else input_path.parent / "schema_report.json"
    )
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    logger.info("schema_report_written | %s | records=%d", str(output_path), processed)
    print(f"Schema report written to {output_path}")
    if args.print_report:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
