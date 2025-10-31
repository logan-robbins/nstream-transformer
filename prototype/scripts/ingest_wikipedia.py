"""CLI entry point for downloading and manifesting Wikipedia snapshots."""

from __future__ import annotations

# Ensure local src/ is on sys.path when running from the repo without installation
import os as _os, sys as _sys
_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_SRC_PATH = _os.path.join(_REPO_ROOT, "src")
if _SRC_PATH not in _sys.path and _os.path.isdir(_SRC_PATH):
    _sys.path.insert(0, _SRC_PATH)

import argparse
from pathlib import Path
import logging
import sys
import os
import atexit

from nstream_transformer.data.ingestion import (
    DEFAULT_SHARD_SIZE,
    WikipediaIngestionConfig,
    ingest_wikipedia_snapshot,
)
from nstream_transformer.utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        required=True,
        help="Wikipedia dataset snapshot identifier (e.g. 20220301.en)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to download (default: train)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help=(
            "Directory to store JSONL shards. Defaults to data/raw and expands to "
            "data/raw/wikipedia/<snapshot>"
        ),
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional explicit manifest path. Defaults to data/manifests/...",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap on the number of records to download (useful for smoke tests)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help="Number of records per JSONL shard (default: 1000)",
    )
    streaming_group = parser.add_mutually_exclusive_group()
    streaming_group.add_argument(
        "--streaming",
        dest="streaming",
        action="store_true",
        help="Download data in streaming mode (default)",
    )
    streaming_group.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Disable streaming mode and load the dataset into memory",
    )
    parser.set_defaults(streaming=True)
    parser.add_argument(
        "--print-manifest",
        action="store_true",
        help="Print the manifest JSON to stdout after writing it to disk",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = configure_logging(logging.INFO, name="nstream.cli.ingest")
    logger.info(
        "ingest_cli_begin | snapshot=%s | split=%s | out=%s | max_records=%s | streaming=%s | shard_size=%s",
        args.snapshot, args.split, args.output_dir, args.max_records, args.streaming, args.shard_size,
    )
    atexit.register(lambda: print("[ingest-cli] atexit: process exiting", flush=True))

    config = WikipediaIngestionConfig(
        snapshot=args.snapshot,
        split=args.split,
        output_dir=Path(args.output_dir),
        streaming=args.streaming,
        max_records=args.max_records,
        shard_size=args.shard_size,
        manifest_path=Path(args.manifest_path) if args.manifest_path else None,
    )

    manifest = ingest_wikipedia_snapshot(config)

    manifest_path = config.manifest_path or config.default_manifest_path
    print(
        f"Ingested {manifest.total_records} records into {len(manifest.shards)} shards at "
        f"{config.output_dir}. Manifest written to {manifest_path}."
    )
    logger.info(
        "ingest_cli_end | records=%d | shards=%d | out=%s | manifest=%s",
        manifest.total_records, len(manifest.shards), str(config.output_dir), str(manifest_path),
    )
    if args.print_manifest:
        print(manifest.to_json())
    # Ensure output is flushed and the process terminates for debug visibility
    sys.stdout.flush()
    sys.stderr.flush()
    # Forcefully terminate the interpreter to avoid lingering non-daemon threads
    os._exit(0)


if __name__ == "__main__":
    main()
