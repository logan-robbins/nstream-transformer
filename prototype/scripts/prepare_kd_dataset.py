"""Stage-oriented CLI for converting multistream wiki data into KD training records."""

from __future__ import annotations

# Ensure local src/ is discoverable when running from repo root.
import os as _os
import sys as _sys
_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_SRC_PATH = _os.path.join(_REPO_ROOT, "src")
if _SRC_PATH not in _sys.path and _os.path.isdir(_SRC_PATH):
    _sys.path.insert(0, _SRC_PATH)

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import logging
import math

from nstream_transformer.data import (
    TokenizerConfig,
    TwoBranchKDCollatorConfig,
    resolve_tokenizer,
)
from nstream_transformer.integration.gpt_oss.embedder import GptOssEmbedder
from nstream_transformer.utils import configure_logging


@dataclass(slots=True)
class StageContext:
    stage: str
    multistream_path: Path
    staging_dir: Path
    output_path: Path
    max_records: int | None
    tokenizer_path: Path | None
    model_reference: str | None
    logger: logging.Logger


def parse_args(argv: Sequence[str] | None = None) -> StageContext:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["load-multistream", "fan-out-roles", "tokenize-roles", "embed-notes", "pack-kd"],
        help="Pipeline stage to execute",
    )
    parser.add_argument("--multistream", required=True, help="Path to multistream JSONL produced by run_process")
    parser.add_argument("--staging-dir", required=True, help="Directory for intermediate artefacts")
    parser.add_argument("--output", required=True, help="Destination KD JSONL file path")
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap on records to process")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer path override")
    parser.add_argument(
        "--model",
        default=None,
        help="GPT-OSS model identifier or local directory (defaults to tokenizer parent or HF repo)",
    )

    args = parser.parse_args(argv)
    staging_dir = Path(args.staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = Path(args.tokenizer).resolve() if args.tokenizer else None
    if tokenizer_path is not None and not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer path does not exist: {tokenizer_path}")

    model_reference: str | None
    if args.model:
        model_reference = args.model
    elif tokenizer_path is not None:
        model_reference = str(tokenizer_path.parent)
    else:
        model_reference = None

    logger = configure_logging(logging.INFO, name=f"nstream.data_prep.{args.stage}")
    logger.info(
        "stage_begin | stage=%s | multistream=%s | output=%s | staging=%s | max=%s | tokenizer=%s | model=%s",
        args.stage,
        args.multistream,
        args.output,
        staging_dir,
        args.max_records,
        args.tokenizer or "<default>",
        model_reference or "<default>",
    )

    return StageContext(
        stage=args.stage,
        multistream_path=Path(args.multistream),
        staging_dir=staging_dir,
        output_path=Path(args.output),
        max_records=args.max_records,
        tokenizer_path=tokenizer_path,
        model_reference=model_reference,
        logger=logger,
    )


def dispatch_stage(context: StageContext) -> None:
    stages: Mapping[str, Callable[[StageContext], None]] = {
        "load-multistream": stage_load_multistream,
        "fan-out-roles": stage_fan_out_roles,
        "tokenize-roles": stage_tokenize_roles,
        "embed-notes": stage_embed_notes,
        "pack-kd": stage_pack_kd,
    }
    try:
        handler = stages[context.stage]
    except KeyError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Unsupported stage requested: {context.stage}") from exc
    handler(context)


def stage_load_multistream(context: StageContext) -> None:
    """Inspect multistream schema and emit structured stats for downstream stages."""
    path = context.multistream_path
    if not path.exists():
        raise FileNotFoundError(f"Multistream dataset not found: {path}")

    context.logger.info(
        "load_multistream | tokenizer_hint=%s",
        context.tokenizer_path or "gpt-oss-default",
    )

    totals = {
        "total_records": 0,
        "processed_records": 0,
        "plan_lengths": [],
    }
    metadata_flags = Counter()
    role_ranges = defaultdict(
        lambda: {
            "examples": 0,
            "surface_total": 0,
            "surface_min": None,
            "surface_max": None,
            "notes_total": 0,
            "notes_min": None,
            "notes_max": None,
        }
    )
    speculative_note_counts = defaultdict(int)
    samples: list[dict[str, object]] = []
    sample_cap = 3

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            payload = raw.strip()
            if not payload:
                continue

            totals["total_records"] += 1
            if context.max_records is not None and totals["processed_records"] >= context.max_records:
                continue

            try:
                record: dict[str, object] = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            totals["processed_records"] += 1
            plan_tokens = record.get("plan", []) if isinstance(record, dict) else []
            if isinstance(plan_tokens, list):
                totals["plan_lengths"].append(len(plan_tokens))

            metadata = record.get("metadata", {}) if isinstance(record, dict) else {}
            if isinstance(metadata, dict):
                if metadata.get("teacher_plan"):
                    metadata_flags["teacher_plan"] += 1
                if metadata.get("teacher_notes"):
                    metadata_flags["teacher_notes"] += 1
                speculative = metadata.get("speculative_notes", {})
                if isinstance(speculative, dict):
                    for role_name, notes in speculative.items():
                        if isinstance(notes, list):
                            speculative_note_counts[role_name] += len(notes)

            roles = record.get("roles", {}) if isinstance(record, dict) else {}
            if not isinstance(roles, dict):
                continue

            for role_name, role_payload in roles.items():
                if not isinstance(role_payload, dict):
                    continue
                stats = role_ranges[role_name]
                stats["examples"] += 1

                surface_tokens = role_payload.get("surface_tokens", [])
                if isinstance(surface_tokens, list):
                    surface_len = len(surface_tokens)
                    stats["surface_total"] += surface_len
                    stats["surface_min"] = surface_len if stats["surface_min"] is None else min(stats["surface_min"], surface_len)
                    stats["surface_max"] = surface_len if stats["surface_max"] is None else max(stats["surface_max"], surface_len)

                notes_tokens = role_payload.get("notes_tokens", [])
                if isinstance(notes_tokens, list):
                    notes_len = len(notes_tokens)
                    stats["notes_total"] += notes_len
                    stats["notes_min"] = notes_len if stats["notes_min"] is None else min(stats["notes_min"], notes_len)
                    stats["notes_max"] = notes_len if stats["notes_max"] is None else max(stats["notes_max"], notes_len)

            if len(samples) < sample_cap:
                samples.append(record)

    if totals["processed_records"] == 0:
        context.logger.warning("load_multistream | no records processed (check max_records?)")

    overview_path = context.staging_dir / "load_multistream_overview.json"
    sample_path = context.staging_dir / "load_multistream_samples.jsonl"

    role_summary = {}
    for role_name, stats in role_ranges.items():
        examples = stats["examples"] or 1
        role_summary[role_name] = {
            "examples": stats["examples"],
            "surface_tokens": {
                "min": stats["surface_min"],
                "max": stats["surface_max"],
                "mean": stats["surface_total"] / examples,
            },
            "notes_tokens": {
                "min": stats["notes_min"],
                "max": stats["notes_max"],
                "mean": stats["notes_total"] / examples,
            },
        }

    plan_lengths = totals["plan_lengths"] or [0]
    plan_summary = {
        "min": min(plan_lengths),
        "max": max(plan_lengths),
        "mean": (sum(plan_lengths) / len(plan_lengths)) if plan_lengths else 0.0,
    }

    overview_payload = {
        "total_records": totals["total_records"],
        "processed_records": totals["processed_records"],
        "max_records": context.max_records,
        "roles": sorted(role_summary.keys()),
        "role_summary": role_summary,
        "plan_summary": plan_summary,
        "metadata_flags": dict(metadata_flags),
        "speculative_note_counts": dict(speculative_note_counts),
    }

    overview_path.write_text(json.dumps(overview_payload, indent=2), encoding="utf-8")
    context.logger.info("load_multistream | overview_written=%s", overview_path)

    with sample_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")
    context.logger.info("load_multistream | samples_written=%s count=%d", sample_path, len(samples))

    context.logger.info(
        "load_multistream | summary total=%d processed=%d roles=%s",
        overview_payload["total_records"],
        overview_payload["processed_records"],
        ",".join(overview_payload["roles"]),
    )


def stage_fan_out_roles(context: StageContext) -> None:
    """Flatten roles into a stage-oriented manifest for downstream tokenization."""
    overview_path = context.staging_dir / "load_multistream_overview.json"
    if not overview_path.exists():
        raise FileNotFoundError(
            "load-multistream overview missing. Run the first stage before fan-out-roles."
        )

    manifest_path = context.staging_dir / "fan_out_roles_manifest.jsonl"
    detail_path = context.staging_dir / "fan_out_roles_detail.json"

    multistream_records = _iter_multistream(context)
    records_written = 0
    role_records = 0
    plan_summary: Counter[int] = Counter()

    with manifest_path.open("w", encoding="utf-8") as manifest_handle:
        for record in multistream_records:
            metadata = record.get("metadata", {}) if isinstance(record, dict) else {}
            plan = record.get("plan", []) if isinstance(record, dict) else []
            roles = record.get("roles", {}) if isinstance(record, dict) else {}

            article_id = None
            title = None
            if isinstance(metadata, dict):
                article_id = metadata.get("id")
                title = metadata.get("title")

            metadata_teacher_notes = {}
            metadata_student_notes = {}
            if isinstance(metadata, dict):
                teacher_map = metadata.get("teacher_notes", {})
                student_map = metadata.get("speculative_notes", {})
                if isinstance(teacher_map, dict):
                    metadata_teacher_notes = {
                        key: list(value) if isinstance(value, list) else []
                        for key, value in teacher_map.items()
                    }
                if isinstance(student_map, dict):
                    metadata_student_notes = {
                        key: list(value) if isinstance(value, list) else []
                        for key, value in student_map.items()
                    }

            if isinstance(plan, list):
                plan_summary[len(plan)] += 1

            if not isinstance(roles, dict):
                continue

            for role_name, payload in roles.items():
                if not isinstance(payload, dict):
                    continue

                surface_tokens = payload.get("surface_tokens", [])
                notes_tokens = payload.get("notes_tokens", [])
                role_plan = payload.get("plan", plan)
                teacher_notes = payload.get("notes", [])
                student_notes = metadata_student_notes.get(role_name, [])
                if not student_notes:
                    student_notes = metadata_teacher_notes.get(role_name, [])

                manifest_record = {
                    "article_id": article_id,
                    "title": title,
                    "role": role_name,
                    "surface_tokens": surface_tokens,
                    "notes_tokens": notes_tokens,
                    "plan_tokens": role_plan,
                    "teacher_notes": teacher_notes,
                    "student_notes": student_notes,
                }

                manifest_handle.write(json.dumps(manifest_record) + "\n")
                role_records += 1

            records_written += 1

    detail_payload = {
        "records_processed": records_written,
        "role_records": role_records,
        "plan_length_histogram": dict(plan_summary),
        "tokenizer_hint": str(context.tokenizer_path) if context.tokenizer_path else "gpt-oss-default",
    }

    detail_path.write_text(json.dumps(detail_payload, indent=2), encoding="utf-8")
    context.logger.info(
        "fan_out_roles | manifest=%s role_records=%d",
        manifest_path,
        role_records,
    )


def stage_tokenize_roles(context: StageContext) -> None:
    """Tokenize per-role payloads using the GPT-OSS tokenizer."""

    manifest_in = context.staging_dir / "fan_out_roles_manifest.jsonl"
    if not manifest_in.exists():
        raise FileNotFoundError(
            "fan_out_roles_manifest.jsonl missing. Run the fan-out-roles stage first."
        )

    manifest_out = context.staging_dir / "tokenized_roles_manifest.jsonl"
    summary_out = context.staging_dir / "tokenize_roles_summary.json"
    tokenizer_manifest_out = context.staging_dir / "tokenizer_manifest.json"

    if context.tokenizer_path is not None:
        tokenizer_config = TokenizerConfig(custom_path=context.tokenizer_path)
    else:
        tokenizer_config = TokenizerConfig()
    tokenizer, tokenizer_manifest = resolve_tokenizer(tokenizer_config)
    tokenizer_manifest.write(tokenizer_manifest_out)
    context.logger.info(
        "tokenize_roles | tokenizer=%s | source=%s | vocab=%d | added_special=%d",
        tokenizer_manifest.tokenizer_class,
        tokenizer_manifest.source,
        tokenizer_manifest.vocab_size,
        len(tokenizer_manifest.added_tokens),
    )

    collator_max_length = TwoBranchKDCollatorConfig(pad_token_id=0).max_length
    eos_token_id = (
        tokenizer.eos_token_id
        if tokenizer.eos_token_id is not None
        else tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else 0
    )

    def _normalise_ids(ids: list[int]) -> list[int]:
        if ids:
            return ids
        # Ensure downstream padding logic has at least a sentinel token.
        return [eos_token_id]

    seq_stats = {
        "student": {"count": 0, "total": 0, "min": None, "max": None, "overflow": 0},
        "planner": {"count": 0, "total": 0, "min": None, "max": None, "overflow": 0},
    }

    records_processed = 0
    fallback_sequences = 0
    articles = set()

    with manifest_in.open("r", encoding="utf-8") as input_handle, manifest_out.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line_number, raw in enumerate(input_handle, start=1):
            payload = raw.strip()
            if not payload:
                continue

            try:
                record = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in role manifest (line {line_number}): {exc}") from exc

            articles.add(record.get("article_id"))

            surface_tokens = record.get("surface_tokens", [])
            plan_tokens = record.get("plan_tokens", [])
            teacher_notes = record.get("teacher_notes", [])
            student_notes = record.get("student_notes", [])
            notes_tokens = record.get("notes_tokens", [])

            surface_text = " ".join(surface_tokens) if isinstance(surface_tokens, list) else ""
            plan_text = "\n".join(plan_tokens) if isinstance(plan_tokens, list) else ""

            student_ids = tokenizer.encode(surface_text, add_special_tokens=True)
            planner_ids = tokenizer.encode(plan_text, add_special_tokens=True)

            if not student_ids or not planner_ids:
                fallback_sequences += 1
            student_ids = _normalise_ids(student_ids)
            planner_ids = _normalise_ids(planner_ids)

            _update_seq_stats(seq_stats["student"], len(student_ids), collator_max_length)
            _update_seq_stats(seq_stats["planner"], len(planner_ids), collator_max_length)

            tokenized_record = {
                "article_id": record.get("article_id"),
                "title": record.get("title"),
                "role": record.get("role"),
                "student_ids": student_ids,
                "student_labels": list(student_ids),
                "planner_ids": planner_ids,
                "plan_tokens": plan_tokens,
                "notes_tokens": notes_tokens,
                "teacher_notes": teacher_notes,
                "student_notes": student_notes,
            }

            output_handle.write(json.dumps(tokenized_record) + "\n")
            records_processed += 1

    summary_payload = {
        "records_processed": records_processed,
        "articles_seen": len([item for item in articles if item is not None]),
        "sequence_stats": {
            "student_ids": _seq_stats_summary(seq_stats["student"]),
            "planner_ids": _seq_stats_summary(seq_stats["planner"]),
        },
        "collator_max_length": collator_max_length,
        "fallback_sequences": fallback_sequences,
        "tokenizer_manifest": tokenizer_manifest.to_dict(),
    }

    summary_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    context.logger.info(
        "tokenize_roles | manifest=%s records=%d fallback=%d",
        manifest_out,
        records_processed,
        fallback_sequences,
    )


def stage_embed_notes(context: StageContext) -> None:
    """Materialise teacher/student note embeddings using the GPT-OSS trunk."""

    tokenized_manifest = context.staging_dir / "tokenized_roles_manifest.jsonl"
    if not tokenized_manifest.exists():
        raise FileNotFoundError(
            "tokenized_roles_manifest.jsonl missing. Run tokenize-roles before embed-notes."
        )

    embedded_manifest = context.staging_dir / "embedded_roles_manifest.jsonl"
    summary_out = context.staging_dir / "embed_notes_summary.json"

    collator_config = TwoBranchKDCollatorConfig(pad_token_id=0)
    notes_dim = collator_config.notes_dim

    embedder = _resolve_embedder(context, notes_dim)

    records: list[dict[str, object]] = []
    teacher_texts: list[str] = []
    student_texts: list[str] = []

    with tokenized_manifest.open("r", encoding="utf-8") as input_handle:
        for line_number, raw in enumerate(input_handle, start=1):
            payload = raw.strip()
            if not payload:
                continue
            try:
                record = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in tokenized manifest (line {line_number}): {exc}"
                ) from exc

            records.append(record)
            notes_tokens = record.get("notes_tokens", [])
            teacher_texts.append(_flatten_note_text(record.get("teacher_notes"), notes_tokens))
            student_texts.append(_flatten_note_text(record.get("student_notes"), notes_tokens))

    teacher_vectors = _embed_with_fallback(embedder, teacher_texts, notes_dim)
    student_vectors = _embed_with_fallback(embedder, student_texts, notes_dim)

    stats = {
        "records": 0,
        "teacher_zero": 0,
        "student_zero": 0,
        "teacher_norm": 0.0,
        "student_norm": 0.0,
        "role_counts": Counter(),
    }

    with embedded_manifest.open("w", encoding="utf-8") as output_handle:
        for record, teacher_vec, student_vec in zip(records, teacher_vectors, student_vectors):
            role = record.get("role", "core")
            stats["records"] += 1
            stats["role_counts"][role] += 1

            teacher_norm = _l2_norm(teacher_vec)
            student_norm = _l2_norm(student_vec)

            if teacher_norm == 0.0:
                stats["teacher_zero"] += 1
            if student_norm == 0.0:
                stats["student_zero"] += 1

            stats["teacher_norm"] += teacher_norm
            stats["student_norm"] += student_norm

            record["notes_teacher"] = teacher_vec
            record["notes_student"] = student_vec

            output_handle.write(json.dumps(record) + "\n")

    summary_payload = {
        "records": stats["records"],
        "notes_dim": notes_dim,
        "teacher_zero_records": stats["teacher_zero"],
        "student_zero_records": stats["student_zero"],
        "avg_teacher_norm": (stats["teacher_norm"] / stats["records"])
        if stats["records"]
        else 0.0,
        "avg_student_norm": (stats["student_norm"] / stats["records"])
        if stats["records"]
        else 0.0,
        "role_counts": dict(stats["role_counts"]),
    }

    summary_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    context.logger.info(
        "embed_notes | manifest=%s records=%d",
        embedded_manifest,
        stats["records"],
    )


def stage_pack_kd(context: StageContext) -> None:
    """Write the final KD JSONL manifest consumed by the training dataloader."""

    embedded_manifest = context.staging_dir / "embedded_roles_manifest.jsonl"
    if not embedded_manifest.exists():
        raise FileNotFoundError(
            "embedded_roles_manifest.jsonl missing. Run embed-notes before pack-kd."
        )

    output_path = context.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_out = context.staging_dir / "pack_kd_summary.json"

    collator_config = TwoBranchKDCollatorConfig(pad_token_id=0)
    notes_dim = collator_config.notes_dim

    stats = {
        "records": 0,
        "role_counts": Counter(),
        "student_length": {"min": None, "max": 0},
        "planner_length": {"min": None, "max": 0},
        "notes_dim_mismatches": 0,
    }

    required_fields = [
        "student_ids",
        "student_labels",
        "planner_ids",
        "notes_teacher",
        "notes_student",
        "role",
    ]

    with embedded_manifest.open("r", encoding="utf-8") as input_handle, output_path.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line_number, raw in enumerate(input_handle, start=1):
            payload = raw.strip()
            if not payload:
                continue
            try:
                record = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in embedded manifest (line {line_number}): {exc}"
                ) from exc

            for field in required_fields:
                if field not in record:
                    raise KeyError(f"Embedded record missing required field {field!r} (line {line_number}).")

            student_ids = list(record["student_ids"])
            student_labels = list(record["student_labels"])
            planner_ids = list(record["planner_ids"])
            notes_teacher = list(record["notes_teacher"])
            notes_student = list(record["notes_student"])
            role = record["role"]

            if len(notes_teacher) != notes_dim or len(notes_student) != notes_dim:
                stats["notes_dim_mismatches"] += 1
                raise ValueError(
                    f"Notes dim mismatch on line {line_number}. Expected {notes_dim}, "
                    f"got teacher={len(notes_teacher)} student={len(notes_student)}"
                )

            student_len = len(student_ids)
            planner_len = len(planner_ids)
            stats["records"] += 1
            stats["role_counts"][role] += 1
            stats["student_length"]["min"] = (
                student_len
                if stats["student_length"]["min"] is None
                else min(stats["student_length"]["min"], student_len)
            )
            stats["student_length"]["max"] = max(stats["student_length"]["max"], student_len)
            stats["planner_length"]["min"] = (
                planner_len
                if stats["planner_length"]["min"] is None
                else min(stats["planner_length"]["min"], planner_len)
            )
            stats["planner_length"]["max"] = max(stats["planner_length"]["max"], planner_len)

            kd_record = {
                "student_ids": student_ids,
                "student_labels": student_labels,
                "planner_ids": planner_ids,
                "notes_teacher": notes_teacher,
                "notes_student": notes_student,
                "role": role,
            }

            # Preserve optional metadata for traceability
            for optional_field in ("article_id", "title", "plan_tokens", "notes_tokens"):
                if optional_field in record:
                    kd_record[optional_field] = record[optional_field]

            output_handle.write(json.dumps(kd_record) + "\n")

    summary_payload = {
        "records": stats["records"],
        "notes_dim": notes_dim,
        "roles": dict(stats["role_counts"]),
        "student_length": stats["student_length"],
        "planner_length": stats["planner_length"],
        "notes_dim_mismatches": stats["notes_dim_mismatches"],
        "output_path": str(output_path),
    }

    summary_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    context.logger.info(
        "pack_kd | output=%s records=%d",
        output_path,
        stats["records"],
    )


def _flatten_note_text(
    primary_notes: Sequence[str] | None, fallback_tokens: Sequence[str] | None
) -> str:
    content = [text.strip() for text in primary_notes or [] if text and text.strip()]
    if content:
        return "\n\n".join(content)

    tokens = [str(token) for token in fallback_tokens or [] if token]
    if tokens:
        return " ".join(tokens)
    return ""


def _embed_with_fallback(
    embedder: GptOssEmbedder, texts: Sequence[str], notes_dim: int
) -> list[list[float]]:
    vectors: list[list[float]] = [[0.0] * notes_dim for _ in texts]
    non_empty: list[tuple[int, str]] = [
        (index, text) for index, text in enumerate(texts) if text and text.strip()
    ]

    if not non_empty:
        return vectors

    indices, payloads = zip(*non_empty)
    embeddings = embedder.embed(list(payloads), target_dim=notes_dim)

    for idx, vector in zip(indices, embeddings):
        vectors[idx] = [round(float(value), 6) for value in vector]
    return vectors


_EMBEDDER_CACHE: dict[tuple[str, str | None, int], GptOssEmbedder] = {}


def _resolve_embedder(context: StageContext, notes_dim: int) -> GptOssEmbedder:
    model_reference = context.model_reference or "openai/gpt-oss-20b"
    tokenizer_reference = (
        str(context.tokenizer_path)
        if context.tokenizer_path is not None
        else model_reference
    )
    model_path = Path(model_reference)
    if model_path.exists():
        if not any((model_path / candidate).exists() for candidate in ("config.json", "model.safetensors")):
            original = model_path / "original"
            if original.exists():
                model_reference = str(original)
    
    cache_key = (model_reference, tokenizer_reference, notes_dim)
    if cache_key in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[cache_key]

    embedder = GptOssEmbedder(
        model_reference=model_reference,
        tokenizer_reference=tokenizer_reference,
        target_dimension=notes_dim,
    )
    _EMBEDDER_CACHE[cache_key] = embedder
    context.logger.info(
        "embedder_initialised | model=%s | tokenizer=%s | notes_dim=%d",
        model_reference,
        tokenizer_reference,
        notes_dim,
    )
    return embedder


def _l2_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values)) if values else 0.0


def _update_seq_stats(bucket: dict, value: int, limit: int) -> None:
    bucket["count"] += 1
    bucket["total"] += value
    bucket["min"] = value if bucket["min"] is None else min(bucket["min"], value)
    bucket["max"] = value if bucket["max"] is None else max(bucket["max"], value)
    if value > limit:
        bucket["overflow"] += 1


def _seq_stats_summary(bucket: dict) -> dict[str, float | int | None]:
    count = bucket.get("count", 0) or 0
    mean = bucket["total"] / count if count else 0.0
    return {
        "count": count,
        "min": bucket.get("min"),
        "max": bucket.get("max"),
        "mean": mean,
        "overflow": bucket.get("overflow", 0),
    }


def _iter_multistream(context: StageContext):
    count = 0
    with context.multistream_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            payload = raw.strip()
            if not payload:
                continue
            try:
                record = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in multistream dataset: {exc}") from exc
            yield record
            count += 1
            if context.max_records is not None and count >= context.max_records:
                context.logger.info(
                    "iter_multistream | max_records hit (%d), stopping early",
                    context.max_records,
                )
                break

def main(argv: Sequence[str] | None = None) -> None:
    context = parse_args(argv)
    dispatch_stage(context)


if __name__ == "__main__":
    main()
