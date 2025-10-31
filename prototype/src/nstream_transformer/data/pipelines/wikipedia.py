"""Wikipedia preprocessing pipeline producing planner and multi-stream datasets.

Enhanced with detailed logging for debugging small smoke runs.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from nstream_transformer.data.validation import WikipediaSchemaMetricsAggregator
from nstream_transformer.data.extraction import TeacherPlanner, TeacherPlan
from nstream_transformer.data.teacher_runner import TeacherRunner, TeacherRunnerConfig


_PLANNER_CACHE: dict[tuple[str, str], TeacherPlanner] = {}
_TEACHER_RUNNER_CACHE: dict[tuple[str, str], TeacherRunner] = {}
from nstream_transformer.utils import configure_logging
import logging

SECTION_HEADING_RE = re.compile(r"^={2,}\s*(.+?)\s*={2,}$", re.MULTILINE)
ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
REFERENCE_TITLES = {
    "references",
    "external links",
    "see also",
    "further reading",
    "notes",
}


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [block.strip() for block in text.replace("\r", "").split("\n\n") if block.strip()]
    return paragraphs or [text.strip()]


def _resolve_teacher_planner(config: WikipediaPipelineConfig) -> TeacherPlanner:
    key = (config.teacher_provider, config.teacher_model)
    if key not in _PLANNER_CACHE:
        _PLANNER_CACHE[key] = TeacherPlanner(
            provider=config.teacher_provider,
            model=config.teacher_model,
        )
    return _PLANNER_CACHE[key]


def _resolve_teacher_runner(config: WikipediaPipelineConfig) -> TeacherRunner:
    runner_cfg = config.teacher_runner
    if runner_cfg is None:
        raise ValueError("teacher_runner configuration is required to generate supervision.")
    if runner_cfg.cache_dir is None:
        default_cache = Path(config.output_dir) / config.run_id / "teacher_runner_cache"
        runner_cfg.cache_dir = str(default_cache)
    key = (runner_cfg.provider, runner_cfg.model, runner_cfg.cache_dir or "")
    if key not in _TEACHER_RUNNER_CACHE:
        _TEACHER_RUNNER_CACHE[key] = TeacherRunner(runner_cfg)
    return _TEACHER_RUNNER_CACHE[key]


def _generate_teacher_plan(
    title: str,
    paragraphs: Sequence[str],
    config: WikipediaPipelineConfig,
) -> TeacherPlan:
    logger = configure_logging(logging.INFO, name="nstream.pipeline.wiki")
    planner = _resolve_teacher_planner(config)
    logger.info(
        "teacher_plan_begin | provider=%s | model=%s | title=%s | paragraphs=%d",
        config.teacher_provider,
        config.teacher_model,
        title,
        len(paragraphs),
    )
    teacher_plan = planner.build_plan(title, paragraphs)
    logger.info(
        "teacher_plan_end | provider=%s | model=%s | title=%s | summaries=%d | segments=%d",
        config.teacher_provider,
        config.teacher_model,
        title,
        len(teacher_plan.summaries),
        len(teacher_plan.segments),
    )
    return teacher_plan


@dataclass(slots=True)
class WikipediaPipelineConfig:
    """Configuration describing how to run the Wikipedia preprocessing pipeline."""

    raw_dir: Path | str
    output_dir: Path | str = Path("data/processed")
    run_id: str = "wikipedia_run"
    max_records: int | None = None
    include_references: bool = False
    use_teacher_planner: bool = True
    teacher_provider: str = "ollama"
    teacher_model: str = "llama3.1:70b-instruct-q5_K_M"
    min_section_length: int = 80
    teacher_runner: TeacherRunnerConfig | None = None

    def __post_init__(self) -> None:
        self.raw_dir = Path(self.raw_dir)
        self.output_dir = Path(self.output_dir)
        if not self.run_id:
            raise ValueError("run_id must be provided")
        if self.max_records is not None and self.max_records <= 0:
            raise ValueError("max_records must be positive when supplied")
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw directory {self.raw_dir} does not exist")

    @property
    def processed_dir(self) -> Path:
        return self.output_dir / self.run_id


@dataclass(slots=True)
class Histogram:
    values: list[int] = field(default_factory=list)

    def add(self, value: int) -> None:
        self.values.append(value)

    def to_summary(self) -> dict[str, float | int]:
        if not self.values:
            return {"min": 0, "max": 0, "mean": 0.0}
        return {
            "min": min(self.values),
            "max": max(self.values),
            "mean": float(sum(self.values) / len(self.values)),
        }


@dataclass(slots=True)
class PipelineStats:
    articles: int = 0
    plan_tokens: Histogram = field(default_factory=Histogram)
    surface_tokens: Histogram = field(default_factory=Histogram)
    notes_tokens: Histogram = field(default_factory=Histogram)

    def to_dict(self) -> dict[str, Any]:
        return {
            "articles": self.articles,
            "plan_tokens": self.plan_tokens.to_summary(),
            "surface_tokens": self.surface_tokens.to_summary(),
            "notes_tokens": self.notes_tokens.to_summary(),
        }


@dataclass(slots=True)
class WikipediaPipelineRunResult:
    processed: int
    planner_path: Path
    multistream_path: Path
    stats_path: Path
    schema_report_path: Path


def run_wikipedia_pipeline(config: WikipediaPipelineConfig) -> WikipediaPipelineRunResult:
    """Run the Wikipedia preprocessing pipeline and materialise dataset artefacts."""
    logger = configure_logging(logging.INFO, name="nstream.pipeline.wiki")
    logger.info(
        "pipeline_begin | raw_dir=%s | out=%s | run_id=%s | max_records=%s | include_refs=%s",
        str(config.raw_dir), str(config.output_dir), config.run_id, config.max_records, config.include_references,
    )
    processed_dir = config.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    planner_path = processed_dir / "planner.jsonl"
    multistream_path = processed_dir / "multistream.jsonl"
    stats_path = processed_dir / "stats.json"
    schema_report_path = processed_dir / "schema_report.json"

    stats = PipelineStats()
    schema_aggregator = WikipediaSchemaMetricsAggregator()

    files = sorted(config.raw_dir.glob("*.jsonl"))
    logger.info("discovered_shards | count=%d | examples=%s", len(files), [p.name for p in files[:3]])

    with planner_path.open("w", encoding="utf-8") as planner_file, multistream_path.open(
        "w", encoding="utf-8"
    ) as multistream_file:
        for index, record in enumerate(_iter_raw_records(config.raw_dir)):
            if config.max_records is not None and index >= config.max_records:
                break
            if index % 50 == 0:
                logger.info("progress | read=%d", index)
            processed = _process_record(record, config, schema_aggregator, stats)
            if processed is None:
                continue
            planner_payload, multistream_payload = processed
            planner_json = json.dumps(planner_payload, ensure_ascii=False, sort_keys=True)
            multistream_json = json.dumps(
                multistream_payload,
                ensure_ascii=False,
                sort_keys=True,
            )
            planner_file.write(planner_json.replace("\n", "\\n") + "\n")
            multistream_file.write(multistream_json.replace("\n", "\\n") + "\n")
            stats.articles += 1
            if stats.articles % 50 == 0:
                logger.info("progress | processed=%d", stats.articles)

    stats_path.write_text(json.dumps(stats.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    schema_report = schema_aggregator.finalize()
    schema_report_path.write_text(
        json.dumps(schema_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    logger.info(
        "pipeline_end | processed=%d | planner=%s | multistream=%s | stats=%s | schema=%s",
        stats.articles, planner_path.name, multistream_path.name, stats_path.name, schema_report_path.name,
    )
    return WikipediaPipelineRunResult(
        processed=stats.articles,
        planner_path=planner_path,
        multistream_path=multistream_path,
        stats_path=stats_path,
        schema_report_path=schema_report_path,
    )


def _iter_raw_records(raw_dir: Path) -> Iterator[dict[str, Any]]:
    logger = configure_logging(logging.INFO, name="nstream.pipeline.wiki")
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"No JSONL shards discovered in {raw_dir}. Did you run the ingestion step?"
        )
    for path in files:
        logger.info("reading_shard | %s", path.name)
        with path.open(encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception as err:
                    logger.error("jsonl_parse_error | file=%s | line=%d | err=%s", path.name, line_idx + 1, err)
                    raise


def _process_record(
    record: Mapping[str, Any],
    config: WikipediaPipelineConfig,
    schema_aggregator: WikipediaSchemaMetricsAggregator,
    stats: PipelineStats,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    logger = configure_logging(logging.INFO, name="nstream.pipeline.wiki")
    title = str(record.get("title", "")).strip()
    text = str(record.get("text", "")).strip()
    if not title or not text:
        return None

    paragraphs = _split_paragraphs(text)

    teacher_plan: TeacherPlan | None = None
    role_overrides: dict[str, list[str]] | None = None
    teacher_notes: dict[str, list[str]] | None = None
    teacher_metadata: dict[str, Any] | None = None
    teacher_snapshots_meta: list[dict[str, Any]] = []

    if config.use_teacher_planner:
        try:
            teacher_plan = _generate_teacher_plan(title, paragraphs, config)
        except Exception as exc:  # pragma: no cover - teacher is required
            logger.error("teacher_plan_failed | title=%s | err=%s", title, exc)
            raise

    intro_text, sections = _segment_article(text)
    if not intro_text and not sections:
        return None

    sections, references = _separate_references(sections)

    schema = _build_schema(title, intro_text, sections, references)

    if teacher_plan is not None:
        plan_tokens = [f"{summary.role.upper()}::{summary.summary}" for summary in teacher_plan.summaries]
        role_overrides = {}
        for segment in teacher_plan.segments:
            role = segment.role.lower()
            chunk = "\n\n".join(paragraphs[segment.paragraph_start:segment.paragraph_end]).strip()
            role_overrides[role] = [chunk] if chunk else [""]
        for role in ("intro", "core", "wrap"):
            role_overrides.setdefault(role, [""])
        teacher_notes = {
            summary.role.lower(): list(summary.notes)
            for summary in teacher_plan.summaries
        }
        teacher_metadata = teacher_plan.raw
    else:
        plan_tokens = _build_plan_tokens(title, intro_text, sections)

    if config.teacher_runner is not None:
        runner = _resolve_teacher_runner(config)
        example_identifier = str(
            record.get("id")
            or record.get("article_id")
            or record.get("title")
            or f"{config.run_id}-{stats.articles}"
        )
        runner_example = {
            "example_id": example_identifier,
            "metadata": {
                "document_text": text,
                "document_paragraphs": paragraphs,
                "teacher_plan": teacher_metadata or (teacher_plan.raw if teacher_plan is not None else {}),
                "title": title,
            },
        }
        teacher_run = runner.run(runner_example)
        teacher_notes = {role: list(notes) for role, notes in teacher_run.role_notes.items()}
        teacher_metadata = teacher_run.teacher_plan
        teacher_snapshots_meta = [
            {
                "version": snapshot.version,
                "stride": snapshot.stride,
                "role_notes": snapshot.role_notes,
                "coverage": snapshot.coverage,
            }
            for snapshot in teacher_run.snapshots
        ]

    plan_payload = _build_plan_payload(
        title,
        plan_tokens,
        schema,
        num_sections=len(sections),
    )
    multistream_payload = _build_multistream_payload(
        record,
        intro_text,
        sections,
        references,
        schema,
        plan_payload["plan_tokens"],
        stats,
        config,
        role_overrides=role_overrides,
        teacher_metadata=teacher_metadata,
        teacher_notes=teacher_notes,
        teacher_snapshots=teacher_snapshots_meta,
        document_text=text,
        document_paragraphs=paragraphs,
    )

    stats.plan_tokens.add(len(plan_payload["plan_tokens"]))
    schema_aggregator.update(multistream_payload, schema)

    return plan_payload, multistream_payload


def _segment_article(text: str) -> tuple[str, list[dict[str, str]]]:
    matches = list(SECTION_HEADING_RE.finditer(text))
    if not matches:
        return text.strip(), []

    intro_end = matches[0].start()
    intro = text[:intro_end].strip()
    sections: list[dict[str, str]] = []

    for idx, match in enumerate(matches):
        title = match.group(1).strip()
        content_start = match.end()
        content_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section_text = text[content_start:content_end].strip()
        if section_text:
            sections.append({"title": title, "content": section_text})
    return intro, sections


def _separate_references(sections: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    primary: list[dict[str, str]] = []
    references: list[dict[str, str]] = []
    for section in sections:
        title = section.get("title", "").strip()
        if title.lower() in REFERENCE_TITLES:
            references.append({"title": title, "content": section.get("content", "")})
        else:
            primary.append({"title": title, "content": section.get("content", "")})
    return primary, references


def _build_schema(
    title: str,
    intro_text: str,
    sections: Sequence[Mapping[str, str]],
    references: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    entities = _extract_entities(title, intro_text, sections)
    definitions = _extract_definitions(title, intro_text, sections)
    facts = _extract_facts(title, sections)
    coverage = _summarise_coverage(sections, entities)
    cross_refs = _build_cross_references(sections)

    schema = {
        "entities": entities,
        "definitions": definitions,
        "facts": facts,
        "coverage": coverage,
        "cross_references": cross_refs,
        "references": [ref["title"] for ref in references],
    }
    return schema


def _extract_entities(
    title: str,
    intro_text: str,
    sections: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    sources = [("title", title), ("introduction", intro_text)]
    for section in sections:
        sources.append((section.get("title", "section"), section.get("content", "")))
    seen = {}
    for source, text in sources:
        for match in ENTITY_RE.finditer(text):
            candidate = match.group(1).strip()
            if len(candidate) < 2:
                continue
            lower = candidate.lower()
            if lower not in seen:
                seen[lower] = {"name": candidate, "source": source}
    return list(seen.values())


def _extract_definitions(
    title: str,
    intro_text: str,
    sections: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []
    first_sentence = _first_sentence(intro_text)
    if first_sentence and " is " in first_sentence.lower():
        definitions.append({"entity": title, "definition": first_sentence})
    for section in sections:
        sentence = _first_sentence(section.get("content", ""))
        if sentence and " is " in sentence.lower():
            definitions.append({"entity": section.get("title", ""), "definition": sentence})
    return definitions


def _extract_facts(title: str, sections: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for section in sections:
        sentence = _first_sentence(section.get("content", ""))
        if not sentence:
            continue
        facts.append(
            {
                "subject": title,
                "predicate": f"has section {section.get('title', '')}",
                "object": sentence,
                "section": section.get("title", ""),
            }
        )
    return facts


def _summarise_coverage(
    sections: Sequence[Mapping[str, str]],
    entities: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    coverage: list[dict[str, Any]] = []
    entity_lookup = defaultdict(list)
    for entity in entities:
        entity_lookup[entity["source"]].append(entity["name"])
    for section in sections:
        title = section.get("title", "")
        text = section.get("content", "")
        summary = _first_sentence(text) or text[:160]
        completeness = "complete" if len(text) >= 400 else "partial"
        coverage.append(
            {
                "section": title,
                "summary": summary,
                "entities": entity_lookup.get(title, []),
                "completeness": completeness,
            }
        )
    return coverage


def _build_cross_references(sections: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    cross_refs: list[dict[str, Any]] = []
    for previous, current in zip(sections, sections[1:]):
        cross_refs.append(
            {
                "from": previous.get("title", ""),
                "to": current.get("title", ""),
                "relation": "ELABORATION",
            }
        )
    return cross_refs


def _build_plan_payload(
    title: str,
    plan_tokens: Sequence[str],
    schema: Mapping[str, Any],
    num_sections: int,
) -> dict[str, Any]:
    schema_summary = {
        "entity_count": len(schema.get("entities", [])),
        "fact_count": len(schema.get("facts", [])),
        "definition_count": len(schema.get("definitions", [])),
    }
    return {
        "prompt": title,
        "plan_tokens": list(plan_tokens),
        "labels": {"title": title, "num_sections": num_sections},
        "schema_summary": schema_summary,
    }


def _build_plan_tokens(
    title: str,
    intro_text: str,
    sections: Sequence[Mapping[str, str]],
) -> list[str]:
    tokens: list[str] = []
    intro_summary = _truncate_summary(_first_sentence(intro_text) or intro_text, max_chars=160)
    if intro_summary:
        tokens.append(f"INTRO::{intro_summary}")
    for section in sections:
        title_token = section.get("title", "Section")
        section_summary = _truncate_summary(
            _first_sentence(section.get("content", "")) or section.get("content", ""),
            max_chars=200,
        )
        if section_summary:
            tokens.append(f"{title_token}::{section_summary}")
    return tokens


def _build_multistream_payload(
    record: Mapping[str, Any],
    intro_text: str,
    sections: Sequence[Mapping[str, str]],
    references: Sequence[Mapping[str, str]],
    schema: Mapping[str, Any],
    plan_tokens: Sequence[str],
    stats: PipelineStats,
    config: WikipediaPipelineConfig,
    *,
    role_overrides: dict[str, list[str]] | None = None,
    teacher_metadata: Mapping[str, Any] | None = None,
    teacher_notes: Mapping[str, Sequence[str]] | None = None,
    teacher_snapshots: Sequence[Mapping[str, Any]] | None = None,
    document_text: str | None = None,
    document_paragraphs: Sequence[str] | None = None,
) -> dict[str, Any]:
    if role_overrides is not None:
        roles = {role: list(chunks) for role, chunks in role_overrides.items()}
        for role in ("intro", "core", "wrap"):
            roles.setdefault(role, [""])
    else:
        roles = _assign_roles(intro_text, sections)

    role_payloads: dict[str, Any] = {}
    role_plan = list(plan_tokens)
    for role_name, role_sections in roles.items():
        surface_text = "\n\n".join(role_sections)
        surface_tokens = _tokenize(surface_text)
        raw_notes = list(teacher_notes.get(role_name, [])) if teacher_notes else []
        notes_tokens = _tokenize(" ".join(raw_notes))

        stats.surface_tokens.add(len(surface_tokens))
        stats.notes_tokens.add(len(notes_tokens))

        role_payloads[role_name] = {
            "surface_tokens": surface_tokens,
            "notes_tokens": notes_tokens,
            "schema": _select_role_schema(role_name, schema),
            "plan": role_plan,
            "notes": raw_notes,
        }

    metadata = {
        "id": record.get("id"),
        "title": record.get("title"),
        "url": record.get("url"),
        "references": references if config.include_references else [ref["title"] for ref in references],
        "role_surface_lengths": {role: len(payload["surface_tokens"]) for role, payload in role_payloads.items()},
    }
    if teacher_notes is not None:
        metadata["teacher_notes"] = {role: list(notes) for role, notes in teacher_notes.items()}
    if teacher_metadata is not None:
        metadata["teacher_plan"] = teacher_metadata
    if teacher_snapshots is not None:
        metadata["teacher_snapshots"] = list(teacher_snapshots)
    if document_text is not None:
        metadata["document_text"] = document_text
    if document_paragraphs is not None:
        metadata["document_paragraphs"] = list(document_paragraphs)

    return {
        "plan": list(plan_tokens),
        "roles": role_payloads,
        "metadata": metadata,
    }


def _assign_roles(intro_text: str, sections: Sequence[Mapping[str, str]]) -> dict[str, list[str]]:
    primary_texts = [section.get("content", "") for section in sections]

    roles: dict[str, list[str]] = {"intro": [], "core": [], "wrap": []}

    intro_clean = intro_text.strip()
    if intro_clean:
        roles["intro"].append(intro_clean)

    if primary_texts:
        if len(primary_texts) == 1:
            roles["core"].append(primary_texts[0])
        else:
            roles["core"].extend(primary_texts[:-1])
            roles["wrap"].append(primary_texts[-1])
    elif intro_clean:
        roles["wrap"].append(intro_clean)

    # Ensure no role is empty to maintain downstream assumptions.
    for role_name, bucket in roles.items():
        if not bucket:
            bucket.append("")
    return roles


def _select_role_schema(role_name: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    if role_name == "intro":
        keys = ("entities", "definitions")
    elif role_name == "core":
        keys = ("entities", "facts", "coverage")
    else:
        keys = ("facts", "cross_references")
    return {key: schema.get(key, []) for key in keys}


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    tokens = re.findall(r"\w+", text)
    return [token.lower() for token in tokens]




def _first_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    match = re.search(r"([^.?!]+[.?!])", text)
    if match:
        return match.group(1).strip()
    return text


def _truncate_summary(text: str, max_chars: int = 200) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


__all__ = [
    "WikipediaPipelineConfig",
    "WikipediaPipelineRunResult",
    "run_wikipedia_pipeline",
]
