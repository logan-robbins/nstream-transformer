"""Schema validation metrics for Wikipedia preprocessing outputs."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

TOKEN_RE = re.compile(r"\w+")
YEAR_RE = re.compile(r"\b(\d{4})\b")
RANGE_PATTERN = re.compile(r"\d{4}\D+(?:to|through|between|and|-)\D+\d{4}")
POLARITY_TOKENS = {
    "largest": "largest",
    "smallest": "smallest",
    "first": "first",
    "last": "last",
    "former": "former",
    "current": "current",
    "not": "negation",
    "never": "negation",
}
ROLE_ORDER = ["intro", "core", "wrap"]


@dataclass(slots=True)
class WikipediaSchemaMetricsAggregator:
    """Aggregate schema-centric validation metrics across a dataset."""

    target_ratio_min: float = 0.10
    target_ratio_max: float = 0.20
    articles: int = 0
    entity_counter: Counter[str] = field(default_factory=Counter)
    fact_count: int = 0
    definition_count: int = 0
    coverage_count: int = 0
    cross_ref_count: int = 0
    reconstruction_scores: list[float] = field(default_factory=list)
    fact_hits: int = 0
    fact_total: int = 0
    compactness_ratios: list[float] = field(default_factory=list)
    compactness_within_target: int = 0
    entity_persistence_hits: int = 0
    entity_persistence_checks: int = 0
    inconsistency_entities: int = 0
    inconsistency_checks: int = 0

    def update(
        self,
        record: Mapping[str, Any],
        schema: Mapping[str, Any] | None = None,
    ) -> None:
        """Update aggregate metrics using a single multistream record."""

        roles = record.get("roles", {})
        if not roles:
            return
        self.articles += 1

        combined_schema = schema or self._merge_role_schema(roles.values())

        self._tally_schema_counts(combined_schema)
        self._update_compactness(roles.values())
        self._update_reconstruction(roles)
        self._update_fact_coverage(combined_schema, roles.values())
        self._update_entity_persistence(combined_schema, roles.values())
        self._update_consistency(combined_schema)

    def finalize(self) -> dict[str, Any]:
        """Summarise collected metrics into a JSON-serialisable dictionary."""

        unique_entities = len(self.entity_counter)
        total_entities = sum(self.entity_counter.values())
        avg_entities = (total_entities / self.articles) if self.articles else 0.0

        reconstruction = (
            sum(self.reconstruction_scores) / len(self.reconstruction_scores)
            if self.reconstruction_scores
            else 0.0
        )
        fact_coverage = (self.fact_hits / self.fact_total) if self.fact_total else 0.0
        compactness_mean = (
            sum(self.compactness_ratios) / len(self.compactness_ratios)
            if self.compactness_ratios
            else 0.0
        )
        compactness_within = (
            self.compactness_within_target / len(self.compactness_ratios)
            if self.compactness_ratios
            else 0.0
        )
        entity_persistence = (
            self.entity_persistence_hits / self.entity_persistence_checks
            if self.entity_persistence_checks
            else 0.0
        )
        inconsistency_rate = (
            self.inconsistency_entities / self.inconsistency_checks
            if self.inconsistency_checks
            else 0.0
        )

        report = {
            "articles": self.articles,
            "unique_entities": unique_entities,
            "total_entities": total_entities,
            "avg_entities_per_article": avg_entities,
            "fact_count": self.fact_count,
            "definition_count": self.definition_count,
            "coverage_entries": self.coverage_count,
            "cross_references": self.cross_ref_count,
            "reconstruction_score": reconstruction,
            "fact_coverage": fact_coverage,
            "entity_persistence": entity_persistence,
            "compactness_mean": compactness_mean,
            "compactness_within_range_ratio": compactness_within,
            "notes_ratio_target": {
                "min": self.target_ratio_min,
                "max": self.target_ratio_max,
            },
            "inconsistency_rate": inconsistency_rate,
        }
        return report

    def _tally_schema_counts(self, schema: Mapping[str, Any]) -> None:
        for entity in schema.get("entities", []):
            name = str(entity.get("name", "")).strip()
            if name:
                self.entity_counter[name] += 1
        self.fact_count += len(schema.get("facts", []))
        self.definition_count += len(schema.get("definitions", []))
        self.coverage_count += len(schema.get("coverage", []))
        self.cross_ref_count += len(schema.get("cross_references", []))

    def _update_compactness(self, roles: Iterable[Mapping[str, Any]]) -> None:
        for role in roles:
            surface_tokens = role.get("surface_tokens", [])
            notes_tokens = role.get("notes_tokens", [])
            surface_len = len(surface_tokens)
            notes_len = len(notes_tokens)
            if surface_len == 0 and notes_len == 0:
                self.compactness_ratios.append(0.0)
                continue
            ratio = notes_len / surface_len if surface_len else 0.0
            self.compactness_ratios.append(ratio)
            if self.target_ratio_min <= ratio <= self.target_ratio_max:
                self.compactness_within_target += 1

    def _update_reconstruction(self, roles: Mapping[str, Mapping[str, Any]]) -> None:
        ordered_roles = [role for role in ROLE_ORDER if role in roles]
        remaining_roles = [role for role in roles.keys() if role not in ordered_roles]
        ordered_roles.extend(sorted(remaining_roles))

        accumulated_notes: set[str] = set()
        for role_name in ordered_roles:
            role = roles[role_name]
            surface_tokens = set(role.get("surface_tokens", []))
            if not surface_tokens:
                self.reconstruction_scores.append(0.0)
            else:
                overlap = len(surface_tokens & accumulated_notes)
                score = overlap / len(surface_tokens)
                self.reconstruction_scores.append(score)
            accumulated_notes.update(role.get("notes_tokens", []))

    def _update_fact_coverage(
        self,
        schema: Mapping[str, Any],
        roles: Iterable[Mapping[str, Any]],
    ) -> None:
        notes_union: set[str] = set()
        for role in roles:
            notes_union.update(role.get("notes_tokens", []))

        for fact in schema.get("facts", []):
            fact_text = str(fact.get("object", ""))
            fact_tokens = {token.lower() for token in TOKEN_RE.findall(fact_text)}
            if not fact_tokens:
                continue
            self.fact_total += 1
            overlap = fact_tokens & {token.lower() for token in notes_union}
            if overlap and len(overlap) / len(fact_tokens) >= 0.5:
                self.fact_hits += 1

    def _update_entity_persistence(
        self,
        schema: Mapping[str, Any],
        roles: Iterable[Mapping[str, Any]],
    ) -> None:
        role_surfaces = [set(role.get("surface_tokens", [])) for role in roles]
        if not role_surfaces:
            return
        threshold = 1 if len(role_surfaces) == 1 else 2

        for entity in schema.get("entities", []):
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            entity_tokens = {token.lower() for token in TOKEN_RE.findall(name)}
            if not entity_tokens:
                continue
            appearances = 0
            for surface_tokens in role_surfaces:
                if entity_tokens.issubset(surface_tokens):
                    appearances += 1
            self.entity_persistence_checks += 1
            if appearances >= threshold:
                self.entity_persistence_hits += 1

    def _update_consistency(self, schema: Mapping[str, Any]) -> None:
        entity_numbers: defaultdict[str, set[str]] = defaultdict(set)
        entity_polarities: defaultdict[str, set[str]] = defaultdict(set)
        entity_range_context: defaultdict[str, bool] = defaultdict(bool)

        for definition in schema.get("definitions", []):
            entity = str(definition.get("entity", "unknown"))
            text = str(definition.get("definition", ""))
            entity_numbers[entity].update(YEAR_RE.findall(text))
            if _looks_like_range(text):
                entity_range_context[entity] = True
            self._accumulate_polarities(entity_polarities, entity, text)

        for fact in schema.get("facts", []):
            entity = str(fact.get("subject", "unknown"))
            text = str(fact.get("object", ""))
            entity_numbers[entity].update(YEAR_RE.findall(text))
            if _looks_like_range(text):
                entity_range_context[entity] = True
            self._accumulate_polarities(entity_polarities, entity, text)

        considered_entities = set(entity_numbers.keys()) | set(entity_polarities.keys())
        if not considered_entities:
            return

        inconsistent_entities: set[str] = set()
        for entity, numbers in entity_numbers.items():
            if len(numbers) > 1 and not entity_range_context.get(entity, False):
                inconsistent_entities.add(entity)
        for entity, tags in entity_polarities.items():
            if {"largest", "smallest"}.issubset(tags):
                inconsistent_entities.add(entity)
            if "negation" in tags and ("current" in tags or "former" in tags):
                inconsistent_entities.add(entity)
        self.inconsistency_checks += len(considered_entities)
        self.inconsistency_entities += len(inconsistent_entities)

    def _accumulate_polarities(
        self,
        entity_polarities: defaultdict[str, set[str]],
        entity: str,
        text: str,
    ) -> None:
        lowered = text.lower()
        for token, label in POLARITY_TOKENS.items():
            if token in lowered:
                entity_polarities[entity].add(label)

    @staticmethod
    def _merge_role_schema(roles: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        merged: dict[str, Any] = {
            "entities": [],
            "definitions": [],
            "facts": [],
            "coverage": [],
            "cross_references": [],
        }
        for role in roles:
            role_schema = role.get("schema", {}) or {}
            for key in merged.keys():
                value = role_schema.get(key)
                if not value:
                    continue
                if isinstance(value, list):
                    merged[key].extend(value)
        return merged


def validate_multistream_file(input_path: Path) -> dict[str, Any]:
    """Compute schema validation metrics for the provided multistream JSONL file."""

    aggregator = WikipediaSchemaMetricsAggregator()
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            aggregator.update(record)
    return aggregator.finalize()


def _looks_like_range(text: str) -> bool:
    """Heuristic to detect year ranges (e.g. 1900-1950) rather than contradictions."""

    return bool(RANGE_PATTERN.search(text.lower()))
