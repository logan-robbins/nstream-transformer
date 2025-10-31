"""Rich hierarchical schema for state-of-the-art teacher note extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Sequence


class NoteLevel(str, Enum):
    """Granularity level of the extracted note."""

    DOCUMENT = "document"
    SECTION = "section"
    SENTENCE = "sentence"
    SPAN = "span"


class EntityType(str, Enum):
    """Fine-grained entity type taxonomy."""

    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"
    DATE = "DATE"
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"
    PRODUCT = "PRODUCT"
    QUANTITY = "QUANTITY"
    OTHER = "OTHER"


class RelationType(str, Enum):
    """Semantic relation types between entities."""

    DEFINITION = "DEFINITION"
    CAUSATION = "CAUSATION"
    TEMPORAL = "TEMPORAL"
    ELABORATION = "ELABORATION"
    COMPARISON = "COMPARISON"
    CONTRADICTION = "CONTRADICTION"
    ATTRIBUTION = "ATTRIBUTION"
    PART_OF = "PART_OF"
    INSTANCE_OF = "INSTANCE_OF"


class DiscourseType(str, Enum):
    """Discourse-level structural annotations."""

    INTRODUCTION = "INTRODUCTION"
    BACKGROUND = "BACKGROUND"
    MAIN_POINT = "MAIN_POINT"
    EVIDENCE = "EVIDENCE"
    CONCLUSION = "CONCLUSION"
    TRANSITION = "TRANSITION"
    ASIDE = "ASIDE"


@dataclass
class TextSpan:
    """Represents a grounded span in the source text."""

    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid span: start={self.start}, end={self.end}")


@dataclass
class RichEntity:
    """Enhanced entity representation with type, salience, and grounding."""

    name: str
    entity_type: EntityType
    spans: List[TextSpan]
    salience: float  # 0.0-1.0, importance for understanding the document
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.salience <= 1.0:
            raise ValueError(f"Salience must be in [0,1], got {self.salience}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0,1], got {self.confidence}")


@dataclass
class RichRelation:
    """Structured relation between entities or concepts."""

    relation_type: RelationType
    subject: str
    object: str
    predicate: str
    spans: List[TextSpan]
    confidence: float = 1.0
    context: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0,1], got {self.confidence}")


@dataclass
class DiscourseAnnotation:
    """Discourse-level structural information."""

    discourse_type: DiscourseType
    spans: List[TextSpan]
    summary: str
    importance: float  # 0.0-1.0
    dependencies: List[str] = field(default_factory=list)  # References to other segments

    def __post_init__(self) -> None:
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError(f"Importance must be in [0,1], got {self.importance}")


@dataclass
class DocumentAnalysis:
    """High-level document understanding metadata."""

    domain: str  # e.g., "scientific", "biographical", "technical"
    primary_topics: List[str]
    document_type: str  # e.g., "article", "biography", "tutorial"
    key_themes: List[str]
    complexity_score: float  # 0.0-1.0, estimated reading difficulty
    structure_summary: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.complexity_score <= 1.0:
            raise ValueError(f"Complexity must be in [0,1], got {self.complexity_score}")


@dataclass
class ExtractedNoteSet:
    """Complete set of hierarchical notes extracted from a document."""

    document_id: str
    analysis: DocumentAnalysis
    entities: List[RichEntity]
    relations: List[RichRelation]
    discourse_annotations: List[DiscourseAnnotation]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_legacy_format(self) -> List[Dict[str, Any]]:
        """Convert to simple note format for backward compatibility."""
        legacy_notes = []

        # Entity notes
        for entity in self.entities:
            legacy_notes.append({
                "note_type": entity.entity_type.value,
                "content": entity.name,
                "span_indices": [span.start for span in entity.spans],
                "attributes": {
                    "salience": str(entity.salience),
                    "confidence": str(entity.confidence),
                    **entity.attributes,
                },
            })

        # Relation notes
        for relation in self.relations:
            legacy_notes.append({
                "note_type": relation.relation_type.value,
                "content": f"{relation.subject} {relation.predicate} {relation.object}",
                "span_indices": [span.start for span in relation.spans],
                "attributes": {"confidence": str(relation.confidence)},
            })

        # Discourse notes
        for discourse in self.discourse_annotations:
            legacy_notes.append({
                "note_type": discourse.discourse_type.value,
                "content": discourse.summary,
                "span_indices": [span.start for span in discourse.spans],
                "attributes": {"importance": str(discourse.importance)},
            })

        return legacy_notes

    def get_salience_ranked_entities(self, top_k: int = 10) -> List[RichEntity]:
        """Return top-k entities by salience score."""
        return sorted(self.entities, key=lambda e: e.salience, reverse=True)[:top_k]

    def get_relations_by_type(self, relation_type: RelationType) -> List[RichRelation]:
        """Filter relations by type."""
        return [r for r in self.relations if r.relation_type == relation_type]

    def get_discourse_by_type(self, discourse_type: DiscourseType) -> List[DiscourseAnnotation]:
        """Filter discourse annotations by type."""
        return [d for d in self.discourse_annotations if d.discourse_type == discourse_type]


@dataclass
class ExtractionResult:
    """Result of the complete extraction pipeline."""

    success: bool
    note_set: ExtractedNoteSet | None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "NoteLevel",
    "EntityType",
    "RelationType",
    "DiscourseType",
    "TextSpan",
    "RichEntity",
    "RichRelation",
    "DiscourseAnnotation",
    "DocumentAnalysis",
    "ExtractedNoteSet",
    "ExtractionResult",
]
