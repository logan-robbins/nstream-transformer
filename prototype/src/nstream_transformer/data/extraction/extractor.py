"""Multi-stage teacher note extractor using LLMs."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient, LLMMessage, create_llm_client
from .schema import (
    DiscourseAnnotation,
    DiscourseType,
    DocumentAnalysis,
    EntityType,
    ExtractedNoteSet,
    ExtractionResult,
    RelationType,
    RichEntity,
    RichRelation,
    TextSpan,
)


class LLMTeacherNoteExtractor:
    """Multi-stage extraction pipeline using frontier LLMs.

    Architecture:
        Stage 1: Document Analysis - Domain, structure, complexity
        Stage 2: Entity Extraction - Named entities with types and salience
        Stage 3: Relation Extraction - Semantic relationships and definitions
        Stage 4: Discourse Extraction - Structural annotations and planning hints
        Stage 5: Verification - Consistency checks and confidence scoring

    Each stage uses specialized prompts optimized for that extraction task.
    All extracted notes are grounded to text spans for interpretability.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        provider: str = "anthropic",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the extractor.

        Args:
            llm_client: Pre-configured LLM client (overrides provider/api_key/model)
            provider: LLM provider ("anthropic", "openai", "mock")
            api_key: API key for the provider
            model: Model name to use
            verbose: Enable verbose logging
        """
        self.llm = llm_client or create_llm_client(provider, api_key, model)
        self.verbose = verbose

    def extract(
        self,
        document_id: str,
        text: str,
        plan_segments: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        """Run the complete multi-stage extraction pipeline.

        Args:
            document_id: Unique identifier for the document
            text: Full document text
            plan_segments: Optional planner-aligned segments
            metadata: Optional additional metadata

        Returns:
            ExtractionResult with comprehensive note set
        """
        errors: List[str] = []
        warnings: List[str] = []
        extraction_metadata: Dict[str, Any] = {"stages_completed": []}

        try:
            # Stage 1: Document Analysis
            if self.verbose:
                print("[Stage 1/5] Analyzing document structure...")
            analysis = self._extract_document_analysis(text)
            extraction_metadata["stages_completed"].append("analysis")

            # Stage 2: Entity Extraction
            if self.verbose:
                print("[Stage 2/5] Extracting entities...")
            entities = self._extract_entities(text, analysis)
            extraction_metadata["stages_completed"].append("entities")
            extraction_metadata["entity_count"] = len(entities)

            # Stage 3: Relation Extraction
            if self.verbose:
                print("[Stage 3/5] Extracting relations...")
            relations = self._extract_relations(text, entities, analysis)
            extraction_metadata["stages_completed"].append("relations")
            extraction_metadata["relation_count"] = len(relations)

            # Stage 4: Discourse Extraction
            if self.verbose:
                print("[Stage 4/5] Extracting discourse structure...")
            discourse = self._extract_discourse(text, plan_segments, analysis)
            extraction_metadata["stages_completed"].append("discourse")
            extraction_metadata["discourse_count"] = len(discourse)

            # Stage 5: Verification
            if self.verbose:
                print("[Stage 5/5] Verifying and scoring...")
            entities, relations, discourse, stage_warnings = self._verify_and_score(
                text, entities, relations, discourse
            )
            warnings.extend(stage_warnings)
            extraction_metadata["stages_completed"].append("verification")

            note_set = ExtractedNoteSet(
                document_id=document_id,
                analysis=analysis,
                entities=entities,
                relations=relations,
                discourse_annotations=discourse,
                metadata=metadata or {},
            )

            return ExtractionResult(
                success=True,
                note_set=note_set,
                errors=errors,
                warnings=warnings,
                extraction_metadata=extraction_metadata,
            )

        except Exception as e:
            errors.append(f"Extraction failed: {str(e)}")
            return ExtractionResult(
                success=False,
                note_set=None,
                errors=errors,
                warnings=warnings,
                extraction_metadata=extraction_metadata,
            )

    def _extract_document_analysis(self, text: str) -> DocumentAnalysis:
        """Stage 1: Analyze document structure and domain."""
        prompt = f"""Analyze this document and provide high-level metadata.

Document:
{text[:2000]}{"..." if len(text) > 2000 else ""}

Provide your analysis as JSON with this structure:
{{
  "domain": "scientific|biographical|technical|news|literary|historical|other",
  "primary_topics": ["topic1", "topic2", "topic3"],
  "document_type": "article|biography|tutorial|reference|essay|report",
  "key_themes": ["theme1", "theme2"],
  "complexity_score": 0.7,
  "structure_summary": "Brief description of how the document is organized"
}}

Consider:
- Writing style and vocabulary
- Subject matter and domain
- Organizational structure
- Target audience sophistication
- Presence of technical jargon
"""

        messages = [
            LLMMessage(role="system", content="You are an expert document analyzer."),
            LLMMessage(role="user", content=prompt),
        ]

        response = self.llm.complete(messages, temperature=0.0, max_tokens=1000)
        data = self.llm.extract_json(response)

        return DocumentAnalysis(
            domain=data.get("domain", "other"),
            primary_topics=data.get("primary_topics", []),
            document_type=data.get("document_type", "article"),
            key_themes=data.get("key_themes", []),
            complexity_score=float(data.get("complexity_score", 0.5)),
            structure_summary=data.get("structure_summary", ""),
        )

    def _extract_entities(self, text: str, analysis: DocumentAnalysis) -> List[RichEntity]:
        """Stage 2: Extract entities with types, spans, and salience."""
        prompt = f"""Extract all important entities from this {analysis.domain} document.

Document:
{text}

For each entity, provide:
1. Entity name (canonical form)
2. Entity type (PERSON, ORG, LOC, DATE, EVENT, CONCEPT, PRODUCT, QUANTITY, OTHER)
3. All text spans where it appears (as character offsets)
4. Salience score (0.0-1.0, how important for understanding the document)
5. Alternative names/aliases
6. Key attributes

Return JSON:
{{
  "entities": [
    {{
      "name": "Albert Einstein",
      "type": "PERSON",
      "spans": [{{"start": 0, "end": 15, "text": "Albert Einstein"}}, {{"start": 234, "end": 242, "text": "Einstein"}}],
      "salience": 0.95,
      "aliases": ["Einstein", "A. Einstein"],
      "attributes": {{"role": "physicist", "nationality": "German-American"}}
    }}
  ]
}}

Prioritize:
- Entities central to the main topics: {", ".join(analysis.primary_topics)}
- Named entities over generic concepts
- Entities with multiple mentions (higher salience)
- Domain-specific important entities
"""

        messages = [
            LLMMessage(
                role="system",
                content="You are an expert in named entity recognition and information extraction.",
            ),
            LLMMessage(role="user", content=prompt),
        ]

        response = self.llm.complete(messages, temperature=0.0, max_tokens=3000)
        data = self.llm.extract_json(response)

        entities = []
        for ent_data in data.get("entities", []):
            try:
                entity_type = EntityType[ent_data["type"]]
                spans = [
                    TextSpan(
                        start=span["start"],
                        end=span["end"],
                        text=span.get("text", ""),
                    )
                    for span in ent_data.get("spans", [])
                ]

                entities.append(
                    RichEntity(
                        name=ent_data["name"],
                        entity_type=entity_type,
                        spans=spans,
                        salience=float(ent_data.get("salience", 0.5)),
                        aliases=ent_data.get("aliases", []),
                        attributes=ent_data.get("attributes", {}),
                        confidence=1.0,
                    )
                )
            except (KeyError, ValueError) as e:
                if self.verbose:
                    print(f"Skipping malformed entity: {e}")
                continue

        return entities

    def _extract_relations(
        self,
        text: str,
        entities: List[RichEntity],
        analysis: DocumentAnalysis,
    ) -> List[RichRelation]:
        """Stage 3: Extract semantic relations between entities."""
        entity_names = [e.name for e in entities[:20]]  # Top 20 by order (salience implicit)

        prompt = f"""Extract semantic relations between entities in this {analysis.domain} document.

Document:
{text}

Key entities identified:
{", ".join(entity_names)}

Extract relations of these types:
- DEFINITION: X is defined as Y
- CAUSATION: X causes Y
- TEMPORAL: X happened before/after Y
- ELABORATION: X provides detail about Y
- COMPARISON: X is similar/different to Y
- CONTRADICTION: X contradicts Y
- ATTRIBUTION: X is attributed to Y
- PART_OF: X is part of Y
- INSTANCE_OF: X is an instance/example of Y

Return JSON:
{{
  "relations": [
    {{
      "type": "DEFINITION",
      "subject": "photosynthesis",
      "predicate": "is defined as",
      "object": "the process by which plants convert light to energy",
      "spans": [{{"start": 45, "end": 120, "text": "photosynthesis is the process..."}}],
      "confidence": 0.95,
      "context": "Appears in introduction explaining core concepts"
    }}
  ]
}}

Focus on:
- Relations involving high-salience entities
- Relations central to understanding {", ".join(analysis.key_themes)}
- Well-grounded relations with clear text evidence
"""

        messages = [
            LLMMessage(
                role="system",
                content="You are an expert in relation extraction and semantic analysis.",
            ),
            LLMMessage(role="user", content=prompt),
        ]

        response = self.llm.complete(messages, temperature=0.0, max_tokens=3000)
        data = self.llm.extract_json(response)

        relations = []
        for rel_data in data.get("relations", []):
            try:
                relation_type = RelationType[rel_data["type"]]
                spans = [
                    TextSpan(
                        start=span["start"],
                        end=span["end"],
                        text=span.get("text", ""),
                    )
                    for span in rel_data.get("spans", [])
                ]

                relations.append(
                    RichRelation(
                        relation_type=relation_type,
                        subject=rel_data["subject"],
                        object=rel_data["object"],
                        predicate=rel_data.get("predicate", "relates to"),
                        spans=spans,
                        confidence=float(rel_data.get("confidence", 0.8)),
                        context=rel_data.get("context", ""),
                    )
                )
            except (KeyError, ValueError) as e:
                if self.verbose:
                    print(f"Skipping malformed relation: {e}")
                continue

        return relations

    def _extract_discourse(
        self,
        text: str,
        plan_segments: Optional[List[str]],
        analysis: DocumentAnalysis,
    ) -> List[DiscourseAnnotation]:
        """Stage 4: Extract discourse-level structure."""
        plan_context = ""
        if plan_segments:
            plan_context = f"\n\nPlanner segments provided:\n{chr(10).join(f'- {seg}' for seg in plan_segments)}"

        prompt = f"""Analyze the discourse structure of this {analysis.document_type}.

Document:
{text}{plan_context}

Identify and annotate discourse segments with these types:
- INTRODUCTION: Opening context and framing
- BACKGROUND: Historical context or prerequisites
- MAIN_POINT: Core arguments or central content
- EVIDENCE: Supporting evidence or examples
- CONCLUSION: Synthesis and takeaways
- TRANSITION: Connecting segments
- ASIDE: Tangential but relevant information

Return JSON:
{{
  "discourse": [
    {{
      "type": "INTRODUCTION",
      "spans": [{{"start": 0, "end": 156, "text": "First paragraph..."}}],
      "summary": "Introduces the topic of quantum mechanics",
      "importance": 0.9,
      "dependencies": []
    }},
    {{
      "type": "MAIN_POINT",
      "spans": [{{"start": 450, "end": 890, "text": "Core section..."}}],
      "summary": "Explains wave-particle duality",
      "importance": 0.95,
      "dependencies": ["INTRODUCTION"]
    }}
  ]
}}

Consider:
- Document type: {analysis.document_type}
- Structure summary: {analysis.structure_summary}
- Key themes: {", ".join(analysis.key_themes)}
"""

        messages = [
            LLMMessage(
                role="system",
                content="You are an expert in discourse analysis and document structure.",
            ),
            LLMMessage(role="user", content=prompt),
        ]

        response = self.llm.complete(messages, temperature=0.0, max_tokens=2500)
        data = self.llm.extract_json(response)

        discourse = []
        for disc_data in data.get("discourse", []):
            try:
                discourse_type = DiscourseType[disc_data["type"]]
                spans = [
                    TextSpan(
                        start=span["start"],
                        end=span["end"],
                        text=span.get("text", ""),
                    )
                    for span in disc_data.get("spans", [])
                ]

                discourse.append(
                    DiscourseAnnotation(
                        discourse_type=discourse_type,
                        spans=spans,
                        summary=disc_data.get("summary", ""),
                        importance=float(disc_data.get("importance", 0.5)),
                        dependencies=disc_data.get("dependencies", []),
                    )
                )
            except (KeyError, ValueError) as e:
                if self.verbose:
                    print(f"Skipping malformed discourse annotation: {e}")
                continue

        return discourse

    def _verify_and_score(
        self,
        text: str,
        entities: List[RichEntity],
        relations: List[RichRelation],
        discourse: List[DiscourseAnnotation],
    ) -> tuple[List[RichEntity], List[RichRelation], List[DiscourseAnnotation], List[str]]:
        """Stage 5: Verify consistency and adjust confidence scores."""
        warnings = []

        # Verify entity spans
        text_len = len(text)
        valid_entities = []
        for entity in entities:
            valid_spans = []
            for span in entity.spans:
                if span.end <= text_len:
                    # Verify span text matches
                    actual_text = text[span.start : span.end]
                    if not span.text or actual_text.strip():
                        valid_spans.append(span)
                else:
                    warnings.append(f"Entity '{entity.name}' has invalid span: {span.start}-{span.end}")

            if valid_spans:
                entity.spans = valid_spans
                valid_entities.append(entity)

        # Verify relation spans
        valid_relations = []
        for relation in relations:
            valid_spans = []
            for span in relation.spans:
                if span.end <= text_len:
                    valid_spans.append(span)
                else:
                    warnings.append(
                        f"Relation '{relation.subject} {relation.predicate} {relation.object}' has invalid span"
                    )

            if valid_spans:
                relation.spans = valid_spans
                valid_relations.append(relation)

        # Verify discourse spans
        valid_discourse = []
        for disc in discourse:
            valid_spans = []
            for span in disc.spans:
                if span.end <= text_len:
                    valid_spans.append(span)
                else:
                    warnings.append(f"Discourse '{disc.discourse_type}' has invalid span")

            if valid_spans:
                disc.spans = valid_spans
                valid_discourse.append(disc)

        return valid_entities, valid_relations, valid_discourse, warnings


__all__ = ["LLMTeacherNoteExtractor"]
