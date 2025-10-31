"""LLM-backed planner that derives intro/core/wrap summaries and breakpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .llm_client import LLMClient, LLMMessage, create_llm_client


@dataclass
class PlanSegment:
    """Single planner segment with role and paragraph range."""

    role: str
    paragraph_start: int
    paragraph_end: int


@dataclass
class PlanSummary:
    """Plan summary describing responsibilities and note scaffolds."""

    role: str
    summary: str
    notes: List[str]


@dataclass
class TeacherPlan:
    """Planner output containing summaries, segments, and raw teacher data."""

    summaries: List[PlanSummary]
    segments: List[PlanSegment]
    raw: dict[str, Any]

    def get_role(self, role: str) -> PlanSegment | None:
        lowered = role.lower()
        for segment in self.segments:
            if segment.role.lower() == lowered:
                return segment
        return None

    def get_summary(self, role: str) -> str | None:
        lowered = role.lower()
        for summary in self.summaries:
            if summary.role.lower() == lowered:
                return summary.summary
        return None


class TeacherPlanner:
    """Generate intro/core/wrap plan and segmentation using an LLM teacher."""

    def __init__(
        self,
        *,
        provider: str = "ollama",
        model: str = "llama3.1:70b-instruct-q5_K_M",
        api_key: str | None = None,
        client: LLMClient | None = None,
    ) -> None:
        if client is not None:
            self.client = client
        else:
            self.client = create_llm_client(provider=provider, api_key=api_key, model=model)

    def build_plan(self, title: str, paragraphs: Sequence[str]) -> TeacherPlan:
        """Request an LLM to plan and segment the article into three sections."""
        excerpt_lines: List[str] = []
        for idx, para in enumerate(paragraphs):
            snippet = " ".join(para.strip().split())
            excerpt_lines.append(f"[{idx}] {snippet}")

        prompt = f"""You are preparing a training dataset for a multistream writer.
The article is written collaboratively by three roles:
- intro (role A) sets context and seeding information.
- core  (role B) develops the main exposition with only minimal cues from intro and wrap.
- wrap  (role C) delivers conclusions, implications, and references.

Your tasks:
1. Provide a high-level plan specifying what each role should cover. For each role, describe in 1–2 sentences what content it should emphasise and, for core and wrap, reference (briefly) what intro already established and what the following role will handle.
2. For each role, supply a list of 2–4 short note strings (max ~120 characters each) highlighting the concrete facts, entities, or arguments that the role must surface.
3. Divide the numbered paragraphs into exactly three contiguous segments—intro, core, wrap—ensuring:
   • Every paragraph is assigned to exactly one segment (no gaps or overlaps).
   • Intro and wrap together use no more than ~40% of the paragraphs; the core must receive the majority (at least ~40%).
   • Use zero-based paragraph indices and return integer boundaries where paragraph_end is the index immediately after the segment.

Return valid JSON with this schema (no additional text):
{{
  "plan": [
    {{"role": "intro", "summary": "...", "notes": ["...", "..."]}},
    {{"role": "core", "summary": "...", "notes": ["...", "..."]}},
    {{"role": "wrap", "summary": "...", "notes": ["...", "..."]}}
  ],
  "segments": [
    {{"role": "intro", "paragraph_start": 0, "paragraph_end": 4}},
    {{"role": "core", "paragraph_start": 4, "paragraph_end": 18}},
    {{"role": "wrap", "paragraph_start": 18, "paragraph_end": {len(paragraphs)}}}
  ]
}}

Article title: {title}
Paragraphs:\n{chr(10).join(excerpt_lines)}
"""

        messages = [
            LLMMessage(
                role="system",
                content=(
                    "You are an expert long-form editor. Produce precise JSON with contiguous segments."
                ),
            ),
            LLMMessage(role="user", content=prompt),
        ]

        response_schema = {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string"},
                            "summary": {"type": "string"},
                            "notes": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["role", "summary", "notes"],
                    },
                },
                "segments": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string"},
                            "paragraph_start": {"type": "integer"},
                            "paragraph_end": {"type": "integer"},
                        },
                        "required": ["role", "paragraph_start", "paragraph_end"],
                    },
                },
            },
            "required": ["plan", "segments"],
        }

        response = self.client.complete(
            messages,
            temperature=0.0,
            response_format=response_schema,
        )
        try:
            data = self.client.extract_json(response)
        except ValueError:
            retry_messages = messages + [
                LLMMessage(role="assistant", content=response.content),
                LLMMessage(
                    role="user",
                    content="Respond with JSON ONLY that matches the exact schema provided earlier. No prose.",
                ),
            ]
            response = self.client.complete(
                retry_messages,
                temperature=0.0,
                response_format=response_schema,
            )
            data = self.client.extract_json(response)

        raw_plan = data.get("plan", [])
        raw_segments = data.get("segments", [])

        if len(raw_plan) != 3 or len(raw_segments) != 3:
            raise ValueError(f"Teacher returned unexpected structure: {json.dumps(data, indent=2)}")

        summaries: List[PlanSummary] = []
        summary_by_role: dict[str, str] = {}
        for item in raw_plan:
            role = item.get("role")
            summary = item.get("summary")
            notes_raw = item.get("notes", [])
            if not role or summary is None:
                raise ValueError(f"Plan item missing role/summary: {item}")
            if not isinstance(notes_raw, list):
                raise ValueError(f"Plan item notes must be a list: {item}")
            notes = [str(note).strip() for note in notes_raw if str(note).strip()]
            summaries.append(PlanSummary(role=role, summary=summary, notes=notes))
            summary_by_role[role.lower()] = summary

        plan_segments: List[PlanSegment] = []

        for segment in raw_segments:
            role = segment.get("role")
            if role is None:
                raise ValueError(f"Segment missing role: {segment}")
            start = int(segment.get("paragraph_start"))
            end = int(segment.get("paragraph_end"))
            if start < 0 or end < 0 or start >= end or end > len(paragraphs):
                raise ValueError(f"Invalid paragraph range for role {role}: {segment}")
            plan_segments.append(
                PlanSegment(
                    role=role,
                    paragraph_start=start,
                    paragraph_end=end,
                )
            )

        # Sort segments in article order to guarantee contiguity downstream.
        plan_segments.sort(key=lambda seg: seg.paragraph_start)

        return TeacherPlan(summaries=summaries, segments=plan_segments, raw=data)


__all__ = [
    "PlanSegment",
    "TeacherPlan",
    "TeacherPlanner",
]
