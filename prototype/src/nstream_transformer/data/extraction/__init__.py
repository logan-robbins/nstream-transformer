"""Teacher note extraction using LLMs."""

from .extractor import LLMTeacherNoteExtractor
from .llm_client import (
    AnthropicClient,
    LLMClient,
    LLMMessage,
    LLMResponse,
    MockLLMClient,
    OllamaClient,
    OpenAIClient,
    create_llm_client,
)
from .planner import PlanSegment, TeacherPlan, TeacherPlanner
from .schema import (
    DiscourseAnnotation,
    DiscourseType,
    DocumentAnalysis,
    EntityType,
    ExtractedNoteSet,
    ExtractionResult,
    NoteLevel,
    RelationType,
    RichEntity,
    RichRelation,
    TextSpan,
)

__all__ = [
    # Schema
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
    # LLM Clients
    "LLMClient",
    "LLMMessage",
    "LLMResponse",
    "AnthropicClient",
    "OpenAIClient",
    "OllamaClient",
    "MockLLMClient",
    "create_llm_client",
    # Planner
    "PlanSegment",
    "TeacherPlan",
    "TeacherPlanner",
    # Extractor
    "LLMTeacherNoteExtractor",
]
