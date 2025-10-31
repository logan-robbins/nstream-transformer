"""Model primitives for the GPT-OSS backed N-Stream Transformer."""

from .nstream_transformer import NStreamModelConfig, NStreamTransformer
from .role_adapters import RoleAdapterConfig, RoleAdapters
from .heads import (
    PlannerHead,
    PlannerHeadConfig,
    NotesHead,
    NotesHeadConfig,
    SpeculationHead,
    SpeculationHeadConfig,
    AgreementHead,
    AgreementHeadConfig,
)

__all__ = [
    "NStreamTransformer",
    "NStreamModelConfig",
    "RoleAdapterConfig",
    "RoleAdapters",
    "PlannerHead",
    "PlannerHeadConfig",
    "NotesHead",
    "NotesHeadConfig",
    "SpeculationHead",
    "SpeculationHeadConfig",
    "AgreementHead",
    "AgreementHeadConfig",
]
