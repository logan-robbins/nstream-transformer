"""Role-specific heads for the GPT-OSS integration."""

from .planner import PlannerHead, PlannerHeadConfig
from .notes import NotesHead, NotesHeadConfig
from .speculation import SpeculationHead, SpeculationHeadConfig
from .agreement import AgreementHead, AgreementHeadConfig
from .coverage import CoverageHead, CoverageHeadConfig
from .role_classifier import RoleClassifierConfig, RoleClassifierHead

__all__ = [
    "PlannerHead",
    "PlannerHeadConfig",
    "NotesHead",
    "NotesHeadConfig",
    "SpeculationHead",
    "SpeculationHeadConfig",
    "AgreementHead",
    "AgreementHeadConfig",
    "CoverageHead",
    "CoverageHeadConfig",
    "RoleClassifierHead",
    "RoleClassifierConfig",
]
