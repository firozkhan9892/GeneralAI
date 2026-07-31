"""Skills package exports."""

from app.core.registry.base_registry import BaseRegistry
from app.kernel.skills.builtins import register_builtin_skills
from app.kernel.skills.executor import SkillSelector, SkillExecutor
from app.kernel.skills.models import (
    Skill,
    SkillDescriptor,
    SkillResult,
    ToolRequirement,
)

SkillRegistry = BaseRegistry[SkillDescriptor]

__all__ = [
    "SkillSelector",
    "SkillExecutor",
    "SkillRegistry",
    "Skill",
    "SkillDescriptor",
    "SkillResult",
    "ToolRequirement",
    "register_builtin_skills",
]
