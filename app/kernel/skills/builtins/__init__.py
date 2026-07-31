"""Built-in skill implementations and registration."""

from __future__ import annotations

from typing import Any

from app.kernel.skills.builtins import (
    calculator,
    echo,
    respond,
    search_memory,
    summarize,
)
from app.kernel.skills.models import SkillDescriptor


def register_builtin_skills(registry: Any) -> None:
    """Register all built-in skills with a SkillRegistry.

    Args:
        registry: A SkillRegistry instance (or compatible).
    """
    skills = [
        SkillDescriptor(
            name="echo",
            description="Echo the input message",
            required_tools=(),
            input_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Message to echo"}
                },
                "required": ["message"],
            },
        ),
        SkillDescriptor(
            name="calculator",
            description="Evaluate mathematical expressions",
            required_tools=(),
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"],
            },
        ),
        SkillDescriptor(
            name="summarize",
            description="Deterministically summarize text",
            required_tools=(),
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to summarize"},
                    "max_sentences": {"type": "integer", "default": 3},
                },
                "required": ["text"],
            },
        ),
        SkillDescriptor(
            name="search_memory",
            description="Search experience records",
            required_tools=(),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        ),
        SkillDescriptor(
            name="respond",
            description="Build a response message",
            required_tools=(),
            input_schema={
                "type": "object",
                "properties": {
                    "content": {"description": "Response content"},
                    "format": {"type": "string", "default": "text"},
                },
                "required": ["content"],
            },
        ),
    ]

    for desc in skills:
        registry.register_skill(desc)


def get_skill_handler(skill_name: str) -> Any:
    """Get the handler function for a built-in skill.

    Args:
        skill_name: Name of the skill.

    Returns:
        Async handler function.

    Raises:
        KeyError: If the skill is not a built-in.
    """
    handlers: dict[str, Any] = {
        "echo": echo.execute,
        "calculator": calculator.execute,
        "summarize": summarize.execute,
        "search_memory": search_memory.execute,
        "respond": respond.execute,
    }
    if skill_name not in handlers:
        raise KeyError(f"Unknown built-in skill: {skill_name}")
    return handlers[skill_name]


__all__ = [
    "calculator",
    "echo",
    "respond",
    "search_memory",
    "summarize",
    "register_builtin_skills",
    "get_skill_handler",
]
