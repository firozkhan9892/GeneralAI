"""Built-in tool implementations and registration."""

from __future__ import annotations

from typing import Any

from app.kernel.tools.builtins import calculator, clock, json, text_utils, uuid
from app.kernel.tools.models import ToolDescriptor


def register_builtin_tools(registry: Any) -> None:
    """Register all built-in tools with a ToolRegistry.

    Args:
        registry: A ToolRegistry instance (or compatible).
    """
    tools = [
        ToolDescriptor(
            name="calculator",
            description="Evaluate mathematical expressions",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"],
            },
        ),
        ToolDescriptor(
            name="clock",
            description="Get current time and date",
            input_schema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["iso", "unix", "human"],
                        "default": "iso",
                    }
                },
            },
        ),
        ToolDescriptor(
            name="uuid",
            description="Generate unique identifiers",
            input_schema={
                "type": "object",
                "properties": {
                    "version": {"type": "integer", "enum": [1, 4], "default": 4}
                },
            },
        ),
        ToolDescriptor(
            name="json",
            description="Parse, stringify, or validate JSON",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["parse", "stringify", "validate"],
                        "default": "parse",
                    },
                    "data": {"description": "JSON string or object"},
                },
                "required": ["operation"],
            },
        ),
        ToolDescriptor(
            name="text_utils",
            description="Text manipulation utilities",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "uppercase",
                            "lowercase",
                            "strip",
                            "truncate",
                            "word_count",
                            "char_count",
                            "replace",
                            "split_paragraphs",
                        ],
                        "default": "strip",
                    },
                    "text": {"type": "string", "description": "Input text"},
                },
                "required": ["operation"],
            },
        ),
    ]

    for desc in tools:
        registry.register_tool(desc)


def get_tool_handler(tool_name: str) -> Any:
    """Get the handler function for a built-in tool.

    Args:
        tool_name: Name of the tool.

    Returns:
        Async handler function.

    Raises:
        KeyError: If the tool is not a built-in.
    """
    handlers: dict[str, Any] = {
        "calculator": calculator.execute,
        "clock": clock.execute,
        "uuid": uuid.execute,
        "json": json.execute,
        "text_utils": text_utils.execute,
    }
    if tool_name not in handlers:
        raise KeyError(f"Unknown built-in tool: {tool_name}")
    return handlers[tool_name]


__all__ = [
    "calculator",
    "clock",
    "json",
    "text_utils",
    "uuid",
    "register_builtin_tools",
    "get_tool_handler",
]
