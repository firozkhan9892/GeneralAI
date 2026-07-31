"""Catalogue of built-in tools, grouped by category.

The catalogue lists concrete tool instances that ship with the platform.
It is a plain data module so it can be consumed by the registry (for
``discover``) and by callers that want the default tool set, without
introducing import cycles.
"""

from __future__ import annotations

from app.tools.base import Tool
from app.tools.categories.builtin import (
    CalculatorTool,
    ClockTool,
    EchoTool,
    TextUtilsTool,
)
from app.tools.categories.file import FileListTool, FileReadTool, FileWriteTool
from app.tools.categories.http import HttpRequestTool
from app.tools.categories.python import PythonEvalTool
from app.tools.categories.shell import ShellRunTool
from app.tools.categories.web import WebFetchTool
from app.tools.models import ToolCategory

#: The default set of tools available out of the box.
DEFAULT_TOOLS: tuple[Tool, ...] = (
    CalculatorTool(),
    ClockTool(),
    EchoTool(),
    TextUtilsTool(),
    FileReadTool(),
    FileWriteTool(),
    FileListTool(),
    WebFetchTool(),
    ShellRunTool(),
    PythonEvalTool(),
    HttpRequestTool(),
)


def tools_by_category(category: ToolCategory) -> list[Tool]:
    """Return the default tools belonging to *category*.

    Args:
        category: The category to filter by.

    Returns:
        Tools whose category matches.
    """
    return [tool for tool in DEFAULT_TOOLS if tool.category == category]
