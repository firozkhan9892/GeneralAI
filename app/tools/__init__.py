"""Tool System.

A production-grade tool execution framework: frozen domain models, a
base tool abstraction, a registry, a policy-driven executor with
timeout/cancellation/retry support, a permission system, concrete
category tools (file, web, shell, python, http, builtin), and a
deterministic mock for testing.
"""

from __future__ import annotations

from app.tools.base import Tool
from app.tools.bootstrap import register_tool_components
from app.tools.context import (
    CancellationToken,
    ExecutionContext,
    Memory,
    ToolContext,
    ToolSession,
)
from app.tools.exceptions import (
    PermissionDeniedError,
    ToolAlreadyRegisteredError,
    ToolCancelledError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolTimeoutError,
    ToolValidationError,
)
from app.tools.executor import ToolExecutor
from app.tools.mock import MockTool
from app.tools.models import (
    ToolCategory,
    ToolMetadata,
    ToolParameter,
    ToolResult,
)
from app.tools.network import HttpClient, HttpResponse, UrllibHttpClient
from app.tools.permissions import (
    PermissionDecision,
    PermissionResult,
    PermissionSystem,
)
from app.tools.registry import ToolRegistry

__all__ = [
    "Tool",
    "register_tool_components",
    "CancellationToken",
    "ExecutionContext",
    "Memory",
    "ToolContext",
    "ToolSession",
    "PermissionDeniedError",
    "ToolAlreadyRegisteredError",
    "ToolCancelledError",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolTimeoutError",
    "ToolValidationError",
    "ToolExecutor",
    "MockTool",
    "ToolCategory",
    "ToolMetadata",
    "ToolParameter",
    "ToolResult",
    "HttpClient",
    "HttpResponse",
    "UrllibHttpClient",
    "PermissionDecision",
    "PermissionResult",
    "PermissionSystem",
    "ToolRegistry",
]
