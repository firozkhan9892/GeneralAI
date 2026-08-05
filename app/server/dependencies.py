"""FastAPI dependencies that read from the application state."""

from __future__ import annotations

from fastapi import Request

from app.agents.manager import AgentManager
from app.automation.workflow import WorkflowService
from app.kernel.memory.engine import MemoryEngine
from app.server.config import ServerSettings
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


def get_settings(request: Request) -> ServerSettings:
    """Return the server settings from the application state."""
    return request.app.state.settings


def get_container(request: Request):
    """Return the DI container from the application state."""
    return request.app.state.container


def get_agent_manager(request: Request) -> AgentManager:
    """Return the shared AgentManager."""
    return request.app.state.agent_manager


def get_memory_engine(request: Request) -> MemoryEngine:
    """Return the shared MemoryEngine."""
    return request.app.state.memory_engine


def get_tool_registry(request: Request) -> ToolRegistry:
    """Return the shared ToolRegistry."""
    return request.app.state.tool_registry


def get_tool_executor(request: Request) -> ToolExecutor:
    """Return the shared ToolExecutor."""
    return request.app.state.tool_executor


def get_rate_limiter(request: Request):
    """Return the shared RateLimiter."""
    return request.app.state.rate_limiter


def get_workflow_service(request: Request) -> WorkflowService:
    """Return the shared WorkflowService."""
    return request.app.state.workflow_service
