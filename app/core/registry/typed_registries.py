"""Typed sub-registries for the GeneralAI platform.

Each registry is a thin typed wrapper around :class:`BaseRegistry`
with a domain-specific name for clarity and future extension.
"""

from __future__ import annotations

from app.core.interfaces.iagent import IAgent
from app.core.interfaces.ibrain import IBrain
from app.core.interfaces.imemory import IMemory
from app.core.interfaces.iplanner import IPlanner
from app.core.interfaces.iplugin import IPlugin
from app.core.interfaces.itool import ITool
from app.core.interfaces.iworkflow import IWorkflow
from app.core.registry.base_registry import BaseRegistry


class BrainRegistry(BaseRegistry[IBrain]):
    """Registry for brain module implementations."""


class MemoryRegistry(BaseRegistry[IMemory]):
    """Registry for memory module implementations."""


class ToolRegistry(BaseRegistry[ITool]):
    """Registry for tool implementations."""


class AgentRegistry(BaseRegistry[IAgent]):
    """Registry for agent implementations."""


class PlannerRegistry(BaseRegistry[IPlanner]):
    """Registry for planner implementations."""


class PluginRegistry(BaseRegistry[IPlugin]):
    """Registry for loaded plugin instances."""


class WorkflowRegistry(BaseRegistry[IWorkflow]):
    """Registry for workflow definitions."""
