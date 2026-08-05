"""Plugin integration with the workflow automation subsystem.

Covers the additive PluginContext fields (workflow_service /
workflow_registry / step_type_registry) and the PluginManager cleanup
that unregisters plugin-owned workflows during unload.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock


from app.automation.models import WorkflowDefinition, WorkflowStep, WorkflowStepType
from app.automation.registries import WorkflowRegistry
from app.plugins.base import PluginBase, PluginContext
from app.plugins.manager import PluginManager
from app.plugins.models import (
    PluginLifecycleState,
    PluginManifest,
    PluginState,
    PluginType,
)


def test_plugin_context_has_workflow_fields() -> None:
    context = PluginContext()
    assert context.workflow_service is None
    assert context.workflow_registry is None
    assert context.step_type_registry is None

    filled = PluginContext(
        workflow_service=object(),
        workflow_registry=WorkflowRegistry(),
        step_type_registry=object(),
    )
    assert filled.workflow_service is not None
    assert filled.workflow_registry is not None
    assert filled.step_type_registry is not None


def _definition() -> WorkflowDefinition:
    return WorkflowDefinition(
        id="plugin-wf",
        version="1.0.0",
        steps=(WorkflowStep(id="a", type=WorkflowStepType.TASK),),
    )


class WorkflowPlugin(PluginBase):
    """A plugin that registers a workflow during enable."""

    def __init__(self, registry: WorkflowRegistry) -> None:
        super().__init__()
        self._registry = registry
        self._manifest = PluginManifest(
            name="wf_plugin",
            version="1.0.0",
            plugin_type=PluginType.WORKFLOW,
        )

    @property
    def manifest(self) -> PluginManifest:
        return self._manifest

    async def enable(self, context: PluginContext) -> list[str]:
        definition = _definition()
        if context.workflow_registry is not None:
            context.workflow_registry.register(definition)
        return [definition.id]

    async def unregister(self, context: PluginContext) -> None:
        if context.workflow_registry is not None:
            context.workflow_registry.unregister("plugin-wf")


def test_plugin_manager_unloads_plugin_workflows() -> None:
    registry = WorkflowRegistry()
    plugin = WorkflowPlugin(registry)
    state = PluginState(
        name="wf_plugin",
        plugin_type=PluginType.WORKFLOW,
        lifecycle_state=PluginLifecycleState.INSTALLED,
        manifest=plugin.manifest,
    )

    manager = PluginManager()
    manager._registry.register_plugin(plugin, state)
    manager._registry.set_state("wf_plugin", PluginLifecycleState.LOADED)
    context = PluginContext(workflow_registry=registry)
    manager.context = context

    asyncio.run(plugin.enable(context))
    assert registry.has("plugin-wf", "1.0.0")

    manager.enable("wf_plugin")
    assert registry.has("plugin-wf", "1.0.0")

    manager.disable("wf_plugin")
    manager.unload("wf_plugin")

    assert not registry.has("plugin-wf", "1.0.0")


def test_plugin_context_accepts_mock_workflow_service() -> None:
    service = MagicMock()
    context = PluginContext(workflow_service=service)
    assert context.workflow_service is service
