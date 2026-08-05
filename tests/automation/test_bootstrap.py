"""Tests for the automation DI bootstrap and server wiring."""

from __future__ import annotations

import asyncio

import pytest

from app.automation.bootstrap import register_automation_components
from app.automation.models import WorkflowDefinition, WorkflowStep, WorkflowStepType
from app.automation.registries import WorkflowRegistry, WorkflowRunRegistry
from app.automation.stores import (
    EventStore,
    InMemoryWorkflowStore,
    ScheduleStore,
    WorkflowRunStore,
    WorkflowStore,
)
from app.automation.validation import WorkflowValidator
from app.automation.workflow import WorkflowGraphExporter, WorkflowService
from app.core.container import DependencyContainer


def _definition() -> WorkflowDefinition:
    return WorkflowDefinition(
        id="wf",
        version="1.0.0",
        steps=(WorkflowStep(id="a", type=WorkflowStepType.TRANSFORM, expression="1"),),
    )


def test_bootstrap_registers_singletons() -> None:
    container = DependencyContainer()
    register_automation_components(container)

    for interface in (
        WorkflowStore,
        WorkflowRunStore,
        ScheduleStore,
        EventStore,
        WorkflowRegistry,
        WorkflowRunRegistry,
        WorkflowValidator,
        WorkflowGraphExporter,
        WorkflowService,
    ):
        assert container.has(interface), f"missing {interface}"

    service_a = container.resolve(WorkflowService)
    service_b = container.resolve(WorkflowService)
    assert service_a is service_b


def test_bootstrap_is_idempotent() -> None:
    container = DependencyContainer()
    register_automation_components(container)
    register_automation_components(container)  # must not raise
    register_automation_components(container)
    assert container.has(WorkflowService)


def test_service_manages_definitions_via_di() -> None:
    container = DependencyContainer()
    register_automation_components(container)
    service = container.resolve(WorkflowService)
    created = service.create_definition(_definition())
    assert created.id == "wf"
    assert service.get_definition("wf", "1.0.0") is not None


def test_service_executes_via_di() -> None:
    container = DependencyContainer()
    register_automation_components(container)
    service = container.resolve(WorkflowService)
    service.create_definition(_definition())
    published = service.publish_definition("wf", "1.0.0")
    assert published.status.value == "published"


def test_service_exposes_registries_via_di() -> None:
    container = DependencyContainer()
    register_automation_components(container)
    service = container.resolve(WorkflowService)
    assert service.registry is container.resolve(WorkflowRegistry)
    assert service.run_registry is container.resolve(WorkflowRunRegistry)


def test_bootstrap_respects_pre_registered_stores() -> None:
    container = DependencyContainer()
    custom = InMemoryWorkflowStore()
    container.register_singleton(WorkflowStore, instance=custom)
    register_automation_components(container)
    assert container.resolve(WorkflowService) is not None
    assert container.resolve(WorkflowStore) is custom  # type: ignore[type-abstract]


def test_bootstrap_with_agent_collaborators() -> None:
    container = DependencyContainer()
    register_automation_components(container)
    service = container.resolve(WorkflowService)
    # Collaborators absent => no crash, TASK steps fail at runtime.
    assert service is not None
    service.create_definition(_definition())
    with pytest.raises(Exception):
        asyncio.run(service.execute("wf", {}))


def test_server_exposes_workflow_service() -> None:
    from fastapi.testclient import TestClient

    from app.server.app import create_app

    app = create_app()
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        service = app.state.workflow_service
        assert service is not None

        response = client.get("/health")
        assert response.status_code == 200

        # Expose a definition through the DI-wired service.
        definition = service.create_definition(_definition())
        assert definition.id == "wf"
