"""Dependency-injection wiring for the automation module.

Registers the :class:`WorkflowService` and its collaborators (registries,
executor, validator, graph exporter and the durable stores) with the
application's :class:`DependencyContainer`.  Registration is idempotent:
re-running :func:`register_automation_components` never raises, so it is
safe to call from the server factory and the lifespan.
"""

from __future__ import annotations

import logging

from app.automation.registries import (
    StepTypeRegistry,
    WorkflowRegistry,
    WorkflowRunRegistry,
)
from app.automation.stores import (
    EventStore,
    InMemoryEventStore,
    InMemoryScheduleStore,
    InMemoryWorkflowRunStore,
    InMemoryWorkflowStore,
    ScheduleStore,
    WorkflowRunStore,
    WorkflowStore,
)
from app.automation.validation import WorkflowValidator
from app.automation.workflow import WorkflowGraphExporter, WorkflowService
from app.core.container import DependencyContainer

log = logging.getLogger(__name__)


def register_automation_components(container: DependencyContainer) -> None:
    """Register workflow automation components with a DI container.

    Idempotent: each type is registered only when absent so repeated
    calls (server factory + lifespan) are safe.  All components are
    singletons.  Collaborators owned by other modules (tool executor,
    agent manager, LLM router) are resolved lazily by the service
    factory when present, otherwise left ``None``.

    Args:
        container: The application's ``DependencyContainer``.
    """
    # In-memory stores (default).  A persistent/JSON-backed store can be
    # registered beforehand — the container.has() guard honours it.
    if not container.has(WorkflowStore):
        container.register_singleton(
            WorkflowStore, factory=lambda: InMemoryWorkflowStore()
        )
    if not container.has(WorkflowRunStore):
        container.register_singleton(
            WorkflowRunStore, factory=lambda: InMemoryWorkflowRunStore()
        )
    if not container.has(ScheduleStore):
        container.register_singleton(
            ScheduleStore, factory=lambda: InMemoryScheduleStore()
        )
    if not container.has(EventStore):
        container.register_singleton(EventStore, factory=lambda: InMemoryEventStore())

    # Registries and stateless collaborators.
    if not container.has(WorkflowRegistry):
        container.register_singleton(WorkflowRegistry)
    if not container.has(WorkflowRunRegistry):
        container.register_singleton(WorkflowRunRegistry)
    if not container.has(StepTypeRegistry):
        container.register_singleton(StepTypeRegistry)
    if not container.has(WorkflowValidator):
        container.register_singleton(WorkflowValidator)
    if not container.has(WorkflowGraphExporter):
        container.register_singleton(WorkflowGraphExporter)

    # The executor is built by the container (its StepTypeRegistry is
    # injected automatically) and the service is wired last so it can
    # bind the subworkflow runner.
    if not container.has(_executor_type()):
        container.register_singleton(_executor_type())
    if not container.has(WorkflowService):
        container.register_singleton(
            WorkflowService, factory=_make_workflow_service(container)
        )
    log.info("Registered automation components with DI container")


def _make_workflow_service(container: DependencyContainer):
    """Return a factory building a WorkflowService from the container."""

    def _factory() -> WorkflowService:
        service = WorkflowService(
            registry=container.resolve(WorkflowRegistry),
            run_registry=container.resolve(WorkflowRunRegistry),
            executor=container.resolve(_executor_type()),
            validator=container.resolve(WorkflowValidator),
            exporter=container.resolve(WorkflowGraphExporter),
            definition_store=container.resolve(WorkflowStore),  # type: ignore[type-abstract]
            run_store=container.resolve(WorkflowRunStore),  # type: ignore[type-abstract]
            schedule_store=container.resolve(ScheduleStore),  # type: ignore[type-abstract]
            event_store=container.resolve(EventStore),  # type: ignore[type-abstract]
            tool_executor=_try_resolve_tool_executor(container),
            agent_manager=_try_resolve_agent_manager(container),
            llm_router=_try_resolve_llm_router(container),
        )
        return service

    return _factory


def _executor_type():
    """Return the WorkflowExecutor type without a circular import."""
    from app.automation.executor import WorkflowExecutor

    return WorkflowExecutor


def _try_resolve_tool_executor(container: DependencyContainer):
    try:
        from app.tools.executor import ToolExecutor
    except ImportError:  # pragma: no cover - dependency absence guard
        return None
    return container.resolve(ToolExecutor) if container.has(ToolExecutor) else None


def _try_resolve_agent_manager(container: DependencyContainer):
    try:
        from app.agents.manager import AgentManager
    except ImportError:  # pragma: no cover - dependency absence guard
        return None
    return container.resolve(AgentManager) if container.has(AgentManager) else None


def _try_resolve_llm_router(container: DependencyContainer):
    try:
        from app.llm.llm_router import LLMRouter
    except ImportError:  # pragma: no cover - dependency absence guard
        return None
    return container.resolve(LLMRouter) if container.has(LLMRouter) else None
