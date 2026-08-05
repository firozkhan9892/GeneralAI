"""Workflow automation REST endpoints (Phase 12e).

Routers are thin: every endpoint delegates to :class:`WorkflowService`
through the :func:`get_workflow_service` dependency.  Domain errors
raised by the service are mapped to HTTP status codes by the exception
handlers registered in :mod:`app.server.app`.

The workflow router exposes definitions, runs, approvals, graph export
and an SSE event stream; a second router exposes schedules.

Route order matters: FastAPI matches paths in registration order, so the
literal ``/workflows/runs`` routes are declared **before** the
parameterised ``/workflows/{workflow_id}`` routes.  Otherwise a request
for ``GET /workflows/runs`` would be captured with ``workflow_id="runs"``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse

from app.automation.models import (
    ScheduleSpec,
    WorkflowDefinition,
    WorkflowRun,
    WorkflowStatus,
)
from app.automation.workflow import WorkflowService
from app.server.dependencies import get_workflow_service
from app.server.schemas import (
    ApprovalDecisionRequest,
    ScheduleCreateRequest,
    ScheduleListResponse,
    ScheduleUpdateRequest,
    WorkflowListResponse,
    WorkflowPublishRequest,
    WorkflowRunListResponse,
    WorkflowRunRequest,
    WorkflowVersionCreateRequest,
)
from app.server.streaming import sse_format

log = logging.getLogger(__name__)

router = APIRouter(prefix="/workflows", tags=["workflows"])
schedule_router = APIRouter(prefix="/schedules", tags=["schedules"])


# ----------------------------------------------------------------------
# Workflow definitions
# ----------------------------------------------------------------------


@router.post("", response_model=WorkflowDefinition, summary="Create a workflow")
async def create_workflow(
    body: WorkflowDefinition,
    service: WorkflowService = Depends(get_workflow_service),
) -> WorkflowDefinition:
    """Register a new workflow definition (draft)."""
    return service.create_definition(body)


@router.get("", response_model=WorkflowListResponse, summary="List workflows")
async def list_workflows(
    status: str | None = Query(default=None, description="Filter by status"),
    service: WorkflowService = Depends(get_workflow_service),
) -> WorkflowListResponse:
    """Return registered workflow definitions, optionally filtered."""
    status_enum: WorkflowStatus | None = None
    if status is not None:
        try:
            status_enum = WorkflowStatus(status)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status '{status}'. Must be one of "
                f"{[s.value for s in WorkflowStatus]}.",
            ) from exc
    workflows = service.list_definitions(status_enum)
    return WorkflowListResponse(total=len(workflows), workflows=workflows)


# ----------------------------------------------------------------------
# Runs — declared before the parameterised /{workflow_id} routes so the
# literal "runs" segment is never captured as a workflow id.
# ----------------------------------------------------------------------


@router.get("/runs", response_model=WorkflowRunListResponse, summary="List runs")
async def list_runs(
    workflow_id: str | None = Query(default=None, description="Filter by workflow"),
    status: str | None = Query(default=None, description="Filter by run status"),
    service: WorkflowService = Depends(get_workflow_service),
) -> WorkflowRunListResponse:
    """Return workflow runs, newest first, optionally filtered."""
    runs = service.list_runs(workflow_id, status)
    return WorkflowRunListResponse(total=len(runs), runs=runs)


@router.get("/runs/{run_id}", response_model=WorkflowRun, summary="Get a run")
async def get_run(
    run_id: str,
    service: WorkflowService = Depends(get_workflow_service),
) -> WorkflowRun:
    """Return a single workflow run snapshot."""
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return run


@router.post(
    "/runs/{run_id}/cancel", response_model=WorkflowRun, summary="Cancel a run"
)
async def cancel_run(
    run_id: str,
    service: WorkflowService = Depends(get_workflow_service),
) -> WorkflowRun:
    """Cancel a non-terminal workflow run."""
    return service.cancel(run_id)


@router.post(
    "/runs/{run_id}/steps/{step_id}/approve",
    response_model=WorkflowRun,
    summary="Approve a pending step",
)
async def approve_step(
    run_id: str,
    step_id: str,
    body: ApprovalDecisionRequest,
    service: WorkflowService = Depends(get_workflow_service),
) -> WorkflowRun:
    """Record an approval decision for a pending approval step."""
    run, request = _approval_request(service, run_id, step_id)
    return service.approve(
        run.run_id,
        request.request_id,
        decided_by=body.decided_by,
        decision_note=body.decision_note,
    )


@router.post(
    "/runs/{run_id}/steps/{step_id}/reject",
    response_model=WorkflowRun,
    summary="Reject a pending step",
)
async def reject_step(
    run_id: str,
    step_id: str,
    body: ApprovalDecisionRequest,
    service: WorkflowService = Depends(get_workflow_service),
) -> WorkflowRun:
    """Record a rejection decision for a pending approval step."""
    run, request = _approval_request(service, run_id, step_id)
    return service.reject(
        run.run_id,
        request.request_id,
        decided_by=body.decided_by,
        decision_note=body.decision_note,
    )


@router.get("/runs/{run_id}/events", response_model=None, summary="Stream run events")
async def stream_run_events(
    run_id: str,
    service: WorkflowService = Depends(get_workflow_service),
) -> StreamingResponse:
    """Stream a run's events via Server-Sent Events until it terminates."""
    if service.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return StreamingResponse(
        _stream_run_events(service, run_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ----------------------------------------------------------------------
# Single-workflow routes
# ----------------------------------------------------------------------


@router.get(
    "/{workflow_id}", response_model=WorkflowDefinition, summary="Get a workflow"
)
async def get_workflow(
    workflow_id: str,
    version: str | None = Query(default=None, description="Specific version"),
    service: WorkflowService = Depends(get_workflow_service),
) -> WorkflowDefinition:
    """Return a workflow definition (latest published unless *version* is given)."""
    definition = service.get_definition(workflow_id, version)
    if definition is None:
        raise HTTPException(
            status_code=404, detail=f"Workflow '{workflow_id}' not found"
        )
    return definition


@router.get(
    "/{workflow_id}/versions",
    response_model=WorkflowListResponse,
    summary="List workflow versions",
)
async def list_workflow_versions(
    workflow_id: str,
    service: WorkflowService = Depends(get_workflow_service),
) -> WorkflowListResponse:
    """Return every version of a workflow, newest first."""
    versions = service.registry.list_versions(workflow_id)
    if not versions:
        raise HTTPException(
            status_code=404, detail=f"Workflow '{workflow_id}' not found"
        )
    definitions = [
        definition
        for version in versions
        if (definition := service.get_definition(workflow_id, version)) is not None
    ]
    return WorkflowListResponse(total=len(definitions), workflows=definitions)


@router.post(
    "/{workflow_id}/versions",
    response_model=WorkflowDefinition,
    summary="Create a workflow version",
)
async def create_workflow_version(
    workflow_id: str,
    body: WorkflowVersionCreateRequest,
    service: WorkflowService = Depends(get_workflow_service),
) -> WorkflowDefinition:
    """Register a new draft version of an existing workflow."""
    definition = body.definition.model_copy(
        update={"id": workflow_id, "version": body.version}
    )
    return service.create_definition(definition)


@router.post(
    "/{workflow_id}/publish",
    response_model=WorkflowDefinition,
    summary="Publish a workflow",
)
async def publish_workflow(
    workflow_id: str,
    body: WorkflowPublishRequest,
    service: WorkflowService = Depends(get_workflow_service),
) -> WorkflowDefinition:
    """Validate and publish a workflow version."""
    return service.publish_definition(workflow_id, body.version)


@router.post("/{workflow_id}/run", response_model=WorkflowRun, summary="Run a workflow")
async def run_workflow(
    workflow_id: str,
    body: WorkflowRunRequest,
    service: WorkflowService = Depends(get_workflow_service),
) -> WorkflowRun:
    """Start (or deduplicate) a workflow run."""
    return await service.execute(
        workflow_id,
        body.inputs,
        version=body.version,
        idempotency_key=body.idempotency_key,
    )


@router.get("/{workflow_id}/graph", summary="Export the workflow graph")
async def workflow_graph(
    workflow_id: str,
    format: str = Query(
        default="json", pattern="^(json|mermaid|dot)$", description="Output format"
    ),
    version: str | None = Query(default=None, description="Specific version"),
    service: WorkflowService = Depends(get_workflow_service),
):
    """Return the workflow DAG as JSON, Mermaid or Graphviz DOT."""
    graph = service.export_graph(workflow_id, version)
    if format == "mermaid":
        return PlainTextResponse(_to_mermaid(graph), media_type="text/plain")
    if format == "dot":
        return PlainTextResponse(_to_dot(graph), media_type="text/plain")
    return graph


# ----------------------------------------------------------------------
# Schedules
# ----------------------------------------------------------------------


@schedule_router.post("", response_model=ScheduleSpec, summary="Create a schedule")
async def create_schedule(
    body: ScheduleCreateRequest,
    service: WorkflowService = Depends(get_workflow_service),
) -> ScheduleSpec:
    """Create a schedule that fires a workflow on a trigger."""
    spec = service.create_schedule(
        workflow_id=body.workflow_id,
        trigger_type=body.trigger_type,
        cron_expression=body.cron_expression,
        interval_seconds=body.interval_seconds,
        run_at=body.run_at,
        timezone=body.timezone,
        payload=body.payload,
        enabled=body.enabled,
        max_concurrent_runs=body.max_concurrent_runs,
    )
    if body.workflow_version:
        spec = service.update_schedule(
            spec.model_copy(update={"workflow_version": body.workflow_version})
        )
    return spec


@schedule_router.get("", response_model=ScheduleListResponse, summary="List schedules")
async def list_schedules(
    enabled: bool | None = Query(default=None, description="Filter by enabled state"),
    service: WorkflowService = Depends(get_workflow_service),
) -> ScheduleListResponse:
    """Return stored schedules, optionally filtered by enabled state."""
    schedules = service.list_schedules(enabled)
    return ScheduleListResponse(total=len(schedules), schedules=schedules)


@schedule_router.patch(
    "/{schedule_id}", response_model=ScheduleSpec, summary="Update a schedule"
)
async def update_schedule(
    schedule_id: str,
    body: ScheduleUpdateRequest,
    service: WorkflowService = Depends(get_workflow_service),
) -> ScheduleSpec:
    """Partially update a schedule."""
    existing = service.get_schedule(schedule_id)
    if existing is None:
        raise HTTPException(
            status_code=404, detail=f"Schedule '{schedule_id}' not found"
        )
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    return service.update_schedule(existing.model_copy(update=updates))


@schedule_router.delete("/{schedule_id}", status_code=204, summary="Delete a schedule")
async def delete_schedule(
    schedule_id: str,
    service: WorkflowService = Depends(get_workflow_service),
) -> None:
    """Delete a schedule."""
    if not service.delete_schedule(schedule_id):
        raise HTTPException(
            status_code=404, detail=f"Schedule '{schedule_id}' not found"
        )


@schedule_router.post(
    "/{schedule_id}/run", response_model=WorkflowRun, summary="Run a schedule now"
)
async def run_schedule(
    schedule_id: str,
    service: WorkflowService = Depends(get_workflow_service),
) -> WorkflowRun:
    """Fire a schedule's workflow immediately (manual trigger)."""
    spec = service.get_schedule(schedule_id)
    if spec is None:
        raise HTTPException(
            status_code=404, detail=f"Schedule '{schedule_id}' not found"
        )
    return await service.execute(
        spec.workflow_id,
        dict(spec.payload),
        version=spec.workflow_version or None,
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _approval_request(service: WorkflowService, run_id: str, step_id: str):
    """Return the pending approval request for *step_id*, or raise 404."""
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    request = next(
        (approval for approval in run.approval_requests if approval.step_id == step_id),
        None,
    )
    if request is None:
        raise HTTPException(
            status_code=404,
            detail=f"No approval request for step '{step_id}' in run '{run_id}'",
        )
    return run, request


async def _stream_run_events(
    service: WorkflowService, run_id: str
) -> AsyncGenerator[str, None]:
    """Yield SSE frames for a run's event timeline until it terminates."""
    seen = 0
    while True:
        run = service.get_run(run_id)
        if run is None:
            yield sse_format("workflow.error", {"detail": f"Run '{run_id}' not found"})
            return
        events = run.events
        for event in events[seen:]:
            seen += 1
            yield sse_format("workflow.event", event.model_dump(mode="json"))
        if run.is_terminal:
            yield sse_format("workflow.run.completed", run.model_dump(mode="json"))
            return
        await asyncio.sleep(0.05)


def _to_mermaid(graph: dict[str, Any]) -> str:
    """Render the exported graph as a Mermaid flowchart."""
    lines = ["flowchart TD"]
    for node in graph["nodes"]:
        label = node["name"] or node["id"]
        lines.append(f'    {node["id"]}["{label}"]')
    for edge in graph["edges"]:
        lines.append(f"    {edge['source']} --> {edge['target']}")
    return "\n".join(lines) + "\n"


def _to_dot(graph: dict[str, Any]) -> str:
    """Render the exported graph as Graphviz DOT."""
    lines = ["digraph workflow {"]
    for node in graph["nodes"]:
        label = node["name"] or node["id"]
        lines.append(f'    "{node["id"]}" [label="{label}"];')
    for edge in graph["edges"]:
        lines.append(f'    "{edge["source"]}" -> "{edge["target"]}";')
    lines.append("}")
    return "\n".join(lines) + "\n"
