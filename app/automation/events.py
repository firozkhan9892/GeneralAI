"""Workflow event type names.

These constants identify events persisted with a :class:`WorkflowRun`
and published on the application :class:`EventBus`.  Prefixes use the
``workflow.*`` namespace reserved for automation.
"""

from __future__ import annotations

from typing import Final

EVENT_WORKFLOW_RUN_STARTED: Final[str] = "workflow.run.started"
EVENT_WORKFLOW_RUN_COMPLETED: Final[str] = "workflow.run.completed"
EVENT_WORKFLOW_RUN_FAILED: Final[str] = "workflow.run.failed"
EVENT_WORKFLOW_RUN_CANCELLED: Final[str] = "workflow.run.cancelled"
EVENT_WORKFLOW_RUN_TIMED_OUT: Final[str] = "workflow.run.timed_out"
EVENT_WORKFLOW_RUN_PAUSED: Final[str] = "workflow.run.paused"
EVENT_WORKFLOW_RUN_RESUMED: Final[str] = "workflow.run.resumed"

EVENT_WORKFLOW_STEP_STARTED: Final[str] = "workflow.step.started"
EVENT_WORKFLOW_STEP_COMPLETED: Final[str] = "workflow.step.completed"
EVENT_WORKFLOW_STEP_FAILED: Final[str] = "workflow.step.failed"
EVENT_WORKFLOW_STEP_SKIPPED: Final[str] = "workflow.step.skipped"
EVENT_WORKFLOW_STEP_RETRYING: Final[str] = "workflow.step.retrying"
EVENT_WORKFLOW_STEP_TIMED_OUT: Final[str] = "workflow.step.timed_out"

EVENT_WORKFLOW_APPROVAL_REQUESTED: Final[str] = "workflow.approval.requested"
EVENT_WORKFLOW_APPROVAL_DECIDED: Final[str] = "workflow.approval.decided"

EVENT_WORKFLOW_DEFINED: Final[str] = "workflow.defined"
EVENT_WORKFLOW_PUBLISHED: Final[str] = "workflow.published"

EVENT_SCHEDULE_FIRED: Final[str] = "workflow.schedule.fired"
EVENT_SCHEDULE_COMPLETED: Final[str] = "workflow.schedule.completed"
EVENT_SCHEDULE_ERROR: Final[str] = "workflow.schedule.error"
