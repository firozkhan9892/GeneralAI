"""Exceptions for the workflow automation module.

Every exception derives from :class:`WorkflowError` which itself derives
from :class:`GeneralAIError` so that top-level handlers can catch and
report automation failures uniformly.
"""

from __future__ import annotations

from typing import Any

from app.core.exceptions.base import GeneralAIError

_MODULE = "automation"


class WorkflowError(GeneralAIError):
    """Base exception for all workflow automation errors."""

    def __init__(
        self,
        message: str = "",
        *,
        cause: Exception | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            module=_MODULE,
            cause=cause,
            context=context,
        )


class WorkflowNotFoundError(WorkflowError):
    """Raised when a workflow definition cannot be located."""

    def __init__(self, workflow_id: str, *, version: str | None = None) -> None:
        version_note = f" (version '{version}')" if version else ""
        super().__init__(
            f"Workflow '{workflow_id}'{version_note} not found",
            context={"workflow_id": workflow_id, "version": version},
        )


class WorkflowVersionError(WorkflowError):
    """Raised when a version cannot be modified or is unknown."""

    def __init__(
        self,
        message: str,
        *,
        workflow_id: str = "",
        version: str = "",
    ) -> None:
        super().__init__(
            message,
            context={
                "workflow_id": workflow_id,
                "version": version,
            },
        )


class WorkflowValidationError(WorkflowError):
    """Raised when a definition fails validation."""

    def __init__(self, message: str, *, violations: list[str] | None = None) -> None:
        super().__init__(message, context={"violations": violations or []})


class WorkflowExecutionError(WorkflowError):
    """Raised when a workflow run fails at the executor level."""


class WorkflowStepError(WorkflowError):
    """Raised when a single workflow step fails.

    Carries the failing step identifier so callers can attribute the
    error to the correct node in the dependency graph.
    """

    def __init__(
        self, message: str, *, step_id: str, cause: Exception | None = None
    ) -> None:
        super().__init__(
            message,
            cause=cause,
            context={"step_id": step_id},
        )
        self.step_id = step_id


class WorkflowApprovalError(WorkflowError):
    """Raised when an approval request cannot be processed."""


class ApprovalTimeoutError(WorkflowApprovalError):
    """Raised when an approval request times out."""


class WorkflowSchedulerError(WorkflowError):
    """Raised when a schedule is invalid or cannot run."""


class WorkflowConcurrencyError(WorkflowError):
    """Raised when a workflow exceeds its maximum concurrent runs."""


class WorkflowOutputConflictError(WorkflowError):
    """Raised when a step attempts to produce an output it already has."""

    def __init__(self, step_id: str) -> None:
        super().__init__(
            f"Step '{step_id}' already produced an output",
            context={"step_id": step_id},
        )


class WorkflowPersistenceError(WorkflowError):
    """Raised when a workflow store operation fails."""
