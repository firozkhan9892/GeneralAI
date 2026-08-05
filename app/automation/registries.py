"""Thread-safe registries for workflows, runs, schedules and step types."""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Awaitable, Protocol

from app.automation.exceptions import (
    WorkflowNotFoundError,
    WorkflowVersionError,
)
from app.automation.models import (
    ScheduleSpec,
    WorkflowDefinition,
    WorkflowRun,
    WorkflowStatus,
    WorkflowStep,
    WorkflowStepType,
)


class WorkflowRegistry:
    """Registry of workflow definitions keyed by (workflow_id, version).

    Enforces version immutability: a published version can never be
    replaced or removed.  Editing a workflow always produces a new draft
    version.
    """

    def __init__(self) -> None:
        self._versions: dict[str, dict[str, WorkflowDefinition]] = defaultdict(dict)
        self._published: dict[str, WorkflowDefinition] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, definition: WorkflowDefinition) -> None:
        """Register a workflow definition.

        Raises:
            WorkflowVersionError: If a published version with the same
                id/version already exists (published versions are
                immutable).
        """
        with self._lock:
            existing = self._versions[definition.id].get(definition.version)
            if existing is not None and existing.status == WorkflowStatus.PUBLISHED:
                raise WorkflowVersionError(
                    f"Published workflow '{definition.id}' version "
                    f"'{definition.version}' is immutable",
                    workflow_id=definition.id,
                    version=definition.version,
                )
            self._versions[definition.id][definition.version] = definition

    def unregister(self, workflow_id: str, version: str | None = None) -> bool:
        """Remove a definition.

        Raises:
            WorkflowVersionError: If the definition is published.
        """
        with self._lock:
            if version is None:
                removed = len(self._versions.pop(workflow_id, {}))
                self._published.pop(workflow_id, None)
                return removed > 0
            entry = self._versions.get(workflow_id, {}).get(version)
            if entry is None:
                return False
            if entry.status == WorkflowStatus.PUBLISHED:
                raise WorkflowVersionError(
                    f"Published workflow '{workflow_id}' version '{version}' "
                    "cannot be removed",
                    workflow_id=workflow_id,
                    version=version,
                )
            del self._versions[workflow_id][version]
            return True

    def publish(self, workflow_id: str, version: str) -> WorkflowDefinition:
        """Mark a validated definition as published.

        Raises:
            WorkflowNotFoundError: If the definition does not exist.
        """
        with self._lock:
            definition = self._versions.get(workflow_id, {}).get(version)
            if definition is None:
                raise WorkflowNotFoundError(workflow_id, version=version)
            published = definition.with_status(WorkflowStatus.PUBLISHED)
            self._versions[workflow_id][version] = published
            self._published[workflow_id] = published
            return published

    def deprecate(self, workflow_id: str, version: str) -> WorkflowDefinition:
        """Mark a published definition as deprecated."""
        with self._lock:
            definition = self.get(workflow_id, version)
            if definition is None:
                raise WorkflowNotFoundError(workflow_id, version=version)
            if definition.status != WorkflowStatus.PUBLISHED:
                raise WorkflowVersionError(
                    f"Only published workflows can be deprecated; "
                    f"'{workflow_id}' version '{version}' is "
                    f"'{definition.status.value}'",
                    workflow_id=workflow_id,
                    version=version,
                )
            deprecated = definition.with_status(WorkflowStatus.DEPRECATED)
            self._versions[workflow_id][version] = deprecated
            return deprecated

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(
        self, workflow_id: str, version: str | None = None
    ) -> WorkflowDefinition | None:
        """Return a definition by id (and optional version).

        When *version* is ``None``, the highest published version is
        returned, falling back to the highest draft version.
        """
        with self._lock:
            versions = self._versions.get(workflow_id)
            if not versions:
                return None
            if version is not None:
                return versions.get(version)
            published = self._published.get(workflow_id)
            if published is not None:
                return published
            return versions[max(versions, key=_version_key)]

    def get_published(self, workflow_id: str) -> WorkflowDefinition | None:
        """Return the currently published definition, or ``None``."""
        with self._lock:
            return self._published.get(workflow_id)

    def list_versions(self, workflow_id: str) -> list[str]:
        """Return all version strings for a workflow, newest first."""
        with self._lock:
            return sorted(
                self._versions.get(workflow_id, {}), key=_version_key, reverse=True
            )

    def list_all(
        self, status: WorkflowStatus | None = None
    ) -> list[WorkflowDefinition]:
        """Return all registered definitions, optionally filtered by status."""
        with self._lock:
            results: list[WorkflowDefinition] = []
            for versions in self._versions.values():
                for definition in versions.values():
                    if status is None or definition.status == status:
                        results.append(definition)
            return results

    def has(self, workflow_id: str, version: str | None = None) -> bool:
        """Return ``True`` if a definition exists."""
        with self._lock:
            if version is not None:
                return (
                    workflow_id in self._versions
                    and version in self._versions[workflow_id]
                )
            return workflow_id in self._versions

    def __len__(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._versions.values())


class WorkflowRunRegistry:
    """Registry of workflow runs.

    Runs are immutable; registry methods only store and retrieve them.
    """

    def __init__(self) -> None:
        self._runs: dict[str, WorkflowRun] = {}
        self._lock = threading.RLock()

    def save(self, run: WorkflowRun) -> None:
        """Store or update a run."""
        with self._lock:
            self._runs[run.run_id] = run

    def get(self, run_id: str) -> WorkflowRun | None:
        """Return a run by id, or ``None``."""
        with self._lock:
            return self._runs.get(run_id)

    def delete(self, run_id: str) -> bool:
        """Remove a run, returning whether it existed."""
        with self._lock:
            return self._runs.pop(run_id, None) is not None

    def list_all(
        self,
        workflow_id: str | None = None,
        status: str | None = None,
    ) -> list[WorkflowRun]:
        """Return runs, newest first, optionally filtered."""
        with self._lock:
            results = [
                run
                for run in self._runs.values()
                if (workflow_id is None or run.workflow_id == workflow_id)
                and (status is None or run.status.value == status)
            ]
            return sorted(results, key=lambda run: run.created_at, reverse=True)

    def find_by_idempotency(
        self, workflow_id: str, workflow_version: str, key: str
    ) -> WorkflowRun | None:
        """Return a run matching an idempotency key for reproducibility.

        Returns the most recent matching run so duplicate requests with
        the same key resolve to the existing run.
        """
        with self._lock:
            matches = [
                run
                for run in self._runs.values()
                if run.idempotency_key == key
                and run.workflow_id == workflow_id
                and run.workflow_version == workflow_version
            ]
            if not matches:
                return None
            return max(matches, key=lambda run: (run.created_at, run.run_id))

    def __len__(self) -> int:
        with self._lock:
            return len(self._runs)


class ScheduleRegistry:
    """Registry of workflow schedules."""

    def __init__(self) -> None:
        self._schedules: dict[str, ScheduleSpec] = {}
        self._lock = threading.RLock()

    def save(self, schedule: ScheduleSpec) -> None:
        """Store or update a schedule."""
        with self._lock:
            self._schedules[schedule.schedule_id] = schedule

    def get(self, schedule_id: str) -> ScheduleSpec | None:
        """Return a schedule by id, or ``None``."""
        with self._lock:
            return self._schedules.get(schedule_id)

    def delete(self, schedule_id: str) -> bool:
        """Remove a schedule, returning whether it existed."""
        with self._lock:
            return self._schedules.pop(schedule_id, None) is not None

    def list_all(self, enabled: bool | None = None) -> list[ScheduleSpec]:
        """Return schedules, optionally filtered by enabled state."""
        with self._lock:
            return [
                s
                for s in self._schedules.values()
                if enabled is None or s.enabled == enabled
            ]

    def __len__(self) -> int:
        with self._lock:
            return len(self._schedules)


# ----------------------------------------------------------------------
# Step type registry
# ----------------------------------------------------------------------


class StepExecutor(Protocol):
    """Contract for a step executor callable.

    Executors receive the step, its isolated context, the engine, the
    shared engine state, the run, the snapshot, and the shared run
    context.  They return the step's output (JSON-safe).
    """

    async def __call__(
        self,
        step: WorkflowStep,
        step_ctx: Any,
        engine: Any,
        state: Any,
        run: Any,
        snapshot: Any,
        shared: Any,
    ) -> Awaitable[Any]:
        """Execute *step* and return its output."""
        ...


class StepTypeRegistry:
    """Registry mapping step types to executor callables.

    Plugin-defined executors register here without modifying the engine.
    The engine dispatches purely through this registry, so it never
    contains plugin-specific logic.
    """

    def __init__(self) -> None:
        self._executors: dict[str, StepExecutor] = {}
        self._lock = threading.RLock()

    def register(
        self, step_type: WorkflowStepType | str, executor: StepExecutor
    ) -> None:
        """Register an executor for *step_type*.

        Raises:
            ValueError: If *step_type* is already registered.
        """
        key = step_type.value if isinstance(step_type, WorkflowStepType) else step_type
        with self._lock:
            if key in self._executors:
                raise ValueError(f"Step type '{key}' is already registered")
            self._executors[key] = executor

    def unregister(self, step_type: WorkflowStepType | str) -> None:
        """Remove an executor registration."""
        key = step_type.value if isinstance(step_type, WorkflowStepType) else step_type
        with self._lock:
            self._executors.pop(key, None)

    def get(self, step_type: WorkflowStepType | str) -> StepExecutor | None:
        """Return the executor for *step_type*, or ``None``."""
        key = step_type.value if isinstance(step_type, WorkflowStepType) else step_type
        with self._lock:
            return self._executors.get(key)

    def has(self, step_type: WorkflowStepType | str) -> bool:
        """Return ``True`` if *step_type* has a registered executor."""
        key = step_type.value if isinstance(step_type, WorkflowStepType) else step_type
        with self._lock:
            return key in self._executors

    def list(self) -> list[str]:
        """Return all registered step type keys."""
        with self._lock:
            return sorted(self._executors.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._executors)


def _version_key(version: str) -> tuple[int, ...]:
    """Convert a semantic version string to a sortable key."""
    parts = []
    for part in version.split("."):
        if part.isdigit():
            parts.append(int(part))
        else:
            parts.append(0)
    return tuple(parts)
