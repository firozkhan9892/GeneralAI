"""Persistence store abstractions for the automation module.

Stores separate *where* workflow artefacts are kept from the in-memory
registries used by the service layer.  They define a minimal durable
interface (``WorkflowStore``, ``WorkflowRunStore``, ``ScheduleStore``,
``EventStore``) plus thread-safe in-memory implementations.  JSON-backed
implementations live in :mod:`app.automation.persistence`.
"""

from __future__ import annotations

import threading
from typing import Protocol

from app.automation.models import (
    RESUMABLE_RUN_STATUSES,
    ScheduleSpec,
    WorkflowDefinition,
    WorkflowEvent,
    WorkflowRun,
    WorkflowStatus,
)


class WorkflowStore(Protocol):
    """Durable storage for workflow definitions."""

    def save_definition(self, definition: WorkflowDefinition) -> None:
        """Store or update *definition*."""

    def get_definition(
        self, workflow_id: str, version: str | None = None
    ) -> WorkflowDefinition | None:
        """Return a definition by id (and optional version).

        With *version* omitted the most relevant version is returned
        (published first, then the newest draft).
        """

    def list_definitions(
        self, status: WorkflowStatus | None = None
    ) -> list[WorkflowDefinition]:
        """Return all definitions, optionally filtered by *status*."""

    def delete_definition(self, workflow_id: str, version: str | None = None) -> bool:
        """Remove a definition, returning whether it existed."""

    def has(self, workflow_id: str, version: str | None = None) -> bool:
        """Return ``True`` if a definition exists."""


class WorkflowRunStore(Protocol):
    """Durable storage for workflow runs."""

    def save_run(self, run: WorkflowRun) -> None:
        """Store or update *run* (preserving its idempotency key)."""

    def get_run(self, run_id: str) -> WorkflowRun | None:
        """Return a run by id, or ``None``."""

    def list_runs(
        self,
        workflow_id: str | None = None,
        status: str | None = None,
    ) -> list[WorkflowRun]:
        """Return runs, newest first, optionally filtered."""

    def delete_run(self, run_id: str) -> bool:
        """Remove a run, returning whether it existed."""

    def find_by_idempotency(
        self, workflow_id: str, workflow_version: str, key: str
    ) -> WorkflowRun | None:
        """Return the run recorded against an idempotency key, or ``None``."""

    def list_resumable(self) -> list[WorkflowRun]:
        """Return runs that can be resumed after a restart.

        Includes waiting-approval, delay/paused, and other non-terminal
        states — never completed workflows.
        """


class ScheduleStore(Protocol):
    """Durable storage for workflow schedules."""

    def save_schedule(self, schedule: ScheduleSpec) -> None:
        """Store or update *schedule*."""

    def get_schedule(self, schedule_id: str) -> ScheduleSpec | None:
        """Return a schedule by id, or ``None``."""

    def list_schedules(self, enabled: bool | None = None) -> list[ScheduleSpec]:
        """Return schedules, optionally filtered by enabled state."""

    def delete_schedule(self, schedule_id: str) -> bool:
        """Remove a schedule, returning whether it existed."""


class EventStore(Protocol):
    """Durable storage for workflow events (replay / timelines)."""

    def append_event(self, event: WorkflowEvent) -> None:
        """Append *event* to its run's timeline."""

    def list_events(
        self,
        run_id: str | None = None,
        event_type: str | None = None,
    ) -> list[WorkflowEvent]:
        """Return events, optionally filtered by run id and event type."""

    def events_for_run(self, run_id: str) -> list[WorkflowEvent]:
        """Return all events recorded for *run_id* in order."""

    def delete_events(self, run_id: str) -> bool:
        """Remove all events for *run_id*, returning whether any existed."""


# ----------------------------------------------------------------------
# In-memory implementations
# ----------------------------------------------------------------------


class InMemoryWorkflowStore:
    """Thread-safe in-memory :class:`WorkflowStore`."""

    def __init__(self) -> None:
        self._definitions: dict[str, dict[str, WorkflowDefinition]] = {}
        self._lock = threading.RLock()

    def save_definition(self, definition: WorkflowDefinition) -> None:
        with self._lock:
            self._definitions.setdefault(definition.id, {})[definition.version] = (
                definition
            )

    def get_definition(
        self, workflow_id: str, version: str | None = None
    ) -> WorkflowDefinition | None:
        with self._lock:
            versions = self._definitions.get(workflow_id)
            if not versions:
                return None
            if version is not None:
                return versions.get(version)
            for candidate in versions.values():
                if candidate.status == WorkflowStatus.PUBLISHED:
                    return candidate
            return versions[max(versions, key=_version_key)]

    def list_definitions(
        self, status: WorkflowStatus | None = None
    ) -> list[WorkflowDefinition]:
        with self._lock:
            return [
                definition
                for versions in self._definitions.values()
                for definition in versions.values()
                if status is None or definition.status == status
            ]

    def delete_definition(self, workflow_id: str, version: str | None = None) -> bool:
        with self._lock:
            if version is None:
                return self._definitions.pop(workflow_id, None) is not None
            versions = self._definitions.get(workflow_id)
            if versions is None or version not in versions:
                return False
            del versions[version]
            return True

    def has(self, workflow_id: str, version: str | None = None) -> bool:
        with self._lock:
            if version is not None:
                return (
                    workflow_id in self._definitions
                    and version in self._definitions[workflow_id]
                )
            return workflow_id in self._definitions

    def __len__(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._definitions.values())


class InMemoryWorkflowRunStore:
    """Thread-safe in-memory :class:`WorkflowRunStore`."""

    def __init__(self) -> None:
        self._runs: dict[str, WorkflowRun] = {}
        self._lock = threading.RLock()

    def save_run(self, run: WorkflowRun) -> None:
        with self._lock:
            self._runs[run.run_id] = run

    def get_run(self, run_id: str) -> WorkflowRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(
        self,
        workflow_id: str | None = None,
        status: str | None = None,
    ) -> list[WorkflowRun]:
        with self._lock:
            results = [
                run
                for run in self._runs.values()
                if (workflow_id is None or run.workflow_id == workflow_id)
                and (status is None or run.status.value == status)
            ]
            return sorted(results, key=lambda run: run.created_at, reverse=True)

    def delete_run(self, run_id: str) -> bool:
        with self._lock:
            return self._runs.pop(run_id, None) is not None

    def find_by_idempotency(
        self, workflow_id: str, workflow_version: str, key: str
    ) -> WorkflowRun | None:
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

    def list_resumable(self) -> list[WorkflowRun]:
        with self._lock:
            return [
                run
                for run in self._runs.values()
                if run.status in RESUMABLE_RUN_STATUSES
            ]

    def __len__(self) -> int:
        with self._lock:
            return len(self._runs)


class InMemoryScheduleStore:
    """Thread-safe in-memory :class:`ScheduleStore`."""

    def __init__(self) -> None:
        self._schedules: dict[str, ScheduleSpec] = {}
        self._lock = threading.RLock()

    def save_schedule(self, schedule: ScheduleSpec) -> None:
        with self._lock:
            self._schedules[schedule.schedule_id] = schedule

    def get_schedule(self, schedule_id: str) -> ScheduleSpec | None:
        with self._lock:
            return self._schedules.get(schedule_id)

    def list_schedules(self, enabled: bool | None = None) -> list[ScheduleSpec]:
        with self._lock:
            return [
                schedule
                for schedule in self._schedules.values()
                if enabled is None or schedule.enabled == enabled
            ]

    def delete_schedule(self, schedule_id: str) -> bool:
        with self._lock:
            return self._schedules.pop(schedule_id, None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._schedules)


class InMemoryEventStore:
    """Thread-safe in-memory :class:`EventStore`."""

    def __init__(self) -> None:
        self._events: list[WorkflowEvent] = []
        self._lock = threading.RLock()

    def append_event(self, event: WorkflowEvent) -> None:
        with self._lock:
            self._events.append(event)

    def list_events(
        self,
        run_id: str | None = None,
        event_type: str | None = None,
    ) -> list[WorkflowEvent]:
        with self._lock:
            return [
                event
                for event in self._events
                if (run_id is None or event.run_id == run_id)
                and (event_type is None or event.event_type == event_type)
            ]

    def events_for_run(self, run_id: str) -> list[WorkflowEvent]:
        with self._lock:
            return [event for event in self._events if event.run_id == run_id]

    def delete_events(self, run_id: str) -> bool:
        with self._lock:
            before = len(self._events)
            self._events = [event for event in self._events if event.run_id != run_id]
            return len(self._events) != before

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


def _version_key(version: str) -> tuple[int, ...]:
    """Convert a semantic version string to a sortable key."""
    parts = []
    for part in version.split("."):
        parts.append(int(part) if part.isdigit() else 0)
    return tuple(parts)
