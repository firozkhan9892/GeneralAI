"""JSON-backed persistence implementations for automation stores.

Each store persists a single JSON document on disk using Pydantic v2
``model_dump(mode="json")`` for serialisation so datetimes, enums and
nested models round-trip losslessly.  Writes are atomic (temp file +
rename) and guarded by a lock so the stores are safe to share across
threads.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from app.automation.models import (
    RESUMABLE_RUN_STATUSES,
    ScheduleSpec,
    WorkflowDefinition,
    WorkflowEvent,
    WorkflowRun,
    WorkflowStatus,
)


def _dump(model: Any) -> dict[str, Any]:
    """JSON-safe dictionary for *model* (Pydantic ``mode="json"``)."""
    return model.model_dump(mode="json")


def _load(model_type: type[Any], data: dict[str, Any]) -> Any:
    """Rebuild a Pydantic model from JSON-safe *data*."""
    return model_type.model_validate(data)


class JsonDocumentStore:
    """Base class for a single-file JSON document store."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Low-level document I/O
    # ------------------------------------------------------------------

    def _read_document(self) -> dict[str, Any]:
        with self._lock:
            if not self._path.exists():
                return {}
            try:
                with self._path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
            except (json.JSONDecodeError, OSError):
                return {}

    def _write_document(self, document: dict[str, Any]) -> None:
        with self._lock:
            tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=2, default=_json_default)
            os.replace(tmp_path, self._path)


def _json_default(value: Any) -> Any:
    """Fallback serializer for values Pydantic could not encode."""
    raise TypeError(f"Cannot serialise {type(value).__name__}")


class JsonWorkflowStore(JsonDocumentStore):
    """JSON-backed :class:`WorkflowStore`.

    Document layout: ``{"<workflow_id>": {"<version>": {definition}}}``.
    """

    def save_definition(self, definition: WorkflowDefinition) -> None:
        with self._lock:
            document = self._read_document()
            versions = document.setdefault(definition.id, {})
            versions[definition.version] = _dump(definition)
            self._write_document(document)

    def get_definition(
        self, workflow_id: str, version: str | None = None
    ) -> WorkflowDefinition | None:
        with self._lock:
            document = self._read_document()
            versions = document.get(workflow_id)
            if not versions:
                return None
            if version is not None:
                raw = versions.get(version)
                return _load(WorkflowDefinition, raw) if raw is not None else None
            for raw in versions.values():
                if raw.get("status") == WorkflowStatus.PUBLISHED.value:
                    return _load(WorkflowDefinition, raw)
            if versions:
                latest = max(versions.keys(), key=_version_key)
                return _load(WorkflowDefinition, versions[latest])
            return None

    def list_definitions(
        self, status: WorkflowStatus | None = None
    ) -> list[WorkflowDefinition]:
        with self._lock:
            document = self._read_document()
            results: list[WorkflowDefinition] = []
            for versions in document.values():
                for raw in versions.values():
                    if status is None or raw.get("status") == status.value:
                        results.append(_load(WorkflowDefinition, raw))
            return results

    def delete_definition(self, workflow_id: str, version: str | None = None) -> bool:
        with self._lock:
            document = self._read_document()
            versions = document.get(workflow_id)
            if not versions:
                return False
            if version is None:
                del document[workflow_id]
            elif version in versions:
                del versions[version]
            else:
                return False
            self._write_document(document)
            return True

    def has(self, workflow_id: str, version: str | None = None) -> bool:
        with self._lock:
            document = self._read_document()
            versions = document.get(workflow_id)
            if not versions:
                return False
            if version is None:
                return True
            return version in versions


class JsonWorkflowRunStore(JsonDocumentStore):
    """JSON-backed :class:`WorkflowRunStore`.

    Document layout: ``{"<run_id>": {run}}``.
    """

    def save_run(self, run: WorkflowRun) -> None:
        with self._lock:
            document = self._read_document()
            document[run.run_id] = _dump(run)
            self._write_document(document)

    def get_run(self, run_id: str) -> WorkflowRun | None:
        with self._lock:
            raw = self._read_document().get(run_id)
            return _load(WorkflowRun, raw) if raw is not None else None

    def list_runs(
        self,
        workflow_id: str | None = None,
        status: str | None = None,
    ) -> list[WorkflowRun]:
        with self._lock:
            document = self._read_document()
            results = [
                _load(WorkflowRun, raw)
                for raw in document.values()
                if (workflow_id is None or raw.get("workflow_id") == workflow_id)
                and (status is None or raw.get("status") == status)
            ]
            return sorted(results, key=lambda run: run.created_at, reverse=True)

    def delete_run(self, run_id: str) -> bool:
        with self._lock:
            document = self._read_document()
            if run_id not in document:
                return False
            del document[run_id]
            self._write_document(document)
            return True

    def find_by_idempotency(
        self, workflow_id: str, workflow_version: str, key: str
    ) -> WorkflowRun | None:
        with self._lock:
            document = self._read_document()
            matches = [
                _load(WorkflowRun, raw)
                for raw in document.values()
                if raw.get("idempotency_key") == key
                and raw.get("workflow_id") == workflow_id
                and raw.get("workflow_version") == workflow_version
            ]
            if not matches:
                return None
            return max(matches, key=lambda run: (run.created_at, run.run_id))

    def list_resumable(self) -> list[WorkflowRun]:
        with self._lock:
            document = self._read_document()
            resumable_values = {s.value for s in RESUMABLE_RUN_STATUSES}
            return [
                _load(WorkflowRun, raw)
                for raw in document.values()
                if raw.get("status") in resumable_values
            ]


class JsonScheduleStore(JsonDocumentStore):
    """JSON-backed :class:`ScheduleStore`.

    Document layout: ``{"<schedule_id>": {schedule}}``.
    """

    def save_schedule(self, schedule: ScheduleSpec) -> None:
        with self._lock:
            document = self._read_document()
            document[schedule.schedule_id] = _dump(schedule)
            self._write_document(document)

    def get_schedule(self, schedule_id: str) -> ScheduleSpec | None:
        with self._lock:
            raw = self._read_document().get(schedule_id)
            return _load(ScheduleSpec, raw) if raw is not None else None

    def list_schedules(self, enabled: bool | None = None) -> list[ScheduleSpec]:
        with self._lock:
            document = self._read_document()
            return [
                _load(ScheduleSpec, raw)
                for raw in document.values()
                if enabled is None or raw.get("enabled") == enabled
            ]

    def delete_schedule(self, schedule_id: str) -> bool:
        with self._lock:
            document = self._read_document()
            if schedule_id not in document:
                return False
            del document[schedule_id]
            self._write_document(document)
            return True


class JsonEventStore(JsonDocumentStore):
    """JSON-backed :class:`EventStore`.

    Document layout: ``{"<run_id>": [{event}, ...]}``.
    """

    def append_event(self, event: WorkflowEvent) -> None:
        with self._lock:
            document = self._read_document()
            document.setdefault(event.run_id, []).append(_dump(event))
            self._write_document(document)

    def list_events(
        self,
        run_id: str | None = None,
        event_type: str | None = None,
    ) -> list[WorkflowEvent]:
        with self._lock:
            document = self._read_document()
            results: list[WorkflowEvent] = []
            for run_key, events in document.items():
                if run_id is not None and run_key != run_id:
                    continue
                for raw in events:
                    if event_type is None or raw.get("event_type") == event_type:
                        results.append(_load(WorkflowEvent, raw))
            return results

    def events_for_run(self, run_id: str) -> list[WorkflowEvent]:
        with self._lock:
            document = self._read_document()
            return [_load(WorkflowEvent, raw) for raw in document.get(run_id, [])]

    def delete_events(self, run_id: str) -> bool:
        with self._lock:
            document = self._read_document()
            if run_id not in document:
                return False
            del document[run_id]
            self._write_document(document)
            return True


def _version_key(version: str) -> tuple[int, ...]:
    """Convert a semantic version string to a sortable key."""
    parts = []
    for part in version.split("."):
        parts.append(int(part) if part.isdigit() else 0)
    return tuple(parts)
