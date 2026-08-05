"""Asynchronous workflow scheduler.

The scheduler fires workflow runs on cron, interval or one-shot
schedules.  It is async-first and thread-safe, and injects a
:class:`Clock` so time is deterministic in tests — it **never** calls
``datetime.utcnow()`` directly.

State (schedules and their ``next_run_at``) is persisted through a
:class:`ScheduleStore`; after a restart :meth:`WorkflowScheduler.restore`
recomputes timers so completed workflows are never re-run.  Runs that
were waiting (approval / delay / paused) are exposed for resumption via
:meth:`WorkflowScheduler.resume_pending_runs`.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any

from app.automation.events import (
    EVENT_SCHEDULE_COMPLETED,
    EVENT_SCHEDULE_ERROR,
    EVENT_SCHEDULE_FIRED,
)
from app.automation.exceptions import WorkflowSchedulerError
from app.automation.models import (
    ScheduleSpec,
    ScheduleTriggerType,
    WorkflowEvent,
    WorkflowRun,
)
from app.automation.stores import ScheduleStore, WorkflowRunStore
from app.automation.time import Clock, SystemClock

log = logging.getLogger(__name__)


class WorkflowScheduler:
    """Fires due schedules by delegating to an injected run starter.

    The scheduler itself never executes workflows; it hands each due
    :class:`ScheduleSpec` (plus an idempotency key) to ``run_starter``
    and persists the resulting schedule progress.
    """

    def __init__(
        self,
        *,
        schedule_store: ScheduleStore,
        run_store: WorkflowRunStore,
        run_starter: Callable[[ScheduleSpec, str], Awaitable[WorkflowRun]],
        event_sink: Callable[[WorkflowEvent], None] | None = None,
        clock: Clock | None = None,
        max_concurrent_runs: int = 8,
        poll_interval_s: float = 1.0,
    ) -> None:
        self._schedule_store = schedule_store
        self._run_store = run_store
        self._run_starter = run_starter
        self._event_sink = event_sink
        self._clock: Clock = clock or SystemClock()
        self._max_concurrent_runs = max_concurrent_runs
        self._poll_interval_s = poll_interval_s

        self._lock = threading.RLock()
        self._active_runs: dict[str, int] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._loop_task: asyncio.Task[Any] | None = None
        self._stopping = asyncio.Event()

    # ------------------------------------------------------------------
    # Schedule management
    # ------------------------------------------------------------------

    def add_schedule(self, spec: ScheduleSpec) -> ScheduleSpec:
        """Store a schedule, computing ``next_run_at`` when absent."""
        with self._lock:
            if spec.next_run_at is None:
                spec = spec.model_copy(
                    update={"next_run_at": self._initial_next_run(spec)}
                )
            self._schedule_store.save_schedule(spec)
            return spec

    def update_schedule(self, spec: ScheduleSpec) -> ScheduleSpec:
        """Replace an existing schedule."""
        with self._lock:
            self._schedule_store.save_schedule(spec)
            return spec

    def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a schedule, returning whether it existed."""
        with self._lock:
            return self._schedule_store.delete_schedule(schedule_id)

    def get_schedule(self, schedule_id: str) -> ScheduleSpec | None:
        """Return a schedule by id."""
        with self._lock:
            return self._schedule_store.get_schedule(schedule_id)

    def list_schedules(self, enabled: bool | None = None) -> list[ScheduleSpec]:
        """Return stored schedules, optionally filtered by enabled state."""
        with self._lock:
            return self._schedule_store.list_schedules(enabled=enabled)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def restore(self) -> None:
        """Recompute schedule timers after a restart.

        One-shot schedules that already fired are disabled; interval and
        cron schedules advance their ``next_run_at`` from *now* so missed
        occurrences are not fired in a burst.
        """
        now = self._clock.utcnow()
        with self._lock:
            for spec in self._schedule_store.list_schedules():
                restored = self._restore_schedule(spec, now)
                if restored is not spec:
                    self._schedule_store.save_schedule(restored)

    def _restore_schedule(self, spec: ScheduleSpec, now: datetime) -> ScheduleSpec:
        if spec.trigger_type == ScheduleTriggerType.DATETIME:
            if spec.last_run_at is not None:
                return spec.model_copy(update={"enabled": False, "next_run_at": None})
            if spec.run_at is not None:
                return spec.model_copy(update={"next_run_at": spec.run_at})
            return spec
        if spec.trigger_type == ScheduleTriggerType.INTERVAL:
            if spec.last_run_at is None:
                return spec.model_copy(
                    update={
                        "next_run_at": now + timedelta(seconds=spec.interval_seconds)
                    }
                )
            candidate = spec.last_run_at + timedelta(seconds=spec.interval_seconds)
            return spec.model_copy(update={"next_run_at": max(candidate, now)})
        return spec.model_copy(
            update={"next_run_at": next_cron_match(spec.cron_expression, now)}
        )

    async def start(self) -> None:
        """Begin the background polling loop (call from the event loop)."""
        self.restore()
        with self._lock:
            if self._loop_task is not None:
                return
            self._stopping = asyncio.Event()
            self._loop_task = asyncio.create_task(self._run_loop())

    async def shutdown(self) -> None:
        """Stop the poll loop and await in-flight run tasks (graceful)."""
        with self._lock:
            self._stopping.set()
            loop_task = self._loop_task
            self._loop_task = None
        if loop_task is not None:
            loop_task.cancel()
            await asyncio.gather(loop_task, return_exceptions=True)
        await self.drain()

    async def drain(self) -> None:
        """Await all in-flight fire tasks to completion.

        Used by tests and graceful shutdown so scheduled runs that have
        already started finish before the caller proceeds.
        """
        tasks = list(self._tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_loop(self) -> None:
        try:
            while not self._stopping.is_set():
                await self.tick()
                await asyncio.sleep(self._poll_interval_s)
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def tick(self) -> list[ScheduleSpec]:
        """Fire every enabled schedule that is now due; return those fired."""
        now = self._clock.utcnow()
        with self._lock:
            schedules = self._schedule_store.list_schedules(enabled=True)
        due = [
            spec
            for spec in schedules
            if spec.next_run_at is not None and spec.next_run_at <= now
        ]
        due.sort(key=lambda spec: spec.next_run_at or now)
        fired: list[ScheduleSpec] = []
        for spec in due:
            if not self._can_start(spec):
                continue
            self._start_fire(spec, now)
            fired.append(spec)
        return fired

    def _can_start(self, spec: ScheduleSpec) -> bool:
        with self._lock:
            total_active = sum(self._active_runs.values())
            if total_active >= self._max_concurrent_runs:
                return False
            schedule_active = self._active_runs.get(spec.schedule_id, 0)
            return schedule_active < spec.max_concurrent_runs

    def _start_fire(self, spec: ScheduleSpec, scheduled_at: datetime) -> None:
        key = self._idempotency_key(spec, scheduled_at)
        with self._lock:
            self._active_runs[spec.schedule_id] = (
                self._active_runs.get(spec.schedule_id, 0) + 1
            )
        task = asyncio.create_task(self._fire(spec, scheduled_at, key))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _fire(
        self, spec: ScheduleSpec, scheduled_at: datetime, key: str
    ) -> WorkflowRun:
        """Run one schedule occurrence and persist progress."""
        try:
            existing = self._run_store.find_by_idempotency(
                spec.workflow_id, spec.workflow_version, key
            )
            if existing is not None and existing.is_terminal:
                self._emit(EVENT_SCHEDULE_COMPLETED, spec, existing.run_id, {})
                self._advance(spec, scheduled_at, existing.status.value)
                return existing

            self._emit(EVENT_SCHEDULE_FIRED, spec, "", {})
            log.debug(
                "Schedule fired",
                extra={
                    "schedule_id": spec.schedule_id,
                    "workflow_id": spec.workflow_id,
                },
            )
            run = await self._run_starter(spec, key)
            self._emit(EVENT_SCHEDULE_COMPLETED, spec, run.run_id, {})
            self._advance(spec, scheduled_at, run.status.value)
            return run
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Schedule run failed",
                extra={
                    "schedule_id": spec.schedule_id,
                    "workflow_id": spec.workflow_id,
                    "error": str(exc),
                },
            )
            self._emit(EVENT_SCHEDULE_ERROR, spec, "", {"error": str(exc)})
            self._advance(spec, scheduled_at, "error")
            raise
        finally:
            with self._lock:
                active = self._active_runs.get(spec.schedule_id, 0) - 1
                if active <= 0:
                    self._active_runs.pop(spec.schedule_id, None)
                else:
                    self._active_runs[spec.schedule_id] = active

    def _advance(
        self,
        spec: ScheduleSpec,
        fired_at: datetime,
        status: str,
    ) -> None:
        """Persist the schedule's progress after a fire."""
        if spec.trigger_type == ScheduleTriggerType.DATETIME:
            updated = spec.model_copy(
                update={
                    "enabled": False,
                    "last_run_at": fired_at,
                    "last_status": status,
                    "next_run_at": None,
                }
            )
        elif spec.trigger_type == ScheduleTriggerType.INTERVAL:
            updated = spec.model_copy(
                update={
                    "last_run_at": fired_at,
                    "last_status": status,
                    "next_run_at": fired_at + timedelta(seconds=spec.interval_seconds),
                }
            )
        else:
            updated = spec.model_copy(
                update={
                    "last_run_at": fired_at,
                    "last_status": status,
                    "next_run_at": next_cron_match(spec.cron_expression, fired_at),
                }
            )
        with self._lock:
            self._schedule_store.save_schedule(updated)

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    def pending_runs(self) -> list[WorkflowRun]:
        """Return runs that should be resumed after a restart."""
        return self._run_store.list_resumable()

    async def resume_pending_runs(
        self,
        resume_cb: Callable[[WorkflowRun], Awaitable[WorkflowRun]],
    ) -> list[WorkflowRun]:
        """Resume every run waiting after a restart.

        Passes each pending run (approval / delay / paused) to
        *resume_cb*, which the service layer wires to the executor.
        Completed runs are never re-run.
        """
        resumed: list[WorkflowRun] = []
        for run in self.pending_runs():
            resumed.append(await resume_cb(run))
        return resumed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _idempotency_key(self, spec: ScheduleSpec, scheduled_at: datetime) -> str:
        return f"schedule:{spec.schedule_id}:{scheduled_at.isoformat()}"

    def _initial_next_run(self, spec: ScheduleSpec) -> datetime:
        now = self._clock.utcnow()
        if spec.trigger_type == ScheduleTriggerType.DATETIME:
            if spec.run_at is None:
                raise WorkflowSchedulerError(
                    f"Schedule '{spec.schedule_id}' has no run_at for a "
                    "one-shot trigger"
                )
            return spec.run_at
        if spec.trigger_type == ScheduleTriggerType.INTERVAL:
            if spec.interval_seconds <= 0:
                raise WorkflowSchedulerError(
                    f"Schedule '{spec.schedule_id}' must have a positive "
                    "interval_seconds"
                )
            return now + timedelta(seconds=spec.interval_seconds)
        return next_cron_match(spec.cron_expression, now)

    def _emit(
        self,
        event_type: str,
        spec: ScheduleSpec,
        run_id: str,
        data: dict[str, Any],
    ) -> None:
        if self._event_sink is None:
            return
        payload = dict(data)
        payload.setdefault("schedule_id", spec.schedule_id)
        payload.setdefault("workflow_id", spec.workflow_id)
        self._event_sink(
            WorkflowEvent(
                event_type=event_type,
                run_id=run_id,
                timestamp=self._clock.utcnow(),
                data=payload,
            )
        )


# ----------------------------------------------------------------------
# Minimal 5-field cron support
# ----------------------------------------------------------------------


def next_cron_match(expression: str, after: datetime) -> datetime:
    """Return the next datetime matching a 5-field cron *expression*.

    Fields: ``minute hour day-of-month month day-of-week``.  Supports
    ``*``, lists, ranges and steps (``*/5``, ``1-5``, ``1-30/3``).
    Day-of-week uses 0/7 = Sunday … 6 = Saturday.  When both day fields
    are restricted a day matches when **either** field matches (Vixie
    cron semantics).

    Raises:
        WorkflowSchedulerError: If the expression is malformed or no
            match occurs within a 5-year horizon.
    """
    fields = expression.strip().split()
    if len(fields) != 5:
        raise WorkflowSchedulerError(
            f"Invalid cron expression '{expression}': expected 5 fields"
        )
    minutes = _parse_cron_field(fields[0], 0, 59)
    hours = _parse_cron_field(fields[1], 0, 23)
    days_of_month = _parse_cron_field(fields[2], 1, 31)
    months = _parse_cron_field(fields[3], 1, 12)
    days_of_week = {day % 7 for day in _parse_cron_field(fields[4], 0, 7)}

    dom_wildcard = fields[2].strip() == "*"
    dow_wildcard = fields[4].strip() == "*"

    start = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    horizon = 5 * 366
    for day_offset in range(horizon):
        day = start + timedelta(days=day_offset)
        if day.month not in months:
            continue
        dom_ok = day.day in days_of_month
        dow_ok = (day.weekday() + 1) % 7 in days_of_week
        if dom_wildcard and dow_wildcard:
            day_ok = True
        elif dom_wildcard:
            day_ok = dow_ok
        elif dow_wildcard:
            day_ok = dom_ok
        else:
            day_ok = dom_ok or dow_ok
        if not day_ok:
            continue
        for hour in sorted(hours):
            for minute in sorted(minutes):
                candidate = day.replace(hour=hour, minute=minute)
                if candidate > after:
                    return candidate
    raise WorkflowSchedulerError(
        f"No future occurrence found for cron '{expression}' within 5 years"
    )


def _parse_cron_field(field: str, low: int, high: int) -> set[int]:
    """Parse one cron field into the set of allowed values."""
    values: set[int] = set()
    for part in field.split(","):
        part = part.strip()
        if not part:
            raise WorkflowSchedulerError(f"Empty cron field component in '{field}'")
        if part == "*":
            values.update(range(low, high + 1))
            continue
        step = 1
        base = part
        if "/" in part:
            base, _, step_str = part.partition("/")
            try:
                step = int(step_str)
            except ValueError as exc:
                raise WorkflowSchedulerError(
                    f"Invalid cron step '{step_str}' in '{field}'"
                ) from exc
            if step <= 0:
                raise WorkflowSchedulerError(
                    f"Invalid cron step '{step_str}' in '{field}'"
                )
        if base == "*":
            values.update(range(low, high + 1, step))
        elif "-" in base:
            start_str, _, end_str = base.partition("-")
            try:
                start_value = int(start_str)
                end_value = int(end_str)
            except ValueError as exc:
                raise WorkflowSchedulerError(
                    f"Invalid cron range '{base}' in '{field}'"
                ) from exc
            if start_value > end_value:
                raise WorkflowSchedulerError(
                    f"Invalid cron range '{base}' in '{field}'"
                )
            values.update(range(start_value, end_value + 1, step))
        else:
            try:
                values.add(int(base))
            except ValueError as exc:
                raise WorkflowSchedulerError(
                    f"Invalid cron value '{base}' in '{field}'"
                ) from exc
    return {value for value in values if low <= value <= high}
