"""Unit tests for the asynchronous workflow scheduler."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from app.automation.exceptions import WorkflowSchedulerError
from app.automation.models import (
    ScheduleSpec,
    ScheduleTriggerType,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowSnapshot,
    WorkflowStep,
    WorkflowStepType,
)
from app.automation.scheduler import WorkflowScheduler, next_cron_match
from app.automation.stores import (
    InMemoryEventStore,
    InMemoryScheduleStore,
    InMemoryWorkflowRunStore,
)
from app.automation.time import FakeClock

UTC = timezone.utc


def _snapshot() -> WorkflowSnapshot:
    return WorkflowSnapshot(
        workflow_id="wf",
        version="1.0.0",
        step_definitions=(WorkflowStep(id="a", type=WorkflowStepType.TASK),),
    )


def _done_run(run_id: str, status: WorkflowRunStatus = WorkflowRunStatus.SUCCEEDED):
    return WorkflowRun(
        run_id=run_id,
        workflow_id="wf",
        workflow_version="1.0.0",
        status=status,
        snapshot=_snapshot(),
    )


# ----------------------------------------------------------------------
# Cron matching
# ----------------------------------------------------------------------


def test_next_cron_match_every_minute() -> None:
    after = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    assert next_cron_match("* * * * *", after) == datetime(
        2026, 1, 1, 12, 1, 0, tzinfo=UTC
    )


def test_next_cron_match_hour_and_minute() -> None:
    after = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    assert next_cron_match("30 9 * * *", after) == datetime(
        2026, 1, 2, 9, 30, 0, tzinfo=UTC
    )


def test_next_cron_match_dow() -> None:
    after = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)  # Thursday
    assert next_cron_match("0 0 * * 1", after) == datetime(
        2026, 1, 5, 0, 0, 0, tzinfo=UTC
    )  # Monday


def test_next_cron_match_step() -> None:
    after = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    assert next_cron_match("*/5 * * * *", after) == datetime(
        2026, 1, 1, 12, 5, 0, tzinfo=UTC
    )


def test_next_cron_match_invalid_raises() -> None:
    after = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    with pytest.raises(WorkflowSchedulerError):
        next_cron_match("not a cron", after)


# ----------------------------------------------------------------------
# Scheduler basics
# ----------------------------------------------------------------------


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


def _make_scheduler(fake_clock, run_store, events=None):
    schedule_store = InMemoryScheduleStore()
    event_store = events if events is not None else InMemoryEventStore()
    started: list[str] = []

    async def starter(spec: ScheduleSpec, key: str) -> WorkflowRun:
        started.append(spec.schedule_id)
        return _done_run(f"run-{len(started)}")

    scheduler = WorkflowScheduler(
        schedule_store=schedule_store,
        run_store=run_store,
        run_starter=starter,
        event_sink=event_store.append_event,
        clock=fake_clock,
        poll_interval_s=0.01,
    )
    scheduler.started_ids = started
    return scheduler


def test_interval_schedule_fires_on_time(fake_clock) -> None:
    run_store = InMemoryWorkflowRunStore()
    scheduler = _make_scheduler(fake_clock, run_store)
    scheduler.add_schedule(
        ScheduleSpec(
            schedule_id="s1",
            workflow_id="wf",
            trigger_type=ScheduleTriggerType.INTERVAL,
            interval_seconds=60.0,
        )
    )
    spec = scheduler.get_schedule("s1")
    assert spec is not None and spec.next_run_at is not None
    assert spec.next_run_at == fake_clock.utcnow() + timedelta(seconds=60)

    asyncio.run(scheduler.tick())
    assert scheduler.started_ids == []

    fake_clock.advance(60)
    asyncio.run(scheduler.tick())
    assert scheduler.started_ids == ["s1"]

    # Next occurrence advanced by the interval.
    spec = scheduler.get_schedule("s1")
    assert spec is not None and spec.next_run_at is not None
    assert spec.last_run_at is not None
    assert spec.last_status == "succeeded"


def test_interval_schedule_repeats(fake_clock) -> None:
    run_store = InMemoryWorkflowRunStore()
    scheduler = _make_scheduler(fake_clock, run_store)
    scheduler.add_schedule(
        ScheduleSpec(
            schedule_id="s1",
            workflow_id="wf",
            trigger_type=ScheduleTriggerType.INTERVAL,
            interval_seconds=60.0,
        )
    )

    asyncio.run(scheduler.tick())
    fake_clock.advance(60)
    asyncio.run(scheduler.tick())
    fake_clock.advance(60)
    asyncio.run(scheduler.tick())

    assert scheduler.started_ids == ["s1", "s1"]


def test_datetime_one_shot_fires_once_then_disables(fake_clock) -> None:
    run_store = InMemoryWorkflowRunStore()
    scheduler = _make_scheduler(fake_clock, run_store)
    run_at = fake_clock.utcnow() + timedelta(minutes=5)
    scheduler.add_schedule(
        ScheduleSpec(
            schedule_id="one",
            workflow_id="wf",
            trigger_type=ScheduleTriggerType.DATETIME,
            run_at=run_at,
        )
    )

    fake_clock.advance(300)
    asyncio.run(scheduler.tick())
    assert scheduler.started_ids == ["one"]

    spec = scheduler.get_schedule("one")
    assert spec is not None
    assert spec.enabled is False
    assert spec.next_run_at is None

    # Firing again does not happen.
    fake_clock.advance(300)
    asyncio.run(scheduler.tick())
    assert scheduler.started_ids == ["one"]


def test_cron_schedule_fires_at_match(fake_clock) -> None:
    run_store = InMemoryWorkflowRunStore()
    scheduler = _make_scheduler(fake_clock, run_store)
    scheduler.add_schedule(
        ScheduleSpec(
            schedule_id="c1",
            workflow_id="wf",
            trigger_type=ScheduleTriggerType.CRON,
            cron_expression="*/30 * * * *",
        )
    )
    spec = scheduler.get_schedule("c1")
    assert spec is not None
    assert spec.next_run_at == datetime(2026, 1, 1, 0, 30, 0, tzinfo=UTC)

    fake_clock.advance(30 * 60)
    asyncio.run(scheduler.tick())
    assert scheduler.started_ids == ["c1"]
    spec = scheduler.get_schedule("c1")
    assert spec is not None and spec.next_run_at == datetime(
        2026, 1, 1, 1, 0, 0, tzinfo=UTC
    )


def test_disabled_schedule_never_fires(fake_clock) -> None:
    run_store = InMemoryWorkflowRunStore()
    scheduler = _make_scheduler(fake_clock, run_store)
    scheduler.add_schedule(
        ScheduleSpec(
            schedule_id="s1",
            workflow_id="wf",
            trigger_type=ScheduleTriggerType.INTERVAL,
            interval_seconds=60.0,
            enabled=False,
        )
    )
    fake_clock.advance(600)
    asyncio.run(scheduler.tick())
    assert scheduler.started_ids == []


def test_max_concurrent_runs_per_schedule(fake_clock) -> None:
    run_store = InMemoryWorkflowRunStore()
    schedule_store = InMemoryScheduleStore()
    event_store = InMemoryEventStore()

    async def slow_starter(spec: ScheduleSpec, key: str) -> WorkflowRun:
        await asyncio.sleep(0.05)
        return _done_run(key)

    scheduler = WorkflowScheduler(
        schedule_store=schedule_store,
        run_store=run_store,
        run_starter=slow_starter,
        event_sink=event_store.append_event,
        clock=fake_clock,
        max_concurrent_runs=1,
        poll_interval_s=0.01,
    )
    scheduler.add_schedule(
        ScheduleSpec(
            schedule_id="s1",
            workflow_id="wf",
            trigger_type=ScheduleTriggerType.INTERVAL,
            interval_seconds=1.0,
            max_concurrent_runs=1,
        )
    )
    scheduler.add_schedule(
        ScheduleSpec(
            schedule_id="s2",
            workflow_id="wf",
            trigger_type=ScheduleTriggerType.INTERVAL,
            interval_seconds=1.0,
            max_concurrent_runs=1,
        )
    )

    async def scenario() -> None:
        fake_clock.advance(1)
        fired = await scheduler.tick()
        assert len(fired) == 1  # only one starts under the global cap
        assert sum(scheduler._active_runs.values()) == 1
        await scheduler.drain()
        fired = await scheduler.tick()
        assert len(fired) == 1  # the second starts once the first completes

    asyncio.run(scenario())


def test_graceful_shutdown_awaits_in_flight(fake_clock) -> None:
    run_store = InMemoryWorkflowRunStore()
    schedule_store = InMemoryScheduleStore()
    event_store = InMemoryEventStore()
    completed = asyncio.Event()

    async def slow_starter(spec: ScheduleSpec, key: str) -> WorkflowRun:
        await asyncio.sleep(0.05)
        completed.set()
        return _done_run(key)

    scheduler = WorkflowScheduler(
        schedule_store=schedule_store,
        run_store=run_store,
        run_starter=slow_starter,
        event_sink=event_store.append_event,
        clock=fake_clock,
        poll_interval_s=0.01,
    )
    scheduler.add_schedule(
        ScheduleSpec(
            schedule_id="s1",
            workflow_id="wf",
            trigger_type=ScheduleTriggerType.INTERVAL,
            interval_seconds=1.0,
        )
    )

    async def scenario() -> None:
        await scheduler.start()
        fake_clock.advance(1)
        await scheduler.tick()
        await scheduler.shutdown()

    asyncio.run(scenario())
    assert completed.is_set()


def test_run_loop_background_polling(fake_clock) -> None:
    run_store = InMemoryWorkflowRunStore()
    scheduler = _make_scheduler(fake_clock, run_store)
    scheduler.add_schedule(
        ScheduleSpec(
            schedule_id="s1",
            workflow_id="wf",
            trigger_type=ScheduleTriggerType.INTERVAL,
            interval_seconds=1.0,
        )
    )

    async def scenario() -> None:
        await scheduler.start()
        fake_clock.advance(1)
        await asyncio.sleep(0.05)
        await scheduler.shutdown()

    asyncio.run(scenario())
    assert scheduler.started_ids == ["s1"]


def test_restore_recomputes_timers_after_restart(fake_clock) -> None:
    run_store = InMemoryWorkflowRunStore()
    schedule_store = InMemoryScheduleStore()
    scheduler = _make_scheduler(fake_clock, run_store)
    scheduler._schedule_store = schedule_store

    # A schedule that was mid-flight before the restart.
    schedule_store.save_schedule(
        ScheduleSpec(
            schedule_id="s1",
            workflow_id="wf",
            trigger_type=ScheduleTriggerType.INTERVAL,
            interval_seconds=60.0,
            next_run_at=fake_clock.utcnow() + timedelta(seconds=10),
            last_run_at=fake_clock.utcnow() - timedelta(seconds=50),
        )
    )

    scheduler.restore()
    restored = scheduler.get_schedule("s1")
    assert restored is not None
    # last_run_at + interval = now+10s, which is after now.
    assert restored.next_run_at > fake_clock.utcnow()


def test_restore_does_not_rerun_completed_one_shot(fake_clock) -> None:
    run_store = InMemoryWorkflowRunStore()
    scheduler = _make_scheduler(fake_clock, run_store)
    scheduler.add_schedule(
        ScheduleSpec(
            schedule_id="one",
            workflow_id="wf",
            trigger_type=ScheduleTriggerType.DATETIME,
            run_at=fake_clock.utcnow() - timedelta(hours=1),
        )
    )
    # Mark it as already fired before the "restart".
    fired = scheduler.get_schedule("one")
    assert fired is not None
    scheduler.update_schedule(
        fired.model_copy(
            update={
                "last_run_at": fired.run_at,
                "last_status": "succeeded",
                "enabled": True,
            }
        )
    )

    scheduler.restore()
    restored = scheduler.get_schedule("one")
    assert restored is not None
    assert restored.enabled is False
    assert restored.next_run_at is None


def test_resume_pending_runs_restores_waiting_approval(fake_clock) -> None:
    run_store = InMemoryWorkflowRunStore()
    waiting = _done_run("waiting", WorkflowRunStatus.WAITING_APPROVAL)
    completed = _done_run("done", WorkflowRunStatus.SUCCEEDED)
    run_store.save_run(waiting)
    run_store.save_run(completed)

    scheduler = _make_scheduler(fake_clock, run_store)
    resumed: list[str] = []

    async def resume_cb(run: WorkflowRun) -> WorkflowRun:
        resumed.append(run.run_id)
        return run.model_copy(update={"status": WorkflowRunStatus.SUCCEEDED})

    result = asyncio.run(scheduler.resume_pending_runs(resume_cb))
    assert resumed == ["waiting"]
    assert result[0].status == WorkflowRunStatus.SUCCEEDED


def test_schedule_events_recorded(fake_clock) -> None:
    run_store = InMemoryWorkflowRunStore()
    event_store = InMemoryEventStore()
    scheduler = _make_scheduler(fake_clock, run_store, events=event_store)
    scheduler.add_schedule(
        ScheduleSpec(
            schedule_id="s1",
            workflow_id="wf",
            trigger_type=ScheduleTriggerType.INTERVAL,
            interval_seconds=60.0,
        )
    )

    async def scenario() -> None:
        fake_clock.advance(60)
        await scheduler.tick()
        await scheduler.drain()

    asyncio.run(scenario())

    fired = event_store.list_events(event_type="workflow.schedule.fired")
    completed = event_store.list_events(event_type="workflow.schedule.completed")
    assert len(fired) == 1
    assert len(completed) == 1
    assert fired[0].data["schedule_id"] == "s1"
