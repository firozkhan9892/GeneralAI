"""Tests for the agent manager (Phase 8).

Covers the session models, session stores (in-memory + JSON), the
session registry, the AgentManager's concurrent execution with
cancel/pause/resume, retries, timeouts, memory/tool integration,
persistence, and DI bootstrap.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from app.agents import (
    AgentExecutionOptions,
    AgentManager,
    AgentSession,
    InMemorySessionStore,
    JsonSessionStore,
    SessionRegistry,
    SessionStatus,
    SessionStore,
    register_agent_manager_components,
)
from app.agents.exceptions import (
    SessionAlreadyExistsError,
    SessionNotFoundError,
    SessionNotRunnableError,
)
from app.agents.models import TERMINAL_SESSION_STATUSES
from app.agents.persistence import json_safe
from app.core.container import DependencyContainer
from app.kernel.agent import (
    AgentRequest,
    AgentRunConfig,
    AgentRuntime,
)
from app.tools.mock import MockTool
from app.tools.registry import ToolRegistry

QUESTION_TOOLS = ("analyze_question", "retrieve_knowledge", "formulate_answer")


# ── Helpers ──────────────────────────────────────────────────────────


def _registry(
    names: tuple[str, ...] = QUESTION_TOOLS,
    *,
    delay_s: float = 0.0,
    fail: str | None = None,
    fail_first_n: int = 0,
) -> ToolRegistry:
    registry = ToolRegistry()
    for name in names:
        registry.register(
            MockTool(
                name=name,
                delay_s=delay_s,
                fail=fail,
                fail_first_n=fail_first_n,
            )
        )
    return registry


def _runtime(
    names: tuple[str, ...] = QUESTION_TOOLS,
    *,
    delay_s: float = 0.0,
    fail: str | None = None,
    fail_first_n: int = 0,
) -> AgentRuntime:
    return AgentRuntime(
        tool_registry=_registry(
            names, delay_s=delay_s, fail=fail, fail_first_n=fail_first_n
        )
    )


def _manager(
    *,
    runtime: AgentRuntime | None = None,
    store: SessionStore | None = None,
) -> AgentManager:
    registry = SessionRegistry(store=store) if store is not None else None
    return AgentManager(runtime=runtime or _runtime(), registry=registry)


def _session(**overrides: Any) -> AgentSession:
    fields: dict[str, Any] = {
        "session_id": "s-default",
        "request": AgentRequest(raw_input="Hello", session_id="s-default"),
    }
    fields.update(overrides)
    return AgentSession(**fields)


# ── Session models ───────────────────────────────────────────────────


class TestSessionModels:
    def test_status_values(self) -> None:
        assert SessionStatus.RUNNING.value == "running"
        assert SessionStatus.PAUSED.value == "paused"
        assert SessionStatus.COMPLETED.value == "completed"
        assert SessionStatus.TIMED_OUT.value == "timed_out"

    def test_terminal_statuses(self) -> None:
        assert SessionStatus.COMPLETED in TERMINAL_SESSION_STATUSES
        assert SessionStatus.FAILED in TERMINAL_SESSION_STATUSES
        assert SessionStatus.CANCELLED in TERMINAL_SESSION_STATUSES
        assert SessionStatus.TIMED_OUT in TERMINAL_SESSION_STATUSES
        assert SessionStatus.RUNNING not in TERMINAL_SESSION_STATUSES
        assert SessionStatus.PAUSED not in TERMINAL_SESSION_STATUSES

    def test_defaults(self) -> None:
        session = _session()
        assert session.status == SessionStatus.PENDING
        assert session.attempts == 0
        assert session.error is None
        assert session.response is None
        assert session.created_at is not None
        assert session.updated_at is not None
        assert session.is_running is False
        assert session.is_terminal is False

    def test_with_status_running(self) -> None:
        updated = _session().with_status(SessionStatus.RUNNING)
        assert updated.status == SessionStatus.RUNNING
        assert updated.is_running is True

    def test_with_status_completed_stamps_completed_at(self) -> None:
        session = _session()
        updated = session.with_status(SessionStatus.COMPLETED, completed=True)
        assert updated.completed_at is not None
        assert updated.is_terminal is True

    def test_with_status_paused_stamps_paused_at(self) -> None:
        updated = _session().with_status(SessionStatus.PAUSED)
        assert updated.paused_at is not None

    def test_with_error_and_response(self) -> None:
        from app.kernel.agent import AgentResponse

        response = AgentResponse(success=True, session_id="s-default")
        updated = _session().with_status(
            SessionStatus.FAILED, error="boom", response=response
        )
        assert updated.error == "boom"
        assert updated.response is response


class TestAgentExecutionOptions:
    def test_defaults(self) -> None:
        options = AgentExecutionOptions()
        assert options.max_attempts == 1
        assert options.retry_delay_s == 0.0
        assert options.persist is True

    def test_max_attempts_minimum(self) -> None:
        with pytest.raises(Exception):
            AgentExecutionOptions(max_attempts=0)


# ── Session stores ───────────────────────────────────────────────────


class TestInMemorySessionStore:
    @pytest.mark.asyncio
    async def test_save_load_roundtrip(self) -> None:
        store = InMemorySessionStore()
        session = _session(session_id="s1")
        await store.save(session)
        loaded = await store.load("s1")
        assert loaded is not None
        assert loaded.session_id == "s1"
        assert loaded.request.raw_input == "Hello"

    @pytest.mark.asyncio
    async def test_load_missing(self) -> None:
        store = InMemorySessionStore()
        assert await store.load("missing") is None

    @pytest.mark.asyncio
    async def test_list(self) -> None:
        store = InMemorySessionStore()
        await store.save(_session(session_id="a"))
        await store.save(_session(session_id="b"))
        ids = {s.session_id for s in await store.list()}
        assert ids == {"a", "b"}

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        store = InMemorySessionStore()
        await store.save(_session(session_id="a"))
        assert await store.delete("a") is True
        assert await store.delete("a") is False
        assert await store.load("a") is None


class TestJsonSessionStore:
    def _store(self, tmp_path: Path) -> JsonSessionStore:
        return JsonSessionStore(tmp_path)

    @pytest.mark.asyncio
    async def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        await store.save(_session(session_id="s1"))
        assert (tmp_path / "s1.json").exists()
        loaded = await store.load("s1")
        assert loaded is not None
        assert loaded.session_id == "s1"

    @pytest.mark.asyncio
    async def test_load_missing(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        assert await store.load("missing") is None

    @pytest.mark.asyncio
    async def test_directory_created(self, tmp_path: Path) -> None:
        nested = tmp_path / "nested" / "dir"
        store = JsonSessionStore(nested)
        await store.save(_session(session_id="s1"))
        assert nested.exists()

    @pytest.mark.asyncio
    async def test_list_skips_malformed(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        await store.save(_session(session_id="good"))
        (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
        sessions = await store.list()
        assert [s.session_id for s in sessions] == ["good"]

    @pytest.mark.asyncio
    async def test_delete(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        await store.save(_session(session_id="s1"))
        assert await store.delete("s1") is True
        assert await store.delete("s1") is False

    @pytest.mark.asyncio
    async def test_restore_across_instances(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        await store.save(_session(session_id="s1"))
        fresh = JsonSessionStore(tmp_path)
        sessions = await fresh.list()
        assert [s.session_id for s in sessions] == ["s1"]

    @pytest.mark.asyncio
    async def test_persists_completed_response(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        session = _session(session_id="s1")
        await store.save(session)
        raw = json.loads((tmp_path / "s1.json").read_text(encoding="utf-8"))
        assert raw["session_id"] == "s1"
        assert raw["status"] == "pending"


class TestJsonSafe:
    def test_scalars(self) -> None:
        assert json_safe("x") == "x"
        assert json_safe(1) == 1
        assert json_safe(None) is None

    def test_nested_structures(self) -> None:
        value = {"a": [1, {"b": True}], "c": (1, 2)}
        assert json_safe(value) == {"a": [1, {"b": True}], "c": [1, 2]}

    def test_arbitrary_object_stringified(self) -> None:
        class Obj:
            def __str__(self) -> str:
                return "obj!"

        assert json_safe({"x": Obj()}) == {"x": "obj!"}

    def test_pydantic_model_dumped(self) -> None:
        model = AgentExecutionOptions()
        assert json_safe({"opts": model})["opts"]["max_attempts"] == 1


# ── Session registry ─────────────────────────────────────────────────


class TestSessionRegistry:
    def test_register_get_has(self) -> None:
        registry = SessionRegistry()
        session = _session(session_id="s1")
        registry.register(session)
        assert registry.has("s1") is True
        assert registry.get("s1") is session
        assert registry.get("missing") is None

    def test_register_duplicate_raises(self) -> None:
        registry = SessionRegistry()
        registry.register(_session(session_id="s1"))
        with pytest.raises(SessionAlreadyExistsError):
            registry.register(_session(session_id="s1"))

    def test_get_or_raise(self) -> None:
        registry = SessionRegistry()
        with pytest.raises(SessionNotFoundError):
            registry.get_or_raise("missing")
        registry.register(_session(session_id="s1"))
        assert registry.get_or_raise("s1").session_id == "s1"

    def test_update_upserts(self) -> None:
        registry = SessionRegistry()
        registry.update(_session(session_id="s1"))
        assert registry.has("s1")
        updated = registry.get_or_raise("s1").with_status(SessionStatus.RUNNING)
        registry.update(updated)
        assert registry.get_or_raise("s1").status == SessionStatus.RUNNING

    def test_count_and_all_newest_first(self) -> None:
        registry = SessionRegistry()
        old = _session(session_id="a", created_at=datetime(2020, 1, 1))
        new = _session(session_id="b", created_at=datetime(2021, 1, 1))
        registry.register(old)
        registry.register(new)
        assert registry.count() == 2
        assert [s.session_id for s in registry.all()] == ["b", "a"]

    def test_list_filters_by_status(self) -> None:
        registry = SessionRegistry()
        running = _session(session_id="a").with_status(SessionStatus.RUNNING)
        done = _session(session_id="b").with_status(SessionStatus.COMPLETED)
        registry.register(running)
        registry.register(done)
        result = registry.list(status=SessionStatus.COMPLETED)
        assert [s.session_id for s in result] == ["b"]

    def test_list_pagination(self) -> None:
        registry = SessionRegistry()
        for i in range(5):
            registry.register(
                _session(session_id=f"s{i}", created_at=datetime(2020, 1, i + 1))
            )
        page = registry.list(limit=2, offset=1)
        assert [s.session_id for s in page] == ["s3", "s2"]

    def test_remove_and_clear(self) -> None:
        registry = SessionRegistry()
        registry.register(_session(session_id="a"))
        registry.register(_session(session_id="b"))
        assert registry.remove("a") is True
        assert registry.remove("a") is False
        registry.clear()
        assert registry.count() == 0

    @pytest.mark.asyncio
    async def test_persist_and_load_all(self) -> None:
        store = InMemorySessionStore()
        registry = SessionRegistry(store=store)
        registry.register(_session(session_id="a"))
        await registry.persist(registry.get_or_raise("a"))
        fresh = SessionRegistry(store=store)
        assert await fresh.load_all() == 1
        assert fresh.has("a")

    @pytest.mark.asyncio
    async def test_delete_removes_from_store(self) -> None:
        store = InMemorySessionStore()
        registry = SessionRegistry(store=store)
        registry.register(_session(session_id="a"))
        await registry.persist(registry.get_or_raise("a"))
        assert await registry.delete("a") is True
        assert await store.load("a") is None


# ── AgentManager: execution ──────────────────────────────────────────


class TestAgentManagerExecute:
    @pytest.mark.asyncio
    async def test_execute_completes_successfully(self) -> None:
        mgr = _manager()
        session = await mgr.execute(
            AgentRequest(raw_input="What is 2+2?", session_id="s1")
        )
        assert session.session_id == "s1"
        assert session.status == SessionStatus.RUNNING
        done = await mgr.await_completion("s1")
        assert done.status == SessionStatus.COMPLETED
        assert done.response is not None
        assert done.response.success is True
        assert done.error is None
        assert done.attempts == 1

    @pytest.mark.asyncio
    async def test_execute_generates_session_id(self) -> None:
        mgr = _manager()
        session = await mgr.execute(
            AgentRequest(raw_input="What is up?", session_id="")
        )
        assert session.session_id != ""
        done = await mgr.await_completion(session.session_id)
        assert done.status == SessionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_duplicate_session_raises(self) -> None:
        mgr = _manager()
        await mgr.execute(AgentRequest(raw_input="What is up?", session_id="dup"))
        with pytest.raises(SessionAlreadyExistsError):
            await mgr.execute(AgentRequest(raw_input="Hi again", session_id="dup"))

    @pytest.mark.asyncio
    async def test_execute_config_session_id_filled(self) -> None:
        mgr = _manager()
        session = await mgr.execute(
            AgentRequest(raw_input="What is up?", session_id="cfg"),
            config=AgentRunConfig(),
        )
        assert session.config is not None
        assert session.config.session_id == "cfg"
        done = await mgr.await_completion("cfg")
        assert done.status == SessionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_multiple_concurrent_sessions(self) -> None:
        mgr = _manager()
        created = [
            await mgr.execute(AgentRequest(raw_input="What is q?", session_id=f"c{i}"))
            for i in range(5)
        ]
        assert len(created) == 5
        results = await asyncio.gather(
            *(mgr.await_completion(s.session_id) for s in created)
        )
        assert all(r.status == SessionStatus.COMPLETED for r in results)
        assert mgr.active_count() == 0
        assert len(mgr.list_sessions()) == 5

    @pytest.mark.asyncio
    async def test_get_status_and_unknown(self) -> None:
        mgr = _manager()
        await mgr.execute(AgentRequest(raw_input="What is up?", session_id="st"))
        session = mgr.get_status("st")
        assert isinstance(session, AgentSession)
        with pytest.raises(SessionNotFoundError):
            mgr.get_status("missing")

    @pytest.mark.asyncio
    async def test_list_sessions_filter(self) -> None:
        mgr = _manager()
        await mgr.execute(AgentRequest(raw_input="What is up?", session_id="done"))
        await mgr.await_completion("done")
        await mgr.execute(AgentRequest(raw_input="What is up?", session_id="run"))
        done_sessions = mgr.list_sessions(status=SessionStatus.COMPLETED)
        assert [s.session_id for s in done_sessions] == ["done"]

    @pytest.mark.asyncio
    async def test_await_completion_timeout_keeps_run_alive(self) -> None:
        mgr = _manager(runtime=_runtime(delay_s=0.1))
        await mgr.execute(AgentRequest(raw_input="What is slow?", session_id="slow"))
        snapshot = await mgr.await_completion("slow", timeout_s=0.01)
        assert snapshot.status == SessionStatus.RUNNING
        done = await mgr.await_completion("slow")
        assert done.status == SessionStatus.COMPLETED


# ── AgentManager: lifecycle control ──────────────────────────────────


class TestAgentManagerLifecycle:
    @pytest.mark.asyncio
    async def test_cancel_running_session(self) -> None:
        mgr = _manager(runtime=_runtime(delay_s=0.1))
        await mgr.execute(AgentRequest(raw_input="What is up?", session_id="c1"))
        await asyncio.sleep(0.01)
        session = await mgr.cancel("c1", reason="test")
        assert session.status == SessionStatus.CANCELLED
        assert session.error == "test"

    @pytest.mark.asyncio
    async def test_cancel_paused_session(self) -> None:
        mgr = _manager(runtime=_runtime(delay_s=0.1))
        await mgr.execute(AgentRequest(raw_input="What is up?", session_id="c2"))
        await mgr.pause("c2")
        session = await mgr.cancel("c2", reason="test")
        assert session.status == SessionStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_completed_is_noop(self) -> None:
        mgr = _manager()
        await mgr.execute(AgentRequest(raw_input="What is up?", session_id="c3"))
        await mgr.await_completion("c3")
        session = await mgr.cancel("c3")
        assert session.status == SessionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_cancel_unknown_raises(self) -> None:
        mgr = _manager()
        with pytest.raises(SessionNotFoundError):
            await mgr.cancel("missing")

    @pytest.mark.asyncio
    async def test_pause_and_resume(self) -> None:
        mgr = _manager(runtime=_runtime(delay_s=0.1))
        await mgr.execute(AgentRequest(raw_input="What is up?", session_id="pr"))
        await asyncio.sleep(0.01)
        paused = await mgr.pause("pr")
        assert paused.status == SessionStatus.PAUSED

        resumed = mgr.resume("pr")
        assert resumed.status == SessionStatus.RUNNING

        done = await mgr.await_completion("pr")
        assert done.status == SessionStatus.COMPLETED
        assert done.attempts == 2

    @pytest.mark.asyncio
    async def test_pause_terminal_raises(self) -> None:
        mgr = _manager()
        await mgr.execute(AgentRequest(raw_input="What is up?", session_id="pt"))
        await mgr.await_completion("pt")
        with pytest.raises(SessionNotRunnableError):
            await mgr.pause("pt")

    @pytest.mark.asyncio
    async def test_pause_twice_raises(self) -> None:
        mgr = _manager(runtime=_runtime(delay_s=0.1))
        await mgr.execute(AgentRequest(raw_input="What is up?", session_id="p2"))
        await mgr.pause("p2")
        with pytest.raises(SessionNotRunnableError):
            await mgr.pause("p2")

    @pytest.mark.asyncio
    async def test_resume_non_paused_raises(self) -> None:
        mgr = _manager()
        await mgr.execute(AgentRequest(raw_input="What is up?", session_id="rnp"))
        await mgr.await_completion("rnp")
        with pytest.raises(SessionNotRunnableError):
            mgr.resume("rnp")

    @pytest.mark.asyncio
    async def test_delete_session(self) -> None:
        mgr = _manager(runtime=_runtime(delay_s=0.1))
        await mgr.execute(AgentRequest(raw_input="What is up?", session_id="d1"))
        assert await mgr.delete_session("d1") is True
        assert mgr.get_session("d1") is None
        assert await mgr.delete_session("d1") is False

    @pytest.mark.asyncio
    async def test_shutdown_cancels_active(self) -> None:
        mgr = _manager(runtime=_runtime(delay_s=0.5))
        await mgr.execute(AgentRequest(raw_input="What is up?", session_id="sd"))
        await mgr.shutdown()
        assert mgr.get_status("sd").status == SessionStatus.CANCELLED
        assert mgr.active_count() == 0


# ── AgentManager: retries ────────────────────────────────────────────


class TestAgentManagerRetry:
    @pytest.mark.asyncio
    async def test_retry_recovers_transient_failure(self) -> None:
        registry = ToolRegistry()
        registry.register(MockTool(name="analyze_question", fail_first_n=1))
        registry.register(MockTool(name="retrieve_knowledge"))
        registry.register(MockTool(name="formulate_answer"))
        mgr = _manager(runtime=AgentRuntime(tool_registry=registry))
        await mgr.execute(
            AgentRequest(raw_input="What is up?", session_id="r1"),
            options=AgentExecutionOptions(max_attempts=3),
        )
        done = await mgr.await_completion("r1")
        assert done.status == SessionStatus.COMPLETED
        assert done.attempts == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted_fails(self) -> None:
        mgr = _manager(runtime=_runtime(fail="permanent error"))
        await mgr.execute(
            AgentRequest(raw_input="What is up?", session_id="r2"),
            options=AgentExecutionOptions(max_attempts=3),
        )
        done = await mgr.await_completion("r2")
        assert done.status == SessionStatus.FAILED
        assert done.attempts == 3
        assert done.error is not None

    @pytest.mark.asyncio
    async def test_no_retry_by_default(self) -> None:
        mgr = _manager(runtime=_runtime(fail="boom"))
        await mgr.execute(AgentRequest(raw_input="What is up?", session_id="r3"))
        done = await mgr.await_completion("r3")
        assert done.status == SessionStatus.FAILED
        assert done.attempts == 1

    @pytest.mark.asyncio
    async def test_failed_session_exposes_response(self) -> None:
        mgr = _manager(runtime=_runtime(fail="boom"))
        await mgr.execute(AgentRequest(raw_input="What is up?", session_id="r4"))
        done = await mgr.await_completion("r4")
        assert done.response is not None
        assert done.response.success is False


# ── AgentManager: timeout ────────────────────────────────────────────


class TestAgentManagerTimeout:
    @pytest.mark.asyncio
    async def test_slow_run_times_out(self) -> None:
        mgr = _manager(runtime=_runtime(delay_s=2.0))
        await mgr.execute(
            AgentRequest(raw_input="What is slow?", session_id="to1"),
            config=AgentRunConfig(session_id="to1", overall_timeout_s=1),
        )
        done = await mgr.await_completion("to1")
        assert done.status == SessionStatus.TIMED_OUT
        assert done.error is not None


# ── AgentManager: memory and tools ───────────────────────────────────


class TestAgentManagerMemoryAndTools:
    @pytest.mark.asyncio
    async def test_memory_records_for_session(self) -> None:
        mgr = _manager()
        await mgr.execute(AgentRequest(raw_input="What is 2+2?", session_id="m1"))
        await mgr.await_completion("m1")
        records = await mgr.memory_for_session("m1")
        assert len(records) == 3
        assert all(r.session_id == "m1" for r in records)

    def test_tools_registry_shared(self) -> None:
        registry = _registry()
        runtime = AgentRuntime(tool_registry=registry)
        mgr = AgentManager(runtime=runtime)
        assert mgr.tools is registry

    def test_register_tool(self) -> None:
        mgr = _manager()
        assert "clock" not in mgr.tool_names()
        from app.tools.categories.builtin import ClockTool

        mgr.register_tool(ClockTool())
        assert "clock" in mgr.tool_names()

    def test_tool_names(self) -> None:
        mgr = _manager()
        assert set(mgr.tool_names()) == set(QUESTION_TOOLS)

    def test_memory_property(self) -> None:
        mgr = _manager()
        assert mgr.memory is mgr.runtime.memory


# ── AgentManager: persistence ────────────────────────────────────────


class TestAgentManagerPersistence:
    @pytest.mark.asyncio
    async def test_restore_loads_persisted_sessions(self, tmp_path: Path) -> None:
        store = JsonSessionStore(tmp_path)
        mgr = _manager(store=store)
        await mgr.execute(AgentRequest(raw_input="What is up?", session_id="p1"))
        await mgr.await_completion("p1")

        fresh = _manager(store=JsonSessionStore(tmp_path))
        assert fresh.get_session("p1") is None
        assert await fresh.restore() == 1
        restored = fresh.get_status("p1")
        assert restored.status == SessionStatus.COMPLETED
        assert restored.response is not None

    @pytest.mark.asyncio
    async def test_delete_removes_from_store(self, tmp_path: Path) -> None:
        store = JsonSessionStore(tmp_path)
        mgr = _manager(store=store)
        await mgr.execute(AgentRequest(raw_input="What is up?", session_id="p2"))
        await mgr.await_completion("p2")
        assert await mgr.delete_session("p2") is True
        fresh = SessionRegistry(store=JsonSessionStore(tmp_path))
        assert await fresh.load_all() == 0

    @pytest.mark.asyncio
    async def test_restore_skips_running_sessions_in_memory(
        self, tmp_path: Path
    ) -> None:
        store = JsonSessionStore(tmp_path)
        mgr = _manager(store=store)
        await mgr.execute(AgentRequest(raw_input="What is up?", session_id="p3"))
        await mgr.await_completion("p3")
        assert await mgr.restore() == 1
        # Re-running restore is idempotent (does not duplicate).
        assert await mgr.restore() == 1
        assert len(mgr.list_sessions()) == 1


# ── AgentManager: DI bootstrap ───────────────────────────────────────


class TestAgentManagerBootstrap:
    def test_register_components(self) -> None:
        container = DependencyContainer()
        register_agent_manager_components(container)
        assert container.has(AgentManager)
        assert container.has(SessionRegistry)
        assert container.has(SessionStore)

    def test_resolve_manager_shares_tools(self) -> None:
        container = DependencyContainer()
        register_agent_manager_components(container)
        registry = container.resolve(ToolRegistry)
        for name in QUESTION_TOOLS:
            registry.register(MockTool(name=name))
        mgr = container.resolve(AgentManager)
        assert mgr.tools is registry

    @pytest.mark.asyncio
    async def test_bootstrap_run_success(self) -> None:
        container = DependencyContainer()
        register_agent_manager_components(container)
        registry = container.resolve(ToolRegistry)
        for name in QUESTION_TOOLS:
            registry.register(MockTool(name=name))
        mgr = container.resolve(AgentManager)
        await mgr.execute(AgentRequest(raw_input="What is 2+2?", session_id="di1"))
        done = await mgr.await_completion("di1")
        assert done.status == SessionStatus.COMPLETED

    def test_idempotent(self) -> None:
        container = DependencyContainer()
        register_agent_manager_components(container)
        register_agent_manager_components(container)
        assert container.has(AgentManager)
