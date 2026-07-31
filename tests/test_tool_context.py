"""Tests for tool execution context components."""

from __future__ import annotations

import pytest

from app.tools.context import (
    CancellationToken,
    ExecutionContext,
    Memory,
    ToolContext,
    ToolSession,
)
from app.tools.exceptions import ToolCancelledError


class TestCancellationToken:
    def test_starts_uncancelled(self) -> None:
        assert CancellationToken().is_cancelled is False

    def test_cancel(self) -> None:
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled is True

    def test_cancel_idempotent(self) -> None:
        token = CancellationToken()
        token.cancel()
        token.cancel()
        assert token.is_cancelled is True

    def test_raise_if_cancelled(self) -> None:
        token = CancellationToken()
        token.cancel()
        with pytest.raises(ToolCancelledError):
            token.raise_if_cancelled()

    def test_raise_if_not_cancelled(self) -> None:
        CancellationToken().raise_if_cancelled()


class TestMemory:
    def test_get_default(self) -> None:
        memory = Memory()
        assert memory.get("missing", "fallback") == "fallback"

    def test_set_get(self) -> None:
        memory = Memory()
        memory.set("key", "value")
        assert memory.get("key") == "value"

    def test_has(self) -> None:
        memory = Memory(initial={"a": 1})
        assert memory.has("a")
        assert not memory.has("b")

    def test_keys_items(self) -> None:
        memory = Memory(initial={"a": 1, "b": 2})
        assert sorted(memory.keys()) == ["a", "b"]
        assert memory.items() == {"a": 1, "b": 2}

    def test_len_and_clear(self) -> None:
        memory = Memory(initial={"a": 1})
        assert len(memory) == 1
        memory.clear()
        assert len(memory) == 0


class TestToolSession:
    def test_generates_id(self) -> None:
        first = ToolSession()
        second = ToolSession()
        assert first.session_id
        assert first.session_id != second.session_id

    def test_explicit_id(self) -> None:
        session = ToolSession("abc")
        assert session.session_id == "abc"

    def test_metadata(self) -> None:
        session = ToolSession(metadata={"user": "alice"})
        assert session.metadata == {"user": "alice"}


class TestExecutionContext:
    def test_generates_request_id(self) -> None:
        ctx = ExecutionContext()
        assert ctx.request_id

    def test_attempts(self) -> None:
        ctx = ExecutionContext(attempt=2, max_attempts=3)
        assert ctx.attempt == 2
        assert ctx.max_attempts == 3


class TestToolContext:
    def test_defaults(self) -> None:
        ctx = ToolContext()
        assert ctx.session is None
        assert ctx.session_id is None
        assert ctx.cancelled is False
        assert isinstance(ctx.memory, Memory)

    def test_session_id(self) -> None:
        session = ToolSession("s1")
        ctx = ToolContext(session=session)
        assert ctx.session_id == "s1"

    def test_shared_memory(self) -> None:
        memory = Memory()
        ctx = ToolContext(memory=memory)
        ctx.memory.set("k", "v")
        assert memory.get("k") == "v"

    def test_cancel(self) -> None:
        ctx = ToolContext()
        ctx.cancel()
        assert ctx.cancelled is True

    def test_explicit_token(self) -> None:
        token = CancellationToken()
        ctx = ToolContext(token=token)
        token.cancel()
        assert ctx.cancelled is True

    def test_raise_if_cancelled(self) -> None:
        ctx = ToolContext()
        ctx.cancel()
        with pytest.raises(ToolCancelledError):
            ctx.raise_if_cancelled()
