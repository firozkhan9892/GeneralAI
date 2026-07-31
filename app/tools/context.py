"""Execution context for tools.

A :class:`ToolContext` carries everything a tool may need while it runs:
the owning session, a shared memory store, per-invocation execution
metadata, and a cooperative cancellation token.  Contexts are immutable
facades over mutable stores so cancellation and memory can be shared
across a session without data races.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any

from app.tools.exceptions import ToolCancelledError


class CancellationToken:
    """A cooperative, thread-safe cancellation flag.

    Tools call :meth:`raise_if_cancelled` at safe points to observe
    cancellation without the framework needing to interrupt them.
    """

    def __init__(self) -> None:
        self._cancelled = False
        self._lock = threading.Lock()

    def cancel(self) -> None:
        """Request cancellation. Idempotent and thread-safe."""
        with self._lock:
            self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        """Return ``True`` if cancellation has been requested."""
        with self._lock:
            return self._cancelled

    def raise_if_cancelled(self) -> None:
        """Raise :class:`ToolCancelledError` if cancellation was requested."""
        if self.is_cancelled:
            raise ToolCancelledError(
                "Tool execution was cancelled",
                module="tools.context",
            )


class Memory:
    """A thread-safe key/value store shared within a session."""

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = dict(initial or {})
        self._lock = threading.RLock()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    def items(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class ToolSession:
    """Identifies and describes a session a tool execution belongs to."""

    def __init__(
        self,
        session_id: str | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.session_id: str = session_id or uuid.uuid4().hex
        self.metadata: dict[str, Any] = dict(metadata or {})


class ExecutionContext:
    """Per-invocation metadata about an execution."""

    def __init__(
        self,
        *,
        request_id: str | None = None,
        attempt: int = 1,
        max_attempts: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.request_id: str = request_id or uuid.uuid4().hex
        self.attempt: int = attempt
        self.max_attempts: int = max_attempts
        self.metadata: dict[str, Any] = dict(metadata or {})


class ToolContext:
    """Aggregated context handed to a tool during execution.

    Args:
        session: Optional owning :class:`ToolSession`.
        memory: Optional shared :class:`Memory`; created on demand.
        execution: Optional :class:`ExecutionContext`.
        token: Optional :class:`CancellationToken`; created on demand.
        metadata: Extra context metadata.
    """

    def __init__(
        self,
        *,
        session: ToolSession | None = None,
        memory: Memory | None = None,
        execution: ExecutionContext | None = None,
        token: CancellationToken | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.session: ToolSession | None = session
        self.memory: Memory = memory if memory is not None else Memory()
        self.execution: ExecutionContext = execution or ExecutionContext()
        self.token: CancellationToken = token or CancellationToken()
        self.metadata: dict[str, Any] = dict(metadata or {})

    @property
    def session_id(self) -> str | None:
        """Return the owning session id, if any."""
        return self.session.session_id if self.session else None

    @property
    def cancelled(self) -> bool:
        """Return ``True`` if the cancellation token has been tripped."""
        return self.token.is_cancelled

    def cancel(self) -> None:
        """Request cancellation of this execution."""
        self.token.cancel()

    def raise_if_cancelled(self) -> None:
        """Raise :class:`ToolCancelledError` if cancelled."""
        self.token.raise_if_cancelled()
