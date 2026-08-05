"""Injectable clock abstraction.

Scheduler and executor logic must **never** call ``datetime.utcnow()``
directly.  They accept a :class:`Clock` so that time is deterministic and
easily testable (fake clocks can be advanced manually).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Protocol


class Clock(Protocol):
    """Interface for reading the current wall-clock time."""

    def utcnow(self) -> datetime:
        """Return the current UTC time as an aware :class:`datetime`."""

    def now(self) -> datetime:
        """Return the current local time as an aware :class:`datetime`."""

    def time(self) -> float:
        """Return the current monotonic time in seconds."""


class SystemClock:
    """Real clock backed by :func:`datetime.utcnow`.

    All datetimes returned are timezone-aware UTC.
    """

    def utcnow(self) -> datetime:
        return datetime.now(timezone.utc)

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def time(self) -> float:
        return time.monotonic()


class FakeClock:
    """Deterministic clock for tests.

    The clock starts at *start* (default ``2026-01-01T00:00:00Z``) and
    only advances when :meth:`advance` or :meth:`set` is called.
    """

    def __init__(self, start: datetime | None = None) -> None:
        base = start or datetime(2026, 1, 1, tzinfo=timezone.utc)
        if base.tzinfo is None:
            base = base.replace(tzinfo=timezone.utc)
        self._current = base
        self._monotonic = 0.0

    def utcnow(self) -> datetime:
        return self._current

    def now(self) -> datetime:
        return self._current

    def time(self) -> float:
        return self._monotonic

    def advance(self, seconds: float) -> None:
        """Advance the fake clock by *seconds*."""
        self._current = self._current + timedelta(seconds=seconds)
        self._monotonic += seconds

    def set(self, value: datetime) -> None:
        """Set the clock to an exact datetime."""
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        self._current = value
