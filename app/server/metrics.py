"""Request metrics collection for the FastAPI server.

A lightweight, thread-safe counter that tracks the total number of
requests, per-path counts, and error counts.  Wired in via an HTTP
middleware so ``/metrics`` can report live server activity.
"""

from __future__ import annotations

from typing import Any
import threading


class MetricsCollector:
    """Thread-safe request counters.

    Args:
        enabled: When ``False``, recording is a no-op but counters stay
            readable (starts at zero).
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._requests_total = 0
        self._errors_total = 0
        self._requests_by_path: dict[str, int] = {}
        self._lock = threading.Lock()

    def record(self, path: str, status_code: int) -> None:
        """Record a single completed request.

        Args:
            path: Request URL path.
            status_code: HTTP status code returned by the route.
        """
        if not self._enabled:
            return
        with self._lock:
            self._requests_total += 1
            self._requests_by_path[path] = self._requests_by_path.get(path, 0) + 1
            if status_code >= 500:
                self._errors_total += 1

    def snapshot(self) -> dict[str, Any]:
        """Return the current counter values."""
        with self._lock:
            return {
                "requests_total": self._requests_total,
                "errors_total": self._errors_total,
                "requests_by_path": dict(self._requests_by_path),
            }

    def clear(self) -> None:
        """Reset all counters."""
        with self._lock:
            self._requests_total = 0
            self._errors_total = 0
            self._requests_by_path.clear()
