"""Retry and fallback policies for the agent runtime.

Deterministic, rule-based policies that control how the agent loop
handles step failures: how many times to retry a failed tool call and
which tool to fall back to when none is selected.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_RETRYABLE_MARKERS = (
    "timeout",
    "timed out",
    "temporary",
    "retry",
    "connection",
    "unavailable",
)


class RetryPolicy:
    """Deterministic retry policy for failed agent steps.

    A step is retried when the error looks transient (contains a
    retryable marker) and the attempt budget has not been exhausted.
    """

    def __init__(
        self,
        max_retries: int = 2,
        *,
        retryable_markers: tuple[str, ...] = _RETRYABLE_MARKERS,
    ) -> None:
        self._max_retries = max(0, max_retries)
        self._retryable_markers = tuple(retryable_markers)

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def should_retry(self, attempts_used: int, error: str | None) -> bool:
        """Decide whether a failed attempt should be retried.

        Args:
            attempts_used: Number of attempts already consumed.
            error: Error message from the failed attempt.

        Returns:
            ``True`` if a retry should be attempted.
        """
        if attempts_used > self._max_retries:
            return False
        if not error:
            return False
        lowered = error.lower()
        return any(marker in lowered for marker in self._retryable_markers)

    def is_retryable(self, error: str | None) -> bool:
        """Classify whether an error looks transient.

        Args:
            error: Error message to classify.

        Returns:
            ``True`` if the error is considered retryable.
        """
        if not error:
            return False
        lowered = error.lower()
        return any(marker in lowered for marker in self._retryable_markers)


class FallbackPolicy:
    """Deterministic fallback-tool selection.

    When a step produces no tool match, the policy supplies a default
    tool so the loop can still make progress instead of failing.
    """

    def __init__(
        self,
        fallback_tool: str = "echo",
        *,
        available_tools: tuple[str, ...] = (),
    ) -> None:
        self._fallback_tool = fallback_tool
        self._available = tuple(sorted(available_tools))

    @property
    def fallback_tool(self) -> str:
        return self._fallback_tool

    def set_available_tools(self, tools: tuple[str, ...] | list[str] | Any) -> None:
        """Record the set of tools currently available.

        Args:
            tools: Iterable of tool names.
        """
        self._available = tuple(sorted(tools))

    def select_fallback(self, requested: str | None = None) -> str | None:
        """Return the fallback tool to use.

        Prefers an explicit request when it is available; otherwise
        returns the configured default fallback when available.

        Args:
            requested: Optional explicitly requested tool name.

        Returns:
            A tool name, or ``None`` if nothing is available.
        """
        if requested and requested in self._available:
            return requested
        if self._fallback_tool in self._available:
            return self._fallback_tool
        return None
