"""Deterministic mock tool for testing.

Produces predictable, repeatable output with no side effects so unit and
integration tests can exercise the full tool contract (validation,
permissions, timeouts, cancellation, retries) without depending on real
resources.  Optional behaviours can be configured to simulate delays,
failures, and custom handlers.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.exceptions import ToolExecutionError
from app.tools.models import ToolCategory, ToolParameter


class MockTool(Tool):
    """A configurable, deterministic tool for testing.

    Args:
        name: Tool name (defaults to ``mock``).
        echo_input: If ``True``, returns ``Echo: <input>``.
        result: Fixed output returned when ``echo_input`` is ``False``.
        delay_s: Optional sleep duration to simulate latency.
        fail: If given, raise a :class:`ToolExecutionError` with this
            message every run.
        fail_first_n: Fail the first ``n`` invocations, then succeed.
        on_run: Optional callable invoked with arguments and context;
            its return value becomes the tool output.
    """

    name = "mock"
    description = "Deterministic mock tool for testing"
    category = ToolCategory.BUILTIN
    parameters = (
        ToolParameter(
            name="input",
            description="Arbitrary input",
            param_type="string",
        ),
    )

    def __init__(
        self,
        *,
        name: str = "mock",
        echo_input: bool = True,
        result: Any = None,
        delay_s: float = 0.0,
        fail: str | None = None,
        fail_first_n: int = 0,
        on_run: Callable[[Mapping[str, Any], ToolContext | None], Any] | None = None,
    ) -> None:
        self.name = name
        self._echo_input = echo_input
        self._result = result
        self._delay_s = delay_s
        self._fail = fail
        self._fail_first_n = fail_first_n
        self._on_run = on_run
        self.call_count = 0

    def run(
        self, arguments: Mapping[str, Any], context: ToolContext | None = None
    ) -> Any:
        """Return deterministic output according to configuration."""
        self.call_count += 1
        if self._delay_s > 0:
            time.sleep(self._delay_s)
        if self._fail is not None:
            raise ToolExecutionError(
                self._fail,
                module="tools.mock",
            )
        if self.call_count <= self._fail_first_n:
            raise ToolExecutionError(
                f"Simulated failure (attempt {self.call_count})",
                module="tools.mock",
            )
        if self._on_run is not None:
            return self._on_run(arguments, context)
        if self._echo_input:
            return f"Echo: {arguments.get('input', '')}"
        return self._result
