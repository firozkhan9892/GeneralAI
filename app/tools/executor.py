"""Tool executor.

Orchestrates tool invocations end-to-end: permission checks, argument
validation, cooperative cancellation, timeouts, retries, and result
shaping.  All failures are captured into a :class:`ToolResult` rather
than raised, so callers have a uniform success/error contract.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.exceptions import (
    ToolCancelledError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolTimeoutError,
    ToolValidationError,
)
from app.tools.models import ToolResult
from app.tools.permissions import PermissionDecision, PermissionSystem
from app.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

ConfirmationHandler = Callable[[Tool, Mapping[str, Any]], bool]


def _error_text(exc: Exception) -> str:
    """Return a clean error message from an exception."""
    message = getattr(exc, "message", "")
    if message:
        return str(message)
    return str(exc) or exc.__class__.__name__


class ToolExecutor:
    """Executes tools with policy enforcement and failure capture.

    Args:
        registry: Optional registry used to resolve tool names.
        permission_system: Optional policy gate.
        confirmation_handler: Optional callback consulted when a tool
            requires confirmation.
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        permission_system: PermissionSystem | None = None,
        confirmation_handler: ConfirmationHandler | None = None,
    ) -> None:
        self._registry = registry
        self._permission_system = permission_system
        self._confirmation_handler = confirmation_handler

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        name_or_tool: str | Tool,
        arguments: Mapping[str, Any] | None = None,
        *,
        context: ToolContext | None = None,
        timeout_s: float | None = None,
        max_retries: int = 0,
        retry_delay_s: float = 0.0,
    ) -> ToolResult:
        """Execute a tool synchronously.

        Args:
            name_or_tool: Registered tool name or a :class:`Tool` instance.
            arguments: Invocation arguments.
            context: Optional execution context.
            timeout_s: Optional per-attempt timeout override.
            max_retries: Number of retries after a failed attempt.
            retry_delay_s: Delay between retries in seconds.

        Returns:
            A :class:`ToolResult` describing the outcome.
        """
        return asyncio.run(
            self.execute_async(
                name_or_tool,
                arguments,
                context=context,
                timeout_s=timeout_s,
                max_retries=max_retries,
                retry_delay_s=retry_delay_s,
            )
        )

    async def execute_async(
        self,
        name_or_tool: str | Tool,
        arguments: Mapping[str, Any] | None = None,
        *,
        context: ToolContext | None = None,
        timeout_s: float | None = None,
        max_retries: int = 0,
        retry_delay_s: float = 0.0,
    ) -> ToolResult:
        """Execute a tool asynchronously.

        See :meth:`execute` for parameter semantics.
        """
        started = time.perf_counter()

        try:
            tool = self._resolve_tool(name_or_tool)
        except ToolNotFoundError as exc:
            return self._failure(
                tool_name="",
                error=str(exc),
                started=started,
                metadata={"phase": "resolve"},
            )

        ctx = context or ToolContext()

        denial = self._check_permission(tool, dict(arguments or {}))
        if denial is not None:
            return self._failure(
                tool_name=tool.name,
                error=denial,
                started=started,
                metadata={"phase": "permission"},
            )

        try:
            clean_arguments = tool.validate(arguments or {})
        except ToolValidationError as exc:
            return self._failure(
                tool_name=tool.name,
                error=str(exc),
                started=started,
                metadata={"phase": "validation"},
            )

        effective_timeout = timeout_s if timeout_s is not None else tool.timeout_s
        attempts = 0
        last_error: str | None = None
        timed_out = False

        while attempts <= max_retries:
            attempts += 1
            if ctx.cancelled:
                return self._failure(
                    tool_name=tool.name,
                    error="Tool execution was cancelled",
                    started=started,
                    metadata={"phase": "cancelled", "attempts": attempts},
                )

            try:
                output = await self._run_attempt(
                    tool, clean_arguments, ctx, effective_timeout
                )
                return ToolResult(
                    tool_name=tool.name,
                    success=True,
                    output=output,
                    execution_time=time.perf_counter() - started,
                    metadata={"attempts": attempts, "timeout_s": effective_timeout},
                )
            except ToolCancelledError:
                return self._failure(
                    tool_name=tool.name,
                    error="Tool execution was cancelled",
                    started=started,
                    metadata={"phase": "cancelled", "attempts": attempts},
                )
            except ToolTimeoutError:
                last_error = f"Tool '{tool.name}' timed out after {effective_timeout}s"
                timed_out = True
            except ToolExecutionError as exc:
                last_error = _error_text(exc)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "Tool '%s' failed on attempt %d: %s", tool.name, attempts, exc
                )
                last_error = _error_text(exc)

            if attempts <= max_retries and retry_delay_s > 0:
                await asyncio.sleep(retry_delay_s)

        return self._failure(
            tool_name=tool.name,
            error=last_error or "Tool execution failed",
            started=started,
            metadata={"attempts": attempts, "timed_out": timed_out},
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_tool(self, name_or_tool: str | Tool) -> Tool:
        """Resolve a name or instance to a tool."""
        if isinstance(name_or_tool, Tool):
            return name_or_tool
        if self._registry is None:
            raise ToolNotFoundError(
                f"Tool '{name_or_tool}' is not registered and no registry is configured",
                module="tools.executor",
            )
        return self._registry.get_or_raise(name_or_tool)

    def _check_permission(self, tool: Tool, arguments: Mapping[str, Any]) -> str | None:
        """Return an error message if the invocation is not permitted."""
        if self._permission_system is None:
            return None
        result = self._permission_system.check(tool.name, dict(arguments))
        if result.decision is PermissionDecision.ALLOW:
            return None
        if result.decision is PermissionDecision.CONFIRM:
            if self._confirmation_handler is None:
                return f"Tool '{tool.name}' requires confirmation"
            if not self._confirmation_handler(tool, dict(arguments)):
                return f"Confirmation declined for tool '{tool.name}'"
            return None
        reason = result.reason or "permission denied"
        return f"Tool '{tool.name}' is not allowed: {reason}"

    async def _run_attempt(
        self,
        tool: Tool,
        arguments: Mapping[str, Any],
        context: ToolContext,
        timeout_s: float | None,
    ) -> Any:
        """Run one attempt, enforcing timeout via :meth:`asyncio.wait_for`."""
        if timeout_s is None or timeout_s <= 0:
            return await tool.arun(arguments, context)
        try:
            return await asyncio.wait_for(
                tool.arun(arguments, context),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise ToolTimeoutError(
                f"Tool '{tool.name}' timed out after {timeout_s}s",
                module="tools.executor",
                cause=exc,
            ) from exc

    @staticmethod
    def _failure(
        *,
        tool_name: str,
        error: str,
        started: float,
        metadata: dict[str, Any],
    ) -> ToolResult:
        """Build a failure result with elapsed time."""
        return ToolResult(
            tool_name=tool_name,
            success=False,
            error=error,
            execution_time=time.perf_counter() - started,
            metadata=metadata,
        )
