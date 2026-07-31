"""Tool resolver and executor — stages 10-11 of the cognitive pipeline."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable

from app.core.registry.base_registry import BaseRegistry
from app.kernel.tools.builtins import get_tool_handler
from app.kernel.tools.models import (
    ToolBinding,
    ToolDescriptor,
    ToolResult,
)

log = logging.getLogger(__name__)


class ToolResolver:
    """Maps tool requests to concrete registered tools."""

    def __init__(self) -> None:
        self._registry: BaseRegistry[ToolDescriptor] = BaseRegistry()

    def register_tool(self, descriptor: ToolDescriptor) -> None:
        """Register a tool descriptor.

        Args:
            descriptor: Tool descriptor.
        """
        self._registry.register(descriptor.name, descriptor, overwrite=True)
        log.debug("Registered tool '%s'", descriptor.name)

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return self._registry.has(tool_name)

    def list_tools(self) -> list[str]:
        """Return all registered tool names."""
        return self._registry.keys()

    async def resolve(
        self, tool_name: str, parameters: dict[str, Any] | None = None
    ) -> ToolBinding:
        """Resolve a tool name to a binding with validated parameters.

        Args:
            tool_name: Name of the tool to resolve.
            parameters: Optional tool parameters.

        Returns:
            Resolved tool binding.

        Raises:
            KeyError: If the tool is not registered.
        """
        descriptor = self._registry.get_or_raise(tool_name)
        params = parameters or {}
        return ToolBinding(
            tool_name=tool_name,
            descriptor=descriptor,
            parameters=params,
        )


class ToolExecutor:
    """Executes tools and returns results."""

    def __init__(self) -> None:
        self._resolver: ToolResolver | None = None
        self._max_retries: int = 3
        self._default_timeout_s: float = 30.0

    def set_resolver(self, resolver: ToolResolver) -> None:
        """Set the tool resolver for dependency injection."""
        self._resolver = resolver

    def _get_resolver(self) -> ToolResolver:
        if self._resolver is None:
            raise RuntimeError("ToolExecutor has no resolver configured")
        return self._resolver

    def _get_handler(self, tool_name: str) -> Callable[..., Awaitable[Any]]:
        try:
            return get_tool_handler(tool_name)
        except KeyError:
            raise KeyError(f"No handler available for tool: {tool_name}")

    async def execute(
        self,
        binding: ToolBinding,
        timeout_s: float | None = None,
        max_retries: int | None = None,
        cancellation_token: Any | None = None,
    ) -> ToolResult:
        """Execute a tool with validated parameters.

        Args:
            binding: Resolved tool binding.
            timeout_s: Optional timeout override (seconds).
            max_retries: Optional retry count override.
            cancellation_token: Optional cancellation token with is_cancelled.

        Returns:
            Tool execution result.
        """
        resolver = self._get_resolver()
        if not resolver.has_tool(binding.tool_name):
            return ToolResult(
                tool_name=binding.tool_name,
                success=False,
                error=f"Tool '{binding.tool_name}' is not registered",
            )

        handler = self._get_handler(binding.tool_name)
        timeout = timeout_s if timeout_s is not None else binding.descriptor.timeout_s
        retries = max_retries if max_retries is not None else self._max_retries

        started_at = time.monotonic()
        last_error: str | None = None

        for attempt in range(retries + 1):
            if cancellation_token is not None and getattr(
                cancellation_token, "is_cancelled", False
            ):
                return ToolResult(
                    tool_name=binding.tool_name,
                    success=False,
                    error="Tool execution cancelled",
                    duration_ms=int((time.monotonic() - started_at) * 1000),
                )

            try:
                result = await asyncio.wait_for(
                    handler(binding.parameters),
                    timeout=float(timeout),
                )
                duration_ms = int((time.monotonic() - started_at) * 1000)
                return ToolResult(
                    tool_name=binding.tool_name,
                    output=result,
                    duration_ms=duration_ms,
                    success=True,
                )
            except asyncio.TimeoutError:
                last_error = f"Tool '{binding.tool_name}' timed out after {timeout}s"
                log.warning(
                    "Tool '%s' attempt %d/%d timed out",
                    binding.tool_name,
                    attempt + 1,
                    retries + 1,
                )
                if attempt >= retries:
                    return ToolResult(
                        tool_name=binding.tool_name,
                        success=False,
                        error=last_error,
                        duration_ms=int((time.monotonic() - started_at) * 1000),
                    )
            except Exception as exc:
                last_error = str(exc)
                log.warning(
                    "Tool '%s' attempt %d/%d failed: %s",
                    binding.tool_name,
                    attempt + 1,
                    retries + 1,
                    exc,
                )
                if attempt >= retries:
                    return ToolResult(
                        tool_name=binding.tool_name,
                        success=False,
                        error=last_error,
                        duration_ms=int((time.monotonic() - started_at) * 1000),
                    )
                await asyncio.sleep(0.01 * (attempt + 1))

        return ToolResult(
            tool_name=binding.tool_name,
            success=False,
            error=last_error or "Unknown error",
            duration_ms=int((time.monotonic() - started_at) * 1000),
        )
