"""Skill selector and executor — stage 9 of the cognitive pipeline."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable

from app.core.registry.base_registry import BaseRegistry
from app.kernel.skills.builtins import get_skill_handler
from app.kernel.skills.models import Skill, SkillDescriptor, SkillResult
from app.kernel.tools.executor import ToolExecutor, ToolResolver

log = logging.getLogger(__name__)


class SkillSelector:
    """Matches plan steps to registered skills."""

    def __init__(self) -> None:
        self._registry: BaseRegistry[SkillDescriptor] = BaseRegistry()

    def register_skill(self, descriptor: SkillDescriptor) -> None:
        """Register a skill.

        Args:
            descriptor: Skill descriptor.
        """
        self._registry.register(descriptor.name, descriptor, overwrite=True)
        log.debug("Registered skill '%s'", descriptor.name)

    def has_skill(self, skill_name: str) -> bool:
        """Check if a skill is registered."""
        return self._registry.has(skill_name)

    def list_skills(self) -> list[str]:
        """Return all registered skill names."""
        return self._registry.keys()

    async def select(
        self, skill_name: str, parameters: dict[str, Any] | None = None
    ) -> Skill:
        """Resolve a skill name to a binding.

        Args:
            skill_name: Name of the skill to select.
            parameters: Optional skill parameters.

        Returns:
            Resolved skill binding.

        Raises:
            KeyError: If the skill is not registered.
        """
        descriptor = self._registry.get_or_raise(skill_name)
        params = parameters or {}
        resolved_tools: tuple[str, ...] = tuple(
            req.tool_name for req in descriptor.required_tools
        )
        return Skill(
            name=skill_name,
            descriptor=descriptor,
            parameters=params,
            resolved_tools=resolved_tools,
        )


class SkillExecutor:
    """Executes skills by orchestrating their internal workflows."""

    def __init__(self) -> None:
        self._selector: SkillSelector | None = None
        self._tool_resolver: ToolResolver | None = None
        self._tool_executor: ToolExecutor | None = None
        self._max_retries: int = 3
        self._default_timeout_s: float = 30.0

    def set_selector(self, selector: SkillSelector) -> None:
        """Set the skill selector for dependency injection."""
        self._selector = selector

    def set_tool_resolver(self, resolver: ToolResolver) -> None:
        """Set the tool resolver for dependency injection."""
        self._tool_resolver = resolver

    def set_tool_executor(self, executor: ToolExecutor) -> None:
        """Set the tool executor for dependency injection."""
        self._tool_executor = executor

    def _get_selector(self) -> SkillSelector:
        if self._selector is None:
            raise RuntimeError("SkillExecutor has no selector configured")
        return self._selector

    def _get_handler(self, skill_name: str) -> Callable[..., Awaitable[Any]]:
        try:
            return get_skill_handler(skill_name)
        except KeyError:
            raise KeyError(f"No handler available for skill: {skill_name}")

    async def _resolve_tool(
        self, tool_name: str, parameters: dict[str, Any] | None = None
    ) -> Any:
        """Resolve and execute a tool, returning its output."""
        if self._tool_resolver is None or self._tool_executor is None:
            raise RuntimeError("Tool resolver/executor not configured")

        binding = await self._tool_resolver.resolve(tool_name, parameters)
        result = await self._tool_executor.execute(binding)
        if not result.success:
            raise RuntimeError(f"Tool '{tool_name}' failed: {result.error}")
        return result.output

    async def execute(
        self,
        binding: Skill,
        timeout_s: float | None = None,
        max_retries: int | None = None,
        cancellation_token: Any | None = None,
    ) -> SkillResult:
        """Execute a skill.

        Args:
            binding: Resolved skill binding.
            timeout_s: Optional timeout override (seconds).
            max_retries: Optional retry count override.
            cancellation_token: Optional cancellation token with is_cancelled.

        Returns:
            Skill execution result.
        """
        selector = self._get_selector()
        if not selector.has_skill(binding.name):
            return SkillResult(
                skill_name=binding.name,
                success=False,
                error=f"Skill '{binding.name}' is not registered",
            )

        handler = self._get_handler(binding.name)
        timeout = timeout_s if timeout_s is not None else self._default_timeout_s
        retries = max_retries if max_retries is not None else self._max_retries

        started_at = time.monotonic()
        last_error: str | None = None
        token_cost = 0

        for attempt in range(retries + 1):
            if cancellation_token is not None and getattr(
                cancellation_token, "is_cancelled", False
            ):
                return SkillResult(
                    skill_name=binding.name,
                    success=False,
                    error="Skill execution cancelled",
                    duration_ms=int((time.monotonic() - started_at) * 1000),
                    token_cost=token_cost,
                )

            try:
                result = await asyncio.wait_for(
                    handler(binding.parameters),
                    timeout=float(timeout),
                )
                duration_ms = int((time.monotonic() - started_at) * 1000)
                return SkillResult(
                    skill_name=binding.name,
                    output=result,
                    duration_ms=duration_ms,
                    token_cost=token_cost,
                    success=True,
                )
            except asyncio.TimeoutError:
                last_error = f"Skill '{binding.name}' timed out after {timeout}s"
                log.warning(
                    "Skill '%s' attempt %d/%d timed out",
                    binding.name,
                    attempt + 1,
                    retries + 1,
                )
                if attempt >= retries:
                    return SkillResult(
                        skill_name=binding.name,
                        success=False,
                        error=last_error,
                        duration_ms=int((time.monotonic() - started_at) * 1000),
                        token_cost=token_cost,
                    )
            except Exception as exc:
                last_error = str(exc)
                log.warning(
                    "Skill '%s' attempt %d/%d failed: %s",
                    binding.name,
                    attempt + 1,
                    retries + 1,
                    exc,
                )
                if attempt >= retries:
                    return SkillResult(
                        skill_name=binding.name,
                        success=False,
                        error=last_error,
                        duration_ms=int((time.monotonic() - started_at) * 1000),
                        token_cost=token_cost,
                    )
                await asyncio.sleep(0.01 * (attempt + 1))

        return SkillResult(
            skill_name=binding.name,
            success=False,
            error=last_error or "Unknown error",
            duration_ms=int((time.monotonic() - started_at) * 1000),
            token_cost=token_cost,
        )
