"""Agent loop — iterative step-by-step plan execution.

The loop drives a plan's steps one at a time, selecting a tool via the
Decision Engine, gating the action through the Policy Engine, executing
it with the Phase-5 ToolExecutor, recording memory after each completed
task, and applying retry / fallback policies on failure.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime

from app.kernel.agent.models import (
    AgentRunConfig,
    AgentStep,
    AgentStepStatus,
)
from app.kernel.agent.policies import FallbackPolicy, RetryPolicy
from app.kernel.decision.engine import DecisionEngine
from app.kernel.decision.models import ActionCandidate, Decision, DecisionReason
from app.kernel.memory.engine import MemoryEngine
from app.kernel.planning.models import Plan, SkillStep
from app.kernel.policy.engine import PolicyEngine
from app.kernel.policy.models import VerdictType
from app.tools.context import CancellationToken, ToolContext, ToolSession
from app.tools.executor import ToolExecutor
from app.tools.models import ToolResult
from app.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


def _tool_confidence(skill_name: str, tool_name: str) -> float:
    """Deterministic relevance score between a skill and a tool.

    Exact matches score 1.0, substring matches 0.6, otherwise 0.1.
    """
    if skill_name == tool_name:
        return 1.0
    if skill_name in tool_name or tool_name in skill_name:
        return 0.6
    return 0.1


class AgentLoop:
    """Executes a plan step-by-step with policy and memory integration.

    Args:
        decision_engine: Selects tools for each step.
        policy_engine: Gates each step action.
        tool_executor: Phase-5 executor that runs tools.
        tool_registry: Registry enumerating available tools.
        memory_engine: Optional memory engine for task recording.
        retry_policy: Policy controlling step retries.
        fallback_policy: Policy supplying a fallback tool.
    """

    def __init__(
        self,
        *,
        decision_engine: DecisionEngine | None = None,
        policy_engine: PolicyEngine | None = None,
        tool_executor: ToolExecutor | None = None,
        tool_registry: ToolRegistry | None = None,
        memory_engine: MemoryEngine | None = None,
        retry_policy: RetryPolicy | None = None,
        fallback_policy: FallbackPolicy | None = None,
    ) -> None:
        self._decision = decision_engine or DecisionEngine()
        self._policy = policy_engine or PolicyEngine()
        self._registry = tool_registry if tool_registry is not None else ToolRegistry()
        self._executor = (
            tool_executor
            if tool_executor is not None
            else ToolExecutor(registry=self._registry)
        )
        self._memory = memory_engine
        self._retry_policy = retry_policy or RetryPolicy()
        self._fallback_policy = fallback_policy or FallbackPolicy()
        self._fallback_policy.set_available_tools(self._registry.names())

    # ── Public API ────────────────────────────────────────────────────

    async def execute(
        self,
        plan: Plan,
        *,
        config: AgentRunConfig,
        cancellation_token: CancellationToken | None = None,
    ) -> tuple[AgentStep, ...]:
        """Execute a plan's steps sequentially.

        Args:
            plan: The plan to execute.
            config: Run configuration.
            cancellation_token: Optional cooperative cancellation signal.

        Returns:
            A tuple of per-step outcomes.  Steps not reached are marked
            ``SKIPPED``.
        """
        completed: list[AgentStep] = []
        steps = plan.steps
        budget = min(len(steps), config.max_iterations)

        for index in range(budget):
            if cancellation_token is not None and cancellation_token.is_cancelled:
                log.info("Agent loop cancelled after %d step(s)", index)
                return self._with_skipped(completed, steps, index)

            step = steps[index]
            result = await self._execute_step(
                step,
                order=index,
                config=config,
                cancellation_token=cancellation_token,
            )
            completed.append(result)

            if result.status != AgentStepStatus.SUCCEEDED:
                log.info(
                    "Agent loop stopped after failed step %d (%s)",
                    index,
                    result.status.value,
                )
                return self._with_skipped(completed, steps, index + 1)

        return self._with_skipped(completed, steps, budget)

    # ── Step execution ────────────────────────────────────────────────

    async def _execute_step(
        self,
        step: SkillStep,
        *,
        order: int,
        config: AgentRunConfig,
        cancellation_token: CancellationToken | None,
    ) -> AgentStep:
        started = datetime.utcnow()
        started_mono = time.monotonic()

        tool_name = await self._select_tool(step, config)
        decision = self._build_decision(step, tool_name)
        verdict = await self._policy.evaluate(decision)

        if verdict.verdict != VerdictType.ALLOW:
            log.info("Step %d denied by policy: %s", order, verdict.verdict.value)
            return AgentStep(
                order=order,
                skill_name=step.skill_name,
                description=step.description,
                status=AgentStepStatus.FAILED,
                tool_name=tool_name,
                error=verdict.denial_reason
                or f"Policy verdict: {verdict.verdict.value}",
                retries=0,
                decision=decision,
                policy_verdict=verdict,
                started_at=started,
                completed_at=datetime.utcnow(),
            )

        try:
            result, retries = await asyncio.wait_for(
                self._execute_with_retries(step, tool_name, config, cancellation_token),
                timeout=config.step_timeout_s,
            )
        except asyncio.TimeoutError:
            log.warning("Step %d timed out after %ss", order, config.step_timeout_s)
            return AgentStep(
                order=order,
                skill_name=step.skill_name,
                description=step.description,
                status=AgentStepStatus.FAILED,
                tool_name=tool_name,
                error=f"Step timed out after {config.step_timeout_s}s",
                retries=0,
                decision=decision,
                policy_verdict=verdict,
                started_at=started,
                completed_at=datetime.utcnow(),
            )

        memory_id: str | None = None
        if config.memory_enabled and self._memory is not None:
            memory_id = await self._record_memory(step, tool_name, result, config)
            log.debug("Step %d memory record %s", order, memory_id)

        completed_at = datetime.utcnow()
        status = AgentStepStatus.SUCCEEDED if result.success else AgentStepStatus.FAILED
        return AgentStep(
            order=order,
            skill_name=step.skill_name,
            description=step.description,
            status=status,
            tool_name=tool_name,
            tool_result=result,
            error=None if result.success else (result.error or "Tool failed"),
            retries=retries,
            memory_record_id=memory_id,
            decision=decision,
            policy_verdict=verdict,
            started_at=started,
            completed_at=completed_at,
            metadata={"duration_ms": int((time.monotonic() - started_mono) * 1000)},
        )

    async def _execute_with_retries(
        self,
        step: SkillStep,
        tool_name: str,
        config: AgentRunConfig,
        cancellation_token: CancellationToken | None,
    ) -> tuple[ToolResult, int]:
        attempts = 0
        retries = 0
        result: ToolResult | None = None

        while attempts <= config.max_retries:
            if cancellation_token is not None and cancellation_token.is_cancelled:
                raise asyncio.CancelledError()

            attempts += 1
            result = await self._run_tool(step, tool_name, config)
            if result.success:
                return result, retries

            if retries >= config.max_retries:
                break
            if not self._retry_policy.should_retry(attempts, result.error):
                break
            retries += 1
            log.info(
                "Step '%s' retry %d/%d after error: %s",
                step.skill_name,
                retries,
                config.max_retries,
                result.error,
            )

        return result or ToolResult(
            tool_name=tool_name, success=False, error="No attempt"
        ), retries

    async def _run_tool(
        self,
        step: SkillStep,
        tool_name: str,
        config: AgentRunConfig,
    ) -> ToolResult:
        context = ToolContext(
            session=ToolSession(session_id=config.session_id),
        )
        try:
            return await self._executor.execute_async(
                tool_name,
                dict(step.parameters),
                context=context,
                timeout_s=config.step_timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Tool '%s' raised %s", tool_name, exc)
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool raised {type(exc).__name__}: {exc}",
            )

    # ── Tool selection ────────────────────────────────────────────────

    async def _select_tool(self, step: SkillStep, config: AgentRunConfig) -> str:
        candidates = [
            ActionCandidate(
                action_type="tool_call",
                description=f"Invoke tool '{name}'",
                parameters={"tool_name": name},
                confidence=_tool_confidence(step.skill_name, name),
                estimated_cost=0,
                source="agent_loop",
            )
            for name in self._registry.names()
        ]

        if candidates:
            ranked = await self._decision.rank_candidates(candidates)
            top = ranked[0]
            if top.confidence >= 0.5:
                name = top.parameters.get("tool_name") or ""
                if name in self._registry.names():
                    return name

        fallback = self._fallback_policy.select_fallback()
        if fallback:
            return fallback

        log.warning(
            "No tool selected for step '%s' and no fallback available",
            step.skill_name,
        )
        return ""

    def _build_decision(self, step: SkillStep, tool_name: str) -> Decision:
        candidate = ActionCandidate(
            action_type="tool_call",
            description=f"Execute step '{step.skill_name}' via tool '{tool_name}'",
            parameters={"tool_name": tool_name, "skill_name": step.skill_name},
            confidence=1.0,
            source="agent_loop",
        )
        return Decision(
            selected_action=candidate,
            candidates=(candidate,),
            reason=DecisionReason(
                primary_rationale=f"Selected tool '{tool_name}' for step '{step.skill_name}'",
            ),
            strategy_used="agent_loop",
            status="pending",
        )

    # ── Memory ────────────────────────────────────────────────────────

    async def _record_memory(
        self,
        step: SkillStep,
        tool_name: str,
        result: ToolResult,
        config: AgentRunConfig,
    ) -> str:
        assert self._memory is not None
        content = (
            f"Task '{step.skill_name}' completed via tool '{tool_name}'"
            if result.success
            else f"Task '{step.skill_name}' failed via tool '{tool_name}'"
        )
        return await self._memory.remember(
            content=content,
            session_id=config.session_id,
            tags=("task", step.skill_name),
            importance=0.9 if result.success else 0.7,
            metadata={
                "skill_name": step.skill_name,
                "tool_name": tool_name,
                "success": result.success,
                "error": result.error,
            },
        )

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _with_skipped(
        completed: list[AgentStep],
        steps: tuple[SkillStep, ...],
        start_index: int,
    ) -> tuple[AgentStep, ...]:
        """Append SKIPPED outcomes for steps never reached."""
        skipped = [
            AgentStep(
                order=index,
                skill_name=step.skill_name,
                description=step.description,
                status=AgentStepStatus.SKIPPED,
                started_at=datetime.utcnow(),
            )
            for index, step in enumerate(steps)
            if index >= start_index
        ]
        return tuple(completed + skipped)
