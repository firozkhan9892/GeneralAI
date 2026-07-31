"""Agent runtime — the execution brain of GeneralAI.

The AgentRuntime integrates every completed cognitive engine into a
single, deterministic execution pipeline:

    Percept → Intent → Goal → Plan → [reasoning] → loop → reflection
    → experience → memory → response

It drives plans through the AgentLoop (tool selection via the Decision
Engine, policy gating, Phase-5 ToolExecutor, and memory updates), then
performs reflection, records an experience, and builds the final output.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.kernel.agent.loop import AgentLoop
from app.kernel.agent.models import (
    AgentRequest,
    AgentResponse,
    AgentRunConfig,
    AgentRunSummary,
    AgentStatus,
    AgentStepStatus,
)
from app.kernel.agent.policies import FallbackPolicy, RetryPolicy
from app.kernel.decision.engine import DecisionEngine
from app.kernel.experience.engine import ExperienceEngine
from app.kernel.experience.models import (
    DecisionSummary,
    Experience,
    LessonCategory,
    LessonLearned,
)
from app.kernel.goals.engine import GoalEngine
from app.kernel.intent.engine import IntentEngine
from app.kernel.memory.engine import MemoryEngine
from app.kernel.perception.engine import PerceptionEngine
from app.kernel.perception.models import RawMessage
from app.kernel.planning.engine import PlanningEngine
from app.kernel.policy.engine import PolicyEngine
from app.kernel.reasoning.engine import ReasoningEngine
from app.kernel.reasoning.models import ReasoningRequest
from app.kernel.reflection.engine import ReflectionEngine
from app.kernel.reflection.models import ReflectionRequest
from app.kernel.response.builder import ResponseBuilder
from app.tools.context import CancellationToken
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


class AgentRuntime:
    """Integrates the cognitive engines into a single agent execution.

    All engines are injected (dependency injection friendly) and default
    to deterministic in-memory implementations, so the runtime executes
    fully offline.

    Args:
        perception: PerceptionEngine instance.
        intent: IntentEngine instance.
        goals: GoalEngine instance.
        planning: PlanningEngine instance.
        reasoning: ReasoningEngine instance.
        decision: DecisionEngine instance.
        policy: PolicyEngine instance.
        memory: MemoryEngine instance.
        experience: ExperienceEngine instance.
        reflection: ReflectionEngine instance.
        response: ResponseBuilder instance.
        tool_registry: Phase-5 ToolRegistry.
        tool_executor: Phase-5 ToolExecutor.
        loop: Optional AgentLoop (built from engines if omitted).
        retry_policy: Retry policy used by the loop.
        fallback_policy: Fallback policy used by the loop.
        config: Default run configuration.
    """

    def __init__(
        self,
        *,
        perception: PerceptionEngine | None = None,
        intent: IntentEngine | None = None,
        goals: GoalEngine | None = None,
        planning: PlanningEngine | None = None,
        reasoning: ReasoningEngine | None = None,
        decision: DecisionEngine | None = None,
        policy: PolicyEngine | None = None,
        memory: MemoryEngine | None = None,
        experience: ExperienceEngine | None = None,
        reflection: ReflectionEngine | None = None,
        response: ResponseBuilder | None = None,
        tool_registry: ToolRegistry | None = None,
        tool_executor: ToolExecutor | None = None,
        loop: AgentLoop | None = None,
        retry_policy: RetryPolicy | None = None,
        fallback_policy: FallbackPolicy | None = None,
        config: AgentRunConfig | None = None,
    ) -> None:
        self._perception = perception or PerceptionEngine()
        self._intent = intent or IntentEngine()
        self._goals = goals or GoalEngine()
        self._planning = planning or PlanningEngine()
        self._reasoning = reasoning or ReasoningEngine()
        self._decision = decision or DecisionEngine()
        self._policy = policy or PolicyEngine()
        self._memory = memory or MemoryEngine()
        self._experience = experience or ExperienceEngine()
        self._reflection = reflection or ReflectionEngine()
        self._response = response or ResponseBuilder()
        self._retry_policy = retry_policy or RetryPolicy()
        self._fallback_policy = fallback_policy or FallbackPolicy()

        self._registry = tool_registry if tool_registry is not None else ToolRegistry()
        self._executor = (
            tool_executor
            if tool_executor is not None
            else ToolExecutor(registry=tool_registry)
        )

        self._loop = loop or AgentLoop(
            decision_engine=self._decision,
            policy_engine=self._policy,
            tool_executor=self._executor,
            tool_registry=self._registry,
            memory_engine=self._memory,
            retry_policy=self._retry_policy,
            fallback_policy=self._fallback_policy,
        )
        self._config = config or AgentRunConfig()
        self._active_sessions: dict[str, CancellationToken] = {}

    # ── Public API ────────────────────────────────────────────────────

    async def run(
        self,
        request: AgentRequest,
        *,
        config: AgentRunConfig | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> AgentResponse:
        """Execute a full agent run.

        Args:
            request: The agent request (raw input + session).
            config: Optional per-run config override.
            cancellation_token: Optional cooperative cancellation signal.

        Returns:
            The final agent response.

        Raises:
            asyncio.CancelledError: If execution is cancelled.
            TimeoutError: If the overall run deadline is exceeded.
        """
        cfg = config or request.config or self._config
        if cfg.session_id == "" and request.session_id != "":
            cfg = cfg.model_copy(update={"session_id": request.session_id})

        token = cancellation_token or CancellationToken()
        self._active_sessions[cfg.session_id] = token

        started = time.monotonic()
        try:
            response = await asyncio.wait_for(
                self._run_internal(request, cfg, token),
                timeout=cfg.overall_timeout_s,
            )
        except asyncio.CancelledError:
            token.cancel()
            log.info("Agent run cancelled: session=%s", cfg.session_id)
            return self._failure_response(
                cfg,
                AgentStatus.CANCELLED,
                "Agent run cancelled",
                started,
            )
        except asyncio.TimeoutError:
            log.warning("Agent run timed out after %ss", cfg.overall_timeout_s)
            return self._failure_response(
                cfg,
                AgentStatus.TIMED_OUT,
                f"Agent run timed out after {cfg.overall_timeout_s}s",
                started,
            )
        finally:
            self._active_sessions.pop(cfg.session_id, None)

        summary = self._build_summary(response.steps, started)
        return response.model_copy(update={"summary": summary})

    def cancel(self, session_id: str, reason: str = "user_requested") -> None:
        """Request cancellation of an active session.

        Args:
            session_id: The session to cancel.
            reason: Cancellation reason.
        """
        token = self._active_sessions.get(session_id)
        if token is not None:
            token.cancel()
            log.info("Cancelled agent session %s (%s)", session_id, reason)
        else:
            log.warning("Cancel requested for unknown session %s", session_id)

    # ── Internal execution ────────────────────────────────────────────

    async def _run_internal(
        self,
        request: AgentRequest,
        config: AgentRunConfig,
        token: CancellationToken,
    ) -> AgentResponse:
        session_id = config.session_id

        # 1. Perception
        raw = RawMessage(content=request.raw_input)
        percept = await self._perception.perceive(raw, session_id=session_id)

        # 2. Intent
        intent = await self._intent.understand(percept)

        # 3. Goal
        hierarchy = await self._goals.resolve(intent)

        # 4. Plan
        plan = await self._planning.create_plan(hierarchy.root)

        # 5. Reasoning (optional)
        reasoning_trace = None
        if config.reasoning_enabled:
            reasoning_trace = await self._reasoning.reason(
                ReasoningRequest(
                    problem=percept.normalized_content or request.raw_input,
                    context={"session_id": session_id, "intent": intent.primary.value},
                )
            )

        # 6. Execute plan via the agent loop
        steps = await self._loop.execute(
            plan,
            config=config,
            cancellation_token=token,
        )

        # 7. Reflection (optional)
        reflection_report = None
        if config.reflection_enabled:
            reflection_report = await self._reflection.evaluate(
                ReflectionRequest(
                    output=[s.tool_result for s in steps],
                    trace=reasoning_trace,
                    context={"session_id": session_id},
                    mode="standard",
                )
            )

        # 8. Experience (optional)
        experience = None
        if config.experience_enabled:
            experience = await self._record_experience(
                session_id, intent, hierarchy, plan, steps, reasoning_trace
            )

        # 9. Memory summary
        memory_summary = await self._memory.summarize()

        # 10. Response
        output = await self._response.build(
            {
                "session_id": session_id,
                "percept": percept,
                "intent": intent,
                "goal_hierarchy": hierarchy,
                "plan": plan,
                "reasoning_trace": reasoning_trace,
                "decision": steps[-1].decision if steps else None,
                "policy_verdict": steps[-1].policy_verdict if steps else None,
                "reflection": reflection_report,
                "experience": experience,
            }
        )

        succeeded = (
            all(s.status == AgentStepStatus.SUCCEEDED for s in steps)
            if steps
            else False
        )
        if token.is_cancelled:
            log.info("Agent run cancelled: session=%s", session_id)
            return AgentResponse(
                success=False,
                status=AgentStatus.CANCELLED,
                output=output,
                session_id=session_id,
                intent=intent,
                goal_hierarchy=hierarchy,
                plan=plan,
                reasoning_trace=reasoning_trace,
                reflection_report=reflection_report,
                memory_summary=memory_summary,
                experience=experience,
                steps=steps,
                error="Agent run cancelled",
            )

        status = AgentStatus.SUCCEEDED if succeeded else AgentStatus.FAILED
        error = None
        if not succeeded and steps:
            first_failed = next(
                (s for s in steps if s.status == AgentStepStatus.FAILED), None
            )
            if first_failed is not None:
                error = first_failed.error

        return AgentResponse(
            success=succeeded,
            status=status,
            output=output,
            session_id=session_id,
            intent=intent,
            goal_hierarchy=hierarchy,
            plan=plan,
            reasoning_trace=reasoning_trace,
            reflection_report=reflection_report,
            memory_summary=memory_summary,
            experience=experience,
            steps=steps,
            error=error,
        )

    # ── Experience recording ──────────────────────────────────────────

    async def _record_experience(
        self,
        session_id: str,
        intent: Any,
        hierarchy: Any,
        plan: Any,
        steps: tuple[Any, ...],
        reasoning_trace: Any,
    ) -> Experience:
        succeeded = [s for s in steps if s.status == AgentStepStatus.SUCCEEDED]
        decisions = [
            DecisionSummary(
                action_type=s.tool_name or s.skill_name,
                confidence=1.0,
                success=True,
            )
            for s in succeeded
        ]
        tools_used = tuple(sorted({s.tool_name for s in steps if s.tool_name}))
        skills_used = tuple(s.skill_name for s in plan.steps)

        lessons: tuple[LessonLearned, ...] = ()
        if reasoning_trace is not None and reasoning_trace.conclusion:
            lessons = (
                LessonLearned(
                    description=reasoning_trace.conclusion,
                    category=LessonCategory.STRATEGY,
                    applicability=(intent.primary,),
                    confidence=0.8,
                ),
            )

        experience = Experience(
            session_id=session_id,
            goal_type=intent.primary,
            goal_description=hierarchy.root.description,
            plan_summary=f"{len(plan.steps)} step(s)",
            skills_used=skills_used,
            tools_used=tools_used,
            decisions=tuple(decisions),
            outcome_score=1.0 if succeeded else 0.0,
            success=bool(succeeded) and len(succeeded) == len(steps),
            failure_reason=(
                None if len(succeeded) == len(steps) else "Some plan steps failed"
            ),
            lessons=lessons,
            token_cost=reasoning_trace.token_cost if reasoning_trace else 0,
        )
        await self._experience.record(experience)
        return experience

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_summary(steps: tuple[Any, ...], started: float) -> AgentRunSummary:
        tools = sorted({s.tool_name for s in steps if s.tool_name})
        return AgentRunSummary(
            total_steps=len(steps),
            succeeded=sum(1 for s in steps if s.status == AgentStepStatus.SUCCEEDED),
            failed=sum(1 for s in steps if s.status == AgentStepStatus.FAILED),
            skipped=sum(1 for s in steps if s.status == AgentStepStatus.SKIPPED),
            retries=sum(s.retries for s in steps),
            tools_invoked=tuple(tools),
            memory_records=sum(1 for s in steps if s.memory_record_id is not None),
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    @staticmethod
    def _failure_response(
        config: AgentRunConfig,
        status: AgentStatus,
        message: str,
        started: float,
    ) -> AgentResponse:
        return AgentResponse(
            success=False,
            status=status,
            session_id=config.session_id,
            error=message,
            summary=AgentRunSummary(
                duration_ms=int((time.monotonic() - started) * 1000)
            ),
        )

    # ── Introspection ─────────────────────────────────────────────────

    def get_active_sessions(self) -> list[str]:
        """Return currently active session identifiers."""
        return list(self._active_sessions.keys())

    @property
    def loop(self) -> AgentLoop:
        return self._loop

    @property
    def memory(self) -> MemoryEngine:
        return self._memory

    @property
    def experience(self) -> ExperienceEngine:
        return self._experience

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def config(self) -> AgentRunConfig:
        return self._config
