"""Stage definitions — maps pipeline stages to engine dispatch logic.

Each ``StageDefinition`` describes how to:
    1. Build a contract request from the pipeline context.
    2. Call the correct method on the correct engine.
    3. Extract the result and store it back in the pipeline context.

This is the central wiring that makes the orchestrator work without
engines calling each other directly.
"""

from __future__ import annotations

from typing import Any

from app.kernel.contracts.base import EngineType
from app.kernel.decision.engine import DecisionEngine
from app.kernel.experience.engine import ExperienceEngine
from app.kernel.experience.models import Experience
from app.kernel.goals.engine import GoalEngine
from app.kernel.intent.engine import IntentEngine
from app.kernel.intent.models import IntentType
from app.kernel.perception.engine import PerceptionEngine
from app.kernel.planning.engine import PlanningEngine
from app.kernel.policy.engine import PolicyEngine
from app.kernel.reasoning.engine import ReasoningEngine
from app.kernel.reasoning.models import ReasoningRequest
from app.kernel.reflection.engine import ReflectionEngine
from app.kernel.reflection.models import ReflectionRequest
from app.kernel.response.builder import ResponseBuilder
from app.kernel.pipeline.dispatcher import StageDefinition


def _extract_response_field(response: Any, field_name: str) -> Any:
    if response is None:
        return None
    if hasattr(response, field_name):
        return getattr(response, field_name)
    return response


def build_stage_definitions() -> list[StageDefinition]:
    """Build the ordered list of stage definitions for the cognitive pipeline."""
    stages: list[StageDefinition] = []

    # Stage 1: Perception → Percept
    stages.append(
        StageDefinition(
            engine_type=EngineType.PERCEPTION,
            name="perception",
            engine_attr="perception",
            method_name="perceive",
            request_field="percept",
            response_field="percept",
            request_builder=lambda ctx, ec: ctx.percept,
            response_extractor=lambda resp, ctx: resp,
            engine_class=PerceptionEngine,
        )
    )

    # Stage 2: Intent
    stages.append(
        StageDefinition(
            engine_type=EngineType.INTENT,
            name="intent",
            engine_attr="intent",
            method_name="understand",
            request_field="percept",
            response_field="intent",
            request_builder=lambda ctx, ec: ctx.percept,
            response_extractor=lambda resp, ctx: resp,
            engine_class=IntentEngine,
        )
    )

    # Stage 3: Goal
    stages.append(
        StageDefinition(
            engine_type=EngineType.GOAL,
            name="goal",
            engine_attr="goals",
            method_name="resolve",
            request_field="intent",
            response_field="goal_hierarchy",
            request_builder=lambda ctx, ec: ctx.intent,
            response_extractor=lambda resp, ctx: resp,
            engine_class=GoalEngine,
        )
    )

    # Stage 4: Planning
    stages.append(
        StageDefinition(
            engine_type=EngineType.PLANNER,
            name="planning",
            engine_attr="planning",
            method_name="create_plan",
            request_field="goal_hierarchy",
            response_field="plan",
            request_builder=lambda ctx, ec: (
                ctx.goal_hierarchy.root if ctx.goal_hierarchy else None
            ),
            response_extractor=lambda resp, ctx: resp,
            engine_class=PlanningEngine,
        )
    )

    # Stage 5: Reasoning
    stages.append(
        StageDefinition(
            engine_type=EngineType.REASONING,
            name="reasoning",
            engine_attr="reasoning",
            method_name="reason",
            request_field="plan",
            response_field="reasoning_trace",
            request_builder=lambda ctx, ec: ReasoningRequest(
                problem=(
                    ctx.goal_hierarchy.root.description
                    if ctx.goal_hierarchy and ctx.goal_hierarchy.root
                    else str(ctx.plan.goal_id if ctx.plan else "Unknown")
                )
            ),
            response_extractor=lambda resp, ctx: resp,
            engine_class=ReasoningEngine,
        )
    )

    # Stage 6: Decision
    stages.append(
        StageDefinition(
            engine_type=EngineType.DECISION,
            name="decision",
            engine_attr="decision",
            method_name="decide",
            request_field="plan",
            response_field="decision",
            request_builder=lambda ctx, ec: ctx.plan,
            response_extractor=lambda resp, ctx: resp,
            engine_class=DecisionEngine,
        )
    )

    # Stage 7: Policy
    stages.append(
        StageDefinition(
            engine_type=EngineType.POLICY,
            name="policy",
            engine_attr="policy",
            method_name="evaluate",
            request_field="decision",
            response_field="policy_verdict",
            request_builder=lambda ctx, ec: ctx.decision,
            response_extractor=lambda resp, ctx: resp,
            engine_class=PolicyEngine,
        )
    )

    # Stage 8: Reflection
    stages.append(
        StageDefinition(
            engine_type=EngineType.REFLECTION,
            name="reflection",
            engine_attr="reflection",
            method_name="evaluate",
            request_field="reasoning_trace",
            response_field="reflection",
            request_builder=lambda ctx, ec: ReflectionRequest(
                output=ctx.reasoning_trace.conclusion if ctx.reasoning_trace else "",
                trace=ctx.reasoning_trace,
                mode="fallback",
            ),
            response_extractor=lambda resp, ctx: resp,
            engine_class=ReflectionEngine,
        )
    )

    # Stage 9: Experience
    stages.append(
        StageDefinition(
            engine_type=EngineType.EXPERIENCE,
            name="experience",
            engine_attr="experience",
            method_name="record",
            request_field="reflection",
            response_field="experience",
            request_builder=lambda ctx, ec: Experience(
                goal_type=ctx.intent.primary if ctx.intent else IntentType.UNKNOWN,
                goal_description=(
                    ctx.goal_hierarchy.root.description
                    if ctx.goal_hierarchy and ctx.goal_hierarchy.root
                    else ""
                ),
                outcome_score=ctx.reflection.overall_score if ctx.reflection else 0.0,
                success=ctx.reflection.verdict == "pass" if ctx.reflection else True,
                session_id=ec.session_id,
            ),
            response_extractor=lambda resp, ctx: resp,
            engine_class=ExperienceEngine,
        )
    )

    # Stage 10: Response
    stages.append(
        StageDefinition(
            engine_type=EngineType.RESPONSE,
            name="response",
            engine_attr="response",
            method_name="build",
            request_field="",
            response_field="response",
            request_builder=lambda ctx, ec: ctx,
            response_extractor=lambda resp, ctx: resp,
            engine_class=ResponseBuilder,
        )
    )

    return stages
