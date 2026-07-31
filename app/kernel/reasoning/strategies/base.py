"""Reasoning strategy implementations."""

from __future__ import annotations

from app.kernel.reasoning.models import (
    ReasoningRequest,
    ReasoningStep,
    ReasoningStrategy,
    ReasoningTrace,
    StepType,
)
from app.kernel.reasoning.strategies import IReasoningStrategy


class ChainOfThoughtStrategy(IReasoningStrategy):
    """Deterministic chain-of-thought reasoning strategy.

    Produces a fixed sequence of reasoning steps based on the
    problem statement, without any LLM or external API calls.
    """

    def __init__(self) -> None:
        self._name = ReasoningStrategy.CHAIN_OF_THOUGHT.value

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, request: ReasoningRequest) -> ReasoningTrace:
        steps: list[ReasoningStep] = []
        step_id = 0

        steps.append(
            ReasoningStep(
                id=f"{self._name}_step_{step_id}",
                type=StepType.THINK,
                content=f"Analyzing problem: {request.problem}",
            )
        )
        step_id += 1

        if request.context:
            keys = list(request.context.keys())
            steps.append(
                ReasoningStep(
                    id=f"{self._name}_step_{step_id}",
                    type=StepType.OBSERVE,
                    content=f"Context provided with keys: {', '.join(keys)}",
                )
            )
            step_id += 1

        if request.constraints:
            constraints_desc = "; ".join(
                f"{k}: {v}" for k, v in request.constraints.items()
            )
            steps.append(
                ReasoningStep(
                    id=f"{self._name}_step_{step_id}",
                    type=StepType.EVALUATE,
                    content=f"Constraints: {constraints_desc}",
                )
            )
            step_id += 1

        steps.append(
            ReasoningStep(
                id=f"{self._name}_step_{step_id}",
                type=StepType.THINK,
                content="Working through solution step by step",
            )
        )
        step_id += 1

        steps.append(
            ReasoningStep(
                id=f"{self._name}_step_{step_id}",
                type=StepType.ACT,
                content=f"Conclude based on analysis of: {request.problem[:100]}",
            )
        )

        token_cost = len(steps) * 10

        return ReasoningTrace(
            steps=tuple(steps),
            conclusion=f"Deterministic reasoning complete for: {request.problem}",
            strategy_used=ReasoningStrategy.CHAIN_OF_THOUGHT,
            token_cost=token_cost,
        )


class DecompositionStrategy(IReasoningStrategy):
    """Deterministic decomposition reasoning strategy.

    Breaks the problem into sub-problems and solves each.
    """

    def __init__(self) -> None:
        self._name = ReasoningStrategy.DECOMPOSITION.value

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, request: ReasoningRequest) -> ReasoningTrace:
        words = request.problem.split()
        chunk_size = max(1, len(words) // 3) if len(words) > 3 else len(words)

        steps: list[ReasoningStep] = []
        step_id = 0

        steps.append(
            ReasoningStep(
                id=f"{self._name}_step_{step_id}",
                type=StepType.THINK,
                content="Decomposing problem into sub-problems",
            )
        )
        step_id += 1

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            steps.append(
                ReasoningStep(
                    id=f"{self._name}_step_{step_id}",
                    type=StepType.THINK,
                    content=f"Sub-problem: {chunk}",
                )
            )
            step_id += 1

        steps.append(
            ReasoningStep(
                id=f"{self._name}_step_{step_id}",
                type=StepType.ACT,
                content=f"Combined {len(words)} concepts into cohesive solution",
            )
        )

        return ReasoningTrace(
            steps=tuple(steps),
            conclusion=f"Decomposed and analyzed: {request.problem[:100]}",
            strategy_used=ReasoningStrategy.DECOMPOSITION,
            token_cost=len(steps) * 10,
        )


class FallbackStrategy(IReasoningStrategy):
    """Simple fallback strategy when no specific strategy is configured.

    Produces minimal reasoning trace.
    """

    def __init__(self) -> None:
        self._name = "fallback"

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, request: ReasoningRequest) -> ReasoningTrace:
        steps = (
            ReasoningStep(
                id="fallback_step_0",
                type=StepType.THINK,
                content=f"Processing: {request.problem}",
            ),
            ReasoningStep(
                id="fallback_step_1",
                type=StepType.ACT,
                content="No specific strategy configured — using fallback",
            ),
        )

        return ReasoningTrace(
            steps=steps,
            conclusion=f"Fallback reasoning for: {request.problem}",
            strategy_used=request.strategy,
            token_cost=20,
        )
