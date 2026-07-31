"""Reasoning engine — stage 5 of the cognitive pipeline."""

from __future__ import annotations

import logging

from app.kernel.reasoning.models import (
    ReasoningRequest,
    ReasoningStep,
    ReasoningStrategy,
    ReasoningTrace,
    StepType,
)
from app.kernel.reasoning.strategies import IReasoningStrategy
from app.kernel.reasoning.strategies.base import (
    ChainOfThoughtStrategy,
    DecompositionStrategy,
    FallbackStrategy,
)

log = logging.getLogger(__name__)

_STRATEGY_MAP: dict[ReasoningStrategy, type[IReasoningStrategy]] = {
    ReasoningStrategy.CHAIN_OF_THOUGHT: ChainOfThoughtStrategy,
    ReasoningStrategy.DECOMPOSITION: DecompositionStrategy,
}


class ReasoningEngine:
    """Core thinking component.

    Produces structured reasoning traces using pluggable strategies.
    """

    def __init__(self) -> None:
        self._strategies: dict[str, IReasoningStrategy] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        for strat_type, strat_cls in _STRATEGY_MAP.items():
            instance = strat_cls()
            self._strategies[strat_type.value] = instance

    async def reason(self, request: ReasoningRequest) -> ReasoningTrace:
        """Execute reasoning using the appropriate strategy.

        Args:
            request: The reasoning request.

        Returns:
            A complete reasoning trace.
        """
        strategy = self._select_strategy(request)
        trace = await strategy.execute(request)

        log.info(
            "Reasoning complete — strategy=%s, steps=%d, tokens=%d",
            trace.strategy_used.value,
            len(trace.steps),
            trace.token_cost,
        )

        return trace

    def _select_strategy(self, request: ReasoningRequest) -> IReasoningStrategy:
        strategy_name = request.strategy.value
        strategy = self._strategies.get(strategy_name)
        if strategy is not None:
            return strategy
        log.warning("Strategy '%s' not found, using fallback", strategy_name)
        return FallbackStrategy()

    def register_strategy(self, name: str, strategy: IReasoningStrategy) -> None:
        """Register a reasoning strategy.

        Args:
            name: Strategy name.
            strategy: Strategy implementation.
        """
        self._strategies[name] = strategy
        log.debug("Registered reasoning strategy: %s", name)

    def unregister_strategy(self, name: str) -> None:
        """Remove a registered strategy.

        Args:
            name: Strategy name to remove.
        """
        self._strategies.pop(name, None)
        log.debug("Unregistered reasoning strategy: %s", name)

    async def refine(self, trace: ReasoningTrace, feedback: str) -> ReasoningTrace:
        """Refine a reasoning trace based on feedback.

        Creates a new trace with the feedback incorporated as an
        additional evaluate step.

        Args:
            trace: The original reasoning trace.
            feedback: Feedback to guide refinement.

        Returns:
            Refined reasoning trace.
        """
        new_step = ReasoningStep(
            id="refine_step",
            type=StepType.EVALUATE,
            content=f"Refinement feedback: {feedback}",
        )

        refined_steps = list(trace.steps) + [new_step]

        log.info(
            "Trace refined — original steps=%d, refined steps=%d",
            len(trace.steps),
            len(refined_steps),
        )

        return ReasoningTrace(
            steps=tuple(refined_steps),
            conclusion=trace.conclusion,
            strategy_used=trace.strategy_used,
            token_cost=trace.token_cost + 5,
            metadata={"refined": True, "feedback": feedback},
        )
