"""Decision engine — stage 6 of the cognitive pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.kernel.decision.criteria import DecisionCriterion
from app.kernel.decision.models import (
    ActionCandidate,
    Decision,
    DecisionReason,
    DecisionScore,
)
from app.kernel.decision.strategies import IDecisionStrategy
from app.kernel.planning.models import Plan

if TYPE_CHECKING:
    from app.kernel.context.models import CognitiveContext
    from app.kernel.reasoning.models import ReasoningTrace

log = logging.getLogger(__name__)


def _step_to_candidate(
    step_order: int, skill_name: str, description: str
) -> ActionCandidate:
    return ActionCandidate(
        action_type="skill_call",
        description=description or f"Execute step {step_order}: {skill_name}",
        parameters={"skill_name": skill_name, "order": step_order},
        confidence=1.0 / (step_order + 1),
        estimated_cost=0,
        source="planning",
    )


def _rank_candidates(
    candidates: list[ActionCandidate],
) -> list[ActionCandidate]:
    return sorted(
        candidates,
        key=lambda c: (-c.confidence, c.estimated_cost),
    )


def _build_decision_scores(
    candidates: list[ActionCandidate],
    selected_index: int,
) -> tuple[DecisionScore, ...]:
    return (
        DecisionScore(
            criterion_name="confidence",
            score=candidates[selected_index].confidence,
            weight=1.0,
            rationale=(
                f"Selected '{candidates[selected_index].action_type}' "
                f"with confidence {candidates[selected_index].confidence:.2f}"
            ),
        ),
    )


def _build_decision_reason(
    candidates: list[ActionCandidate],
    selected_index: int,
) -> DecisionReason:
    trade_offs: list[str] = []
    for i, c in enumerate(candidates):
        if i != selected_index:
            trade_offs.append(
                f"Rejected '{c.action_type}' (confidence {c.confidence:.2f})"
            )

    return DecisionReason(
        primary_rationale=f"Selected action with highest confidence: "
        f"'{candidates[selected_index].action_type}'",
        criteria_scores=_build_decision_scores(candidates, selected_index),
        trade_offs=tuple(trade_offs),
    )


class DecisionEngine:
    """Makes choices based on reasoning output and context.

    Scores candidates against criteria and selects the best action.
    """

    def __init__(self) -> None:
        self._criteria: dict[str, DecisionCriterion] = {}
        self._strategies: dict[str, IDecisionStrategy] = {}

    async def decide(self, plan: Plan) -> Decision:
        """Select the next action from a plan.

        Each plan step becomes an action candidate; the highest-confidence
        actionable step is selected.

        Args:
            plan: The plan to decide upon.

        Returns:
            A Decision selecting the best next action.
        """
        if not plan.steps:
            candidates = [
                ActionCandidate(
                    action_type="noop",
                    description="No steps available in plan",
                    confidence=0.0,
                    source="planning",
                )
            ]
            return Decision(
                selected_action=candidates[0],
                candidates=(candidates[0],),
                reason=DecisionReason(
                    primary_rationale="Plan has no steps — no-op selected",
                ),
                strategy_used="greedy",
                status="pending",
            )

        candidates = [
            _step_to_candidate(step.order, step.skill_name, step.description)
            for step in plan.steps
        ]

        return await self._select_from_candidates(candidates)

    async def _select_from_candidates(
        self, candidates: list[ActionCandidate]
    ) -> Decision:
        ranked = _rank_candidates(candidates)
        selected = ranked[0]
        selected_index = candidates.index(selected)

        decision = Decision(
            selected_action=selected,
            candidates=tuple(candidates),
            reason=_build_decision_reason(candidates, selected_index),
            strategy_used="greedy",
            status="pending",
        )

        log.info(
            "Decision: %s (confidence=%.2f, %d candidate(s))",
            selected.action_type,
            selected.confidence,
            len(candidates),
        )

        return decision

    async def rank_candidates(
        self, candidates: list[ActionCandidate]
    ) -> list[ActionCandidate]:
        """Rank action candidates by priority.

        Sorts by confidence descending, then estimated cost ascending.

        Args:
            candidates: List of candidates to rank.

        Returns:
            Ranked list (highest priority first).
        """
        ranked = _rank_candidates(candidates)

        log.debug("Ranked %d candidate(s)", len(ranked))

        return ranked

    def register_criterion(self, name: str, criterion: DecisionCriterion) -> None:
        """Register a decision criterion.

        Args:
            name: Criterion name.
            criterion: Criterion implementation.
        """
        self._criteria[name] = criterion

    # ── Backward-compatible alias ───────────────────────────────────────

    async def evaluate(
        self, trace: ReasoningTrace, context: CognitiveContext
    ) -> Decision:
        """Evaluate reasoning output and select an action.

        Delegates to a simple confidence-based decision for now.
        """
        candidates = [
            ActionCandidate(
                action_type="respond",
                description="Respond based on reasoning trace",
                confidence=0.7,
                source="reasoning",
            )
        ]
        return await self._select_from_candidates(candidates)
