"""Decision criterion interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.kernel.decision.models import ActionCandidate
from app.kernel.context.models import CognitiveContext


class DecisionCriterion(ABC):
    """Evaluates a single dimension of an action candidate."""

    @abstractmethod
    async def score(
        self, candidate: ActionCandidate, context: CognitiveContext
    ) -> float:
        """Score *candidate* on this criterion.

        Args:
            candidate: The action candidate to evaluate.
            context: The current cognitive context.

        Returns:
            A score between 0.0 and 1.0.
        """
