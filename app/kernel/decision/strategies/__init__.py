"""Decision strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.kernel.decision.models import ActionCandidate, Decision


class IDecisionStrategy(ABC):
    """Pluggable strategy for selecting among action candidates."""

    @abstractmethod
    async def select(self, candidates: list[ActionCandidate]) -> Decision:
        """Select the best action from *candidates*.

        Args:
            candidates: List of possible actions to choose from.

        Returns:
            The selected decision.
        """
