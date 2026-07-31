"""Reasoning strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.kernel.reasoning.models import ReasoningRequest, ReasoningTrace


class IReasoningStrategy(ABC):
    """Pluggable reasoning strategy for the ReasoningEngine."""

    @abstractmethod
    async def execute(self, request: ReasoningRequest) -> ReasoningTrace:
        """Execute this reasoning strategy.

        Args:
            request: The reasoning request containing the problem and context.

        Returns:
            A complete reasoning trace.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name."""
