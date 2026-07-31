"""Reflection strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.kernel.reflection.models import ReflectionReport, ReflectionRequest


class IReflectionStrategy(ABC):
    """Pluggable reflection mode for the ReflectionEngine."""

    @abstractmethod
    async def evaluate(self, request: ReflectionRequest) -> ReflectionReport:
        """Evaluate output quality using this reflection strategy.

        Args:
            request: The reflection request containing output and trace.

        Returns:
            Reflection result with scores and errors.
        """
