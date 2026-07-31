"""Planner interface placeholder."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from app.core.interfaces.base import IModule


class IPlanner(IModule):
    """Contract for task planning implementations.

    Planners decompose high-level goals into sequences of executable
    steps and handle re-planning on failure.
    """

    @abstractmethod
    async def plan(
        self, goal: str, context: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Decompose *goal* into a list of steps.

        Args:
            goal: The high-level objective to plan for.
            context: Optional contextual key-value pairs.

        Returns:
            Ordered list of step descriptors (each step is a dict
            with at least ``action`` and ``parameters`` keys).
        """

    @abstractmethod
    async def replan(
        self, goal: str, feedback: str, previous_plan: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Revise an existing plan based on feedback.

        Args:
            goal: The original objective.
            feedback: Description of what went wrong.
            previous_plan: The plan that failed.

        Returns:
            Revised list of steps.
        """
