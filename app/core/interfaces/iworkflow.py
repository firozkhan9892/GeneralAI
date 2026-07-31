"""Workflow interface placeholder."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from app.core.interfaces.base import IModule


class IWorkflow(IModule):
    """Contract for workflow automation implementations.

    Workflows are directed acyclic graphs (DAGs) of steps executed
    in sequence or parallel.
    """

    @abstractmethod
    async def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run the workflow with the given *inputs*.

        Args:
            inputs: Initial key-value pairs for the workflow.

        Returns:
            The final output key-value pairs after all steps.
        """

    @property
    @abstractmethod
    def steps(self) -> list[dict[str, Any]]:
        """Return the workflow's step definitions."""
