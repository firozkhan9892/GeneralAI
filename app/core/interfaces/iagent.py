"""Agent interface placeholder."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from app.core.interfaces.base import IModule


class IAgent(IModule):
    """Contract for agent implementations.

    Agents are autonomous entities that perceive, reason, and act.
    They own a brain, memory, and tools.
    """

    @abstractmethod
    async def run(self, task: str, context: dict[str, Any] | None = None) -> Any:
        """Execute *task* and return the result.

        Args:
            task: Natural-language description of what to do.
            context: Optional contextual key-value pairs.

        Returns:
            The outcome of the agent's work.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Request the agent to stop its current execution gracefully."""
