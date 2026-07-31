"""Tool interface placeholder."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from app.core.interfaces.base import IModule


class ITool(IModule):
    """Contract for tool implementations.

    Tools are callable capabilities (e.g. web search, code execution,
    file I/O) that the agent can invoke.
    """

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with the given parameters.

        Args:
            **kwargs: Tool-specific parameters.

        Returns:
            The tool's output.
        """

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of what this tool does."""

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """Return the JSON Schema for this tool's parameters."""
