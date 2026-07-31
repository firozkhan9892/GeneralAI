"""Registry of registered tools."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Iterator

from app.core.registry.base_registry import BaseRegistry
from app.tools.base import Tool
from app.tools.exceptions import ToolNotFoundError
from app.tools.models import ToolCategory, ToolMetadata

log = logging.getLogger(__name__)


class ToolRegistry:
    """Thread-safe registry mapping tool names to instances."""

    def __init__(self) -> None:
        self._registry: BaseRegistry[Tool] = BaseRegistry()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, tool: Tool, overwrite: bool = False) -> None:
        """Register a tool instance under its ``name``.

        Args:
            tool: The tool to register.
            overwrite: If ``True``, replace an existing entry.

        Raises:
            ValueError: If the tool name already exists and ``overwrite``
                is ``False``.
        """
        self._registry.register(tool.name, tool, overwrite=overwrite)
        log.debug("Registered tool '%s'", tool.name)

    def unregister(self, name: str) -> None:
        """Remove a registered tool.

        Args:
            name: Tool name to remove.
        """
        self._registry.unregister(name)
        log.debug("Unregistered tool '%s'", name)

    def clear(self) -> None:
        """Remove all registered tools."""
        self._registry.clear()

    def discover(
        self,
        tools: Iterable[Tool] | None = None,
        *,
        category: ToolCategory | None = None,
    ) -> int:
        """Register the default tool set (optionally filtered).

        Args:
            tools: Tools to register; defaults to the built-in catalogue.
            category: If given, only register tools in this category.

        Returns:
            The number of tools registered.
        """
        if tools is None:
            from app.tools.catalog import DEFAULT_TOOLS

            tools = (
                DEFAULT_TOOLS
                if category is None
                else (t for t in DEFAULT_TOOLS if t.category == category)
            )
        count = 0
        for tool in tools:
            self.register(tool, overwrite=True)
            count += 1
        log.debug("Discovered %d tools", count)
        return count

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def has(self, name: str) -> bool:
        """Return ``True`` if *name* is registered."""
        return self._registry.has(name)

    def get(self, name: str) -> Tool | None:
        """Return the registered tool, or ``None``."""
        return self._registry.get(name)

    def get_or_raise(self, name: str) -> Tool:
        """Return the registered tool or raise :class:`ToolNotFoundError`."""
        try:
            return self._registry.get_or_raise(name)
        except KeyError as exc:
            raise ToolNotFoundError(
                f"Tool '{name}' is not registered",
                module="tools.registry",
                cause=exc,
            ) from exc

    def names(self) -> list[str]:
        """Return all registered tool names."""
        return self._registry.keys()

    def tools(self) -> list[Tool]:
        """Return all registered tool instances."""
        return self._registry.values()

    def list_tools(self, category: ToolCategory | None = None) -> list[ToolMetadata]:
        """Return metadata for registered tools, optionally filtered.

        Args:
            category: If given, only return tools in this category.

        Returns:
            Tool descriptors.
        """
        return [
            tool.metadata
            for tool in self._registry.values()
            if category is None or tool.category == category
        ]

    @property
    def count(self) -> int:
        """Return the number of registered tools."""
        return self._registry.count

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self.tools())

    def __len__(self) -> int:
        return self.count
