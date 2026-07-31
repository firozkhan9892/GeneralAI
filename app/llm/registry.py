"""Registry of registered LLM providers."""

from __future__ import annotations

import logging
from typing import Iterator

from app.core.registry.base_registry import BaseRegistry
from app.llm.base import BaseLLMProvider

log = logging.getLogger(__name__)


class ProviderRegistry:
    """Thread-safe registry mapping provider names to instances."""

    def __init__(self) -> None:
        self._registry: BaseRegistry[BaseLLMProvider] = BaseRegistry()

    def register(self, provider: BaseLLMProvider, overwrite: bool = False) -> None:
        """Register a provider instance under its ``name``.

        Args:
            provider: The provider to register.
            overwrite: If ``True``, replace an existing entry.

        Raises:
            ValueError: If the provider name already exists and
                ``overwrite`` is ``False``.
        """
        self._registry.register(provider.name, provider, overwrite=overwrite)
        log.debug("Registered LLM provider '%s'", provider.name)

    def unregister(self, name: str) -> None:
        """Remove a registered provider.

        Args:
            name: Provider name to remove.
        """
        self._registry.unregister(name)
        log.debug("Unregistered LLM provider '%s'", name)

    def clear(self) -> None:
        """Remove all registered providers."""
        self._registry.clear()

    def has(self, name: str) -> bool:
        """Return ``True`` if *name* is registered."""
        return self._registry.has(name)

    def get(self, name: str) -> BaseLLMProvider | None:
        """Return the registered provider, or ``None``."""
        return self._registry.get(name)

    def get_or_raise(self, name: str) -> BaseLLMProvider:
        """Return the registered provider or raise :class:`KeyError`."""
        return self._registry.get_or_raise(name)

    def names(self) -> list[str]:
        """Return all registered provider names."""
        return self._registry.keys()

    def providers(self) -> list[BaseLLMProvider]:
        """Return all registered provider instances."""
        return self._registry.values()

    @property
    def count(self) -> int:
        """Return the number of registered providers."""
        return self._registry.count

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __iter__(self) -> Iterator[BaseLLMProvider]:
        return iter(self.providers())

    def __len__(self) -> int:
        return self.count
