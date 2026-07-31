"""Factory for constructing LLM provider instances."""

from __future__ import annotations

import logging
from typing import Any, Callable

from app.llm.base import BaseLLMProvider

log = logging.getLogger(__name__)

ProviderBuilder = Callable[..., BaseLLMProvider]


class ProviderFactory:
    """Creates provider instances from a registry of builder callables.

    Built-in builders are registered for ``mock``, ``openai``,
    ``openrouter``, ``gemini``, and ``ollama``.  Additional builders
    can be registered at runtime, making the factory easy to extend
    without modifying its callers.
    """

    def __init__(self) -> None:
        self._builders: dict[str, ProviderBuilder] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register the built-in provider builders."""
        from app.llm.providers.gemini import GeminiProvider
        from app.llm.providers.mock import MockProvider
        from app.llm.providers.ollama import OllamaProvider
        from app.llm.providers.openai import OpenAIProvider
        from app.llm.providers.openrouter import OpenRouterProvider

        self.register("mock", MockProvider)
        self.register("openai", OpenAIProvider)
        self.register("openrouter", OpenRouterProvider)
        self.register("gemini", GeminiProvider)
        self.register("ollama", OllamaProvider)

    def register(self, name: str, builder: ProviderBuilder) -> None:
        """Register a builder callable under *name*.

        Args:
            name: Unique builder name.
            builder: Zero-argument-or-keyword callable returning a
                provider instance.
        """
        self._builders[name] = builder
        log.debug("Registered LLM provider builder '%s'", name)

    def unregister(self, name: str) -> None:
        """Remove a registered builder.

        Args:
            name: Builder name to remove.
        """
        self._builders.pop(name, None)

    def has(self, name: str) -> bool:
        """Return ``True`` if a builder is registered for *name*."""
        return name in self._builders

    def names(self) -> list[str]:
        """Return all registered builder names."""
        return list(self._builders.keys())

    def create(self, name: str, **kwargs: Any) -> BaseLLMProvider:
        """Create a provider instance using the registered builder.

        Args:
            name: Registered builder name.
            **kwargs: Keyword arguments forwarded to the builder.

        Returns:
            A configured provider instance.

        Raises:
            KeyError: If no builder is registered for *name*.
        """
        if name not in self._builders:
            raise KeyError(f"No LLM provider builder registered for '{name}'")
        builder = self._builders[name]
        provider = builder(**kwargs)
        log.debug("Created LLM provider '%s'", name)
        return provider

    def create_mock(self, **kwargs: Any) -> BaseLLMProvider:
        """Create a deterministic mock provider.

        Args:
            **kwargs: Optional overrides forwarded to :class:`MockProvider`.

        Returns:
            A mock provider instance.
        """
        return self.create("mock", **kwargs)

    def create_default(self, name: str = "mock", **kwargs: Any) -> BaseLLMProvider:
        """Create a provider using *name* (``mock`` by default).

        Args:
            name: Registered builder name.
            **kwargs: Keyword arguments forwarded to the builder.

        Returns:
            A configured provider instance.
        """
        return self.create(name, **kwargs)
