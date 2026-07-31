"""Tests for ProviderFactory."""

from __future__ import annotations

import pytest

from app.llm.factory import ProviderFactory
from app.llm.providers import (
    GeminiProvider,
    MockProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
)


@pytest.fixture
def factory() -> ProviderFactory:
    return ProviderFactory()


class TestProviderFactoryBuiltins:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("mock", MockProvider),
            ("openai", OpenAIProvider),
            ("openrouter", OpenRouterProvider),
            ("gemini", GeminiProvider),
            ("ollama", OllamaProvider),
        ],
    )
    def test_create_builtin(
        self, factory: ProviderFactory, name: str, expected: type
    ) -> None:
        provider = factory.create(name)
        assert isinstance(provider, expected)

    def test_create_mock(self, factory: ProviderFactory) -> None:
        provider = factory.create_mock()
        assert isinstance(provider, MockProvider)

    def test_create_default_is_mock(self, factory: ProviderFactory) -> None:
        assert isinstance(factory.create_default(), MockProvider)

    def test_create_default_custom(self, factory: ProviderFactory) -> None:
        provider = factory.create_default("ollama")
        assert isinstance(provider, OllamaProvider)

    def test_names_include_builtins(self, factory: ProviderFactory) -> None:
        names = factory.names()
        assert "mock" in names
        assert "openai" in names
        assert "openrouter" in names
        assert "gemini" in names
        assert "ollama" in names


class TestProviderFactoryCustom:
    def test_register_and_create(self, factory: ProviderFactory) -> None:
        factory.register("custom", MockProvider)
        provider = factory.create("custom")
        assert isinstance(provider, MockProvider)

    def test_create_unknown_raises(self, factory: ProviderFactory) -> None:
        with pytest.raises(KeyError):
            factory.create("unknown")

    def test_unregister(self, factory: ProviderFactory) -> None:
        factory.register("custom", MockProvider)
        assert factory.has("custom")
        factory.unregister("custom")
        assert factory.has("custom") is False

    def test_create_kwargs_forwarded(self, factory: ProviderFactory) -> None:
        provider = factory.create("mock", model="custom-model")
        assert provider.default_model == "custom-model"
