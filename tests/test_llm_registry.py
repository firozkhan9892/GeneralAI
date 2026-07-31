"""Tests for ProviderRegistry."""

from __future__ import annotations

import pytest

from app.llm.registry import ProviderRegistry
from app.llm.providers.mock import MockProvider


@pytest.fixture
def registry() -> ProviderRegistry:
    return ProviderRegistry()


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()


class TestProviderRegistryRegistration:
    def test_register_and_get(self, registry: ProviderRegistry) -> None:
        provider = MockProvider()
        registry.register(provider)
        assert registry.has("mock")
        assert registry.get("mock") is provider

    def test_get_missing_returns_none(self, registry: ProviderRegistry) -> None:
        assert registry.get("missing") is None

    def test_get_or_raise(self, registry: ProviderRegistry) -> None:
        provider = MockProvider()
        registry.register(provider)
        assert registry.get_or_raise("mock") is provider

    def test_get_or_raise_missing(self, registry: ProviderRegistry) -> None:
        with pytest.raises(KeyError):
            registry.get_or_raise("missing")

    def test_duplicate_register_raises(self, registry: ProviderRegistry) -> None:
        registry.register(MockProvider())
        with pytest.raises(ValueError):
            registry.register(MockProvider())

    def test_register_overwrite(self, registry: ProviderRegistry) -> None:
        first = MockProvider(model="a")
        second = MockProvider(model="b")
        registry.register(first)
        registry.register(second, overwrite=True)
        assert registry.get("mock") is second

    def test_unregister(self, registry: ProviderRegistry) -> None:
        registry.register(MockProvider())
        registry.unregister("mock")
        assert registry.has("mock") is False

    def test_clear(self, registry: ProviderRegistry) -> None:
        registry.register(MockProvider())
        registry.clear()
        assert registry.count == 0


class TestProviderRegistryQuery:
    def test_names(self, registry: ProviderRegistry) -> None:
        registry.register(MockProvider())
        assert registry.names() == ["mock"]

    def test_providers(self, registry: ProviderRegistry) -> None:
        provider = MockProvider()
        registry.register(provider)
        assert registry.providers() == [provider]

    def test_count(self, registry: ProviderRegistry) -> None:
        assert registry.count == 0
        registry.register(MockProvider())
        assert registry.count == 1

    def test_contains(self, registry: ProviderRegistry) -> None:
        registry.register(MockProvider())
        assert "mock" in registry
        assert "missing" not in registry

    def test_len(self, registry: ProviderRegistry) -> None:
        registry.register(MockProvider())
        assert len(registry) == 1

    def test_iteration(self, registry: ProviderRegistry) -> None:
        registry.register(MockProvider())
        assert [p.name for p in registry] == ["mock"]
