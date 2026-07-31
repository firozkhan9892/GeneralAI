"""Tests for LLM layer DI wiring and package exports."""

from __future__ import annotations

from app.core.container import DependencyContainer
from app.llm.bootstrap import register_llm_components
from app.llm.factory import ProviderFactory
from app.llm.providers.mock import MockProvider
from app.llm.registry import ProviderRegistry


class TestBootstrap:
    def test_registers_registry_singleton(self) -> None:
        container = DependencyContainer()
        register_llm_components(container)
        registry = container.resolve(ProviderRegistry)
        assert isinstance(registry, ProviderRegistry)
        assert container.resolve(ProviderRegistry) is registry

    def test_registers_factory_singleton(self) -> None:
        container = DependencyContainer()
        register_llm_components(container)
        factory = container.resolve(ProviderFactory)
        assert isinstance(factory, ProviderFactory)

    def test_factory_creates_mock_through_container(self) -> None:
        container = DependencyContainer()
        register_llm_components(container)
        factory = container.resolve(ProviderFactory)
        assert isinstance(factory.create("mock"), MockProvider)


class TestPackageExports:
    def test_top_level_imports(self) -> None:
        import app.llm as llm

        symbols = (
            "BaseLLMProvider",
            "ChatRequest",
            "ChatResponse",
            "Message",
            "ModelInfo",
            "OpenAIProvider",
            "ProviderFactory",
            "ProviderRegistry",
            "Role",
            "StreamChunk",
            "ToolCall",
            "ToolDefinition",
            "Usage",
        )
        assert all(getattr(llm, name, None) is not None for name in symbols)
