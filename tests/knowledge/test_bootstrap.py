"""Tests for the knowledge DI bootstrap and wiring."""

from __future__ import annotations

from app.core.container import DependencyContainer
from app.knowledge.base import Retriever
from app.knowledge.bootstrap import register_knowledge_components
from app.knowledge.registry import (
    ChunkerRegistry,
    CitationBuilderRegistry,
    ContextCompressorRegistry,
    EmbeddingProviderRegistry,
    LoaderRegistry,
    QueryRewriterRegistry,
    RerankerRegistry,
    RetrieverRegistry,
    VectorStoreRegistry,
)


class _DummyRetriever(Retriever):
    name = "dummy"

    async def retrieve(self, query, *, context):  # type: ignore[no-untyped-def]
        return []


ALL_REGISTRIES = (
    LoaderRegistry,
    ChunkerRegistry,
    EmbeddingProviderRegistry,
    VectorStoreRegistry,
    RetrieverRegistry,
    QueryRewriterRegistry,
    ContextCompressorRegistry,
    RerankerRegistry,
    CitationBuilderRegistry,
)


def test_bootstrap_registers_all_registries() -> None:
    container = DependencyContainer()
    register_knowledge_components(container)
    for registry_type in ALL_REGISTRIES:
        assert container.has(registry_type), f"missing {registry_type}"


def test_bootstrap_resolves_singletons() -> None:
    container = DependencyContainer()
    register_knowledge_components(container)
    a = container.resolve(RetrieverRegistry)
    b = container.resolve(RetrieverRegistry)
    assert a is b


def test_bootstrap_is_idempotent() -> None:
    container = DependencyContainer()
    register_knowledge_components(container)
    register_knowledge_components(container)  # must not raise
    register_knowledge_components(container)
    assert container.has(RetrieverRegistry)


def test_bootstrap_registries_are_independent_singletons() -> None:
    container = DependencyContainer()
    register_knowledge_components(container)
    assert container.resolve(LoaderRegistry) is not container.resolve(RetrieverRegistry)


def test_bootstrap_respects_pre_registered_registry() -> None:
    container = DependencyContainer()
    custom = RetrieverRegistry()
    custom.register("bm25", _DummyRetriever())
    container.register_singleton(RetrieverRegistry, instance=custom)
    register_knowledge_components(container)
    assert container.resolve(RetrieverRegistry) is custom
    assert container.resolve(RetrieverRegistry).has("bm25")
