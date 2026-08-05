"""Dependency-injection wiring for the knowledge module.

Registers the knowledge component registries with the application's
:class:`DependencyContainer`.  Registration is idempotent: re-running
:func:`register_knowledge_components` never raises, so it is safe to
call from the server factory and the lifespan.

Phase 13a registers only the foundation registries (the abstract
stage contracts).  Concrete loaders, chunkers, embedding providers and
vector stores are registered in later phases once implemented.
"""

from __future__ import annotations

import logging

from app.core.container import DependencyContainer
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

log = logging.getLogger(__name__)


def register_knowledge_components(container: DependencyContainer) -> None:
    """Register knowledge component registries with a DI container.

    Idempotent: each type is registered only when absent so repeated
    calls (server factory + lifespan) are safe.  All components are
    singletons and are created by the container with no dependencies.

    Args:
        container: The application's ``DependencyContainer``.
    """
    if not container.has(LoaderRegistry):
        container.register_singleton(LoaderRegistry)
    if not container.has(ChunkerRegistry):
        container.register_singleton(ChunkerRegistry)
    if not container.has(EmbeddingProviderRegistry):
        container.register_singleton(EmbeddingProviderRegistry)
    if not container.has(VectorStoreRegistry):
        container.register_singleton(VectorStoreRegistry)
    if not container.has(RetrieverRegistry):
        container.register_singleton(RetrieverRegistry)
    if not container.has(QueryRewriterRegistry):
        container.register_singleton(QueryRewriterRegistry)
    if not container.has(ContextCompressorRegistry):
        container.register_singleton(ContextCompressorRegistry)
    if not container.has(RerankerRegistry):
        container.register_singleton(RerankerRegistry)
    if not container.has(CitationBuilderRegistry):
        container.register_singleton(CitationBuilderRegistry)
    log.info("Registered knowledge components with DI container")
