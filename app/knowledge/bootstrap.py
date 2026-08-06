"""Dependency-injection wiring for the knowledge module.

Registers the knowledge component registries with the application's
:class:`DependencyContainer`.  Registration is idempotent: re-running
:func:`register_knowledge_components` never raises, so it is safe to
call from the server factory and the lifespan.

Phase 13a registers foundation registries (abstract stage contracts).
Phase 13b adds concrete loaders, chunkers, the format parser,
collection registry, namespace registry, and knowledge settings.
"""

from __future__ import annotations

import logging

from app.core.container import DependencyContainer
from app.knowledge.config import KnowledgeSettings
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
    # ── Foundation registries (13a) ───────────────────────────────────
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

    # ── Settings (13b) ────────────────────────────────────────────────
    if not container.has(KnowledgeSettings):
        container.register_singleton(KnowledgeSettings)

    # ── Concrete loader registrations (13b) ──────────────────────────
    from app.knowledge.documents.loaders.html import HtmlLoader
    from app.knowledge.documents.loaders.json_loader import JsonLoader
    from app.knowledge.documents.loaders.markdown import MarkdownLoader
    from app.knowledge.documents.loaders.pdf import PdfLoader
    from app.knowledge.documents.loaders.text import TextLoader

    loader_registry = container.resolve(LoaderRegistry)
    for loader_instance in [
        TextLoader(),
        MarkdownLoader(),
        JsonLoader(),
        PdfLoader(),
        HtmlLoader(),
    ]:
        fmt = loader_instance.format.value
        if fmt not in loader_registry:
            loader_registry.register(fmt, loader_instance)

    # ── Concrete chunker registrations (13b) ─────────────────────────
    from app.knowledge.documents.chunkers.fixed import FixedChunker
    from app.knowledge.documents.chunkers.paragraph import ParagraphChunker
    from app.knowledge.documents.chunkers.recursive import RecursiveChunker
    from app.knowledge.documents.chunkers.sentence import SentenceChunker

    chunker_registry = container.resolve(ChunkerRegistry)
    for chunker_instance in [
        FixedChunker(),
        ParagraphChunker(),
        SentenceChunker(),
        RecursiveChunker(),
    ]:
        if chunker_instance.name not in chunker_registry:
            chunker_registry.register(chunker_instance.name, chunker_instance)

    # ── Collection & namespace registries (13b) ──────────────────────
    from app.knowledge.collection_registry import CollectionRegistry
    from app.knowledge.namespace_registry import NamespaceRegistry

    if not container.has(CollectionRegistry):
        container.register_singleton(CollectionRegistry)
    if not container.has(NamespaceRegistry):
        container.register_singleton(NamespaceRegistry)

    # ── Format parser (13b) ──────────────────────────────────────────
    from app.knowledge.documents.parser import FormatParser

    if not container.has(FormatParser):
        container.register_singleton(FormatParser)

    log.info("Registered knowledge components with DI container")
