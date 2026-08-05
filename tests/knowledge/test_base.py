"""Tests for the knowledge pipeline stage abstract base classes."""

from __future__ import annotations

import asyncio

import pytest

from app.knowledge.base import (
    Chunker,
    CitationBuilder,
    ContextCompressor,
    DocumentLoader,
    EmbeddingProvider,
    QueryRewriter,
    Reranker,
    Retriever,
    VectorStore,
)
from app.knowledge.models import (
    CitationResult,
    DocumentFormat,
    EmbeddingModelInfo,
    KnowledgeChunk,
    KnowledgeDocument,
    MetadataFilter,
    RetrievalHit,
    RetrievalQuery,
    VectorSearchHit,
)

ABSTRACT_STAGES = (
    DocumentLoader,
    Chunker,
    EmbeddingProvider,
    VectorStore,
    Retriever,
    QueryRewriter,
    ContextCompressor,
    Reranker,
    CitationBuilder,
)


def test_all_stages_are_abstract() -> None:
    for stage in ABSTRACT_STAGES:
        with pytest.raises(TypeError):
            stage()  # type: ignore[abstract]


def test_document_loader_sync_async_offload() -> None:
    class Loader(DocumentLoader):
        format = DocumentFormat.TXT

        def load(self, content, *, source_uri="", metadata=None) -> KnowledgeDocument:  # type: ignore[no-untyped-def]
            return KnowledgeDocument(
                doc_id="d",
                collection_id="c",
                format=self.format,
                content=content.decode(),
                metadata=dict(metadata or {}),
            )

    loader = Loader()
    doc = asyncio.run(
        loader.load_async(b"hello", source_uri="s.txt", metadata={"a": 1})
    )
    assert doc.content == "hello"
    assert doc.metadata == {"a": 1}


def test_chunker_sync_async_offload() -> None:
    class ChunkerImpl(Chunker):
        name = "fixed"

        def chunk(self, document: KnowledgeDocument) -> list[KnowledgeChunk]:
            return [
                KnowledgeChunk(
                    chunk_id="c1",
                    doc_id=document.doc_id,
                    collection_id=document.collection_id,
                    content=document.content,
                )
            ]

    doc = KnowledgeDocument(doc_id="d", collection_id="c", format=DocumentFormat.TXT)
    chunks = asyncio.run(ChunkerImpl().chunk_async(doc))
    assert chunks[0].content == ""


def test_embedding_provider_async_offload() -> None:
    class Provider(EmbeddingProvider):
        name = "det"
        dimensions = 2

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 2.0]] * len(texts)

        def model_info(self) -> EmbeddingModelInfo:
            return EmbeddingModelInfo(
                name=self.name,
                provider=self.name,
                dimensions=self.dimensions,
                model="m",
            )

    vectors = asyncio.run(Provider().embed_async(["a", "b"]))
    assert len(vectors) == 2
    assert vectors[0] == [1.0, 2.0]


def test_vector_store_async_offload() -> None:
    class Store(VectorStore):
        name = "in_memory"
        dimensions = 2

        def add(self, chunks, vectors):  # type: ignore[no-untyped-def]
            pass

        def delete(self, chunk_ids):  # type: ignore[no-untyped-def]
            pass

        def delete_by_document(self, doc_id, namespace):  # type: ignore[no-untyped-def]
            pass

        def search(self, vector, *, top_k, filters=()):  # type: ignore[no-untyped-def]
            return []

        def count(self) -> int:
            return 0

        def clear(self) -> None:
            pass

    store = Store()

    async def scenario() -> None:
        await store.add_async([], [])
        await store.delete_async([])
        hits = await store.search_async([0.0, 0.0], top_k=5, filters=())
        assert hits == []

    asyncio.run(scenario())


def test_retriever_context_typing() -> None:
    from app.knowledge.base import RetrievalContext

    context = RetrievalContext(namespace="prod", collection_id="c1")
    assert context.namespace == "prod"
    assert context.vector_store is None
    assert context.filters == ()


def test_metadata_filter_tuple_in_context() -> None:
    from app.knowledge.base import RetrievalContext

    filters = (MetadataFilter(field="author", op="eq", value="alice"),)
    context = RetrievalContext(filters=filters)
    assert context.filters == filters


def test_citation_builder_return_type() -> None:
    class Builder(CitationBuilder):
        name = "default"

        def build(self, hits: list[RetrievalHit]) -> CitationResult:
            return CitationResult()

    assert Builder().build([]) == CitationResult()


def test_search_hit_model() -> None:
    hit = VectorSearchHit(chunk_id="c1", doc_id="d1", collection_id="c1", score=0.5)
    assert hit.score == 0.5


def test_retrieval_query_model() -> None:
    query = RetrievalQuery(query="hello")
    assert query.top_k == 10
    assert query.strategy == "hybrid"
