"""Tests for the knowledge typed registries."""

from __future__ import annotations

import threading

import pytest

from app.knowledge.base import Chunker, DocumentLoader, EmbeddingProvider, Retriever
from app.knowledge.models import (
    DocumentFormat,
    EmbeddingModelInfo,
    KnowledgeChunk,
    KnowledgeDocument,
    RetrievalQuery,
)
from app.knowledge.registry import (
    ChunkerRegistry,
    EmbeddingProviderRegistry,
    LoaderRegistry,
    RetrieverRegistry,
)


class _FakeLoader(DocumentLoader):
    format = DocumentFormat.TXT

    def load(self, content, *, source_uri="", metadata=None) -> KnowledgeDocument:  # type: ignore[no-untyped-def]
        return KnowledgeDocument(
            doc_id="d", collection_id="c", format=self.format, content=content.decode()
        )


class _FakeChunker(Chunker):
    name = "fixed"

    def chunk(self, document: KnowledgeDocument) -> list[KnowledgeChunk]:
        return [
            KnowledgeChunk(
                chunk_id="c1",
                doc_id=document.doc_id,
                collection_id="c1",
                content=document.content,
            )
        ]


class _FakeProvider(EmbeddingProvider):
    name = "det"
    dimensions = 4

    def embed(self, texts: list[str]) -> list[list[float]]:  # type: ignore[override]
        return [[0.0] * self.dimensions for _ in texts]

    def model_info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            name=self.name, provider=self.name, dimensions=self.dimensions, model="m"
        )


class _FakeRetriever(Retriever):
    name = "bm25"

    async def retrieve(self, query: RetrievalQuery, *, context):  # type: ignore[no-untyped-def]
        return []


def test_register_get_unregister() -> None:
    registry: RetrieverRegistry = RetrieverRegistry()
    retriever = _FakeRetriever()
    registry.register("bm25", retriever)
    assert registry.has("bm25")
    assert registry.get("bm25") is retriever
    assert registry.count == 1

    registry.unregister("bm25")
    assert not registry.has("bm25")
    assert registry.get("bm25") is None
    assert registry.count == 0


def test_duplicate_registration_raises() -> None:
    registry = RetrieverRegistry()
    registry.register("bm25", _FakeRetriever())
    with pytest.raises(ValueError):
        registry.register("bm25", _FakeRetriever())
    registry.register("bm25", _FakeRetriever(), overwrite=True)


def test_unregister_unknown_is_noop() -> None:
    registry = RetrieverRegistry()
    registry.unregister("missing")  # must not raise
    assert registry.count == 0


def test_enumeration_and_count() -> None:
    registry = RetrieverRegistry()
    registry.register("a", _FakeRetriever())
    registry.register("b", _FakeRetriever())
    assert set(registry.keys()) == {"a", "b"}
    assert len(registry.values()) == 2
    assert len(registry) == 2
    assert registry.is_empty is False


def test_immutable_snapshot() -> None:
    registry = RetrieverRegistry()
    retriever = _FakeRetriever()
    registry.register("a", retriever)

    snapshot = registry.items()
    assert snapshot == {"a": retriever}
    snapshot["a"] = _FakeRetriever()  # type: ignore[index]  # mutating copy is harmless
    assert registry.get("a") is retriever

    entries = registry.snapshot()
    assert entries == (("a", retriever),)


def test_loader_registry_keys_by_format() -> None:
    registry = LoaderRegistry()
    registry.register(DocumentFormat.TXT.value, _FakeLoader())
    assert registry.has("txt")


def test_chunker_registry_by_name() -> None:
    registry = ChunkerRegistry()
    registry.register("fixed", _FakeChunker())
    assert registry.get("fixed") is not None


def test_provider_registry_by_name() -> None:
    registry = EmbeddingProviderRegistry()
    provider = _FakeProvider()
    registry.register("det", provider)
    assert registry.get("det") is provider


def test_typed_registries_are_isolated() -> None:
    retriever_registry = RetrieverRegistry()
    chunker_registry = ChunkerRegistry()
    retriever_registry.register("bm25", _FakeRetriever())
    assert chunker_registry.count == 0


def test_registry_is_thread_safe() -> None:
    registry = RetrieverRegistry()
    errors: list[Exception] = []
    barrier = threading.Barrier(8)

    def worker(i: int) -> None:
        try:
            barrier.wait()
            for n in range(50):
                key = f"w{i}-{n}"
                registry.register(key, _FakeRetriever())
                assert registry.has(key)
                registry.unregister(key)
        except Exception as exc:  # pragma: no cover - failure capture
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert registry.count == 0
