"""Tests for in-memory vector store."""

import pytest

from app.knowledge.models import MetadataFilter
from app.knowledge.vectorstores.in_memory import InMemoryVectorStore


def _make_chunk(
    chunk_id: str, doc_id: str = "d1", ns: str = "ns", col: str = "col", **meta
):
    from app.knowledge.models import KnowledgeChunk

    return KnowledgeChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        collection_id=col,
        namespace=ns,
        content=f"content {chunk_id}",
        metadata=meta,
    )


def test_add_and_count() -> None:
    store = InMemoryVectorStore(dimensions=4)
    chunks = [_make_chunk("c1"), _make_chunk("c2")]
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    store.add(chunks, vectors)
    assert store.count() == 2


def test_search_basic() -> None:
    store = InMemoryVectorStore(dimensions=4)
    chunks = [_make_chunk("c1"), _make_chunk("c2")]
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    store.add(chunks, vectors)
    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    assert results[0].score > 0.99


def test_search_empty_store() -> None:
    store = InMemoryVectorStore(dimensions=4)
    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=5)
    assert results == []


def test_delete_by_ids() -> None:
    store = InMemoryVectorStore(dimensions=4)
    chunks = [_make_chunk("c1"), _make_chunk("c2"), _make_chunk("c3")]
    vectors = [[1.0, 0.0, 0.0, 0.0]] * 3
    store.add(chunks, vectors)
    store.delete(["c1", "c3"])
    assert store.count() == 1
    remaining = store.search([1.0, 0.0, 0.0, 0.0], top_k=5)
    assert remaining[0].chunk_id == "c2"


def test_delete_by_document() -> None:
    store = InMemoryVectorStore(dimensions=4)
    chunks = [
        _make_chunk("c1", doc_id="d1"),
        _make_chunk("c2", doc_id="d1"),
        _make_chunk("c3", doc_id="d2"),
    ]
    vectors = [[1.0, 0.0, 0.0, 0.0]] * 3
    store.add(chunks, vectors)
    store.delete_by_document("d1", "ns")
    assert store.count() == 1


def test_clear() -> None:
    store = InMemoryVectorStore(dimensions=4)
    chunks = [_make_chunk("c1")]
    store.add(chunks, [[1.0, 0.0, 0.0, 0.0]])
    store.clear()
    assert store.count() == 0


def test_metadata_filter_eq() -> None:
    store = InMemoryVectorStore(dimensions=4)
    chunks = [
        _make_chunk("c1", category="A"),
        _make_chunk("c2", category="B"),
    ]
    vectors = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    store.add(chunks, vectors)
    f = MetadataFilter(field="category", op="eq", value="A")
    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=10, filters=(f,))
    assert len(results) == 1
    assert results[0].chunk_id == "c1"


def test_metadata_filter_in() -> None:
    store = InMemoryVectorStore(dimensions=4)
    chunks = [
        _make_chunk("c1", tag="x"),
        _make_chunk("c2", tag="y"),
        _make_chunk("c3", tag="z"),
    ]
    vectors = [[1.0, 0.0, 0.0, 0.0]] * 3
    store.add(chunks, vectors)
    f = MetadataFilter(field="tag", op="in", value=["x", "z"])
    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=10, filters=(f,))
    ids = {r.chunk_id for r in results}
    assert "c1" in ids
    assert "c3" in ids
    assert "c2" not in ids


def test_metadata_filter_gt() -> None:
    store = InMemoryVectorStore(dimensions=4)
    chunks = [
        _make_chunk("c1", score=5),
        _make_chunk("c2", score=10),
        _make_chunk("c3", score=15),
    ]
    vectors = [[1.0, 0.0, 0.0, 0.0]] * 3
    store.add(chunks, vectors)
    f = MetadataFilter(field="score", op="gt", value=7)
    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=10, filters=(f,))
    ids = {r.chunk_id for r in results}
    assert "c2" in ids
    assert "c3" in ids
    assert "c1" not in ids


def test_metadata_filter_contains() -> None:
    store = InMemoryVectorStore(dimensions=4)
    chunks = [
        _make_chunk("c1", text="hello world"),
        _make_chunk("c2", text="foo bar"),
    ]
    vectors = [[1.0, 0.0, 0.0, 0.0]] * 2
    store.add(chunks, vectors)
    f = MetadataFilter(field="text", op="contains", value="hello")
    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=10, filters=(f,))
    assert len(results) == 1
    assert results[0].chunk_id == "c1"


def test_namespace_isolation_in_delete() -> None:
    store = InMemoryVectorStore(dimensions=4)
    chunks = [
        _make_chunk("c1", ns="ns1"),
        _make_chunk("c2", ns="ns2"),
    ]
    vectors = [[1.0, 0.0, 0.0, 0.0]] * 2
    store.add(chunks, vectors)
    store.delete_by_document("d1", "ns1")
    assert store.count() == 1
    remaining = store.search([1.0, 0.0, 0.0, 0.0], top_k=5)
    assert remaining[0].namespace == "ns2"


def test_add_mismatched_lengths() -> None:
    store = InMemoryVectorStore(dimensions=4)
    with pytest.raises(ValueError):
        store.add([_make_chunk("c1")], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


def test_search_top_k() -> None:
    store = InMemoryVectorStore(dimensions=4)
    chunks = [_make_chunk(f"c{i}") for i in range(10)]
    # Use different directions so similarity differs
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
    ]
    store.add(chunks, vectors)
    # Query most similar to c9 (first component = 1)
    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=3)
    assert len(results) == 3
    # c0 and c9 both have [1,0,0,0], so they're most similar (sim=1.0)
    top_ids = {r.chunk_id for r in results}
    assert "c0" in top_ids
    assert "c9" in top_ids
