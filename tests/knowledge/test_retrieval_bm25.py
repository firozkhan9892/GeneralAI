"""Tests for pure-python BM25 retriever."""

import asyncio
import threading

from app.knowledge.models import MetadataFilter
from app.knowledge.retrieval.bm25 import BM25Index, BM25Retriever


def _make_index() -> BM25Index:
    idx = BM25Index()
    idx.add(
        "c1",
        "the cat sat on the mat",
        doc_id="d1",
        collection_id="col1",
        namespace="ns1",
    )
    idx.add(
        "c2",
        "the dog chased the cat",
        doc_id="d2",
        collection_id="col1",
        namespace="ns1",
    )
    idx.add(
        "c3",
        "the bird flew over the dog",
        doc_id="d3",
        collection_id="col1",
        namespace="ns1",
    )
    idx.add(
        "c4",
        "python is a programming language",
        doc_id="d4",
        collection_id="col2",
        namespace="ns2",
    )
    idx.add(
        "c5",
        "java is also a programming language",
        doc_id="d5",
        collection_id="col2",
        namespace="ns2",
    )
    return idx


class TestBM25Index:
    def test_add_and_count(self) -> None:
        idx = BM25Index()
        assert idx.doc_count == 0
        idx.add("c1", "hello world")
        assert idx.doc_count == 1

    def test_add_many(self) -> None:
        idx = BM25Index()
        idx.add_many(["c1", "c2", "c3"], ["hello", "world", "test"])
        assert idx.doc_count == 3

    def test_delete(self) -> None:
        idx = _make_index()
        removed = idx.delete(["c1", "c2"])
        assert removed == 2
        assert idx.doc_count == 3

    def test_delete_nonexistent(self) -> None:
        idx = _make_index()
        removed = idx.delete(["nonexistent"])
        assert removed == 0
        assert idx.doc_count == 5

    def test_clear(self) -> None:
        idx = _make_index()
        idx.clear()
        assert idx.doc_count == 0

    def test_search_basic(self) -> None:
        idx = _make_index()
        hits = idx.search("cat", top_k=5)
        assert len(hits) > 0
        # c1 and c2 mention "cat"
        chunk_ids = [h.chunk_id for h in hits]
        assert "c1" in chunk_ids
        assert "c2" in chunk_ids

    def test_search_ranking(self) -> None:
        idx = _make_index()
        hits = idx.search("cat", top_k=5)
        # c1 has "cat" once, c2 has "cat" once
        # Both should score, but the exact ranking depends on IDF
        assert len(hits) >= 2

    def test_search_empty_index(self) -> None:
        idx = BM25Index()
        hits = idx.search("hello", top_k=5)
        assert hits == []

    def test_search_empty_query(self) -> None:
        idx = _make_index()
        hits = idx.search("", top_k=5)
        assert hits == []

    def test_search_top_k(self) -> None:
        idx = _make_index()
        hits = idx.search("programming", top_k=1)
        assert len(hits) == 1

    def test_search_namespace_filter(self) -> None:
        idx = _make_index()
        hits = idx.search("programming", top_k=10, namespace="ns2")
        assert all(h.namespace == "ns2" for h in hits)

    def test_search_namespace_excludes(self) -> None:
        idx = _make_index()
        hits = idx.search("cat", top_k=10, namespace="ns2")
        assert len(hits) == 0  # cat only in ns1

    def test_search_collection_filter(self) -> None:
        idx = _make_index()
        hits = idx.search("programming", top_k=10, collection_id="col2")
        assert all(h.collection_id == "col2" for h in hits)

    def test_search_metadata_filter(self) -> None:
        idx = BM25Index()
        idx.add("c1", "hello world", metadata={"type": "faq"})
        idx.add("c2", "hello world", metadata={"type": "guide"})
        filters = (MetadataFilter(field="type", op="eq", value="faq"),)
        hits = idx.search("hello", top_k=10, filters=filters)
        assert len(hits) == 1
        assert hits[0].chunk_id == "c1"

    def test_search_returns_retrieval_hits(self) -> None:
        idx = _make_index()
        hits = idx.search("cat", top_k=5)
        for hit in hits:
            assert "bm25" in hit.ranks
            assert hit.score > 0

    def test_thread_safety(self) -> None:
        idx = BM25Index()
        errors: list[Exception] = []

        def add_docs() -> None:
            try:
                for i in range(100):
                    idx.add(f"c{i}", f"document number {i}")
            except Exception as e:
                errors.append(e)

        def search_docs() -> None:
            try:
                for _ in range(50):
                    idx.search("document", top_k=5)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_docs),
            threading.Thread(target=add_docs),
            threading.Thread(target=search_docs),
            threading.Thread(target=search_docs),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


class TestBM25Retriever:
    def test_retrieve(self) -> None:
        idx = _make_index()
        retriever = BM25Retriever(index=idx)
        from app.knowledge.base import RetrievalContext
        from app.knowledge.models import RetrievalQuery

        query = RetrievalQuery(query="cat", top_k=5)
        context = RetrievalContext(namespace="ns1", collection_id="col1")
        hits = asyncio.run(retriever.retrieve(query, context=context))
        assert len(hits) > 0

    def test_name(self) -> None:
        assert BM25Retriever().name == "bm25"
