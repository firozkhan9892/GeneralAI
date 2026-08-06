"""Pure-python BM25 retriever.

Implements the BM25 (Best Matching 25) ranking function without any
external dependencies.  The index is maintained in-memory and supports
incremental add/delete operations.

BM25 formula (per query term *t*)::

    score = IDF(t) * (tf(t, d) * (k1 + 1)) /
            (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

where *tf* is term frequency, *|d|* is document length, *avgdl* is
average document length, *k1* and *b* are tunable parameters.
"""

from __future__ import annotations

import math
import re
import threading
from typing import Any, Iterable

from app.knowledge.base import Retriever, RetrievalContext
from app.knowledge.constants import BM25_B, BM25_K1, DEFAULT_TOP_K
from app.knowledge.models import (
    MetadataFilter,
    RetrievalHit,
    RetrievalQuery,
)


class BM25Index:
    """Thread-safe in-memory BM25 index.

    Stores documents as text with metadata and answers ranked queries.
    The index is rebuilt lazily from added documents; incremental
    add/delete operations update the underlying data and mark the
    index as dirty for the next query.
    """

    def __init__(
        self,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        self._k1 = k1
        self._b = b
        self._lock = threading.RLock()
        self._documents: dict[str, _BM25Document] = {}
        self._dirty = True
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._doc_count: int = 0

    @property
    def doc_count(self) -> int:
        """Return the number of indexed documents."""
        with self._lock:
            return len(self._documents)

    def add(
        self,
        chunk_id: str,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
        namespace: str = "",
        collection_id: str = "",
        doc_id: str = "",
    ) -> None:
        """Add a document to the index.

        Args:
            chunk_id: Unique identifier for this document/chunk.
            text: The document text content.
            metadata: Optional metadata attached to this document.
            namespace: Namespace scope.
            collection_id: Collection scope.
            doc_id: Source document identifier.
        """
        tokens = self._tokenize(text)
        with self._lock:
            self._documents[chunk_id] = _BM25Document(
                chunk_id=chunk_id,
                doc_id=doc_id,
                namespace=namespace,
                collection_id=collection_id,
                text=text,
                tokens=tokens,
                metadata=metadata or {},
            )
            self._dirty = True

    def add_many(
        self,
        chunk_ids: list[str],
        texts: list[str],
        *,
        metadatas: list[dict[str, Any]] | None = None,
        namespaces: list[str] | None = None,
        collection_ids: list[str] | None = None,
        doc_ids: list[str] | None = None,
    ) -> None:
        """Add multiple documents in one locked operation."""
        if metadatas is None:
            metadatas = [{}] * len(texts)
        if namespaces is None:
            namespaces = [""] * len(texts)
        if collection_ids is None:
            collection_ids = [""] * len(texts)
        if doc_ids is None:
            doc_ids = [""] * len(texts)

        with self._lock:
            for i, (cid, text) in enumerate(zip(chunk_ids, texts, strict=True)):
                tokens = self._tokenize(text)
                self._documents[cid] = _BM25Document(
                    chunk_id=cid,
                    doc_id=doc_ids[i],
                    namespace=namespaces[i],
                    collection_id=collection_ids[i],
                    text=text,
                    tokens=tokens,
                    metadata=metadatas[i],
                )
            self._dirty = True

    def delete(self, chunk_ids: Iterable[str]) -> int:
        """Remove documents by chunk_id.  Returns count deleted."""
        to_remove = set(chunk_ids)
        with self._lock:
            removed = 0
            for cid in to_remove:
                if cid in self._documents:
                    del self._documents[cid]
                    removed += 1
            if removed:
                self._dirty = True
            return removed

    def search(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        filters: tuple[MetadataFilter, ...] = (),
        namespace: str = "",
        collection_id: str = "",
    ) -> list[RetrievalHit]:
        """Search the index and return ranked hits.

        Args:
            query: The search query string.
            top_k: Maximum hits to return.
            filters: Optional metadata filters (conjunctive).
            namespace: Scope to a specific namespace.
            collection_id: Scope to a specific collection.

        Returns:
            Hits ranked by BM25 score (descending).
        """
        from app.knowledge.retrieval.filter import evaluate_filters

        query_tokens = self._tokenize(query)

        with self._lock:
            self._rebuild_if_dirty()

            if not self._documents or not query_tokens:
                return []

            scores: list[tuple[str, float, _BM25Document]] = []
            for doc in self._documents.values():
                if namespace and doc.namespace != namespace:
                    continue
                if collection_id and doc.collection_id != collection_id:
                    continue
                if not evaluate_filters(doc.metadata, filters):
                    continue
                score = self._score(query_tokens, doc)
                if score > 0:
                    scores.append((doc.chunk_id, score, doc))

            scores.sort(key=lambda x: x[1], reverse=True)

            hits: list[RetrievalHit] = []
            for chunk_id, score, doc in scores[:top_k]:
                hits.append(
                    RetrievalHit(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        collection_id=doc.collection_id,
                        namespace=doc.namespace,
                        content=doc.text,
                        score=score,
                        ranks={"bm25": score},
                        metadata=dict(doc.metadata),
                    )
                )
            return hits

    def clear(self) -> None:
        """Remove all documents from the index."""
        with self._lock:
            self._documents.clear()
            self._idf.clear()
            self._avgdl = 0.0
            self._doc_count = 0
            self._dirty = True

    def _rebuild_if_dirty(self) -> None:
        """Recompute IDF and average document length if needed."""
        if not self._dirty:
            return

        n = len(self._documents)
        self._doc_count = n

        if n == 0:
            self._idf.clear()
            self._avgdl = 0.0
            self._dirty = False
            return

        # Document frequency per term
        df: dict[str, int] = {}
        total_len = 0
        for doc in self._documents.values():
            unique_tokens = set(doc.tokens)
            total_len += len(doc.tokens)
            for token in unique_tokens:
                df[token] = df.get(token, 0) + 1

        self._avgdl = total_len / n if n > 0 else 0.0

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        self._idf = {}
        for term, freq in df.items():
            self._idf[term] = math.log((n - freq + 0.5) / (freq + 0.5) + 1.0)

        self._dirty = False

    def _score(self, query_tokens: list[str], doc: _BM25Document) -> float:
        """Score a single document against query tokens."""
        score = 0.0
        doc_len = len(doc.tokens)
        tf_map: dict[str, int] = {}
        for token in doc.tokens:
            tf_map[token] = tf_map.get(token, 0) + 1

        for qt in query_tokens:
            if qt not in self._idf:
                continue
            tf = tf_map.get(qt, 0)
            idf = self._idf[qt]
            numerator = tf * (self._k1 + 1)
            denominator = tf + self._k1 * (
                1 - self._b + self._b * doc_len / max(self._avgdl, 1)
            )
            score += idf * numerator / denominator

        return score

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase and split *text* on non-alphanumeric characters."""
        return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


class _BM25Document:
    """Internal record for an indexed document."""

    __slots__ = (
        "chunk_id",
        "doc_id",
        "namespace",
        "collection_id",
        "text",
        "tokens",
        "metadata",
    )

    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        namespace: str,
        collection_id: str,
        text: str,
        tokens: list[str],
        metadata: dict[str, Any],
    ) -> None:
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.namespace = namespace
        self.collection_id = collection_id
        self.text = text
        self.tokens = tokens
        self.metadata = metadata


class BM25Retriever(Retriever):
    """Retrieves chunks using BM25 lexical matching.

    Wraps a :class:`BM25Index` and answers retrieval queries.  The
    index is maintained externally (by the pipeline or service) and
    shared across queries.
    """

    name: str = "bm25"

    def __init__(self, index: BM25Index | None = None) -> None:
        self._index = index or BM25Index()

    @property
    def index(self) -> BM25Index:
        """Return the underlying BM25 index."""
        return self._index

    async def retrieve(
        self, query: RetrievalQuery, *, context: RetrievalContext
    ) -> list[RetrievalHit]:
        """Search the BM25 index for *query*.

        Args:
            query: The retrieval query (uses ``rewritten_query`` or
                ``query``).
            context: Collection / namespace / filter context.

        Returns:
            Hits ranked by BM25 score.
        """
        q = query.rewritten_query or query.query
        return self._index.search(
            q,
            top_k=query.top_k,
            filters=query.filters,
            namespace=context.namespace,
            collection_id=context.collection_id,
        )
