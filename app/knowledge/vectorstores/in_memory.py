"""In-memory vector store using numpy brute-force cosine similarity.

Zero external dependencies beyond numpy (which is a required dep of
the knowledge module).  Thread-safe via ``RLock``.
"""

from __future__ import annotations

import threading
from typing import Any, Iterable

import numpy as np

from app.knowledge.base import VectorStore
from app.knowledge.models import MetadataFilter, VectorSearchHit


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _evaluate_filter(metadata: dict[str, Any], f: MetadataFilter) -> bool:
    """Evaluate a single metadata filter against a metadata dict."""
    val = metadata.get(f.field)
    op = f.op
    if op == "eq":
        return val == f.value
    if op == "neq":
        return val != f.value
    if op == "in":
        return val in (f.value or [])
    if op == "not_in":
        return val not in (f.value or [])
    if op == "gt":
        return val is not None and val > f.value
    if op == "gte":
        return val is not None and val >= f.value
    if op == "lt":
        return val is not None and val < f.value
    if op == "lte":
        return val is not None and val <= f.value
    if op == "exists":
        return f.field in metadata
    if op == "contains":
        return isinstance(val, str) and isinstance(f.value, str) and f.value in val
    return True


class InMemoryVectorStore(VectorStore):
    """Brute-force cosine similarity vector store.

    Parameters
    ----------
    name:
        Store name (for registry identification).
    dimensions:
        Expected vector dimensionality.
    filter_oversample:
        Multiplier for oversampling before metadata filtering.
    """

    name = "in_memory"

    def __init__(
        self,
        dimensions: int = 128,
        filter_oversample: int = 8,
    ) -> None:
        self.dimensions = dimensions
        self._filter_oversample = filter_oversample
        self._lock = threading.RLock()
        self._chunk_ids: list[str] = []
        self._doc_ids: list[str] = []
        self._namespaces: list[str] = []
        self._collection_ids: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        self._vectors: list[np.ndarray] = []

    def add(self, chunks: list[Any], vectors: list[list[float]]) -> None:
        """Store vectors for the given chunks."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        with self._lock:
            for chunk, vec in zip(chunks, vectors, strict=True):
                self._chunk_ids.append(chunk.chunk_id)
                self._doc_ids.append(chunk.doc_id)
                self._namespaces.append(getattr(chunk, "namespace", ""))
                self._collection_ids.append(getattr(chunk, "collection_id", ""))
                self._metadata.append(dict(getattr(chunk, "metadata", {})))
                self._vectors.append(np.array(vec, dtype=np.float32))

    def delete(self, chunk_ids: Iterable[str]) -> None:
        """Remove vectors for the given chunk IDs."""
        id_set = set(chunk_ids)
        with self._lock:
            keep = [i for i, cid in enumerate(self._chunk_ids) if cid not in id_set]
            self._apply_keep(keep)

    def delete_by_document(self, doc_id: str, namespace: str) -> None:
        """Remove all vectors belonging to *doc_id* in *namespace*."""
        with self._lock:
            keep = [
                i
                for i, (d, n) in enumerate(
                    zip(self._doc_ids, self._namespaces, strict=True)
                )
                if not (d == doc_id and n == namespace)
            ]
            self._apply_keep(keep)

    def _apply_keep(self, keep: list[int]) -> None:
        """Rebuild internal lists keeping only indices in *keep*."""
        self._chunk_ids = [self._chunk_ids[i] for i in keep]
        self._doc_ids = [self._doc_ids[i] for i in keep]
        self._namespaces = [self._namespaces[i] for i in keep]
        self._collection_ids = [self._collection_ids[i] for i in keep]
        self._metadata = [self._metadata[i] for i in keep]
        self._vectors = [self._vectors[i] for i in keep]

    def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        filters: tuple[MetadataFilter, ...] = (),
    ) -> list[VectorSearchHit]:
        """Return the *top_k* nearest vectors, optionally filtered."""
        with self._lock:
            if not self._vectors:
                return []

            query = np.array(vector, dtype=np.float32)

            # Compute similarities
            sims = [_cosine_similarity(query, v) for v in self._vectors]

            # If filters, oversample and filter post-search
            if filters:
                oversample = min(len(sims), top_k * self._filter_oversample)
                ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[
                    :oversample
                ]
                ranked = [
                    i
                    for i in ranked
                    if all(_evaluate_filter(self._metadata[i], f) for f in filters)
                ][:top_k]
            else:
                ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[
                    :top_k
                ]

            return [
                VectorSearchHit(
                    chunk_id=self._chunk_ids[i],
                    doc_id=self._doc_ids[i],
                    namespace=self._namespaces[i],
                    collection_id=self._collection_ids[i],
                    score=sims[i],
                    metadata=self._metadata[i],
                )
                for i in ranked
            ]

    def count(self) -> int:
        """Return the total number of stored vectors."""
        with self._lock:
            return len(self._vectors)

    def clear(self) -> None:
        """Remove all stored vectors."""
        with self._lock:
            self._chunk_ids.clear()
            self._doc_ids.clear()
            self._namespaces.clear()
            self._collection_ids.clear()
            self._metadata.clear()
            self._vectors.clear()
