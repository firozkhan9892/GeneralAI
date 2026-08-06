"""FAISS vector store.

Uses ``faiss-cpu`` (lazy-imported) for approximate nearest-neighbour
search.  Falls back to raising ``KnowledgeIndexError`` when the
dependency is unavailable.
"""

from __future__ import annotations

import threading
from typing import Any, Iterable

from app.knowledge.exceptions import KnowledgeIndexError
from app.knowledge.models import MetadataFilter, VectorSearchHit


def _import_faiss():  # type: ignore[no-untyped-def]
    """Lazy-import ``faiss``."""
    try:
        import faiss as _faiss

        return _faiss
    except ImportError as exc:
        raise KnowledgeIndexError(
            "faiss-cpu is required for FAISSVectorStore.  "
            "Install it with: pip install faiss-cpu",
            cause=exc,
        ) from exc


def _evaluate_filter(metadata: dict[str, Any], f: MetadataFilter) -> bool:
    """Evaluate a single metadata filter."""
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


class FAISSVectorStore:
    """FAISS-based vector store using IndexFlatIP (inner product on normalised vectors).

    Parameters
    ----------
    dimensions:
        Vector dimensionality.
    filter_oversample:
        Multiplier for oversampling before metadata filtering.
    """

    name = "faiss"

    def __init__(
        self,
        dimensions: int = 128,
        filter_oversample: int = 8,
    ) -> None:
        faiss = _import_faiss()
        self.dimensions = dimensions
        self._filter_oversample = filter_oversample
        self._lock = threading.RLock()
        self._index = faiss.IndexFlatIP(dimensions)
        self._chunk_ids: list[str] = []
        self._doc_ids: list[str] = []
        self._namespaces: list[str] = []
        self._collection_ids: list[str] = []
        self._metadata: list[dict[str, Any]] = []

    def add(self, chunks: list[Any], vectors: list[list[float]]) -> None:
        """Store vectors for the given chunks."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        import numpy as np

        with self._lock:
            arr = np.array(vectors, dtype=np.float32)
            self._index.add(arr)
            for chunk in chunks:
                self._chunk_ids.append(chunk.chunk_id)
                self._doc_ids.append(chunk.doc_id)
                self._namespaces.append(getattr(chunk, "namespace", ""))
                self._collection_ids.append(getattr(chunk, "collection_id", ""))
                self._metadata.append(dict(getattr(chunk, "metadata", {})))

    def delete(self, chunk_ids: Iterable[str]) -> None:
        """Remove vectors by chunk IDs.

        FAISS IndexFlatIP does not support in-place deletion, so we
        rebuild the index excluding the deleted IDs.
        """
        id_set = set(chunk_ids)
        with self._lock:
            keep = [i for i, cid in enumerate(self._chunk_ids) if cid not in id_set]
            self._rebuild(keep)

    def delete_by_document(self, doc_id: str, namespace: str) -> None:
        """Remove all vectors for *doc_id* in *namespace*."""
        with self._lock:
            keep = [
                i
                for i, (d, n) in enumerate(
                    zip(self._doc_ids, self._namespaces, strict=True)
                )
                if not (d == doc_id and n == namespace)
            ]
            self._rebuild(keep)

    def _rebuild(self, keep: list[int]) -> None:
        """Rebuild the FAISS index keeping only *keep* indices."""
        import numpy as np

        faiss = _import_faiss()
        if not keep:
            self._index = faiss.IndexFlatIP(self.dimensions)
            self._chunk_ids.clear()
            self._doc_ids.clear()
            self._namespaces.clear()
            self._collection_ids.clear()
            self._metadata.clear()
            return

        # Extract kept vectors from the index (reconstruct from flat storage)
        all_vectors = np.vstack([self._index.reconstruct(i) for i in keep]).astype(
            np.float32
        )

        self._index = faiss.IndexFlatIP(self.dimensions)
        self._index.add(all_vectors)

        self._chunk_ids = [self._chunk_ids[i] for i in keep]
        self._doc_ids = [self._doc_ids[i] for i in keep]
        self._namespaces = [self._namespaces[i] for i in keep]
        self._collection_ids = [self._collection_ids[i] for i in keep]
        self._metadata = [self._metadata[i] for i in keep]

    def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        filters: tuple[MetadataFilter, ...] = (),
    ) -> list[VectorSearchHit]:
        """Return the *top_k* nearest vectors, optionally filtered."""
        import numpy as np

        with self._lock:
            if self._index.ntotal == 0:
                return []

            query = np.array([vector], dtype=np.float32)

            if filters:
                oversample = min(self._index.ntotal, top_k * self._filter_oversample)
                distances, idx_arr = self._index.search(query, oversample)
                scores = distances[0].tolist()
                indices = idx_arr[0].tolist()
                results = []
                for score, idx in zip(scores, indices, strict=True):
                    if idx < 0:
                        continue
                    if all(_evaluate_filter(self._metadata[idx], f) for f in filters):
                        results.append(
                            VectorSearchHit(
                                chunk_id=self._chunk_ids[idx],
                                doc_id=self._doc_ids[idx],
                                namespace=self._namespaces[idx],
                                collection_id=self._collection_ids[idx],
                                score=float(score),
                                metadata=self._metadata[idx],
                            )
                        )
                        if len(results) >= top_k:
                            break
                return results
            else:
                distances, idx_arr = self._index.search(query, top_k)
                results = []
                for score, idx in zip(
                    distances[0].tolist(), idx_arr[0].tolist(), strict=True
                ):
                    if idx < 0:
                        continue
                    results.append(
                        VectorSearchHit(
                            chunk_id=self._chunk_ids[idx],
                            doc_id=self._doc_ids[idx],
                            namespace=self._namespaces[idx],
                            collection_id=self._collection_ids[idx],
                            score=float(score),
                            metadata=self._metadata[idx],
                        )
                    )
                return results

    def count(self) -> int:
        """Return the total number of stored vectors."""
        with self._lock:
            return self._index.ntotal

    def clear(self) -> None:
        """Remove all stored vectors."""
        with self._lock:
            self._index = _import_faiss().IndexFlatIP(self.dimensions)
            self._chunk_ids.clear()
            self._doc_ids.clear()
            self._namespaces.clear()
            self._collection_ids.clear()
            self._metadata.clear()
