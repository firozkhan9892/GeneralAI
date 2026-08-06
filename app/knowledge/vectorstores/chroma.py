"""Chroma vector store.

Uses ``chromadb`` (lazy-imported) for persistent or in-memory vector
storage.  Maps each knowledge collection to a Chroma collection.
"""

from __future__ import annotations

import threading
from typing import Any, Iterable

from app.knowledge.exceptions import KnowledgeIndexError
from app.knowledge.models import MetadataFilter, VectorSearchHit


def _import_chromadb():  # type: ignore[no-untyped-def]
    """Lazy-import ``chromadb``."""
    try:
        import chromadb as _chroma

        return _chroma
    except ImportError as exc:
        raise KnowledgeIndexError(
            "chromadb is required for ChromaVectorStore.  "
            "Install it with: pip install chromadb",
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


class ChromaVectorStore:
    """ChromaDB-backed vector store.

    Parameters
    ----------
    collection_name:
        The Chroma collection name.
    dimensions:
        Expected vector dimensionality (used for informational purposes).
    filter_oversample:
        Multiplier for oversampling before metadata filtering.
    """

    name = "chroma"

    def __init__(
        self,
        collection_name: str = "knowledge",
        dimensions: int = 128,
        filter_oversample: int = 8,
    ) -> None:
        chroma = _import_chromadb()
        self.dimensions = dimensions
        self._filter_oversample = filter_oversample
        self._lock = threading.RLock()
        self._client = chroma.Client()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Any], vectors: list[list[float]]) -> None:
        """Store vectors for the given chunks."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        with self._lock:
            ids = [chunk.chunk_id for chunk in chunks]
            metadatas = []
            for chunk in chunks:
                meta = dict(getattr(chunk, "metadata", {}))
                meta["doc_id"] = chunk.doc_id
                meta["namespace"] = getattr(chunk, "namespace", "")
                meta["collection_id"] = getattr(chunk, "collection_id", "")
                metadatas.append(meta)
            documents = [getattr(chunk, "content", "") for chunk in chunks]

            self._collection.add(
                ids=ids,
                embeddings=vectors,
                metadatas=metadatas,
                documents=documents,
            )

    def delete(self, chunk_ids: Iterable[str]) -> None:
        """Remove vectors by chunk IDs."""
        with self._lock:
            self._collection.delete(ids=list(chunk_ids))

    def delete_by_document(self, doc_id: str, namespace: str) -> None:
        """Remove all vectors for *doc_id* in *namespace*."""
        with self._lock:
            self._collection.delete(
                where={"$and": [{"doc_id": doc_id}, {"namespace": namespace}]}
            )

    def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        filters: tuple[MetadataFilter, ...] = (),
    ) -> list[VectorSearchHit]:
        """Return the *top_k* nearest vectors, optionally filtered."""
        with self._lock:
            where = None
            if filters:
                where_parts = []
                for f in filters:
                    if f.op == "eq":
                        where_parts.append({f.field: f.value})
                    elif f.op == "in":
                        where_parts.append({f.field: {"$in": f.value or []}})
                if where_parts:
                    where = (
                        {"$and": where_parts}
                        if len(where_parts) > 1
                        else where_parts[0]
                    )

            kwargs: dict[str, Any] = {
                "query_embeddings": [vector],
                "n_results": top_k * self._filter_oversample if filters else top_k,
            }
            if where is not None:
                kwargs["where"] = where

            results = self._collection.query(**kwargs)

            hits: list[VectorSearchHit] = []
            if results and results.get("ids") and results["ids"][0]:
                ids = results["ids"][0]
                distances = results.get("distances", [[]])[0] or []
                metadatas = results.get("metadatas", [[]])[0] or []
                for idx, chunk_id in enumerate(ids):
                    meta = metadatas[idx] if idx < len(metadatas) else {}
                    score = 1.0 - distances[idx] if idx < len(distances) else 0.0
                    # Apply post-filter for ops not supported by Chroma
                    if filters and not all(_evaluate_filter(meta, f) for f in filters):
                        continue
                    hits.append(
                        VectorSearchHit(
                            chunk_id=chunk_id,
                            doc_id=meta.get("doc_id", ""),
                            namespace=meta.get("namespace", ""),
                            collection_id=meta.get("collection_id", ""),
                            score=float(score),
                            metadata=meta,
                        )
                    )
                    if len(hits) >= top_k:
                        break
            return hits

    def count(self) -> int:
        """Return the total number of stored vectors."""
        with self._lock:
            return self._collection.count()

    def clear(self) -> None:
        """Remove all stored vectors."""
        with self._lock:
            self._client.delete_collection(self._collection.name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection.name,
                metadata={"hnsw:space": "cosine"},
            )
