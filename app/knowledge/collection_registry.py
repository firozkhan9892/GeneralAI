"""Collection registry for managing knowledge collections.

Thread-safe registry that tracks :class:`CollectionMetadata` instances
by collection ID.  Provides CRUD operations, counting, and namespace
scoping.
"""

from __future__ import annotations

import threading
from typing import Iterator

from app.knowledge.exceptions import (
    KnowledgeCollectionNotFoundError,
    KnowledgeDuplicateError,
)
from app.knowledge.models import CollectionMetadata


class CollectionRegistry:
    """Thread-safe in-memory store for :class:`CollectionMetadata`.

    Collections are keyed by ``collection_id``.  Namespace-scoped
    queries return filtered views without mutating the store.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._collections: dict[str, CollectionMetadata] = {}

    def add(self, collection: CollectionMetadata) -> None:
        """Register a collection.

        Raises:
            KnowledgeDuplicateError: If a collection with the same ID
                already exists.
        """
        with self._lock:
            if collection.collection_id in self._collections:
                raise KnowledgeDuplicateError(
                    f"Collection '{collection.collection_id}' already exists",
                    context={"collection_id": collection.collection_id},
                )
            self._collections[collection.collection_id] = collection

    def get(self, collection_id: str) -> CollectionMetadata:
        """Return a collection by ID.

        Raises:
            KnowledgeCollectionNotFoundError: If not found.
        """
        with self._lock:
            coll = self._collections.get(collection_id)
        if coll is None:
            raise KnowledgeCollectionNotFoundError(
                f"Collection '{collection_id}' not found",
                context={"collection_id": collection_id},
            )
        return coll

    def update(self, collection: CollectionMetadata) -> None:
        """Replace an existing collection record.

        Raises:
            KnowledgeCollectionNotFoundError: If not found.
        """
        with self._lock:
            if collection.collection_id not in self._collections:
                raise KnowledgeCollectionNotFoundError(
                    f"Collection '{collection.collection_id}' not found",
                    context={"collection_id": collection.collection_id},
                )
            self._collections[collection.collection_id] = collection

    def delete(self, collection_id: str) -> CollectionMetadata:
        """Remove and return a collection.

        Raises:
            KnowledgeCollectionNotFoundError: If not found.
        """
        with self._lock:
            removed = self._collections.pop(collection_id, None)
        if removed is None:
            raise KnowledgeCollectionNotFoundError(
                f"Collection '{collection_id}' not found",
                context={"collection_id": collection_id},
            )
        return removed

    def list_all(self) -> list[CollectionMetadata]:
        """Return a snapshot of all collections."""
        with self._lock:
            return list(self._collections.values())

    def list_by_namespace(self, namespace: str) -> list[CollectionMetadata]:
        """Return collections in *namespace*."""
        with self._lock:
            return [c for c in self._collections.values() if c.namespace == namespace]

    def count(self) -> int:
        """Return total number of registered collections."""
        with self._lock:
            return len(self._collections)

    def count_by_namespace(self, namespace: str) -> int:
        """Return number of collections in *namespace*."""
        with self._lock:
            return sum(
                1 for c in self._collections.values() if c.namespace == namespace
            )

    def exists(self, collection_id: str) -> bool:
        """Check whether a collection is registered."""
        with self._lock:
            return collection_id in self._collections

    def __iter__(self) -> Iterator[CollectionMetadata]:
        """Iterate over all collections (snapshot-based)."""
        return iter(self.list_all())

    def __len__(self) -> int:
        return self.count()
