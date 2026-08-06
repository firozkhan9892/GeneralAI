"""Namespace registry for managing knowledge namespaces.

Thread-safe registry that tracks :class:`NamespaceMetadata` instances
by namespace name.  Provides CRUD operations and counting.
"""

from __future__ import annotations

import threading
from typing import Iterator

from app.knowledge.constants import DEFAULT_NAMESPACE
from app.knowledge.exceptions import (
    KnowledgeDuplicateError,
    KnowledgeNamespaceNotFoundError,
)
from app.knowledge.models import NamespaceMetadata


class NamespaceRegistry:
    """Thread-safe in-memory store for :class:`NamespaceMetadata`.

    Namespaces are keyed by ``name``.  The ``"default"`` namespace is
    pre-registered on construction.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._namespaces: dict[str, NamespaceMetadata] = {}
        # Pre-register the default namespace
        self._namespaces[DEFAULT_NAMESPACE] = NamespaceMetadata(
            name=DEFAULT_NAMESPACE,
            description="Default namespace",
        )

    def add(self, namespace: NamespaceMetadata) -> None:
        """Register a namespace.

        Raises:
            KnowledgeDuplicateError: If a namespace with the same name
                already exists.
        """
        with self._lock:
            if namespace.name in self._namespaces:
                raise KnowledgeDuplicateError(
                    f"Namespace '{namespace.name}' already exists",
                    context={"namespace": namespace.name},
                )
            self._namespaces[namespace.name] = namespace

    def get(self, name: str) -> NamespaceMetadata:
        """Return a namespace by name.

        Raises:
            KnowledgeNamespaceNotFoundError: If not found.
        """
        with self._lock:
            ns = self._namespaces.get(name)
        if ns is None:
            raise KnowledgeNamespaceNotFoundError(
                f"Namespace '{name}' not found",
                context={"namespace": name},
            )
        return ns

    def update(self, namespace: NamespaceMetadata) -> None:
        """Replace an existing namespace record.

        Raises:
            KnowledgeNamespaceNotFoundError: If not found.
        """
        with self._lock:
            if namespace.name not in self._namespaces:
                raise KnowledgeNamespaceNotFoundError(
                    f"Namespace '{namespace.name}' not found",
                    context={"namespace": namespace.name},
                )
            self._namespaces[namespace.name] = namespace

    def delete(self, name: str) -> NamespaceMetadata:
        """Remove and return a namespace.

        Raises:
            KnowledgeNamespaceNotFoundError: If not found.
            KnowledgeValidationError: If attempting to delete the
                default namespace.
        """
        if name == DEFAULT_NAMESPACE:
            from app.knowledge.exceptions import KnowledgeValidationError

            raise KnowledgeValidationError(
                "Cannot delete the default namespace",
                context={"namespace": name},
            )
        with self._lock:
            removed = self._namespaces.pop(name, None)
        if removed is None:
            raise KnowledgeNamespaceNotFoundError(
                f"Namespace '{name}' not found",
                context={"namespace": name},
            )
        return removed

    def list_all(self) -> list[NamespaceMetadata]:
        """Return a snapshot of all namespaces."""
        with self._lock:
            return list(self._namespaces.values())

    def count(self) -> int:
        """Return total number of registered namespaces."""
        with self._lock:
            return len(self._namespaces)

    def exists(self, name: str) -> bool:
        """Check whether a namespace is registered."""
        with self._lock:
            return name in self._namespaces

    def __iter__(self) -> Iterator[NamespaceMetadata]:
        """Iterate over all namespaces (snapshot-based)."""
        return iter(self.list_all())

    def __len__(self) -> int:
        return self.count()
