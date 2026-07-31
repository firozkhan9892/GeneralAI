"""Generic, thread-safe base registry.

Typed registries inherit from this class to provide type-safe
register / get / list operations.
"""

from __future__ import annotations

import threading
from typing import Generic, Iterator, TypeVar

T = TypeVar("T")


class BaseRegistry(Generic[T]):
    """Thread-safe generic registry.

    Stores items keyed by a string identifier.  All mutation methods
    are protected by a re-entrant lock.
    """

    def __init__(self, max_items: int = 10_000) -> None:
        self._items: dict[str, T] = {}
        self._lock = threading.RLock()
        self._max_items = max_items

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, key: str, item: T, overwrite: bool = False) -> None:
        """Register *item* under *key*.

        Args:
            key: Unique identifier.
            item: The item to store.
            overwrite: If ``True``, replace an existing entry.

        Raises:
            ValueError: If *key* already exists and *overwrite* is ``False``.
        """
        with self._lock:
            if key in self._items and not overwrite:
                raise ValueError(
                    f"Key '{key}' is already registered. Use overwrite=True to replace."
                )
            if len(self._items) >= self._max_items:
                raise ValueError(f"Registry capacity ({self._max_items}) exceeded")
            self._items[key] = item

    def unregister(self, key: str) -> None:
        """Remove the item registered under *key*."""
        with self._lock:
            self._items.pop(key, None)

    def clear(self) -> None:
        """Remove all items."""
        with self._lock:
            self._items.clear()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, key: str) -> T | None:
        """Return the item registered under *key*, or ``None``."""
        with self._lock:
            return self._items.get(key)

    def get_or_raise(self, key: str) -> T:
        """Return the item registered under *key*, or raise :class:`KeyError`."""
        with self._lock:
            if key not in self._items:
                raise KeyError(f"Key '{key}' not found in registry")
            return self._items[key]

    def has(self, key: str) -> bool:
        """Return ``True`` if *key* is registered."""
        with self._lock:
            return key in self._items

    def values(self) -> list[T]:
        """Return a snapshot of all registered items."""
        with self._lock:
            return list(self._items.values())

    def keys(self) -> list[str]:
        """Return a snapshot of all registered keys."""
        with self._lock:
            return list(self._items.keys())

    @property
    def count(self) -> int:
        """Return the number of registered items."""
        with self._lock:
            return len(self._items)

    @property
    def is_empty(self) -> bool:
        """Return ``True`` if no items are registered."""
        with self._lock:
            return len(self._items) == 0

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[T]:
        return iter(self.values())
