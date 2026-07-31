"""Memory interface placeholder."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from app.core.interfaces.base import IModule


class IMemory(IModule):
    """Contract for storage and retrieval implementations.

    Supports storing, querying, and forgetting information with
    optional metadata filtering.
    """

    @abstractmethod
    async def store(
        self, key: str, value: Any, metadata: dict[str, Any] | None = None
    ) -> None:
        """Persist a value identified by *key*.

        Args:
            key: Unique identifier for the stored data.
            value: The data to store.
            metadata: Optional key-value pairs for filtering.
        """

    @abstractmethod
    async def retrieve(self, key: str) -> Any | None:
        """Retrieve a value by its *key*.

        Args:
            key: The identifier used during storage.

        Returns:
            The stored value, or ``None`` if not found.
        """

    @abstractmethod
    async def forget(self, key: str) -> bool:
        """Remove a stored value.

        Args:
            key: The identifier to remove.

        Returns:
            ``True`` if the value existed and was removed.
        """

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search stored values by semantic or keyword query.

        Args:
            query: Natural-language or keyword query string.
            limit: Maximum number of results to return.

        Returns:
            List of matching records as key-value-metadata dicts.
        """
