"""Embedding sub-package.

Provides the embedding cache, concrete providers, and package exports.
"""

from __future__ import annotations

from app.knowledge.embeddings.cache import CacheStats, EmbeddingCache
from app.knowledge.embeddings.mock import MockEmbeddingProvider

__all__ = [
    "EmbeddingCache",
    "CacheStats",
    "MockEmbeddingProvider",
]
