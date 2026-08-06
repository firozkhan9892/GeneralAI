"""Concrete embedding providers.

Re-exports every provider class so callers can do::

    from app.knowledge.embeddings.providers import MockEmbeddingProvider
"""

from __future__ import annotations

from app.knowledge.embeddings.mock import MockEmbeddingProvider
from app.knowledge.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)

__all__ = [
    "MockEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
]
