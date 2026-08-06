"""Reranking implementations.

Provides the identity (no-op) reranker as the default.  An optional
cross-encoder reranker can be added later by implementing the
:class:`Reranker` ABC from :mod:`app.knowledge.base`.
"""

from __future__ import annotations

from app.knowledge.base import Reranker
from app.knowledge.models import RetrievalHit


class IdentityReranker(Reranker):
    """Returns hits in the original order.

    This is the safe default: zero risk, zero latency, deterministic
    in tests.  Use when no cross-encoder model is available or
    reranking is disabled.
    """

    name: str = "identity"

    async def rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        """Return *hits* in their original order."""
        return hits
