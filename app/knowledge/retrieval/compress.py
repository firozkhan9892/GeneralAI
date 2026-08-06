"""Context compression implementations.

Provides the identity (no-op) compressor as the default.  An optional
LLM-based compressor can be added later by implementing the
:class:`ContextCompressor` ABC from :mod:`app.knowledge.base`.
"""

from __future__ import annotations

from app.knowledge.base import ContextCompressor
from app.knowledge.models import RetrievalHit


class IdentityCompressor(ContextCompressor):
    """Returns hits unchanged.

    This is the safe default: zero risk, zero latency, deterministic
    in tests.  Use when no LLM is available or compression is disabled.
    """

    name: str = "identity"

    async def compress(
        self, hits: list[RetrievalHit], *, query: str
    ) -> list[RetrievalHit]:
        """Return *hits* unchanged."""
        return hits
