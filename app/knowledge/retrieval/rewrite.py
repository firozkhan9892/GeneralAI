"""Query rewriting implementations.

Provides the identity (no-op) rewriter as the default.  An optional
LLM-based rewriter can be added later by implementing the
:class:`QueryRewriter` ABC from :mod:`app.knowledge.base`.
"""

from __future__ import annotations

from app.knowledge.base import QueryRewriter, RetrievalContext


class IdentityQueryRewriter(QueryRewriter):
    """Returns the raw query unchanged.

    This is the safe default: zero risk, zero latency, deterministic
    in tests.  Use when no LLM is available or rewriting is disabled.
    """

    name: str = "identity"

    async def rewrite(self, raw: str, *, context: RetrievalContext) -> str:
        """Return *raw* unchanged."""
        return raw
