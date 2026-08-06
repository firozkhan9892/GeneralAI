"""Vector store sub-package.

Provides the concrete vector stores and package exports.
"""

from __future__ import annotations

from app.knowledge.vectorstores.in_memory import InMemoryVectorStore

__all__ = ["InMemoryVectorStore"]
