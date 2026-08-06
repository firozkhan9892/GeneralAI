"""Concrete document chunkers.

Re-exports every chunker class so callers can do::

    from app.knowledge.documents.chunkers import FixedChunker, RecursiveChunker
"""

from __future__ import annotations

from app.knowledge.documents.chunkers.fixed import FixedChunker
from app.knowledge.documents.chunkers.paragraph import ParagraphChunker
from app.knowledge.documents.chunkers.recursive import RecursiveChunker
from app.knowledge.documents.chunkers.sentence import SentenceChunker

__all__ = [
    "FixedChunker",
    "ParagraphChunker",
    "SentenceChunker",
    "RecursiveChunker",
]
