"""Document loading and chunking subpackages.

Provides the format parser, concrete loaders (TXT, Markdown, JSON,
PDF, HTML), and concrete chunkers (fixed, paragraph, sentence,
recursive).
"""

from __future__ import annotations

from app.knowledge.documents.parser import FormatParser

__all__ = ["FormatParser"]
