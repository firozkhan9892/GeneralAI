"""Concrete document loaders.

Re-exports every loader class so callers can do::

    from app.knowledge.documents.loaders import TextLoader, PdfLoader
"""

from __future__ import annotations

from app.knowledge.documents.loaders.html import HtmlLoader
from app.knowledge.documents.loaders.json_loader import JsonLoader
from app.knowledge.documents.loaders.markdown import MarkdownLoader
from app.knowledge.documents.loaders.pdf import PdfLoader
from app.knowledge.documents.loaders.text import TextLoader

__all__ = [
    "TextLoader",
    "MarkdownLoader",
    "JsonLoader",
    "PdfLoader",
    "HtmlLoader",
]
