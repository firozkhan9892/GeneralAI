"""Format dispatch for document loaders.

Maps file extensions (and optional MIME types) to the appropriate
:class:`DocumentLoader` instance.  The parser maintains a thread-safe
registry of extension → loader mappings and provides a single
``parse`` entry point.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from app.knowledge.base import DocumentLoader
from app.knowledge.exceptions import (
    KnowledgeUnsupportedFormatError,
)
from app.knowledge.models import DocumentFormat


class FormatParser:
    """Dispatches raw bytes to the correct loader based on file extension.

    Thread-safe: extension mappings can be updated while parsing
    concurrently.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._extensions: dict[str, DocumentLoader] = {}
        self._mime_types: dict[str, DocumentLoader] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register built-in format handlers."""
        from app.knowledge.documents.loaders.html import HtmlLoader
        from app.knowledge.documents.loaders.json_loader import JsonLoader
        from app.knowledge.documents.loaders.markdown import MarkdownLoader
        from app.knowledge.documents.loaders.pdf import PdfLoader
        from app.knowledge.documents.loaders.text import TextLoader

        loaders = [
            (".txt", None, TextLoader()),
            (".md", None, MarkdownLoader()),
            (".markdown", None, MarkdownLoader()),
            (".json", "application/json", JsonLoader()),
            (".pdf", "application/pdf", PdfLoader()),
            (".html", "text/html", HtmlLoader()),
            (".htm", "text/html", HtmlLoader()),
        ]
        for ext, mime, loader in loaders:
            self._extensions[ext] = loader
            if mime:
                self._mime_types[mime] = loader

    def register(self, ext: str, loader: DocumentLoader) -> None:
        """Register a loader for a file extension (e.g. ``".docx"``)."""
        with self._lock:
            self._extensions[ext.lower()] = loader

    def register_mime(self, mime: str, loader: DocumentLoader) -> None:
        """Register a loader for a MIME type."""
        with self._lock:
            self._mime_types[mime.lower()] = loader

    def get_loader_for_extension(self, ext: str) -> DocumentLoader:
        """Return the loader for *ext* (e.g. ``".pdf"``).

        Raises:
            KnowledgeUnsupportedFormatError: If no loader is registered.
        """
        with self._lock:
            loader = self._extensions.get(ext.lower())
        if loader is None:
            raise KnowledgeUnsupportedFormatError(
                f"No loader registered for extension '{ext}'",
                context={"extension": ext},
            )
        return loader

    def get_loader_for_mime(self, mime: str) -> DocumentLoader:
        """Return the loader for *mime*.

        Raises:
            KnowledgeUnsupportedFormatError: If no loader is registered.
        """
        with self._lock:
            loader = self._mime_types.get(mime.lower())
        if loader is None:
            raise KnowledgeUnsupportedFormatError(
                f"No loader registered for MIME type '{mime}'",
                context={"mime": mime},
            )
        return loader

    def detect_format(self, path: str) -> DocumentFormat:
        """Detect the :class:`DocumentFormat` from a file path."""
        ext = os.path.splitext(path)[1].lower()
        loader = self._extensions.get(ext)
        if loader is not None:
            return loader.format
        return DocumentFormat.TXT

    def parse(
        self,
        content: bytes,
        *,
        source_uri: str,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Parse *content* using the loader matching the file extension of *source_uri*.

        Raises:
            KnowledgeUnsupportedFormatError: If no loader is registered.
            KnowledgeIngestionError: If parsing fails.
        """
        ext = os.path.splitext(source_uri)[1].lower()
        loader = self.get_loader_for_extension(ext)
        return loader.load(content, source_uri=source_uri, metadata=metadata)

    async def parse_async(
        self,
        content: bytes,
        *,
        source_uri: str,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Async variant of :meth:`parse` (thread-offloaded)."""
        import asyncio

        return await asyncio.to_thread(
            self.parse, content, source_uri=source_uri, metadata=metadata
        )
