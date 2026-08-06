"""Markdown document loader.

Parses Markdown files, preserving heading structure as metadata so
downstream chunkers can split on headings.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from app.knowledge.base import DocumentLoader
from app.knowledge.constants import META_KEY_FORMAT, META_KEY_SOURCE_URI
from app.knowledge.exceptions import KnowledgeIngestionError
from app.knowledge.models import (
    DocumentFormat,
    IndexStatus,
    KnowledgeDocument,
)
from app.knowledge.utils import compute_content_hash


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)


def _extract_headings(text: str) -> list[dict[str, Any]]:
    """Return a list of ``{level, text}`` dicts for each Markdown heading."""
    headings: list[dict[str, Any]] = []
    for match in _HEADING_RE.finditer(text):
        headings.append({"level": len(match.group(1)), "text": match.group(2).strip()})
    return headings


def _extract_title(text: str) -> str:
    """Return the first ``#`` heading or first non-empty line."""
    first_heading = _HEADING_RE.search(text)
    if first_heading:
        return first_heading.group(2).strip()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:120]
    return ""


class MarkdownLoader(DocumentLoader):
    """Loader for Markdown files (``.md``)."""

    format = DocumentFormat.MARKDOWN

    def load(
        self,
        content: bytes,
        *,
        source_uri: str,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeDocument:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise KnowledgeIngestionError(
                f"Failed to decode Markdown file: {exc}",
                context={"source_uri": source_uri},
            ) from exc

        headings = _extract_headings(text)
        title = _extract_title(text)

        merged_meta: dict[str, Any] = {
            META_KEY_FORMAT: self.format.value,
            META_KEY_SOURCE_URI: source_uri,
            "headings": headings,
        }
        if metadata:
            merged_meta.update(metadata)

        return KnowledgeDocument(
            doc_id=uuid.uuid4().hex,
            collection_id=merged_meta.pop("collection_id", ""),
            namespace=merged_meta.pop("namespace", ""),
            title=title,
            source_uri=source_uri,
            format=DocumentFormat.MARKDOWN,
            content=text,
            content_hash=compute_content_hash(text),
            status=IndexStatus.PENDING,
            metadata=merged_meta,
        )
