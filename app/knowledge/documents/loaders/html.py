"""HTML document loader.

Uses ``beautifulsoup4`` (lazy-imported) to extract visible text from
HTML, separating block elements into paragraphs.
"""

from __future__ import annotations

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

_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "main",
        "header",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "blockquote",
        "pre",
        "tr",
        "td",
        "th",
    }
)


def _import_bs4():  # type: ignore[no-untyped-def]
    """Lazy-import ``beautifulsoup4``."""
    try:
        from bs4 import BeautifulSoup as _Soup

        return _Soup
    except ImportError as exc:
        raise KnowledgeIngestionError(
            "beautifulsoup4 is required for HTML loading.  Install with: pip install beautifulsoup4",
            cause=exc,
        ) from exc


def _extract_text(soup: Any) -> str:
    """Extract visible text from a BeautifulSoup tree.

    Block-level elements are separated by double newlines; inline
    text within a block is joined by single spaces.
    """
    parts: list[str] = []
    for element in soup.find_all(True):
        if element.name in _BLOCK_TAGS:
            text = element.get_text(separator=" ", strip=True)
            if text:
                parts.append(text)
                parts.append("")  # blank line between blocks
    # Fallback: if no block elements found, just get all text
    if not parts:
        text = soup.get_text(separator=" ", strip=True)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _extract_title(soup: Any) -> str:
    """Extract title from ``<title>``, ``<h1>``, or Open Graph meta."""
    # Try <title>
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        return title_tag.string.strip()[:120]
    # Try <h1>
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)[:120]
    # Try og:title
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return str(og["content"])[:120]
    return ""


class HtmlLoader(DocumentLoader):
    """Loader for HTML files (``.html``)."""

    format = DocumentFormat.HTML

    def load(
        self,
        content: bytes,
        *,
        source_uri: str,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeDocument:
        BeautifulSoup = _import_bs4()

        try:
            html = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise KnowledgeIngestionError(
                f"Failed to decode HTML file: {exc}",
                context={"source_uri": source_uri},
            ) from exc

        soup = BeautifulSoup(html, "html.parser")

        # Remove script/style elements
        for tag in soup.find_all(["script", "style"]):
            tag.decompose()

        text = _extract_text(soup)
        title = _extract_title(soup)

        merged_meta: dict[str, Any] = {
            META_KEY_FORMAT: self.format.value,
            META_KEY_SOURCE_URI: source_uri,
        }
        if metadata:
            merged_meta.update(metadata)

        return KnowledgeDocument(
            doc_id=uuid.uuid4().hex,
            collection_id=merged_meta.pop("collection_id", ""),
            namespace=merged_meta.pop("namespace", ""),
            title=title,
            source_uri=source_uri,
            format=DocumentFormat.HTML,
            content=text,
            content_hash=compute_content_hash(text),
            status=IndexStatus.PENDING,
            metadata=merged_meta,
        )
