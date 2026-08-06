"""PDF document loader.

Uses ``pypdf`` (lazy-imported) to extract text page by page.  If
``pypdf`` is not installed the loader raises a clear error at load
time rather than at import time.
"""

from __future__ import annotations

import uuid
from typing import Any

from app.knowledge.base import DocumentLoader
from app.knowledge.constants import META_KEY_FORMAT, META_KEY_PAGE, META_KEY_SOURCE_URI
from app.knowledge.exceptions import KnowledgeIngestionError
from app.knowledge.models import (
    DocumentFormat,
    IndexStatus,
    KnowledgeDocument,
)
from app.knowledge.utils import compute_content_hash, extract_title_from_content


def _import_pypdf():  # type: ignore[no-untyped-def]
    """Lazy-import ``pypdf`` and raise a clear error if missing."""
    try:
        import pypdf as _pypdf

        return _pypdf
    except ImportError as exc:
        raise KnowledgeIngestionError(
            "pypdf is required for PDF loading.  Install it with: pip install pypdf",
            cause=exc,
        ) from exc


class PdfLoader(DocumentLoader):
    """Loader for PDF files (``.pdf``).

    Extracts text page-by-page and concatenates with page-break
    markers so chunkers can respect page boundaries.
    """

    format = DocumentFormat.PDF

    def load(
        self,
        content: bytes,
        *,
        source_uri: str,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeDocument:
        pypdf = _import_pypdf()
        import io

        try:
            reader = pypdf.PdfReader(io.BytesIO(content))
        except Exception as exc:
            raise KnowledgeIngestionError(
                f"Failed to open PDF: {exc}",
                context={"source_uri": source_uri},
            ) from exc

        pages: list[str] = []
        page_metadata: list[dict[str, Any]] = []
        for idx, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                raise KnowledgeIngestionError(
                    f"Failed to extract text from PDF page {idx + 1}: {exc}",
                    context={"source_uri": source_uri},
                ) from exc
            pages.append(text)
            page_metadata.append({META_KEY_PAGE: idx + 1})

        full_text = "\n\n---\n\n".join(pages)

        merged_meta: dict[str, Any] = {
            META_KEY_FORMAT: self.format.value,
            META_KEY_SOURCE_URI: source_uri,
            "page_count": len(pages),
            "page_metadata": page_metadata,
        }
        if metadata:
            merged_meta.update(metadata)

        return KnowledgeDocument(
            doc_id=uuid.uuid4().hex,
            collection_id=merged_meta.pop("collection_id", ""),
            namespace=merged_meta.pop("namespace", ""),
            title=extract_title_from_content(full_text),
            source_uri=source_uri,
            format=DocumentFormat.PDF,
            content=full_text,
            content_hash=compute_content_hash(full_text),
            status=IndexStatus.PENDING,
            metadata=merged_meta,
        )
