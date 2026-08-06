"""Plain-text document loader.

Handles ``.txt`` and other plain-text formats.  Optionally attempts
to detect structured headers (Markdown ``#`` or underlines) to
populate the document title.
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
from app.knowledge.utils import compute_content_hash, extract_title_from_content


class TextLoader(DocumentLoader):
    """Loader for plain-text files (``.txt``)."""

    format = DocumentFormat.TXT

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
                f"Failed to decode text file: {exc}",
                context={"source_uri": source_uri},
            ) from exc

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
            title=extract_title_from_content(text),
            source_uri=source_uri,
            format=DocumentFormat.TXT,
            content=text,
            content_hash=compute_content_hash(text),
            status=IndexStatus.PENDING,
            metadata=merged_meta,
        )
