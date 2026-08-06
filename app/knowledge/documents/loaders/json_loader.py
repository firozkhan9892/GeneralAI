"""JSON document loader.

Handles JSON files by extracting text content from a configurable
field (default ``"content"``) or by serialising the entire payload
if no field is specified.
"""

from __future__ import annotations

import json
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


class JsonLoader(DocumentLoader):
    """Loader for JSON files (``.json``).

    Parameters
    ----------
    text_field:
        Dot-separated path to the field containing the main text.
        If ``None`` the entire serialised document is used.
    title_field:
        Dot-separated path to the field used as the document title.
    """

    format = DocumentFormat.JSON

    def __init__(
        self,
        text_field: str | None = "content",
        title_field: str | None = "title",
    ) -> None:
        self._text_field = text_field
        self._title_field = title_field

    def _resolve_path(self, data: dict[str, Any], path: str) -> Any:
        """Traverse a dot-separated *path* in *data*."""
        parts = path.split(".")
        current: Any = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    def load(
        self,
        content: bytes,
        *,
        source_uri: str,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeDocument:
        try:
            raw = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise KnowledgeIngestionError(
                f"Failed to decode JSON file: {exc}",
                context={"source_uri": source_uri},
            ) from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise KnowledgeIngestionError(
                f"Invalid JSON: {exc}",
                context={"source_uri": source_uri},
            ) from exc

        if self._text_field and isinstance(data, dict):
            text = str(self._resolve_path(data, self._text_field) or "")
        elif isinstance(data, str):
            text = data
        else:
            text = json.dumps(data, ensure_ascii=False, indent=2)

        title = ""
        if self._title_field and isinstance(data, dict):
            resolved = self._resolve_path(data, self._title_field)
            if resolved is not None:
                title = str(resolved)[:120]
        if not title:
            title = extract_title_from_content(text)

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
            format=DocumentFormat.JSON,
            content=text,
            content_hash=compute_content_hash(text),
            status=IndexStatus.PENDING,
            metadata=merged_meta,
        )
