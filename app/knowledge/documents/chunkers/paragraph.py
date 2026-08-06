"""Paragraph document chunker.

Splits on double-newline paragraph boundaries.  Adjacent small
paragraphs are merged up to the chunk size limit.
"""

from __future__ import annotations

from app.knowledge.base import Chunker
from app.knowledge.constants import DEFAULT_CHUNK_SIZE
from app.knowledge.models import KnowledgeChunk, KnowledgeDocument
from app.knowledge.utils import compute_content_hash, estimate_token_count


class ParagraphChunker(Chunker):
    """Splits a document on paragraph boundaries (``\\n\\n``).

    Parameters
    ----------
    chunk_size:
        Maximum characters per merged chunk.
    separator:
        The paragraph separator pattern.
    """

    name = "paragraph"

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        separator: str = "\n\n",
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self._chunk_size = chunk_size
        self._separator = separator

    def chunk(self, document: KnowledgeDocument) -> list[KnowledgeChunk]:
        text = document.content
        if not text.strip():
            return []

        paragraphs = [p.strip() for p in text.split(self._separator) if p.strip()]

        chunks: list[KnowledgeChunk] = []
        current_parts: list[str] = []
        current_len = 0
        idx = 0

        for para in paragraphs:
            para_len = len(para)
            sep_len = len(self._separator) if current_parts else 0
            if current_len + para_len + sep_len <= self._chunk_size:
                current_parts.append(para)
                current_len += para_len + sep_len
            else:
                if current_parts:
                    chunk_text = self._separator.join(current_parts)
                    chunks.append(self._make_chunk(document, chunk_text, idx))
                    idx += 1
                current_parts = [para]
                current_len = para_len

        if current_parts:
            chunk_text = self._separator.join(current_parts)
            chunks.append(self._make_chunk(document, chunk_text, idx))

        return chunks

    def _make_chunk(
        self, document: KnowledgeDocument, text: str, index: int
    ) -> KnowledgeChunk:
        return KnowledgeChunk(
            chunk_id=f"{document.doc_id}_c{index:04d}",
            doc_id=document.doc_id,
            collection_id=document.collection_id,
            namespace=document.namespace,
            content=text,
            chunk_index=index,
            token_count=estimate_token_count(text),
            content_hash=compute_content_hash(text),
            metadata={**document.metadata, "chunker": "paragraph"},
        )
