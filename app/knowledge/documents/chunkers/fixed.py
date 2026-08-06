"""Fixed-size document chunker.

Splits text into chunks of a maximum character count with optional
overlap.  Respects hard paragraph boundaries where possible.
"""

from __future__ import annotations

from app.knowledge.base import Chunker
from app.knowledge.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from app.knowledge.models import KnowledgeChunk, KnowledgeDocument
from app.knowledge.utils import compute_content_hash, estimate_token_count


class FixedChunker(Chunker):
    """Splits a document into fixed-size character windows.

    Parameters
    ----------
    chunk_size:
        Maximum characters per chunk.
    overlap:
        Character overlap between consecutive chunks.
    """

    name = "fixed"

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be < chunk_size")
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, document: KnowledgeDocument) -> list[KnowledgeChunk]:
        text = document.content
        if not text.strip():
            return []

        chunks: list[KnowledgeChunk] = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self._chunk_size
            chunk_text = text[start:end]

            # Try to break at a paragraph boundary
            if end < len(text):
                last_para = chunk_text.rfind("\n\n")
                if last_para > self._chunk_size // 2:
                    chunk_text = chunk_text[:last_para]
                    end = start + last_para

            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=f"{document.doc_id}_c{idx:04d}",
                        doc_id=document.doc_id,
                        collection_id=document.collection_id,
                        namespace=document.namespace,
                        content=chunk_text,
                        chunk_index=idx,
                        token_count=estimate_token_count(chunk_text),
                        content_hash=compute_content_hash(chunk_text),
                        metadata={**document.metadata, "chunk_size": self._chunk_size},
                    )
                )
                idx += 1

            start = end - self._overlap
            if start <= 0 and end >= len(text):
                break

        return chunks
