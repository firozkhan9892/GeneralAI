"""Recursive structural chunker.

Attempts to split on a hierarchy of separators (e.g. paragraphs →
sentences → words) to produce semantically coherent chunks that
respect the chunk size limit.  Inspired by LangChain's
``RecursiveCharacterTextSplitter``.
"""

from __future__ import annotations

from app.knowledge.base import Chunker
from app.knowledge.constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    RECURSIVE_SEPARATORS,
)
from app.knowledge.models import KnowledgeChunk, KnowledgeDocument
from app.knowledge.utils import compute_content_hash, estimate_token_count


class RecursiveChunker(Chunker):
    """Recursively splits text using a hierarchy of separators.

    Parameters
    ----------
    chunk_size:
        Maximum characters per chunk.
    overlap:
        Character overlap between consecutive chunks.
    separators:
        Ordered list of separator strings, tried from first to last.
    """

    name = "recursive"

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
        separators: tuple[str, ...] = RECURSIVE_SEPARATORS,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._separators = separators

    def _split_recursive(self, text: str, separators: tuple[str, ...]) -> list[str]:
        """Recursively split *text* using *separators*."""
        if len(text) <= self._chunk_size:
            return [text]

        sep = separators[0] if separators else ""
        remaining_separators = separators[1:] if len(separators) > 1 else ()

        if sep:
            parts = text.split(sep)
        else:
            # No separator left: hard split respecting overlap
            step = max(1, self._chunk_size - self._overlap)
            return [text[i : i + self._chunk_size] for i in range(0, len(text), step)]

        result: list[str] = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= self._chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                # If a single part exceeds chunk_size, recurse with finer separators
                if len(part) > self._chunk_size and remaining_separators:
                    result.extend(self._split_recursive(part, remaining_separators))
                    current = ""
                elif len(part) > self._chunk_size:
                    # Last resort: hard split
                    step = max(1, self._chunk_size - self._overlap)
                    result.extend(
                        part[i : i + self._chunk_size]
                        for i in range(0, len(part), step)
                    )
                    current = ""
                else:
                    current = part
                    continue

        if current:
            result.append(current)

        return result

    def chunk(self, document: KnowledgeDocument) -> list[KnowledgeChunk]:
        text = document.content
        if not text.strip():
            return []

        raw_chunks = self._split_recursive(text, self._separators)

        chunks: list[KnowledgeChunk] = []
        idx = 0
        for chunk_text in raw_chunks:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
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
                    metadata={**document.metadata, "chunker": "recursive"},
                )
            )
            idx += 1

        return chunks
