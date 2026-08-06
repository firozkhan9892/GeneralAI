"""Sentence document chunker.

Splits text on sentence boundaries (``. `` followed by an uppercase
letter, ``! ``, ``? ``).  Adjacent sentences are merged up to the
chunk size limit.
"""

from __future__ import annotations

import re

from app.knowledge.base import Chunker
from app.knowledge.constants import DEFAULT_CHUNK_SIZE
from app.knowledge.models import KnowledgeChunk, KnowledgeDocument
from app.knowledge.utils import compute_content_hash, estimate_token_count

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


class SentenceChunker(Chunker):
    """Splits a document on sentence boundaries.

    Parameters
    ----------
    chunk_size:
        Maximum characters per merged chunk.
    """

    name = "sentence"

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self._chunk_size = chunk_size

    def chunk(self, document: KnowledgeDocument) -> list[KnowledgeChunk]:
        text = document.content
        if not text.strip():
            return []

        sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]

        chunks: list[KnowledgeChunk] = []
        current_sentences: list[str] = []
        current_len = 0
        idx = 0

        for sentence in sentences:
            sent_len = len(sentence)
            if (
                current_len + sent_len + (1 if current_sentences else 0)
                <= self._chunk_size
            ):
                current_sentences.append(sentence)
                current_len += sent_len + (1 if len(current_sentences) > 1 else 0)
            else:
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append(self._make_chunk(document, chunk_text, idx))
                    idx += 1
                current_sentences = [sentence]
                current_len = sent_len

        if current_sentences:
            chunk_text = " ".join(current_sentences)
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
            metadata={**document.metadata, "chunker": "sentence"},
        )
