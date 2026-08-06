"""Default citation builder.

Builds :class:`Citation` and :class:`SourceReference` records from a
set of retrieval hits.  The citation ID is a stable SHA-256 hash of
``doc_id + "|" + chunk_ids``.  Source references are deduplicated by
``doc_id`` with confidence set to the maximum hit score.
"""

from __future__ import annotations

import hashlib

from app.knowledge.base import CitationBuilder
from app.knowledge.constants import SNIPPET_LENGTH
from app.knowledge.models import (
    Citation,
    CitationResult,
    RetrievalHit,
    SourceReference,
)


class DefaultCitationBuilder(CitationBuilder):
    """Builds citations and source references from retrieval hits.

    Deterministic and unit-testable — a pure function of the hits.
    """

    name: str = "default"

    def build(self, hits: list[RetrievalHit]) -> CitationResult:
        """Build citation and source metadata for *hits*.

        Args:
            hits: The final hit set (post rerank).

        Returns:
            Citations and deduplicated source references.
        """
        citations: list[Citation] = []
        # Track per-doc_id best score and metadata for source dedup
        doc_sources: dict[str, dict] = {}

        for hit in hits:
            citation_id = _make_citation_id(hit.doc_id, hit.chunk_id)
            snippet = hit.content[:SNIPPET_LENGTH] if hit.content else ""

            citation = Citation(
                citation_id=citation_id,
                doc_id=hit.doc_id,
                doc_title=hit.metadata.get("title", ""),
                chunk_ids=(hit.chunk_id,),
                page=hit.metadata.get("page"),
                source_uri=hit.metadata.get("source_uri", ""),
                snippet=snippet,
                namespace=hit.namespace,
                collection_id=hit.collection_id,
            )
            citations.append(citation)

            # Track best score per doc_id for source confidence
            key = hit.doc_id
            if key not in doc_sources:
                doc_sources[key] = {
                    "doc_title": hit.metadata.get("title", ""),
                    "source_uri": hit.metadata.get("source_uri", ""),
                    "namespace": hit.namespace,
                    "collection_id": hit.collection_id,
                    "version": hit.metadata.get("version", 1),
                    "confidence": hit.score,
                }
            else:
                doc_sources[key]["confidence"] = max(
                    doc_sources[key]["confidence"], hit.score
                )

        sources: list[SourceReference] = []
        for doc_id, data in doc_sources.items():
            sources.append(
                SourceReference(
                    doc_id=doc_id,
                    doc_title=data["doc_title"],
                    source_uri=data["source_uri"],
                    namespace=data["namespace"],
                    collection_id=data["collection_id"],
                    version=data["version"],
                    confidence=data["confidence"],
                )
            )

        return CitationResult(
            citations=tuple(citations),
            sources=tuple(sources),
        )


def _make_citation_id(doc_id: str, chunk_id: str) -> str:
    """Generate a stable 16-char citation ID."""
    raw = f"{doc_id}|{chunk_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
