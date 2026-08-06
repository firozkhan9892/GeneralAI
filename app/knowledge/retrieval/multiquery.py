"""Multi-query retriever.

Expands a single query into N variants (using deterministic surface
variants when no LLM is available), runs the base retriever on each,
and fuses all results via RRF.  This broadens recall by exploring
different phrasings of the same intent.
"""

from __future__ import annotations

import logging
import re

from app.knowledge.base import QueryRewriter, Retriever, RetrievalContext
from app.knowledge.constants import RRF_K
from app.knowledge.models import RetrievalHit, RetrievalQuery
from app.knowledge.retrieval.hybrid import _rrf_fuse

log = logging.getLogger(__name__)

_DEFAULT_N_QUERIES = 3


class MultiQueryRetriever(Retriever):
    """Expands a query into variants and fuses multi-branch retrieval.

    Wraps a base retriever and an optional query rewriter.  When no
    LLM is available, deterministic surface variants are generated
    (keyword extraction, phrase focus, etc.).
    """

    name: str = "multi_query"

    def __init__(
        self,
        base_retriever: Retriever | None = None,
        query_rewriter: QueryRewriter | None = None,
        n_queries: int = _DEFAULT_N_QUERIES,
        rrf_k: int = RRF_K,
    ) -> None:
        self._base_retriever = base_retriever
        self._query_rewriter = query_rewriter
        self._n_queries = n_queries
        self._rrf_k = rrf_k

    async def retrieve(
        self, query: RetrievalQuery, *, context: RetrievalContext
    ) -> list[RetrievalHit]:
        """Expand query and run multi-branch retrieval.

        Args:
            query: The retrieval query.
            context: Collection context.

        Returns:
            Fused hits from all query variants.
        """
        if self._base_retriever is None:
            return []

        raw = query.rewritten_query or query.query
        variants = _generate_variants(raw, self._n_queries)

        all_hits: list[list[RetrievalHit]] = []
        for variant in variants:
            variant_query = RetrievalQuery(
                query=variant,
                rewritten_query=variant,
                namespace=query.namespace,
                collection_id=query.collection_id,
                filters=query.filters,
                strategy=query.strategy,
                top_k=query.top_k,
                vector_weight=query.vector_weight,
                bm25_weight=query.bm25_weight,
            )
            try:
                hits = await self._base_retriever.retrieve(
                    variant_query, context=context
                )
                all_hits.append(hits)
            except Exception:
                log.exception("MultiQueryRetriever: variant retrieval failed")

        if not all_hits:
            return []

        # Flatten all hits into a single list, then fuse with RRF
        flat_hits = [h for branch in all_hits for h in branch]
        return _rrf_fuse(
            flat_hits,
            [],  # no second list — single-list RRF
            vector_weight=1.0,
            bm25_weight=0.0,
            k=self._rrf_k,
            top_k=query.top_k,
        )


def _generate_variants(query: str, n: int) -> list[str]:
    """Generate *n* deterministic query variants.

    When no LLM is available, these surface-level variants help
    broaden recall by emphasising different aspects of the query.
    """
    variants = [query]
    if n <= 1:
        return variants

    # Variant: extract keywords (remove stopwords, keep content words)
    keywords = _extract_keywords(query)
    if keywords and keywords != query:
        variants.append(keywords)

    # Variant: noun-phrase-style (take longest contiguous content words)
    phrases = _extract_phrases(query)
    for p in phrases:
        if p not in variants:
            variants.append(p)
        if len(variants) >= n:
            break

    # Variant: the original with common question words removed
    stripped = _strip_question_words(query)
    if stripped and stripped not in variants:
        variants.append(stripped)

    return variants[:n]


def _extract_keywords(text: str) -> str:
    """Extract content keywords from *text*."""
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "about",
        "what",
        "how",
        "when",
        "where",
        "why",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "or",
        "and",
        "but",
        "not",
        "no",
        "nor",
    }
    words = re.findall(r"[a-zA-Z]+", text.lower())
    keywords = [w for w in words if w not in stopwords]
    return " ".join(keywords)


def _extract_phrases(text: str) -> list[str]:
    """Extract noun-phrase-style chunks from *text*."""
    # Simple heuristic: split on punctuation and conjunctions,
    # keep the longest segments.
    parts = re.split(r"[,;:.!?]|\band\b|\bor\b", text)
    phrases = [p.strip() for p in parts if p.strip()]
    phrases.sort(key=len, reverse=True)
    return phrases


def _strip_question_words(text: str) -> str:
    """Remove leading question words from *text*."""
    question_words = {"what", "how", "when", "where", "why", "which", "who"}
    words = text.split()
    while words and words[0].lower() in question_words:
        words.pop(0)
    return " ".join(words)
