"""Tests for the knowledge domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.knowledge.models import (
    Citation,
    CitationResult,
    CollectionMetadata,
    CollectionStatus,
    DocumentFormat,
    EmbeddingModelInfo,
    EmbeddingVector,
    IndexStatus,
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeEvent,
    MetadataFilter,
    NamespaceMetadata,
    RetrievalHit,
    RetrievalQuery,
    SourceReference,
    VectorSearchHit,
)


def test_document_format_values() -> None:
    assert DocumentFormat.TXT.value == "txt"
    assert DocumentFormat.MARKDOWN.value == "markdown"
    assert DocumentFormat.PDF.value == "pdf"


def test_index_status_default_is_pending() -> None:
    assert IndexStatus.PENDING.value == "pending"


def test_knowledge_document_defaults() -> None:
    doc = KnowledgeDocument(doc_id="d1", collection_id="c1", format=DocumentFormat.TXT)
    assert doc.namespace == ""
    assert doc.title == ""
    assert doc.version == 1
    assert doc.status == IndexStatus.PENDING
    assert doc.metadata == {}
    assert doc.chunk_ids == ()
    assert doc.created_at is not None


def test_knowledge_document_is_frozen() -> None:
    doc = KnowledgeDocument(doc_id="d1", collection_id="c1", format=DocumentFormat.TXT)
    with pytest.raises(ValidationError):
        doc.doc_id = "other"  # type: ignore[misc]


def test_knowledge_document_version_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        KnowledgeDocument(
            doc_id="d", collection_id="c", format=DocumentFormat.TXT, version=0
        )


def test_knowledge_chunk_defaults() -> None:
    chunk = KnowledgeChunk(
        chunk_id="ch1", doc_id="d1", collection_id="c1", content="hi"
    )
    assert chunk.chunk_index == 0
    assert chunk.token_count == 0
    assert chunk.metadata == {}


def test_knowledge_chunk_is_frozen() -> None:
    chunk = KnowledgeChunk(
        chunk_id="ch1", doc_id="d1", collection_id="c1", content="hi"
    )
    with pytest.raises(ValidationError):
        chunk.content = "bye"  # type: ignore[misc]


def test_collection_metadata_defaults() -> None:
    col = CollectionMetadata(collection_id="c1", name="Docs")
    assert col.status == CollectionStatus.ACTIVE
    assert col.document_count == 0
    assert col.chunk_count == 0


def test_namespace_metadata_defaults() -> None:
    ns = NamespaceMetadata(name="prod")
    assert ns.collection_count == 0
    assert ns.metadata == {}


def test_embedding_vector_defaults() -> None:
    vec = EmbeddingVector(
        vector_id="v1", chunk_id="ch1", doc_id="d1", collection_id="c1"
    )
    assert vec.dimensions == 0
    assert vec.model == ""
    assert vec.namespace == ""


def test_embedding_model_info() -> None:
    info = EmbeddingModelInfo(
        name="det", provider="deterministic", dimensions=64, model="hash"
    )
    assert info.dimensions == 64
    assert info.max_input_tokens is None


def test_metadata_filter_any_value() -> None:
    filt = MetadataFilter(field="author", op="eq", value="alice")
    assert filt.value == "alice"


def test_retrieval_query_defaults() -> None:
    q = RetrievalQuery(query="how does x work")
    assert q.strategy == "hybrid"
    assert q.top_k == 10
    assert q.vector_weight == 0.5
    assert q.filters == ()


def test_retrieval_query_top_k_positive() -> None:
    with pytest.raises(ValidationError):
        RetrievalQuery(query="q", top_k=0)


def test_vector_search_hit() -> None:
    hit = VectorSearchHit(chunk_id="ch1", doc_id="d1", collection_id="c1", score=0.9)
    assert hit.namespace == ""


def test_retrieval_hit_ranks_default() -> None:
    hit = RetrievalHit(
        chunk_id="ch1", doc_id="d1", collection_id="c1", content="x", score=1.0
    )
    assert hit.ranks == {}
    assert hit.metadata == {}


def test_citation_and_source() -> None:
    citation = Citation(citation_id="c1", doc_id="d1", doc_title="Doc")
    assert citation.chunk_ids == ()
    source = SourceReference(doc_id="d1", doc_title="Doc")
    assert source.version == 1
    assert source.confidence == 0.0


def test_citation_result_defaults() -> None:
    result = CitationResult()
    assert result.citations == ()
    assert result.sources == ()


def test_knowledge_event_defaults() -> None:
    event = KnowledgeEvent(event_type="knowledge.document.ingested")
    assert event.error is None
    assert event.status == ""
    assert event.data == {}
    assert event.timestamp is not None


def test_knowledge_event_is_frozen() -> None:
    event = KnowledgeEvent(event_type="knowledge.document.ingested")
    with pytest.raises(ValidationError):
        event.event_type = "other"  # type: ignore[misc]


def test_document_serialization_roundtrip() -> None:
    doc = KnowledgeDocument(doc_id="d1", collection_id="c1", format=DocumentFormat.TXT)
    decoded = KnowledgeDocument.model_validate_json(doc.model_dump_json())
    assert decoded == doc


def test_document_model_copy_is_equal() -> None:
    doc = KnowledgeDocument(doc_id="d1", collection_id="c1", format=DocumentFormat.TXT)
    assert doc.model_copy() == doc
