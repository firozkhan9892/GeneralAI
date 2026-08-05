"""Tests for the knowledge exceptions and event constants."""

from __future__ import annotations

import pytest

from app.core.exceptions import GeneralAIError
from app.knowledge.events import (
    EVENT_KNOWLEDGE_COLLECTION_CREATED,
    EVENT_KNOWLEDGE_DOCUMENT_INGESTED,
    EVENT_KNOWLEDGE_INDEX_COMPLETED,
    EVENT_KNOWLEDGE_NAMESPACE_CREATED,
    EVENT_KNOWLEDGE_RETRIEVED,
)
from app.knowledge.exceptions import (
    KnowledgeCollectionNotFoundError,
    KnowledgeDocumentNotFoundError,
    KnowledgeDuplicateError,
    KnowledgeError,
    KnowledgeIndexError,
    KnowledgeIngestionError,
    KnowledgeNamespaceNotFoundError,
    KnowledgeUnsupportedFormatError,
    KnowledgeValidationError,
    KnowledgeVersionError,
)


def test_knowledge_error_is_general_ai_error() -> None:
    assert issubclass(KnowledgeError, GeneralAIError)


@pytest.mark.parametrize(
    "cls",
    [
        KnowledgeValidationError,
        KnowledgeNamespaceNotFoundError,
        KnowledgeCollectionNotFoundError,
        KnowledgeDocumentNotFoundError,
        KnowledgeDuplicateError,
        KnowledgeUnsupportedFormatError,
        KnowledgeIngestionError,
        KnowledgeIndexError,
        KnowledgeVersionError,
    ],
)
def test_knowledge_exceptions_subclass_base(cls) -> None:
    assert issubclass(cls, KnowledgeError)


def test_knowledge_error_defaults() -> None:
    err = KnowledgeError("boom")
    assert err.module == "knowledge"
    assert err.cause is None
    assert err.context == {}


def test_knowledge_error_accepts_context_and_cause() -> None:
    cause = ValueError("inner")
    err = KnowledgeIngestionError("failed", cause=cause, context={"doc_id": "d1"})
    assert err.cause is cause
    assert err.context == {"doc_id": "d1"}


def test_event_constants_use_knowledge_domain() -> None:
    assert EVENT_KNOWLEDGE_NAMESPACE_CREATED == "knowledge.namespace.created"
    assert EVENT_KNOWLEDGE_COLLECTION_CREATED == "knowledge.collection.created"
    assert EVENT_KNOWLEDGE_DOCUMENT_INGESTED == "knowledge.document.ingested"
    assert EVENT_KNOWLEDGE_INDEX_COMPLETED == "knowledge.index.completed"
    assert EVENT_KNOWLEDGE_RETRIEVED == "knowledge.retrieved"
