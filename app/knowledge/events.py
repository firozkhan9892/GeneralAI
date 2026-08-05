"""Knowledge event type names.

These constants identify events published on the application
:class:`EventBus` for the knowledge subsystem.  Prefixes use the
``knowledge.*`` namespace reserved for ingestion and retrieval.
"""

from __future__ import annotations

from typing import Final

EVENT_KNOWLEDGE_NAMESPACE_CREATED: Final[str] = "knowledge.namespace.created"
EVENT_KNOWLEDGE_NAMESPACE_DELETED: Final[str] = "knowledge.namespace.deleted"

EVENT_KNOWLEDGE_COLLECTION_CREATED: Final[str] = "knowledge.collection.created"
EVENT_KNOWLEDGE_COLLECTION_UPDATED: Final[str] = "knowledge.collection.updated"
EVENT_KNOWLEDGE_COLLECTION_DELETED: Final[str] = "knowledge.collection.deleted"

EVENT_KNOWLEDGE_DOCUMENT_INGESTED: Final[str] = "knowledge.document.ingested"
EVENT_KNOWLEDGE_DOCUMENT_UPDATED: Final[str] = "knowledge.document.updated"
EVENT_KNOWLEDGE_DOCUMENT_DELETED: Final[str] = "knowledge.document.deleted"

EVENT_KNOWLEDGE_INDEX_STARTED: Final[str] = "knowledge.index.started"
EVENT_KNOWLEDGE_INDEX_COMPLETED: Final[str] = "knowledge.index.completed"
EVENT_KNOWLEDGE_INDEX_FAILED: Final[str] = "knowledge.index.failed"

EVENT_KNOWLEDGE_RETRIEVED: Final[str] = "knowledge.retrieved"
