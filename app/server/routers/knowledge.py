"""Knowledge / RAG REST API router.

Provides endpoints for document ingestion, retrieval search,
collection management, and namespace management.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.knowledge.exceptions import (
    KnowledgeCollectionNotFoundError,
    KnowledgeDuplicateError,
    KnowledgeError,
    KnowledgeNamespaceNotFoundError,
)
from app.knowledge.indexing.pipeline import IndexingPipeline
from app.knowledge.models import (
    CollectionMetadata,
    NamespaceMetadata,
    RetrievalQuery,
)
from app.knowledge.collection_registry import CollectionRegistry
from app.knowledge.namespace_registry import NamespaceRegistry
from app.knowledge.retrieval.pipeline import RetrievalPipeline
from app.server.dependencies import (
    get_collection_registry,
    get_indexing_pipeline,
    get_namespace_registry,
    get_retrieval_pipeline,
)
from app.server.schemas import (
    KnowledgeCollectionCreateRequest,
    KnowledgeCollectionsResponse,
    KnowledgeIngestRequest,
    KnowledgeIngestResponse,
    KnowledgeNamespaceCreateRequest,
    KnowledgeNamespacesResponse,
    KnowledgeQueryRequest,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    KnowledgeTextIngestRequest,
)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


# ── Document Ingestion ───────────────────────────────────────────────


@router.post(
    "/documents",
    response_model=KnowledgeIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document",
)
async def ingest_document(
    request: KnowledgeIngestRequest,
    pipeline: IndexingPipeline = Depends(get_indexing_pipeline),
) -> KnowledgeIngestResponse:
    """Ingest a document into the knowledge base.

    The document is parsed, chunked, embedded, and stored for retrieval.
    """
    try:
        content_bytes = request.content.encode("utf-8")
        doc = pipeline.ingest(
            content_bytes,
            source_uri=request.source_uri,
            collection_id=request.collection_id,
            namespace=request.namespace,
            metadata=request.metadata,
        )
        return KnowledgeIngestResponse(
            doc_id=doc.doc_id,
            source_uri=request.source_uri,
            chunk_count=len(doc.chunk_ids),
            collection_id=request.collection_id,
        )
    except KnowledgeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=exc.message,
        )


@router.post(
    "/text",
    response_model=KnowledgeIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest raw text",
)
async def ingest_text(
    request: KnowledgeTextIngestRequest,
    pipeline: IndexingPipeline = Depends(get_indexing_pipeline),
) -> KnowledgeIngestResponse:
    """Ingest raw text content into the knowledge base."""
    try:
        content_bytes = request.text.encode("utf-8")
        doc = pipeline.ingest(
            content_bytes,
            source_uri=request.source_uri,
            collection_id=request.collection_id,
            namespace=request.namespace,
        )
        return KnowledgeIngestResponse(
            doc_id=doc.doc_id,
            source_uri=request.source_uri,
            chunk_count=len(doc.chunk_ids),
            collection_id=request.collection_id,
        )
    except KnowledgeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=exc.message,
        )


# ── Retrieval ────────────────────────────────────────────────────────


@router.post(
    "/search",
    response_model=KnowledgeSearchResponse,
    summary="Search the knowledge base",
)
async def search_knowledge(
    request: KnowledgeSearchRequest,
    pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
) -> KnowledgeSearchResponse:
    """Perform a retrieval search against the knowledge base."""
    query = RetrievalQuery(
        query=request.query,
        collection_id=request.collection_id,
        namespace=request.namespace,
        top_k=request.top_k,
        strategy=request.strategy,
    )
    result = await pipeline.retrieve(query)
    return KnowledgeSearchResponse(
        query=request.query,
        total=result.total,
        hits=[result],
        latency_ms=result.latency_ms,
    )


@router.post(
    "/query",
    response_model=KnowledgeSearchResponse,
    summary="Execute a full retrieval query",
)
async def query_knowledge(
    request: KnowledgeQueryRequest,
    pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
) -> KnowledgeSearchResponse:
    """Execute a complete retrieval pipeline query.

    Supports multi-query expansion, compression, reranking, and citation building.
    """
    query = RetrievalQuery(
        query=request.query,
        collection_id=request.collection_id,
        namespace=request.namespace,
        top_k=request.top_k,
        include_sources=request.include_sources,
    )
    result = await pipeline.retrieve(query)
    return KnowledgeSearchResponse(
        query=request.query,
        total=result.total,
        hits=[result],
        latency_ms=result.latency_ms,
    )


# ── Collection Management ────────────────────────────────────────────


@router.get(
    "/collections",
    response_model=KnowledgeCollectionsResponse,
    summary="List collections",
)
async def list_collections(
    registry: CollectionRegistry = Depends(get_collection_registry),
) -> KnowledgeCollectionsResponse:
    """Return all registered knowledge collections."""
    collections = registry.list_all()
    return KnowledgeCollectionsResponse(
        total=len(collections),
        collections=collections,
    )


@router.post(
    "/collections",
    status_code=status.HTTP_201_CREATED,
    summary="Create a collection",
)
async def create_collection(
    request: KnowledgeCollectionCreateRequest,
    registry: CollectionRegistry = Depends(get_collection_registry),
) -> dict:
    """Register a new knowledge collection."""
    try:
        metadata = CollectionMetadata(
            collection_id=request.collection_id,
            name=request.name or request.collection_id,
            namespace=request.namespace,
            description=request.description,
        )
        registry.add(metadata)
        return {"collection_id": request.collection_id, "success": True}
    except KnowledgeDuplicateError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=exc.message,
        )


@router.delete(
    "/collections/{collection_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a collection",
)
async def delete_collection(
    collection_id: str,
    registry: CollectionRegistry = Depends(get_collection_registry),
) -> None:
    """Remove a knowledge collection."""
    try:
        registry.delete(collection_id)
    except KnowledgeCollectionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=exc.message,
        )


# ── Namespace Management ─────────────────────────────────────────────


@router.get(
    "/namespaces",
    response_model=KnowledgeNamespacesResponse,
    summary="List namespaces",
)
async def list_namespaces(
    registry: NamespaceRegistry = Depends(get_namespace_registry),
) -> KnowledgeNamespacesResponse:
    """Return all registered knowledge namespaces."""
    namespaces = registry.list_all()
    return KnowledgeNamespacesResponse(
        total=len(namespaces),
        namespaces=namespaces,
    )


@router.post(
    "/namespaces",
    status_code=status.HTTP_201_CREATED,
    summary="Create a namespace",
)
async def create_namespace(
    request: KnowledgeNamespaceCreateRequest,
    registry: NamespaceRegistry = Depends(get_namespace_registry),
) -> dict:
    """Register a new knowledge namespace."""
    try:
        metadata = NamespaceMetadata(
            name=request.name,
            description=request.description,
        )
        registry.add(metadata)
        return {"namespace": request.name, "success": True}
    except KnowledgeDuplicateError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=exc.message,
        )


@router.delete(
    "/namespaces/{namespace_name}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a namespace",
)
async def delete_namespace(
    namespace_name: str,
    registry: NamespaceRegistry = Depends(get_namespace_registry),
) -> None:
    """Remove a knowledge namespace."""
    try:
        registry.delete(namespace_name)
    except KnowledgeNamespaceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=exc.message,
        )
