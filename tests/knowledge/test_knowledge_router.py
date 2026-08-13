"""Tests for the Knowledge REST API router."""

import pytest
from fastapi.testclient import TestClient

from app.server.app import create_app
from app.server.config import ServerSettings


@pytest.fixture
def app():
    """Create a test application with auth disabled."""
    settings = ServerSettings(
        api_key=None,
        rate_limit_enabled=False,
    )
    return create_app(settings=settings)


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestKnowledgeDocuments:
    """Tests for document ingestion endpoints."""

    def test_ingest_text(self, client):
        """Test ingesting raw text."""
        response = client.post(
            "/knowledge/text",
            json={
                "text": "This is a test document about machine learning. " * 20,
                "source_uri": "test.txt",
                "collection_id": "test-coll",
                "namespace": "test-ns",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert "doc_id" in data
        assert data["chunk_count"] > 0

    def test_ingest_document(self, client):
        """Test ingesting a document."""
        response = client.post(
            "/knowledge/documents",
            json={
                "content": "Document content about artificial intelligence.",
                "source_uri": "doc.txt",
                "collection_id": "test-coll",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert "doc_id" in data

    def test_ingest_text_missing_text(self, client):
        """Test validation when text is empty."""
        response = client.post(
            "/knowledge/text",
            json={
                "text": "",
                "source_uri": "test.txt",
            },
        )
        assert response.status_code == 422


class TestKnowledgeSearch:
    """Tests for retrieval endpoints."""

    def test_search(self, client):
        """Test search endpoint."""
        # First ingest a document
        client.post(
            "/knowledge/text",
            json={
                "text": "Machine learning is a subset of artificial intelligence.",
                "source_uri": "ml.txt",
                "collection_id": "search-coll",
            },
        )

        response = client.post(
            "/knowledge/search",
            json={
                "query": "machine learning",
                "collection_id": "search-coll",
                "top_k": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "latency_ms" in data

    def test_query(self, client):
        """Test query endpoint."""
        response = client.post(
            "/knowledge/query",
            json={
                "query": "What is AI?",
                "collection_id": "search-coll",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "query" in data

    def test_search_missing_query(self, client):
        """Test validation when query is empty."""
        response = client.post(
            "/knowledge/search",
            json={
                "query": "",
            },
        )
        assert response.status_code == 422


class TestKnowledgeCollections:
    """Tests for collection management endpoints."""

    def test_list_collections(self, client):
        """Test listing collections."""
        response = client.get("/knowledge/collections")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "collections" in data

    def test_create_collection(self, client):
        """Test creating a collection."""
        response = client.post(
            "/knowledge/collections",
            json={
                "collection_id": "test-coll-1",
                "name": "Test Collection",
                "description": "A test collection",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True

    def test_create_duplicate_collection(self, client):
        """Test that duplicate collection returns 409."""
        payload = {
            "collection_id": "dup-coll",
            "name": "Duplicate",
        }
        client.post("/knowledge/collections", json=payload)
        response = client.post("/knowledge/collections", json=payload)
        assert response.status_code == 409

    def test_delete_collection(self, client):
        """Test deleting a collection."""
        client.post(
            "/knowledge/collections",
            json={"collection_id": "del-coll"},
        )
        response = client.delete("/knowledge/collections/del-coll")
        assert response.status_code == 204

    def test_delete_nonexistent_collection(self, client):
        """Test deleting nonexistent collection returns 404."""
        response = client.delete("/knowledge/collections/nonexistent")
        assert response.status_code == 404


class TestKnowledgeNamespaces:
    """Tests for namespace management endpoints."""

    def test_list_namespaces(self, client):
        """Test listing namespaces."""
        response = client.get("/knowledge/namespaces")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "namespaces" in data

    def test_create_namespace(self, client):
        """Test creating a namespace."""
        response = client.post(
            "/knowledge/namespaces",
            json={
                "name": "test-ns",
                "description": "Test namespace",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True

    def test_create_duplicate_namespace(self, client):
        """Test that duplicate namespace returns 409."""
        payload = {"name": "dup-ns"}
        client.post("/knowledge/namespaces", json=payload)
        response = client.post("/knowledge/namespaces", json=payload)
        assert response.status_code == 409

    def test_delete_namespace(self, client):
        """Test deleting a namespace."""
        client.post("/knowledge/namespaces", json={"name": "del-ns"})
        response = client.delete("/knowledge/namespaces/del-ns")
        assert response.status_code == 204

    def test_delete_nonexistent_namespace(self, client):
        """Test deleting nonexistent namespace returns 404."""
        response = client.delete("/knowledge/namespaces/nonexistent")
        assert response.status_code == 404


class TestKnowledgeAuth:
    """Tests for authentication on knowledge endpoints."""

    def test_protected_with_api_key(self):
        """Test that endpoints require API key when configured."""
        settings = ServerSettings(
            api_key="test-key",
            rate_limit_enabled=False,
        )
        app = create_app(settings=settings)
        client = TestClient(app)

        response = client.get("/knowledge/collections")
        assert response.status_code == 401

    def test_protected_with_valid_key(self):
        """Test that valid API key grants access."""
        settings = ServerSettings(
            api_key="test-key",
            rate_limit_enabled=False,
        )
        app = create_app(settings=settings)
        client = TestClient(app)

        response = client.get(
            "/knowledge/collections",
            headers={"X-API-Key": "test-key"},
        )
        assert response.status_code == 200


class TestKnowledgeConcurrency:
    """Tests for concurrent access to knowledge endpoints."""

    def test_concurrent_ingestion(self, client):
        """Test concurrent document ingestion."""
        import concurrent.futures

        def ingest(i):
            return client.post(
                "/knowledge/text",
                json={
                    "text": f"Document {i} content about topic {i % 5}.",
                    "source_uri": f"doc-{i}.txt",
                    "collection_id": "concurrent-coll",
                },
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(ingest, i) for i in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert all(r.status_code == 201 for r in results)
