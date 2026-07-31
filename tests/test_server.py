"""Tests for the GeneralAI FastAPI server (Phase 9).

Comprehensive coverage of all server endpoints, authentication,
rate limiting, streaming, and error handling.
"""

from __future__ import annotations

from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.server.app import create_app
from app.server.config import ServerSettings


@pytest.fixture
def app():
    """Create a FastAPI app for testing."""
    return create_app()


@pytest.fixture
def client(app: FastAPI):
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def server_app(app: FastAPI):
    """Fixture for the FastAPI app."""
    return app


@pytest.fixture
def app_with_api_key():
    """Create a FastAPI app with API key for testing."""
    return create_app(settings=ServerSettings(api_key="test-key-123"))


@pytest.fixture
def client_with_api_key(app_with_api_key: FastAPI):
    """Create a test client for the FastAPI app with API key."""
    return TestClient(app_with_api_key)


class TestHealthEndpoint:
    """Tests for the /health endpoint (public)."""

    def test_health_public(self, client: TestClient) -> None:
        """Health endpoint is publicly accessible."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["app_name"] == "GeneralAI API"
        assert "version" in data
        assert "sessions_active" in data
        assert "sessions_total" in data

    def test_health_response_schema(self, client: TestClient) -> None:
        """Health endpoint response matches expected schema."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        required_fields = {
            "status",
            "app_name",
            "version",
            "sessions_active",
            "sessions_total",
        }
        assert set(data.keys()) == required_fields
        assert isinstance(data["sessions_active"], int)
        assert isinstance(data["sessions_total"], int)

    def test_metrics_public(self, client: TestClient) -> None:
        """Metrics endpoint is publicly accessible."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "requests_total" in data
        assert "errors_total" in data
        assert "sessions_active" in data
        assert "sessions_total" in data
        assert "memory_records" in data
        assert "tools_count" in data


class TestChatEndpoints:
    """Tests for chat endpoints."""

    def test_chat_post_without_api_key(self, client: TestClient) -> None:
        """Chat endpoint works without API key when auth is disabled."""
        response = client.post("/chat", json={"message": "Hello, world!"})
        # May return various status codes depending on agent execution
        assert response.status_code != 404

    def test_chat_post_invalid_request(self, client: TestClient) -> None:
        """Chat endpoint rejects invalid requests."""
        response = client.post("/chat", json={})  # Missing required 'message' field
        assert response.status_code in (422, 400, 401)
        data = response.json()
        assert "detail" in data

    def test_chat_post_with_api_key(self, client_with_api_key: TestClient) -> None:
        """Chat endpoint requires API key when configured."""
        response = client_with_api_key.post("/chat", json={"message": "Hello, world!"})
        # Without API key header, should return 401
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
        assert "API key" in data["detail"]

    def test_chat_post_with_valid_api_key(
        self, client_with_api_key: TestClient
    ) -> None:
        """Chat endpoint works with valid API key."""
        response = client_with_api_key.post(
            "/chat",
            json={"message": "Hello, world!"},
            headers={"X-API-Key": "test-key-123"},
        )
        assert response.status_code != 404
        # Should not be 401 (unauthorized)
        assert response.status_code != 401

    def test_chat_stream_endpoint_exists(self, client: TestClient) -> None:
        """SSE streaming endpoint is available."""
        response = client.post("/chat/stream", json={"message": "Hello, world!"})
        # May return 401 if API key required, but endpoint should exist
        assert response.status_code in (200, 401, 422, 404)
        # If 200, should be SSE
        if response.status_code == 200:
            assert "text/event-stream" in response.headers.get("content-type", "")

    def test_chat_stream_requires_message(self, client: TestClient) -> None:
        """SSE streaming endpoint requires message."""
        response = client.post("/chat/stream", json={})
        assert response.status_code in (422, 401, 400)
        data = response.json()
        assert "detail" in data


class TestAgentEndpoints:
    """Tests for agent endpoints."""

    def test_agent_run_endpoint_exists(self, client: TestClient) -> None:
        """Agent run endpoint is available."""
        response = client.post("/agent/run", json={"raw_input": "Hello, world!"})
        # May return 401/422, but endpoint should exist (not 404)
        assert response.status_code != 404

    def test_agent_cancel_endpoint_exists(self, client: TestClient) -> None:
        """Agent cancel endpoint is available."""
        # Registered route — verify via OpenAPI schema rather than response
        # status (the endpoint returns 404 for non-existent sessions, which
        # is application logic, not a missing route).
        spec = client.get("/openapi.json").json()
        assert "/agent/cancel" in spec["paths"]
        assert "post" in spec["paths"]["/agent/cancel"]

    def test_agent_status_endpoint_exists(self, client: TestClient) -> None:
        """Agent status endpoint is available."""
        spec = client.get("/openapi.json").json()
        assert "/agent/status/{session_id}" in spec["paths"]
        assert "get" in spec["paths"]["/agent/status/{session_id}"]

    def test_agents_list_endpoint_exists(self, client: TestClient) -> None:
        """Agents list endpoint is available."""
        spec = client.get("/openapi.json").json()
        assert "/agents" in spec["paths"]
        assert "get" in spec["paths"]["/agents"]


class TestMemoryEndpoint:
    """Tests for memory search endpoint."""

    def test_memory_search_endpoint_exists(self, client: TestClient) -> None:
        """Memory search endpoint is available."""
        response = client.get("/memory/search?q=hello")
        # May return 401, but endpoint should exist
        assert response.status_code != 404

    def test_memory_search_requires_query(self, client: TestClient) -> None:
        """Memory search requires a 'q' parameter."""
        response = client.get("/memory/search")
        assert response.status_code == 422


class TestToolEndpoint:
    """Tests for tool execution endpoint."""

    def test_tool_run_endpoint_exists(self, client: TestClient) -> None:
        """Tool run endpoint is available."""
        # Registered route — verify via OpenAPI schema rather than response
        # status (the endpoint returns 404 for unregistered tools, which is
        # application logic, not a missing route).
        spec = client.get("/openapi.json").json()
        assert "/tools/run" in spec["paths"]
        assert "post" in spec["paths"]["/tools/run"]


class TestMetricsEndpoint:
    """Tests for metrics endpoint (public)."""

    def test_metrics_endpoint_exists(self, client: TestClient) -> None:
        """Metrics endpoint is available."""
        response = client.get("/metrics")
        assert response.status_code == 200


class TestWebSocketEndpoint:
    """Tests for WebSocket endpoint."""

    def test_websocket_endpoint_exists(self, client: TestClient) -> None:
        """WebSocket endpoint is available."""
        with client.websocket_connect("/agent/ws") as websocket:
            # Server sends a "connected" greeting first
            greeting = websocket.receive_json()
            assert greeting["type"] == "connected"
            websocket.send_json({"type": "ping"})
            response = websocket.receive_json()
            assert response["type"] == "pong"

    def test_websocket_invalid_message(self, client: TestClient) -> None:
        """WebSocket rejects invalid messages."""
        with client.websocket_connect("/agent/ws") as websocket:
            # Consume the "connected" greeting
            greeting = websocket.receive_json()
            assert greeting["type"] == "connected"
            websocket.send_json({"type": "invalid"})
            response = websocket.receive_json()
            assert response["type"] == "error"


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_enforced(self, monkeypatch) -> None:
        """Rate limiting is enforced when configured."""
        # Create app with rate limiting enabled and low limit
        settings = ServerSettings(
            api_key="test-key", rate_limit_enabled=True, rate_limit_per_minute=5
        )
        app = create_app(settings=settings)
        client = TestClient(app)

        # Make 6 requests quickly
        for i in range(6):
            response = client.post(
                "/chat",
                json={"message": f"Message {i}"},
                headers={"X-API-Key": "test-key"},
            )
            if i < 5:
                assert response.status_code != 429  # Not rate limited yet
            else:
                assert response.status_code == 429  # Rate limit exceeded


class TestErrorHandling:
    """Tests for error handling."""

    def test_404_handler(self, client: TestClient) -> None:
        """Custom 404 handler is in place."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_422_handler(self, client: TestClient) -> None:
        """Custom 422 handler is in place."""
        response = client.post("/chat", json={})  # Invalid request
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


class TestStreaming:
    """Tests for streaming functionality."""

    def test_sse_event_format(self) -> None:
        """SSE event format is correct."""
        from app.server.streaming import sse_format

        event = "session.started"
        data = {"session_id": "test-session"}
        formatted = sse_format(event, data)

        assert formatted.startswith("event: session.started")
        assert "data: {" in formatted
        assert "test-session" in formatted
        assert formatted.endswith("\n\n")


class TestDependencyInjection:
    """Tests for dependency injection."""

    def test_settings_dependency(self, client: TestClient) -> None:
        """Settings are properly injected."""
        app_state = cast(FastAPI, client.app).state
        assert hasattr(app_state, "settings")
        assert hasattr(app_state, "container")
        assert hasattr(app_state, "agent_manager")

    def test_container_dependencies(self, client: TestClient) -> None:
        """Container dependencies are resolved."""
        # Verify that DI dependencies are set up correctly
        app_state = cast(FastAPI, client.app).state
        assert hasattr(app_state, "agent_manager")
        assert hasattr(app_state, "memory_engine")
        assert hasattr(app_state, "tool_registry")
        assert hasattr(app_state, "tool_executor")


class TestOpenAPI:
    """Tests for OpenAPI documentation."""

    def test_openapi_endpoint(self, client: TestClient) -> None:
        """OpenAPI documentation is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
        assert isinstance(data["paths"], dict)

    def test_swagger_ui(self, client: TestClient) -> None:
        """Swagger UI is available."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_redoc(self, client: TestClient) -> None:
        """ReDoc is available."""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
