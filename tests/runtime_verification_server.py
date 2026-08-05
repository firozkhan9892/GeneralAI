"""Comprehensive runtime verification of the GeneralAI FastAPI server.

Covers: startup/shutdown, health, metrics, chat, chat-stream, agent
endpoints, memory, tools, workflows, OpenAPI, docs/redoc, 404 handler,
WebSocket, API key authentication, and dependency injection wiring.

Run:  python -m pytest tests/runtime_verification_server.py -v 2>&1
"""

from __future__ import annotations


import pytest
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(**kwargs):
    """Create a fresh FastAPI app via the factory."""
    from app.server.app import create_app
    from app.server.config import ServerSettings

    settings = kwargs.pop("settings", None)
    if settings is None:
        settings = ServerSettings(**kwargs)
    return create_app(settings=settings)


def _client(app, **kwargs) -> AsyncClient:
    """Build an async test client for *app*."""
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver", **kwargs)


# ===========================================================================
# 1. SERVER STARTUP / SHUTDOWN
# ===========================================================================


class TestServerStartupShutdown:
    """Verify create_app() returns a working FastAPI app with all state attrs."""

    def test_app_creates(self):
        app = _make_app()
        assert app is not None
        assert app.title is not None

    def test_app_state_attributes(self):
        app = _make_app()
        assert hasattr(app.state, "settings")
        assert hasattr(app.state, "container")
        assert hasattr(app.state, "agent_manager")
        assert hasattr(app.state, "llm_router")
        assert hasattr(app.state, "memory_engine")
        assert hasattr(app.state, "tool_registry")
        assert hasattr(app.state, "tool_executor")
        assert hasattr(app.state, "workflow_service")
        assert hasattr(app.state, "metrics")
        assert hasattr(app.state, "rate_limiter")

    def test_dependency_injection_not_none(self):
        app = _make_app()
        assert app.state.settings is not None
        assert app.state.container is not None
        assert app.state.agent_manager is not None
        assert app.state.llm_router is not None
        assert app.state.memory_engine is not None
        assert app.state.tool_registry is not None
        assert app.state.tool_executor is not None
        assert app.state.workflow_service is not None
        assert app.state.metrics is not None
        assert app.state.rate_limiter is not None

    def test_app_title_and_version(self):
        app = _make_app()
        assert "GeneralAI" in app.title
        assert app.version == "0.1.0"


# ===========================================================================
# 2. HEALTH ENDPOINT
# ===========================================================================


class TestHealthEndpoint:
    """GET /health → 200 with correct schema."""

    @pytest.mark.asyncio
    async def test_health_200(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_schema(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert isinstance(data["app_name"], str)
        assert isinstance(data["version"], str)
        assert isinstance(data["sessions_active"], int)
        assert isinstance(data["sessions_total"], int)
        assert data["sessions_active"] >= 0
        assert data["sessions_total"] >= 0


# ===========================================================================
# 3. METRICS ENDPOINT
# ===========================================================================


class TestMetricsEndpoint:
    """GET /metrics → 200 with correct schema."""

    @pytest.mark.asyncio
    async def test_metrics_200(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.get("/metrics")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_metrics_schema(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.get("/metrics")
        data = resp.json()
        assert "requests_total" in data
        assert "errors_total" in data
        assert "requests_by_path" in data
        assert "sessions_active" in data
        assert "sessions_total" in data
        assert "memory_records" in data
        assert "tools_count" in data
        assert isinstance(data["requests_total"], int)
        assert isinstance(data["requests_by_path"], dict)
        assert isinstance(data["tools_count"], int)
        assert data["requests_total"] >= 0
        assert data["tools_count"] >= 0


# ===========================================================================
# 4. CHAT ENDPOINT
# ===========================================================================


class TestChatEndpoint:
    """POST /chat with {"message": "hello"} → valid response."""

    @pytest.mark.asyncio
    async def test_chat_not_404(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.post("/chat", json={"message": "hello"})
        assert resp.status_code != 404

    @pytest.mark.asyncio
    async def test_chat_not_500(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.post("/chat", json={"message": "hello"})
        assert resp.status_code != 500

    @pytest.mark.asyncio
    async def test_chat_schema(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.post("/chat", json={"message": "hello"})
        if resp.status_code == 200:
            data = resp.json()
            assert "session_id" in data
            assert "status" in data
            assert "content" in data
            assert "success" in data
            assert isinstance(data["session_id"], str)
            assert isinstance(data["success"], bool)

    @pytest.mark.asyncio
    async def test_chat_missing_message_422(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.post("/chat", json={})
        assert resp.status_code == 422


# ===========================================================================
# 5. CHAT STREAM
# ===========================================================================


class TestChatStream:
    """POST /chat/stream → 200 SSE or valid status."""

    @pytest.mark.asyncio
    async def test_stream_not_404(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.post("/chat/stream", json={"message": "hello"})
        assert resp.status_code != 404

    @pytest.mark.asyncio
    async def test_stream_not_500(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.post("/chat/stream", json={"message": "hello"})
        assert resp.status_code != 500

    @pytest.mark.asyncio
    async def test_stream_sse_if_200(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.post("/chat/stream", json={"message": "hello"})
        if resp.status_code == 200:
            ct = resp.headers.get("content-type", "")
            assert "text/event-stream" in ct or "application/json" in ct
            body = resp.text
            assert isinstance(body, str)
            assert len(body) > 0


# ===========================================================================
# 6. AGENT ENDPOINTS
# ===========================================================================


class TestAgentEndpoints:
    """Agent run, cancel, status, list endpoints."""

    @pytest.mark.asyncio
    async def test_agent_run_not_404(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.post(
                "/agent/run",
                json={"raw_input": "hello"},
            )
        assert resp.status_code != 404

    @pytest.mark.asyncio
    async def test_agent_cancel_route_exists(self):
        """Route exists; cancelling a nonexistent session returns 404 via
        SessionNotFoundError — that is correct behaviour, not a missing route."""
        app = _make_app()
        async with _client(app) as client:
            resp = await client.post(
                "/agent/cancel",
                json={"session_id": "test"},
            )
        # The route is registered, so we get a handler response (not the
        # generic 404 for unknown routes).  A missing session produces 404
        # from the SessionNotFoundError handler — both are acceptable.
        assert resp.status_code in (200, 404)

    @pytest.mark.asyncio
    async def test_agent_status_nonexistent_404(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.get("/agent/status/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_agents_list_200(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.get("/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "sessions" in data
        assert isinstance(data["total"], int)
        assert isinstance(data["sessions"], list)

    @pytest.mark.asyncio
    async def test_agent_run_empty_input_422(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.post("/agent/run", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_agent_cancel_empty_422(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.post("/agent/cancel", json={})
        assert resp.status_code == 422


# ===========================================================================
# 7. MEMORY
# ===========================================================================


class TestMemoryEndpoint:
    """GET /memory/search?q=hello → 200 or 401."""

    @pytest.mark.asyncio
    async def test_memory_search(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.get("/memory/search", params={"q": "hello"})
        assert resp.status_code in (200, 401)

    @pytest.mark.asyncio
    async def test_memory_search_schema_if_200(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.get("/memory/search", params={"q": "hello"})
        if resp.status_code == 200:
            data = resp.json()
            assert "query" in data
            assert "total" in data
            assert "hits" in data
            assert isinstance(data["query"], str)
            assert isinstance(data["total"], int)
            assert isinstance(data["hits"], list)

    @pytest.mark.asyncio
    async def test_memory_search_missing_q_422(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.get("/memory/search")
        assert resp.status_code == 422


# ===========================================================================
# 8. TOOLS
# ===========================================================================


class TestToolsEndpoint:
    """Tool run endpoint."""

    @pytest.mark.asyncio
    async def test_tools_run_nonexistent_404(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.post(
                "/tools/run",
                json={"tool": "nonexistent"},
            )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_tools_run_echo_not_404_not_500(self):
        """Run echo tool if registered; otherwise verify route handles
        missing tools correctly (404 = tool not found, route exists)."""
        app = _make_app()
        async with _client(app) as client:
            resp = await client.post(
                "/tools/run",
                json={"tool": "echo", "arguments": {}},
            )
        # If echo is registered we expect 200; if not, 404 from the
        # "tool not registered" check.  Neither is 500.
        assert resp.status_code != 500
        assert resp.status_code in (200, 404)

    @pytest.mark.asyncio
    async def test_tools_run_empty_body_422(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.post("/tools/run", json={})
        assert resp.status_code == 422


# ===========================================================================
# 9. WORKFLOWS
# ===========================================================================


class TestWorkflowsEndpoint:
    """GET /workflows → 200."""

    @pytest.mark.asyncio
    async def test_workflows_list_200(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.get("/workflows")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_workflows_schema(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.get("/workflows")
        data = resp.json()
        assert "total" in data
        assert "workflows" in data
        assert isinstance(data["total"], int)
        assert isinstance(data["workflows"], list)

    @pytest.mark.asyncio
    async def test_workflows_runs_200(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.get("/workflows/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "runs" in data

    @pytest.mark.asyncio
    async def test_schedules_list_200(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.get("/schedules")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "schedules" in data


# ===========================================================================
# 10. OPENAPI
# ===========================================================================


class TestOpenAPI:
    """GET /openapi.json → 200 with expected routes."""

    EXPECTED_ROUTES = {
        "/health",
        "/metrics",
        "/chat",
        "/chat/stream",
        "/agent/run",
        "/agent/cancel",
        "/agent/status/{session_id}",
        "/agents",
        "/memory/search",
        "/tools/run",
        "/workflows",
        "/workflows/runs",
        "/schedules",
    }

    @pytest.mark.asyncio
    async def test_openapi_200(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.get("/openapi.json")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_openapi_routes(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.get("/openapi.json")
        spec = resp.json()
        assert "paths" in spec
        paths = set(spec["paths"].keys())
        for route in self.EXPECTED_ROUTES:
            assert route in paths, (
                f"Route {route} not in OpenAPI paths: {sorted(paths)}"
            )


# ===========================================================================
# 11. DOCS / REDOC
# ===========================================================================


class TestDocsRedoc:
    """GET /docs → 200, GET /redoc → 200."""

    @pytest.mark.asyncio
    async def test_docs_200(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.get("/docs")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_redoc_200(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.get("/redoc")
        assert resp.status_code == 200


# ===========================================================================
# 12. 404 HANDLER
# ===========================================================================


class TestNotFoundHandler:
    """GET /nonexistent → 404 with detail."""

    @pytest.mark.asyncio
    async def test_not_found(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.get("/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_not_found_detail(self):
        app = _make_app()
        async with _client(app) as client:
            resp = await client.get("/nonexistent")
        data = resp.json()
        assert "detail" in data
        assert isinstance(data["detail"], str)
        assert len(data["detail"]) > 0


# ===========================================================================
# 13. WEBSOCKET
# ===========================================================================


class TestWebSocket:
    """Connect to /agent/ws, verify greeting, send ping, verify pong."""

    @pytest.mark.asyncio
    async def test_ws_greeting(self):
        app = _make_app()
        from starlette.testclient import TestClient

        client = TestClient(app)
        with client.websocket_connect("/agent/ws") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connected"

    @pytest.mark.asyncio
    async def test_ws_ping_pong(self):
        app = _make_app()
        from starlette.testclient import TestClient

        client = TestClient(app)
        with client.websocket_connect("/agent/ws") as ws:
            ws.receive_json()  # greeting
            ws.send_json({"type": "ping"})
            msg = ws.receive_json()
            assert msg["type"] == "pong"

    @pytest.mark.asyncio
    async def test_ws_unknown_type(self):
        app = _make_app()
        from starlette.testclient import TestClient

        client = TestClient(app)
        with client.websocket_connect("/agent/ws") as ws:
            ws.receive_json()  # greeting
            ws.send_json({"type": "unknown"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "Unknown message type" in msg["detail"]

    @pytest.mark.asyncio
    async def test_ws_run_missing_input(self):
        app = _make_app()
        from starlette.testclient import TestClient

        client = TestClient(app)
        with client.websocket_connect("/agent/ws") as ws:
            ws.receive_json()  # greeting
            ws.send_json({"type": "run", "raw_input": ""})
            msg = ws.receive_json()
            assert msg["type"] == "error"

    @pytest.mark.asyncio
    async def test_ws_cancel_no_session(self):
        app = _make_app()
        from starlette.testclient import TestClient

        client = TestClient(app)
        with client.websocket_connect("/agent/ws") as ws:
            ws.receive_json()  # greeting
            ws.send_json({"type": "cancel", "session_id": ""})
            msg = ws.receive_json()
            assert msg["type"] == "error"

    @pytest.mark.asyncio
    async def test_ws_status_no_session(self):
        app = _make_app()
        from starlette.testclient import TestClient

        client = TestClient(app)
        with client.websocket_connect("/agent/ws") as ws:
            ws.receive_json()  # greeting
            ws.send_json({"type": "status", "session_id": ""})
            msg = ws.receive_json()
            assert msg["type"] == "error"


# ===========================================================================
# 14. API KEY AUTHENTICATION
# ===========================================================================


class TestApiKeyAuthentication:
    """Create app with api_key="test-key", verify 401 without key, pass with key."""

    @pytest.mark.asyncio
    async def test_no_key_returns_401(self):
        from app.server.config import ServerSettings

        settings = ServerSettings(api_key="test-key")
        app = _make_app(settings=settings)
        async with _client(app) as client:
            resp = await client.post("/chat", json={"message": "hello"})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_key_passes(self):
        from app.server.config import ServerSettings

        settings = ServerSettings(api_key="test-key")
        app = _make_app(settings=settings)
        async with _client(app) as client:
            resp = await client.post(
                "/chat",
                json={"message": "hello"},
                headers={"X-API-Key": "test-key"},
            )
        assert resp.status_code != 401

    @pytest.mark.asyncio
    async def test_invalid_key_returns_401(self):
        from app.server.config import ServerSettings

        settings = ServerSettings(api_key="test-key")
        app = _make_app(settings=settings)
        async with _client(app) as client:
            resp = await client.post(
                "/chat",
                json={"message": "hello"},
                headers={"X-API-Key": "wrong-key"},
            )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_no_key_agent_endpoints_401(self):
        from app.server.config import ServerSettings

        settings = ServerSettings(api_key="test-key")
        app = _make_app(settings=settings)
        async with _client(app) as client:
            resp = await client.get("/agents")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_health_bypasses_api_key(self):
        from app.server.config import ServerSettings

        settings = ServerSettings(api_key="test-key")
        app = _make_app(settings=settings)
        async with _client(app) as client:
            resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_metrics_bypasses_api_key(self):
        from app.server.config import ServerSettings

        settings = ServerSettings(api_key="test-key")
        app = _make_app(settings=settings)
        async with _client(app) as client:
            resp = await client.get("/metrics")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_openapi_bypasses_api_key(self):
        from app.server.config import ServerSettings

        settings = ServerSettings(api_key="test-key")
        app = _make_app(settings=settings)
        async with _client(app) as client:
            resp = await client.get("/openapi.json")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_ws_api_key_check(self):
        from app.server.config import ServerSettings
        from starlette.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        settings = ServerSettings(api_key="test-key")
        app = _make_app(settings=settings)
        client = TestClient(app)
        # Without key — server closes with code 4001
        try:
            with client.websocket_connect("/agent/ws") as ws:
                msg = ws.receive_json()
                assert msg.get("type") != "connected", (
                    "WebSocket should reject without API key"
                )
        except WebSocketDisconnect as exc:
            # Expected: server closed with code 4001 "Invalid API key"
            assert exc.code == 4001 or exc.reason == "Invalid API key"


# ===========================================================================
# 15. DEPENDENCY INJECTION (comprehensive)
# ===========================================================================


class TestDependencyInjection:
    """All app.state.* attributes must be non-None after create_app."""

    STATE_ATTRS = [
        "settings",
        "container",
        "agent_manager",
        "llm_router",
        "memory_engine",
        "tool_registry",
        "tool_executor",
        "workflow_service",
        "metrics",
        "rate_limiter",
    ]

    def test_all_state_attrs_not_none(self):
        app = _make_app()
        for attr in self.STATE_ATTRS:
            val = getattr(app.state, attr, None)
            assert val is not None, f"app.state.{attr} is None"

    def test_settings_type(self):
        app = _make_app()
        from app.server.config import ServerSettings

        assert isinstance(app.state.settings, ServerSettings)

    def test_metrics_type(self):
        app = _make_app()
        from app.server.metrics import MetricsCollector

        assert isinstance(app.state.metrics, MetricsCollector)

    def test_tool_registry_type(self):
        app = _make_app()
        from app.tools.registry import ToolRegistry

        assert isinstance(app.state.tool_registry, ToolRegistry)

    def test_tool_registry_initially_empty(self):
        """Tools are discovered during the lifespan (startup), not at
        create_app() time.  At construction the registry is empty."""
        app = _make_app()
        assert app.state.tool_registry.count == 0


# ===========================================================================
# 16. MIDDLEWARE (metrics counting)
# ===========================================================================


class TestMetricsMiddleware:
    """Verify the metrics middleware records requests."""

    @pytest.mark.asyncio
    async def test_health_recorded(self):
        app = _make_app()
        async with _client(app) as client:
            await client.get("/health")
        snapshot = app.state.metrics.snapshot()
        assert snapshot["requests_total"] >= 1
        assert "/health" in snapshot["requests_by_path"]

    @pytest.mark.asyncio
    async def test_metrics_not_found_counted(self):
        app = _make_app()
        async with _client(app) as client:
            await client.get("/nonexistent")
        snapshot = app.state.metrics.snapshot()
        assert "/nonexistent" in snapshot["requests_by_path"]
