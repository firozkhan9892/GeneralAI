"""Integration tests for LLM cognitive pipeline."""

from __future__ import annotations

from fastapi.testclient import TestClient
from app.server.app import create_app
from app.server.config import ServerSettings
from app.llm.models import ChatResponse, Usage
from app.llm.router_exceptions import FallbackExhaustedError


class TestCognitiveLLMIntegration:
    """Tests for Phase 15 Task 1."""

    def test_chat_path_invokes_llm_router(self, monkeypatch) -> None:
        """Standard /chat path invokes the injected LLMRouter."""
        # Create an app without API keys required
        app = create_app(settings=ServerSettings())
        client = TestClient(app)

        # Get the LLMRouter from the app state
        router = app.state.llm_router
        assert router is not None

        # Spy on generate_async
        call_count = 0
        last_request = None

        async def mock_generate_async(request, *args, **kwargs):
            nonlocal call_count, last_request
            call_count += 1
            last_request = request
            # Return a fake response
            return ChatResponse(
                content="Mocked cognitive response",
                model="test",
                provider="test",
                usage=Usage(),
            )

        monkeypatch.setattr(router, "generate_async", mock_generate_async)

        # Send a chat request
        response = client.post("/chat", json={"message": "Test message"})

        # Verify it went through
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "Mocked cognitive response" in data["content"]

        # Verify the router was actually invoked
        assert call_count == 1
        assert last_request is not None
        assert len(last_request.messages) > 0
        prompt = last_request.messages[0].content
        assert "synthesize this" in prompt
        assert "Test message" in prompt
        assert "STRUCTURED RESULT" in prompt

    def test_chat_llm_router_failure_is_explicit(self, monkeypatch) -> None:
        """A failed LLM call must fail the run rather than fabricate output."""
        app = create_app(settings=ServerSettings())
        client = TestClient(app)
        router = app.state.llm_router

        async def failing_generate_async(request, *args, **kwargs):
            raise FallbackExhaustedError(
                "All configured providers failed",
                module="llm.llm_router",
            )

        monkeypatch.setattr(router, "generate_async", failing_generate_async)
        response = client.post("/chat", json={"message": "Test message"})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["status"] == "failed"
        assert "configured providers failed" in (data["error"] or "")
        assert "Mocked cognitive response" not in data["content"]
