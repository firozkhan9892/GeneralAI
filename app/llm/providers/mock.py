"""Deterministic mock LLM provider for testing.

Produces predictable, repeatable responses with no network access, so
unit and integration tests can exercise the full provider contract
without depending on external services.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Iterator

from app.llm.base import BaseLLMProvider
from app.llm.models import (
    ChatRequest,
    ChatResponse,
    ModelInfo,
    Role,
    StreamChunk,
    ToolCall,
    Usage,
)

log = logging.getLogger(__name__)


class MockProvider(BaseLLMProvider):
    """A deterministic, offline provider.

    The generated content is a pure function of the request, so identical
    requests always produce identical responses.  Optional behaviours can
    be configured to simulate tool calls, JSON output, and streaming.
    """

    name = "mock"

    def __init__(
        self,
        *,
        model: str = "mock-1",
        delay_s: float = 0.0,
        echo_input: bool = True,
        tool_calls_on_tools: bool = True,
        json_mode_enabled: bool = True,
    ) -> None:
        self._model = model
        self._delay_s = delay_s
        self._echo_input = echo_input
        self._tool_calls_on_tools = tool_calls_on_tools
        self._json_mode_enabled = json_mode_enabled

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def default_model(self) -> str:
        """Return the configured model name."""
        return self._model

    def model_info(self, model: str | None = None) -> ModelInfo:
        return ModelInfo(
            name=model or self._model,
            provider=self.name,
            supports_streaming=True,
            supports_tools=True,
            supports_json=True,
            max_context_tokens=8192,
            max_output_tokens=2048,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _delay(self) -> None:
        """Optionally sleep to simulate latency."""
        if self._delay_s > 0:
            time.sleep(self._delay_s)

    def _last_user_message(self, request: ChatRequest) -> str:
        """Return the content of the last user message."""
        for message in reversed(request.messages):
            if message.role == Role.USER:
                return message.content
        return ""

    def _build_content(self, request: ChatRequest) -> str:
        """Deterministically derive response text from the request."""
        prompt = self._last_user_message(request)
        if self._json_mode_enabled and request.response_format is not None:
            return json.dumps(
                {
                    "model": self._get_model(request, self._model),
                    "prompt": prompt,
                    "reply": f"Echo: {prompt}",
                },
                sort_keys=True,
            )
        if self._echo_input:
            return f"Echo: {prompt}"
        digest = hashlib.sha256(
            json.dumps(request.messages, default=str).encode("utf-8")
        ).hexdigest()[:8]
        return f"Mock response [{digest}]"

    def _build_tool_calls(self, request: ChatRequest) -> tuple[ToolCall, ...]:
        """Return a deterministic tool call when tools are offered."""
        if not self._tool_calls_on_tools or not request.tools:
            return ()
        tool = request.tools[0]
        return (
            ToolCall(
                id=f"call_{hashlib.sha256(tool.name.encode()).hexdigest()[:8]}",
                name=tool.name,
                arguments={"input": self._last_user_message(request)},
            ),
        )

    def _estimate_usage(self, request: ChatRequest, content: str) -> Usage:
        """Deterministically estimate token usage from input sizes."""
        prompt_tokens = sum(
            len(message.content.split()) + 4 for message in request.messages
        )
        completion_tokens = len(content.split()) + 4
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, request: ChatRequest) -> ChatResponse:
        """Return a deterministic complete response."""
        self._delay()
        content = self._build_content(request)
        tool_calls = self._build_tool_calls(request)
        if tool_calls:
            content = ""
        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            usage=self._estimate_usage(request, content or str(tool_calls)),
            model=self._get_model(request, self._model),
            provider=self.name,
        )

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        """Yield a deterministic streaming response in fixed-size chunks."""
        self._delay()
        model = self._get_model(request, self._model)
        tool_calls = self._build_tool_calls(request)
        if tool_calls:
            yield StreamChunk(
                content="",
                tool_calls=tool_calls,
                finish_reason="tool_calls",
                model=model,
                provider=self.name,
            )
            return

        content = self._build_content(request)
        chunk_size = 4
        for index in range(0, len(content), chunk_size):
            yield StreamChunk(
                content=content[index : index + chunk_size],
                model=model,
                provider=self.name,
            )
        yield StreamChunk(
            content="",
            finish_reason="stop",
            usage=self._estimate_usage(request, content),
            model=model,
            provider=self.name,
        )
