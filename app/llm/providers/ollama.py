"""Ollama provider — local model server chat API."""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

from app.llm.exceptions import ProviderResponseError
from app.llm.models import (
    ChatRequest,
    ChatResponse,
    StreamChunk,
    ToolCall,
    Usage,
)
from app.llm.providers._base import BaseHttpProvider
from app.llm.transport import HttpResponse, iter_lines

log = logging.getLogger(__name__)


class OllamaProvider(BaseHttpProvider):
    """Provider for a local Ollama server (``/api/chat``)."""

    name = "ollama"
    base_url = "http://localhost:11434"

    def _default_model_name(self) -> str:
        return "llama3.2"

    def _supports_streaming(self) -> bool:
        return True

    def _supports_tools(self) -> bool:
        return True

    def _supports_json(self) -> bool:
        return True

    def _chat_url(self, model: str) -> str:
        return f"{self._base_url}/api/chat"

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def _build_payload(self, request: ChatRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._get_model(request, self._model),
            "messages": [
                {"role": m.role.value, "content": m.content} for m in request.messages
            ],
        }
        options: dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if options:
            payload["options"] = options
        if request.stop:
            payload["stop"] = list(request.stop)
        if request.response_format is not None:
            payload["format"] = "json"
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ]
        return payload

    def _parse_response(self, response: HttpResponse, model: str) -> ChatResponse:
        body = response.json()
        message = body.get("message")
        if not isinstance(message, dict):
            raise ProviderResponseError(
                "Ollama response contained no message",
                module="llm.providers.ollama",
            )
        content = message.get("content") or ""
        tool_calls = _parse_tool_calls(message.get("tool_calls"))
        prompt_tokens = int(body.get("prompt_eval_count", 0))
        completion_tokens = int(body.get("eval_count", 0))
        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="stop" if body.get("done", True) else "stop",
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            model=body.get("model") or model,
            provider=self.name,
        )

    def _iter_events(self, model: str, payload: dict[str, Any]) -> Iterator[Any]:
        chunks = self._transport.post_stream(
            self._chat_url(model),
            headers=self._headers(),
            payload=payload,
            timeout_s=self._timeout_s,
        )
        for line in iter_lines(chunks):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("Skipped malformed Ollama line: %r", line)

    def _parse_stream_chunk(self, event: Any, model: str) -> StreamChunk | None:
        if not isinstance(event, dict):
            return None
        message = event.get("message")
        content = ""
        if isinstance(message, dict):
            content = message.get("content") or ""
        done = event.get("done", False)
        finish_reason = "stop" if done else None
        if not content and finish_reason is None:
            return None
        return StreamChunk(
            content=content,
            finish_reason=finish_reason,
            model=event.get("model") or model,
            provider=self.name,
        )


def _parse_tool_calls(data: Any) -> tuple[ToolCall, ...]:
    """Parse Ollama tool calls into unified tool calls."""
    if not isinstance(data, list):
        return ()
    calls: list[ToolCall] = []
    for call in data:
        function = call.get("function") or {}
        arguments = function.get("arguments") or {}
        calls.append(
            ToolCall(
                id=call.get("id", ""),
                name=function.get("name", ""),
                arguments=arguments if isinstance(arguments, dict) else {},
            )
        )
    return tuple(calls)
