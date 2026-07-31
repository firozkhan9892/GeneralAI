"""OpenAI chat completions provider."""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

from app.llm.exceptions import ProviderResponseError
from app.llm.models import (
    ChatRequest,
    ChatResponse,
    Message,
    StreamChunk,
    ToolCall,
    Usage,
)
from app.llm.providers._base import BaseHttpProvider
from app.llm.transport import HttpResponse, iter_lines, iter_sse_payloads

log = logging.getLogger(__name__)


def _message_to_dict(message: Message) -> dict[str, Any]:
    """Convert a unified message to OpenAI's wire format."""
    data: dict[str, Any] = {"role": message.role.value}
    if message.content:
        data["content"] = message.content
    if message.name is not None:
        data["name"] = message.name
    if message.tool_call_id is not None:
        data["tool_call_id"] = message.tool_call_id
    return data


def _parse_usage(data: dict[str, Any]) -> Usage:
    """Extract usage from an OpenAI response body."""
    usage = data.get("usage") or {}
    return Usage(
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        total_tokens=int(usage.get("total_tokens", 0)),
    )


def _parse_tool_calls(data: list[dict[str, Any]] | None) -> tuple[ToolCall, ...]:
    """Parse OpenAI tool calls into unified tool calls."""
    if not data:
        return ()
    calls: list[ToolCall] = []
    for call in data:
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = {}
        else:
            parsed = arguments or {}
        calls.append(
            ToolCall(
                id=call.get("id", ""),
                name=function.get("name", ""),
                arguments=parsed if isinstance(parsed, dict) else {},
            )
        )
    return tuple(calls)


class OpenAIProvider(BaseHttpProvider):
    """Provider for OpenAI's ``/chat/completions`` API."""

    name = "openai"
    base_url = "https://api.openai.com/v1"

    def _default_model_name(self) -> str:
        return "gpt-4o-mini"

    def _supports_streaming(self) -> bool:
        return True

    def _supports_tools(self) -> bool:
        return True

    def _supports_json(self) -> bool:
        return True

    def _chat_url(self, model: str) -> str:
        return f"{self._base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _build_payload(self, request: ChatRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._get_model(request, self._model),
            "messages": [_message_to_dict(m) for m in request.messages],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop:
            payload["stop"] = list(request.stop)
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
        if request.response_format is not None:
            if request.response_format.type == "json":
                payload["response_format"] = {"type": "json_object"}
            elif request.response_format.type == "json_schema":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": request.response_format.json_schema or {},
                }
        return payload

    def _parse_response(self, response: HttpResponse, model: str) -> ChatResponse:
        body = response.json()
        choices = body.get("choices")
        if not choices:
            raise ProviderResponseError(
                "OpenAI response contained no choices",
                module="llm.providers.openai",
            )
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        tool_calls = _parse_tool_calls(message.get("tool_calls"))
        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            usage=_parse_usage(body),
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
        for payload_text in iter_sse_payloads(iter_lines(chunks)):
            try:
                yield json.loads(payload_text)
            except json.JSONDecodeError:
                log.warning("Skipped malformed SSE payload: %r", payload_text)

    def _parse_stream_chunk(self, event: Any, model: str) -> StreamChunk | None:
        if not isinstance(event, dict):
            return None
        choices = event.get("choices")
        if not choices:
            return None
        choice = choices[0]
        delta = choice.get("delta") or {}
        tool_calls = _parse_tool_calls(delta.get("tool_calls"))
        finish_reason = choice.get("finish_reason")
        usage = None
        if event.get("usage"):
            usage = _parse_usage(event)
        content = delta.get("content")
        if not content and not tool_calls and finish_reason is None and usage is None:
            return None
        return StreamChunk(
            content=content or "",
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            model=event.get("model") or model,
            provider=self.name,
        )
