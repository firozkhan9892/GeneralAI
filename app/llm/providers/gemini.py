"""Google Gemini provider — ``generateContent`` API."""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

from app.llm.exceptions import ProviderResponseError
from app.llm.models import (
    ChatRequest,
    ChatResponse,
    Message,
    Role,
    StreamChunk,
    Usage,
)
from app.llm.providers._base import BaseHttpProvider
from app.llm.transport import HttpResponse, iter_lines, iter_sse_payloads

log = logging.getLogger(__name__)


def _contents_from_messages(messages: tuple[Message, ...]) -> list[dict[str, Any]]:
    """Map unified messages to Gemini ``contents`` (system messages excluded)."""
    contents: list[dict[str, Any]] = []
    for message in messages:
        if message.role == Role.SYSTEM:
            continue
        role = "model" if message.role == Role.ASSISTANT else "user"
        contents.append({"role": role, "parts": [{"text": message.content}]})
    return contents


class GeminiProvider(BaseHttpProvider):
    """Provider for Google's Gemini generative models."""

    name = "gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta"

    def _default_model_name(self) -> str:
        return "gemini-1.5-pro"

    def _supports_streaming(self) -> bool:
        return True

    def _supports_tools(self) -> bool:
        return True

    def _supports_json(self) -> bool:
        return True

    def _chat_url(self, model: str) -> str:
        return f"{self._base_url}/models/{model}:generateContent"

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["x-goog-api-key"] = self._api_key
        return headers

    def _build_payload(self, request: ChatRequest) -> dict[str, Any]:
        system_text = "".join(
            m.content for m in request.messages if m.role == Role.SYSTEM
        )
        payload: dict[str, Any] = {
            "contents": _contents_from_messages(request.messages),
        }
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}
        config: dict[str, Any] = {}
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.max_tokens is not None:
            config["maxOutputTokens"] = request.max_tokens
        if request.stop:
            config["stopSequences"] = list(request.stop)
        if request.response_format is not None:
            config["responseMimeType"] = "application/json"
        if config:
            payload["generationConfig"] = config
        if request.tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        }
                    ]
                }
                for tool in request.tools
            ]
        return payload

    def _parse_response(self, response: HttpResponse, model: str) -> ChatResponse:
        body = response.json()
        candidates = body.get("candidates") or []
        if not candidates:
            raise ProviderResponseError(
                "Gemini response contained no candidates",
                module="llm.providers.gemini",
            )
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        content = "".join(part.get("text", "") for part in parts)
        usage_meta = body.get("usageMetadata") or {}
        prompt_tokens = int(usage_meta.get("promptTokenCount", 0))
        completion_tokens = int(usage_meta.get("candidatesTokenCount", 0))
        return ChatResponse(
            content=content,
            finish_reason=str(candidate.get("finishReason") or "stop"),
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            model=model,
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
        candidates = event.get("candidates") or []
        if not candidates:
            return None
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts)
        finish_reason = candidate.get("finishReason")
        if not text and finish_reason is None:
            return None
        return StreamChunk(
            content=text,
            finish_reason=finish_reason,
            model=model,
            provider=self.name,
        )
