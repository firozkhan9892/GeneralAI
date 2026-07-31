"""Shared base for HTTP-backed LLM providers."""

from __future__ import annotations

import logging
from typing import Any, Iterator

from app.llm.base import BaseLLMProvider
from app.llm.models import ChatRequest, ChatResponse, ModelInfo, StreamChunk
from app.llm.transport import HttpTransport, HttpResponse, UrllibHttpTransport

log = logging.getLogger(__name__)


class BaseHttpProvider(BaseLLMProvider):
    """Base class for providers that call an HTTP chat API.

    Subclasses provide the ``name``, default URL builder, payload
    mapping, and response mapping; the HTTP plumbing and async
    wrappers are inherited.
    """

    base_url: str = ""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float = 60.0,
        transport: HttpTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = (base_url or self.base_url).rstrip("/")
        self._model = model or self._default_model_name()
        self._timeout_s = timeout_s
        self._transport = transport or UrllibHttpTransport()
        log.debug(
            "Initialised %s provider (model=%s, base_url=%s)",
            self.name,
            self._model,
            self._base_url,
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _default_model_name(self) -> str:
        """Return the provider's built-in default model name."""
        raise NotImplementedError

    @property
    def default_model(self) -> str:
        """Return the configured model identifier."""
        return self._model

    def model_info(self, model: str | None = None) -> ModelInfo:
        """Return metadata for a model exposed by this provider."""
        return ModelInfo(
            name=model or self._model,
            provider=self.name,
            supports_streaming=self._supports_streaming(),
            supports_tools=self._supports_tools(),
            supports_json=self._supports_json(),
        )

    def _supports_streaming(self) -> bool:
        """Whether the provider supports streaming."""
        return True

    def _supports_tools(self) -> bool:
        """Whether the provider supports function calling."""
        return False

    def _supports_json(self) -> bool:
        """Whether the provider supports structured JSON output."""
        return False

    # ------------------------------------------------------------------
    # Request plumbing
    # ------------------------------------------------------------------

    def _chat_url(self, model: str) -> str:
        """Return the chat completion URL for *model*."""
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        """Return request headers for this provider."""
        return {}

    def _build_payload(self, request: ChatRequest) -> dict[str, Any]:
        """Translate a unified request into the provider wire format."""
        raise NotImplementedError

    def _parse_response(self, response: HttpResponse, model: str) -> ChatResponse:
        """Translate a provider wire response into a unified response."""
        raise NotImplementedError

    def _parse_stream_chunk(self, event: Any, model: str) -> StreamChunk | None:
        """Translate one provider stream event into a unified chunk.

        Returns ``None`` for events that carry no content.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, request: ChatRequest) -> ChatResponse:
        model = self._get_model(request, self._model)
        response = self._transport.post(
            self._chat_url(model),
            headers=self._headers(),
            payload=self._build_payload(request),
            timeout_s=request.timeout_s or self._timeout_s,
        )
        return self._parse_response(response, model)

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        model = self._get_model(request, self._model)
        payload = self._build_payload(request)
        payload["stream"] = True
        for event in self._iter_events(model, payload):
            chunk = self._parse_stream_chunk(event, model)
            if chunk is not None:
                yield chunk

    def _iter_events(self, model: str, payload: dict[str, Any]) -> Iterator[Any]:
        """Yield parsed stream events from the transport."""
        raise NotImplementedError
