"""Unified streaming layer for LLM providers.

Normalizes streaming output from all providers into a single
:class:`StreamChunk` interface, regardless of the underlying
provider's native streaming format.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Callable, Iterator

from app.llm.base import BaseLLMProvider
from app.llm.models import ChatRequest, ChatResponse, StreamChunk
from app.llm.router_exceptions import RoutingError

log = logging.getLogger(__name__)

StreamHandler = Callable[[StreamChunk], Any]


class UnifiedStreamer:
    """Converts any provider's stream into a uniform :class:`StreamChunk` flow.

    The streamer wraps both synchronous (``Iterator``) and
    asynchronous (``AsyncIterator``) provider streams and exposes
    them through the same async interface.
    """

    async def normalize(
        self,
        provider: BaseLLMProvider,
        request: ChatRequest,
    ) -> AsyncIterator[StreamChunk]:
        """Stream from *provider* and yield normalized :class:`StreamChunk`.

        Args:
            provider: The provider to stream from.
            request: The chat request with ``stream`` enabled.

        Yields:
            Normalized :class:`StreamChunk` objects.

        Raises:
            RoutingError: If the provider does not support streaming.
            ProviderError: If the stream fails.
        """
        info = provider.model_info(request.model)
        if not info.supports_streaming:
            raise RoutingError(
                f"Provider '{provider.name}' does not support streaming",
                module="llm.unified_streamer",
                context={
                    "provider": provider.name,
                    "model": request.model,
                },
            )

        stream_request = request.model_copy(update={"stream": True})
        stream_iter = provider.stream(stream_request)
        async for chunk in self._as_async(stream_iter, request):
            yield chunk

    async def _as_async(
        self,
        stream_iter: Iterator[StreamChunk],
        request: ChatRequest,
    ) -> AsyncIterator[StreamChunk]:
        """Adapt a sync iterator to an async iterator in a thread."""

        def _collect() -> list[StreamChunk]:
            return list(stream_iter)

        chunks = await asyncio.to_thread(_collect)
        for chunk in chunks:
            yield chunk

    async def stream_to_handler(
        self,
        provider: BaseLLMProvider,
        request: ChatRequest,
        handler: StreamHandler,
    ) -> ChatResponse | None:
        """Stream from *provider* and pass each chunk to *handler*.

        Args:
            provider: The provider to stream from.
            request: The chat request with ``stream`` enabled.
            handler: Callable receiving each :class:`StreamChunk`.

        Returns:
            The accumulated :class:`ChatResponse`, or ``None`` if the
            stream produced no content.
        """
        parts: list[str] = []
        final_usage = None
        final_model = request.model
        finish_reason = "stop"

        async for chunk in self.normalize(provider, request):
            handler(chunk)
            if chunk.content:
                parts.append(chunk.content)
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            if chunk.usage:
                final_usage = chunk.usage
            final_model = chunk.model or final_model

        if not parts and not final_usage:
            return None

        from app.llm.models import Usage

        return ChatResponse(
            content="".join(parts),
            model=final_model,
            provider=provider.name,
            finish_reason=finish_reason,
            usage=final_usage if final_usage else Usage(),
        )
