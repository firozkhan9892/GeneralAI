"""Base abstraction for all LLM providers.

Defines the uniform, provider-agnostic contract that every provider
implementation must satisfy.  Providers translate their native wire
formats into the unified :mod:`app.llm.models` types at this boundary,
keeping provider-specific code inside the provider layer.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator

from app.llm.models import (
    ChatRequest,
    ChatResponse,
    ModelInfo,
    StreamChunk,
)


class BaseLLMProvider(ABC):
    """Abstract contract for chat-completion LLM providers.

    Subclasses must implement the synchronous ``generate``/``stream``
    methods.  Async variants are provided on top of the sync ones by
    the base class (unless overridden), so implementations do not need
    to duplicate orchestration logic.
    """

    name: str = ""

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Return the default model identifier for this provider."""

    @abstractmethod
    def model_info(self, model: str | None = None) -> ModelInfo:
        """Describe a model exposed by this provider.

        Args:
            model: Optional model name; defaults to ``default_model``.

        Returns:
            Model metadata.
        """

    # ------------------------------------------------------------------
    # Synchronous generation
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(self, request: ChatRequest) -> ChatResponse:
        """Generate a complete response synchronously.

        Args:
            request: The chat request.

        Returns:
            The complete chat response.

        Raises:
            ProviderError: On any provider failure.
        """

    @abstractmethod
    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        """Stream a response synchronously.

        Args:
            request: The chat request with ``stream`` enabled.

        Yields:
            Chunks of the streaming response.

        Raises:
            ProviderError: On any provider failure.
        """

    # ------------------------------------------------------------------
    # Asynchronous generation
    # ------------------------------------------------------------------

    async def generate_async(self, request: ChatRequest) -> ChatResponse:
        """Generate a complete response asynchronously.

        The default implementation offloads the synchronous
        :meth:`generate` to a worker thread.  Providers with native
        async clients should override this method.

        Args:
            request: The chat request.

        Returns:
            The complete chat response.
        """
        return await asyncio.to_thread(self.generate, request)

    async def stream_async(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Stream a response asynchronously.

        The default implementation offloads the synchronous
        :meth:`stream` to a worker thread.  Providers with native
        async clients should override this method.

        Args:
            request: The chat request with ``stream`` enabled.

        Yields:
            Chunks of the streaming response.
        """

        def _collect() -> list[StreamChunk]:
            return list(self.stream(request))

        for chunk in await asyncio.to_thread(_collect):
            yield chunk

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_model(request: ChatRequest, default_model: str) -> str:
        """Resolve the model name to use for a request."""
        return request.model or default_model
