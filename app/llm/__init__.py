"""LLM Provider Architecture.

Provides a provider-agnostic abstraction over chat-completion LLM
services: unified request/response models, a provider base class,
a registry, a factory, and concrete implementations including a
deterministic mock for testing.
"""

from __future__ import annotations

from app.llm.base import BaseLLMProvider
from app.llm.exceptions import (
    LLMError,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderNotSupportedError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderStreamError,
    ProviderTimeoutError,
)
from app.llm.factory import ProviderFactory
from app.llm.models import (
    ChatRequest,
    ChatResponse,
    Message,
    ModelInfo,
    ResponseFormat,
    Role,
    StreamChunk,
    ToolCall,
    ToolDefinition,
    Usage,
)
from app.llm.providers import (
    GeminiProvider,
    MockProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from app.llm.registry import ProviderRegistry
from app.llm.transport import (
    HttpTransport,
    HttpResponse,
    UrllibHttpTransport,
)

__all__ = [
    "BaseLLMProvider",
    "LLMError",
    "ProviderAuthenticationError",
    "ProviderConfigurationError",
    "ProviderConnectionError",
    "ProviderError",
    "ProviderNotSupportedError",
    "ProviderRateLimitError",
    "ProviderResponseError",
    "ProviderStreamError",
    "ProviderTimeoutError",
    "ProviderFactory",
    "ChatRequest",
    "ChatResponse",
    "Message",
    "ModelInfo",
    "ResponseFormat",
    "Role",
    "StreamChunk",
    "ToolCall",
    "ToolDefinition",
    "Usage",
    "GeminiProvider",
    "MockProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "ProviderRegistry",
    "HttpTransport",
    "HttpResponse",
    "UrllibHttpTransport",
]
