"""LLM Provider Architecture.

Provides a provider-agnostic abstraction over chat-completion LLM
services: unified request/response models, a provider base class,
a registry, a factory, and concrete implementations including a
deterministic mock for testing.
"""

from __future__ import annotations

from app.llm.base import BaseLLMProvider
from app.llm.bootstrap import register_llm_components
from app.llm.capability_matrix import CapabilityMatrix
from app.llm.circuit_breaker import CircuitBreaker
from app.llm.cost_optimizer import CostOptimizer
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
from app.llm.fallback_manager import FallbackManager
from app.llm.health_monitor import ProviderHealthMonitor
from app.llm.llm_router import LLMRouter
from app.llm.load_balancer import LoadBalancer
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
from app.llm.policy_engine import (
    CapabilityRequirementPolicy,
    HealthThresholdPolicy,
    MaxCostPolicy,
    PolicyEngine,
    PolicyRule,
    ProviderAllowListPolicy,
    ProviderBlockListPolicy,
)
from app.llm.prompt_cache import PromptCache
from app.llm.providers import (
    GeminiProvider,
    MockProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from app.llm.registry import ProviderRegistry
from app.llm.request_queue import (
    ProviderRateLimiter,
    RequestPriority,
    RequestQueue,
)
from app.llm.router_exceptions import (
    CircuitBreakerError,
    FallbackExhaustedError,
    NoHealthyProvidersError,
    PolicyViolationError,
    QueueTimeoutError,
    RateLimitExceededError,
    RouterError,
    RoutingError,
)
from app.llm.router_models import (
    AnalyticsEvent,
    CacheKey,
    CapabilityFlag,
    CircuitState,
    CostEstimate,
    LoadBalanceStrategy,
    Priority,
    ProviderCapabilities,
    ProviderHealthSnapshot,
    ProviderRanking,
    ProviderScore,
    RateLimitInfo,
    RouterStrategy,
    RoutingCriteria,
    RoutingDecision,
    TaskType,
)
from app.llm.transport import (
    HttpTransport,
    HttpResponse,
    UrllibHttpTransport,
)
from app.llm.unified_streamer import UnifiedStreamer

from app.llm.analytics import LLMAnalytics

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
    "register_llm_components",
    "LLMRouter",
    "CapabilityMatrix",
    "CircuitBreaker",
    "CostOptimizer",
    "FallbackManager",
    "ProviderHealthMonitor",
    "LoadBalancer",
    "PolicyEngine",
    "PolicyRule",
    "MaxCostPolicy",
    "HealthThresholdPolicy",
    "CapabilityRequirementPolicy",
    "ProviderAllowListPolicy",
    "ProviderBlockListPolicy",
    "PromptCache",
    "RequestQueue",
    "RequestPriority",
    "ProviderRateLimiter",
    "UnifiedStreamer",
    "LLMAnalytics",
    "RouterError",
    "RoutingError",
    "NoHealthyProvidersError",
    "FallbackExhaustedError",
    "CircuitBreakerError",
    "PolicyViolationError",
    "RateLimitExceededError",
    "QueueTimeoutError",
    "RouterStrategy",
    "TaskType",
    "Priority",
    "CircuitState",
    "LoadBalanceStrategy",
    "CapabilityFlag",
    "ProviderCapabilities",
    "RoutingCriteria",
    "RoutingDecision",
    "ProviderHealthSnapshot",
    "ProviderScore",
    "CostEstimate",
    "AnalyticsEvent",
    "ProviderRanking",
    "CacheKey",
    "RateLimitInfo",
]
