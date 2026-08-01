"""Pydantic models for the Multi-LLM Intelligence Layer.

Defines all value objects used by the router, health monitor,
circuit breaker, load balancer, cost optimizer, analytics, etc.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RouterStrategy(str, Enum):
    """Strategies for selecting a provider from the candidate pool."""

    FASTEST = "fastest"
    CHEAPEST = "cheapest"
    BEST = "best"
    SMART = "smart"
    CUSTOM = "custom"


class TaskType(str, Enum):
    """High-level task categories used for routing decisions."""

    REASONING = "reasoning"
    CREATIVE = "creative"
    CONVERSATIONAL = "conversational"
    CODING = "coding"
    EMBEDDING = "embedding"


class Priority(str, Enum):
    """User preference when multiple valid providers exist."""

    SPEED = "speed"
    COST = "cost"
    QUALITY = "quality"


class CircuitState(str, Enum):
    """States for the circuit breaker finite-state machine."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class LoadBalanceStrategy(str, Enum):
    """Strategies for the load balancer."""

    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    LEAST_LATENCY = "least_latency"
    LEAST_ERROR_RATE = "least_error_rate"


class CapabilityFlag(str, Enum):
    """Individual capability flags a provider can advertise."""

    CHAT = "chat"
    VISION = "vision"
    AUDIO = "audio"
    JSON = "json"
    TOOL_CALLING = "tool_calling"
    STREAMING = "streaming"
    LONG_CONTEXT = "long_context"
    REASONING = "reasoning"
    EMBEDDINGS = "embeddings"


class ProviderCapabilities(BaseModel):
    """Capabilities advertised by a provider.

    Attributes:
        chat: Supports chat completion.
        vision: Supports image input.
        audio: Supports audio input/output.
        json_mode: Supports structured JSON output.
        tool_calling: Supports function/tool calling.
        streaming: Supports streaming responses.
        long_context: Supports context windows > 100k tokens.
        reasoning: Supports advanced reasoning (chain-of-thought).
        embeddings: Supports embedding generation.
        context_length: Maximum context window in tokens.
        max_output_tokens: Maximum output tokens per request.
    """

    model_config = ConfigDict(frozen=True)

    chat: bool = Field(default=False, description="Supports chat completion")
    vision: bool = Field(default=False, description="Supports image input")
    audio: bool = Field(default=False, description="Supports audio I/O")
    json_mode: bool = Field(default=False, description="Supports JSON output")
    tool_calling: bool = Field(default=False, description="Supports function calling")
    streaming: bool = Field(default=False, description="Supports streaming")
    long_context: bool = Field(default=False, description="Long context support")
    reasoning: bool = Field(default=False, description="Supports reasoning")
    embeddings: bool = Field(default=False, description="Supports embeddings")
    context_length: int = Field(
        default=4096, ge=1, description="Maximum context window in tokens"
    )
    max_output_tokens: int = Field(
        default=1024, ge=1, description="Maximum output tokens per request"
    )


class RoutingCriteria(BaseModel):
    """Criteria used to evaluate and rank providers.

    Attributes:
        task_type: Type of task (determines required capabilities).
        max_cost: Maximum estimated cost in USD (None = unbounded).
        max_latency: Maximum acceptable latency in seconds.
        requires_streaming: Whether streaming is required.
        requires_tool_calling: Whether tool calling is required.
        requires_vision: Whether vision/image input is required.
        requires_json_mode: Whether JSON output is required.
        min_context_length: Minimum required context window.
        preferred_providers: Ordered list of preferred provider names.
        priority: User priority when multiple providers qualify.
        strategy: Routing strategy to use.
        timeout_s: Overall timeout for the request.
    """

    model_config = ConfigDict(frozen=True)

    task_type: TaskType = Field(
        default=TaskType.CONVERSATIONAL, description="Category of task"
    )
    max_cost: float | None = Field(
        default=None, ge=0.0, description="Maximum estimated cost in USD"
    )
    max_latency: float | None = Field(
        default=None, gt=0, description="Maximum acceptable latency in seconds"
    )
    requires_streaming: bool = Field(
        default=False, description="Whether streaming is required"
    )
    requires_tool_calling: bool = Field(
        default=False, description="Whether tool calling is required"
    )
    requires_vision: bool = Field(
        default=False, description="Whether vision input is required"
    )
    requires_json_mode: bool = Field(
        default=False, description="Whether JSON output is required"
    )
    min_context_length: int = Field(
        default=4096, ge=1, description="Minimum required context window"
    )
    preferred_providers: list[str] = Field(
        default_factory=list, description="Preferred provider names in order"
    )
    priority: Priority = Field(
        default=Priority.QUALITY, description="User priority preference"
    )
    strategy: RouterStrategy = Field(
        default=RouterStrategy.SMART, description="Routing strategy"
    )
    timeout_s: float = Field(default=30.0, gt=0, description="Overall request timeout")

    @field_validator("max_cost", mode="before")
    @classmethod
    def _validate_max_cost(cls, v: Any) -> Any:
        if v is not None and v < 0:
            raise ValueError("max_cost must be non-negative")
        return v


class ProviderHealthSnapshot(BaseModel):
    """Immutable snapshot of a provider's health at a point in time.

    Attributes:
        provider_id: Provider name.
        is_healthy: Whether the provider is currently considered healthy.
        circuit_state: Current circuit breaker state.
        success_count: Total successful requests.
        failure_count: Total failed requests.
        success_rate: Ratio of successes to total requests.
        avg_latency: Average response latency in seconds.
        min_latency: Minimum observed latency.
        max_latency: Maximum observed latency.
        requests_per_minute: Estimated RPM.
        tokens_per_minute: Estimated TPM.
        uptime: Uptime percentage (0.0 - 1.0).
        last_success: Timestamp of last successful request.
        last_failure: Timestamp of last failed request.
        last_failure_error: Message from last failure, if any.
    """

    model_config = ConfigDict(frozen=True)

    provider_id: str = Field(..., description="Provider name")
    is_healthy: bool = Field(..., description="Current health status")
    circuit_state: CircuitState = Field(..., description="Circuit breaker state")
    success_count: int = Field(default=0, ge=0, description="Total successes")
    failure_count: int = Field(default=0, ge=0, description="Total failures")
    success_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Success ratio"
    )
    avg_latency: float = Field(
        default=0.0, ge=0.0, description="Average latency in seconds"
    )
    min_latency: float = Field(
        default=0.0, ge=0.0, description="Minimum latency in seconds"
    )
    max_latency: float = Field(
        default=0.0, ge=0.0, description="Maximum latency in seconds"
    )
    requests_per_minute: float = Field(
        default=0.0, ge=0.0, description="Estimated requests per minute"
    )
    tokens_per_minute: float = Field(
        default=0.0, ge=0.0, description="Estimated tokens per minute"
    )
    uptime: float = Field(default=1.0, ge=0.0, le=1.0, description="Uptime percentage")
    last_success: datetime | None = Field(
        default=None, description="Timestamp of last success"
    )
    last_failure: datetime | None = Field(
        default=None, description="Timestamp of last failure"
    )
    last_failure_error: str | None = Field(
        default=None, description="Error message from last failure"
    )


class ProviderScore(BaseModel):
    """A weighted score for a provider given specific criteria.

    Attributes:
        provider_id: Provider name.
        total_score: Normalized score (0.0 - 1.0).
        health_score: Component score from health (0.0 - 1.0).
        latency_score: Component score from latency (0.0 - 1.0).
        capability_score: Component score from capability match (0.0 - 1.0).
        cost_score: Component score from cost (0.0 - 1.0).
        circuit_penalty: Penalty applied due to circuit breaker state.
        weighted_score: Final weighted score used for ranking.
    """

    model_config = ConfigDict(frozen=True)

    provider_id: str = Field(..., description="Provider name")
    total_score: float = Field(..., ge=0.0, le=1.0, description="Normalized score")
    health_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Health component"
    )
    latency_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Latency component"
    )
    capability_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Capability component"
    )
    cost_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Cost component")
    circuit_penalty: float = Field(
        default=0.0, ge=0.0, description="Penalty from circuit breaker"
    )
    weighted_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Final weighted score"
    )


class RoutingDecision(BaseModel):
    """Result of a routing decision.

    Attributes:
        selected_provider: Name of the selected provider.
        strategy: Strategy used for this decision.
        scores: Scores for all evaluated providers.
        fallback_chain: Ordered list of fallback provider names.
        timestamp: When the decision was made.
        reason: Human-readable explanation of the decision.
    """

    model_config = ConfigDict(frozen=True)

    selected_provider: str = Field(..., description="Selected provider name")
    strategy: RouterStrategy = Field(..., description="Strategy used")
    scores: list[ProviderScore] = Field(
        default_factory=list, description="Scores for evaluated providers"
    )
    fallback_chain: list[str] = Field(
        default_factory=list, description="Fallback provider order"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Decision timestamp"
    )
    reason: str = Field(default="", description="Explanation of decision")


class CostEstimate(BaseModel):
    """Estimated cost for a request via a specific provider.

    Attributes:
        provider_id: Provider name.
        prompt_tokens: Estimated prompt token count.
        completion_tokens: Estimated completion token count.
        total_tokens: Total estimated tokens.
        cost_per_1k_prompt: Provider's cost per 1k prompt tokens.
        cost_per_1k_completion: Provider's cost per 1k completion tokens.
        estimated_cost: Total estimated cost in USD.
    """

    model_config = ConfigDict(frozen=True)

    provider_id: str = Field(..., description="Provider name")
    prompt_tokens: int = Field(default=0, ge=0, description="Estimated prompt tokens")
    completion_tokens: int = Field(
        default=0, ge=0, description="Estimated completion tokens"
    )
    total_tokens: int = Field(default=0, ge=0, description="Total estimated tokens")
    cost_per_1k_prompt: float = Field(
        default=0.0, ge=0.0, description="Cost per 1k prompt tokens (USD)"
    )
    cost_per_1k_completion: float = Field(
        default=0.0, ge=0.0, description="Cost per 1k completion tokens (USD)"
    )
    estimated_cost: float = Field(
        default=0.0, ge=0.0, description="Total estimated cost (USD)"
    )


class AnalyticsEvent(BaseModel):
    """Records a single LLM request event for analytics.

    Attributes:
        provider_id: Provider that handled the request.
        model: Model name used.
        prompt_tokens: Prompt token count.
        completion_tokens: Completion token count.
        total_tokens: Total token count.
        latency: Response latency in seconds.
        first_token_latency: Time to first token in seconds.
        success: Whether the request succeeded.
        error: Error message if failed.
        estimated_cost: Cost in USD.
        cached: Whether the response was served from cache.
        strategy: Routing strategy used.
        timestamp: When the event occurred.
        request_messages: Number of messages in the request.
    """

    model_config = ConfigDict(frozen=False)

    provider_id: str = Field(..., description="Provider name")
    model: str = Field(..., description="Model name used")
    prompt_tokens: int = Field(default=0, ge=0, description="Prompt token count")
    completion_tokens: int = Field(
        default=0, ge=0, description="Completion token count"
    )
    total_tokens: int = Field(default=0, ge=0, description="Total token count")
    latency: float = Field(default=0.0, ge=0.0, description="Response latency (s)")
    first_token_latency: float = Field(
        default=0.0, ge=0.0, description="Time to first token (s)"
    )
    success: bool = Field(..., description="Whether request succeeded")
    error: str | None = Field(default=None, description="Error message if failed")
    estimated_cost: float = Field(default=0.0, ge=0.0, description="Cost in USD")
    cached: bool = Field(default=False, description="Served from cache")
    strategy: RouterStrategy = Field(
        default=RouterStrategy.SMART, description="Routing strategy"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When event occurred"
    )
    request_messages: int = Field(
        default=0, ge=0, description="Number of messages in request"
    )


class ProviderRanking(BaseModel):
    """Aggregated ranking of a provider based on analytics.

    Attributes:
        provider_id: Provider name.
        total_requests: Total number of requests.
        success_rate: Success ratio (0.0 - 1.0).
        avg_latency: Average latency in seconds.
        total_tokens: Total tokens processed.
        total_cost: Total cost in USD.
        cache_hit_rate: Cache hit ratio (0.0 - 1.0).
        requests_per_minute: Estimated RPM.
        tokens_per_minute: Estimated TPM.
        uptime: Uptime percentage (0.0 - 1.0).
        last_used: Timestamp of last request.
    """

    model_config = ConfigDict(frozen=False)

    provider_id: str = Field(..., description="Provider name")
    total_requests: int = Field(default=0, ge=0, description="Total requests")
    success_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Success ratio"
    )
    avg_latency: float = Field(
        default=0.0, ge=0.0, description="Average latency in seconds"
    )
    total_tokens: int = Field(default=0, ge=0, description="Total tokens processed")
    total_cost: float = Field(default=0.0, ge=0.0, description="Total cost in USD")
    cache_hit_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Cache hit ratio"
    )
    requests_per_minute: float = Field(default=0.0, ge=0.0, description="Estimated RPM")
    tokens_per_minute: float = Field(default=0.0, ge=0.0, description="Estimated TPM")
    uptime: float = Field(default=1.0, ge=0.0, le=1.0, description="Uptime percentage")
    last_used: datetime | None = Field(
        default=None, description="Last request timestamp"
    )


class CacheKey(BaseModel):
    """Hash-based cache key for prompt caching.

    Attributes:
        hash: SHA-256 hash of the request content.
        provider_id: Provider that would handle this request.
        model: Model name.
        strategy: Routing strategy used.
    """

    model_config = ConfigDict(frozen=True)

    hash: str = Field(..., min_length=1, description="SHA-256 hash of request")
    provider_id: str = Field(..., description="Provider for this key")
    model: str = Field(..., description="Model name")
    strategy: RouterStrategy = Field(
        default=RouterStrategy.SMART, description="Routing strategy"
    )


class RateLimitInfo(BaseModel):
    """Rate limit information for a provider.

    Attributes:
        requests_per_minute: Request rate limit.
        tokens_per_minute: Token rate limit.
        current_rpm: Current requests in the minute window.
        current_tpm: Current tokens in the minute window.
        reset_at: When the rate limit window resets.
    """

    model_config = ConfigDict(frozen=False)

    requests_per_minute: int = Field(
        default=0, ge=0, description="Request rate limit per minute"
    )
    tokens_per_minute: int = Field(
        default=0, ge=0, description="Token rate limit per minute"
    )
    current_rpm: int = Field(
        default=0, ge=0, description="Current requests this minute"
    )
    current_tpm: int = Field(default=0, ge=0, description="Current tokens this minute")
    reset_at: datetime | None = Field(
        default=None, description="When the window resets"
    )
