"""Exception hierarchy for the Multi-LLM Intelligence Layer.

All router-specific exceptions derive from :class:`RouterError` which in turn
derives from :class:`LLMError` — the same base used for provider-level errors —
so callers can catch router failures uniformly.
"""

from __future__ import annotations

from app.llm.exceptions import LLMError


class RouterError(LLMError):
    """Base exception for the Multi-LLM Intelligence Layer."""


class RoutingError(RouterError):
    """Raised when routing logic fails to select a provider."""


class NoHealthyProvidersError(RoutingError):
    """Raised when no providers are healthy or available."""


class FallbackExhaustedError(RoutingError):
    """Raised when all fallback providers have been exhausted."""


class CircuitBreakerError(RouterError):
    """Raised when a circuit breaker is open and requests are blocked."""


class PolicyViolationError(RouterError):
    """Raised when a policy engine rule prevents a request."""


class RateLimitExceededError(RouterError):
    """Raised when the request queue rejects a job due to rate limits."""


class QueueTimeoutError(RouterError):
    """Raised when a job times out waiting in the request queue."""
