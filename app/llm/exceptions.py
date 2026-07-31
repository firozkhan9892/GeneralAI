"""LLM provider exception hierarchy.

All provider exceptions derive from :class:`LLMError` which in turn
derives from the platform-wide :class:`GeneralAIError`, so the rest of
the application can catch and report provider failures uniformly.
"""

from __future__ import annotations

from app.core.exceptions.base import GeneralAIError


class LLMError(GeneralAIError):
    """Base exception for all LLM provider layer errors."""


class ProviderError(LLMError):
    """Base exception for provider-level failures."""


class ProviderConfigurationError(ProviderError):
    """Raised when a provider is misconfigured or missing credentials."""


class ProviderAuthenticationError(ProviderError):
    """Raised when a provider rejects the supplied credentials."""


class ProviderRateLimitError(ProviderError):
    """Raised when a provider throttles the request."""


class ProviderTimeoutError(ProviderError):
    """Raised when a provider call exceeds its timeout budget."""


class ProviderConnectionError(ProviderError):
    """Raised when a provider endpoint cannot be reached."""


class ProviderResponseError(ProviderError):
    """Raised when a provider returns an unexpected or malformed response."""


class ProviderStreamError(ProviderError):
    """Raised when a provider stream fails mid-generation."""


class ProviderNotSupportedError(ProviderError):
    """Raised when a requested operation is not supported by a provider."""
