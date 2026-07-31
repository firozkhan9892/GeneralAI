"""Tests for the LLM provider exception hierarchy."""

from __future__ import annotations

import pytest

from app.core.exceptions.base import GeneralAIError
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

ALL_EXCEPTIONS = [
    LLMError,
    ProviderError,
    ProviderConfigurationError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderConnectionError,
    ProviderResponseError,
    ProviderStreamError,
    ProviderNotSupportedError,
]


class TestExceptionHierarchy:
    @pytest.mark.parametrize("exc_type", ALL_EXCEPTIONS)
    def test_derives_from_llm_error(self, exc_type: type) -> None:
        assert issubclass(exc_type, LLMError)

    @pytest.mark.parametrize("exc_type", ALL_EXCEPTIONS)
    def test_derives_from_general_ai_error(self, exc_type: type) -> None:
        assert issubclass(exc_type, GeneralAIError)

    @pytest.mark.parametrize("exc_type", ALL_EXCEPTIONS)
    def test_constructible_with_message(self, exc_type: type) -> None:
        exc = exc_type("problem", module="llm.test")
        assert exc.message == "problem"
        assert exc.module == "llm.test"

    def test_catchable_as_general_ai_error(self) -> None:
        with pytest.raises(GeneralAIError):
            raise ProviderRateLimitError("slow down")

    def test_cause_preserved(self) -> None:
        cause = ValueError("root")
        exc = ProviderConnectionError("failed", cause=cause)
        assert exc.cause is cause
