"""Automatic failover between LLM providers.

Implements a configurable fallback chain: when the primary provider
fails, the manager tries the next provider in the chain until a
response is obtained or the chain is exhausted.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable, Optional

from app.llm.base import BaseLLMProvider
from app.llm.models import ChatRequest
from app.llm.router_exceptions import FallbackExhaustedError, RoutingError

log = logging.getLogger(__name__)


class FallbackManager:
    """Configurable provider fallback chain with automatic retry.

    Attributes:
        _chains: Maps primary provider_id → ordered list of fallback provider IDs.
        _provider_resolver: Callable mapping provider_id → :class:`BaseLLMProvider`.
        _max_fallback_attempts: Maximum providers to try per request.
        _lock: Protects ``_chains``.
    """

    def __init__(
        self,
        provider_resolver: Callable[[str], BaseLLMProvider | None] | None = None,
        max_fallback_attempts: int = 3,
    ) -> None:
        self._chains: dict[str, list[str]] = {}
        self._provider_resolver = provider_resolver
        self._max_fallback_attempts = max_fallback_attempts
        self._lock = threading.RLock()

    def set_fallback_chain(
        self,
        primary_provider: str,
        fallback_providers: list[str],
    ) -> None:
        """Configure a fallback chain for a primary provider.

        Args:
            primary_provider: Provider that is attempted first.
            fallback_providers: Ordered fallback providers (no duplicates
                of the primary).
        """
        with self._lock:
            chain = [p for p in fallback_providers if p != primary_provider]
            self._chains[primary_provider] = chain
        log.info(
            "Fallback chain for '%s': %s",
            primary_provider,
            chain or "(none)",
        )

    def get_fallback_chain(self, primary_provider: str) -> list[str]:
        """Return the fallback chain for *primary_provider*.

        Returns an empty list if no chain is configured.
        """
        with self._lock:
            return list(self._chains.get(primary_provider, []))

    def clear_chain(self, primary_provider: str) -> None:
        """Remove the fallback chain for *primary_provider*."""
        with self._lock:
            self._chains.pop(primary_provider, None)

    def clear_all_chains(self) -> None:
        """Remove all fallback chains."""
        with self._lock:
            self._chains.clear()

    def _resolve_provider(self, provider_id: str) -> BaseLLMProvider:
        """Resolve a provider instance by ID via the configured resolver."""
        if self._provider_resolver is None:
            raise RoutingError(
                "No provider resolver configured for fallback manager",
                module="llm.fallback_manager",
            )
        provider = self._provider_resolver(provider_id)
        if provider is None:
            raise RoutingError(
                f"Unable to resolve provider '{provider_id}' for fallback",
                module="llm.fallback_manager",
                context={"provider": provider_id},
            )
        return provider

    async def execute_with_fallback(
        self,
        request: ChatRequest,
        primary_provider: str,
        generate_func: Callable[[BaseLLMProvider, ChatRequest], Any],
        on_success: Callable[[str, BaseLLMProvider, Any], None] | None = None,
        on_failure: Callable[[str, Optional[BaseLLMProvider], Exception], None]
        | None = None,
    ) -> Any:
        """Execute *generate_func* against the fallback chain.

        Args:
            request: The chat request.
            primary_provider: Provider to try first.
            generate_func: Callable that invokes a provider and returns
                a response (e.g., ``provider.generate``).
            on_success: Optional callback after each successful attempt
                (provider_id, provider, result).
            on_failure: Optional callback after each failed attempt
                (provider_id, provider, exception).

        Returns:
            The first successful result.

        Raises:
            FallbackExhaustedError: If all providers in the chain fail.
        """
        chain = [primary_provider] + self.get_fallback_chain(primary_provider)
        chain = chain[: self._max_fallback_attempts]

        last_error: Exception | None = None

        for provider_id in chain:
            try:
                provider = self._resolve_provider(provider_id)
            except Exception as exc:
                last_error = exc
                if on_failure:
                    on_failure(provider_id, None, exc)
                log.warning(
                    "Fallback: could not resolve provider '%s': %s",
                    provider_id,
                    exc,
                )
                continue

            try:
                result = await asyncio.to_thread(generate_func, provider, request)
                if on_success:
                    on_success(provider_id, provider, result)
                log.info(
                    "Fallback: provider '%s' returned response",
                    provider_id,
                )
                return result
            except Exception as exc:
                last_error = exc
                if on_failure:
                    on_failure(provider_id, provider, exc)
                log.warning(
                    "Fallback: provider '%s' failed: %s",
                    provider_id,
                    exc,
                )

        raise FallbackExhaustedError(
            f"All {len(chain)} fallback providers failed for "
            f"'{primary_provider}': {last_error}",
            module="llm.fallback_manager",
            context={
                "primary": primary_provider,
                "chain": chain,
                "last_error": str(last_error) if last_error else None,
            },
            cause=last_error,
        )
