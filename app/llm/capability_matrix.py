"""Capability matrix for LLM providers.

Tracks which capabilities each provider advertises, enabling the router
to filter candidates before making a routing decision.
"""

from __future__ import annotations

import logging
import threading

from app.llm.base import BaseLLMProvider
from app.llm.router_models import CapabilityFlag, ProviderCapabilities
from app.llm.router_exceptions import RoutingError

log = logging.getLogger(__name__)


class CapabilityMatrix:
    """Thread-safe registry of provider capabilities.

    Attributes:
        _capabilities: Maps provider_id to :class:`ProviderCapabilities`.
        _lock: Protects all access to ``_capabilities``.
    """

    def __init__(self) -> None:
        self._capabilities: dict[str, ProviderCapabilities] = {}
        self._lock = threading.RLock()

    def register(self, provider_id: str, caps: ProviderCapabilities) -> None:
        """Register or update capabilities for a provider.

        Args:
            provider_id: Provider name.
            caps: Capabilities to store.
        """
        with self._lock:
            self._capabilities[provider_id] = caps
        log.debug("Registered capabilities for provider '%s'", provider_id)

    def unregister(self, provider_id: str) -> None:
        """Remove a provider's capabilities.

        Args:
            provider_id: Provider name.
        """
        with self._lock:
            self._capabilities.pop(provider_id, None)
        log.debug("Unregistered capabilities for provider '%s'", provider_id)

    def get(self, provider_id: str) -> ProviderCapabilities | None:
        """Return capabilities for *provider_id*, or ``None``."""
        with self._lock:
            return self._capabilities.get(provider_id)

    def has(self, provider_id: str) -> bool:
        """Return ``True`` if capabilities are registered for *provider_id*."""
        with self._lock:
            return provider_id in self._capabilities

    def all_provider_ids(self) -> list[str]:
        """Return all registered provider IDs."""
        with self._lock:
            return list(self._capabilities.keys())

    def get_all(self) -> dict[str, ProviderCapabilities]:
        """Return a shallow copy of all provider capabilities."""
        with self._lock:
            return dict(self._capabilities)

    def supports(self, provider_id: str, flag: CapabilityFlag) -> bool:
        """Return ``True`` if the provider has the given capability flag."""
        caps = self.get(provider_id)
        if caps is None:
            return False
        return getattr(caps, flag.value, False)

    def can_handle(self, provider_id: str, requires: set[CapabilityFlag]) -> bool:
        """Return ``True`` if the provider meets all *requires* capabilities.

        Args:
            provider_id: Provider name.
            requires: Set of capabilities that must be present.

        Returns:
            ``True`` if all required capabilities are available.

        Raises:
            RoutingError: If the provider is not registered.
        """
        with self._lock:
            caps = self._capabilities.get(provider_id)
            if caps is None:
                raise RoutingError(
                    f"Provider '{provider_id}' not registered in capability matrix",
                    module="llm.capability_matrix",
                    context={"provider": provider_id},
                )
            for flag in requires:
                if not getattr(caps, flag.value, False):
                    return False
            return True

    def satisfies_context(self, provider_id: str, min_context_length: int) -> bool:
        """Return ``True`` if the provider's context window meets *min_context_length*."""
        caps = self.get(provider_id)
        if caps is None:
            return False
        return caps.context_length >= min_context_length

    def find_compatible(
        self,
        requires: set[CapabilityFlag],
        min_context_length: int = 0,
    ) -> list[str]:
        """Return provider IDs that meet all capability and context requirements.

        Args:
            requires: Required capability flags.
            min_context_length: Minimum context window (0 = no filter).

        Returns:
            Sorted list of matching provider IDs.
        """
        with self._lock:
            results = []
            for pid, caps in self._capabilities.items():
                ok = True
                for flag in requires:
                    if not getattr(caps, flag.value, False):
                        ok = False
                        break
                if ok and min_context_length > 0:
                    if caps.context_length < min_context_length:
                        ok = False
                if ok:
                    results.append(pid)
            results.sort()
            return results

    @staticmethod
    def from_provider(
        provider: BaseLLMProvider,
    ) -> ProviderCapabilities:
        """Build a :class:`ProviderCapabilities` from a provider instance.

        Uses the provider's ``model_info`` and ``default_model`` to
        populate capability flags and context length.

        Args:
            provider: A :class:`BaseLLMProvider` instance.

        Returns:
            A :class:`ProviderCapabilities` describing the provider.
        """
        try:
            info = provider.model_info()
        except Exception:
            info = None

        caps = ProviderCapabilities(
            context_length=info.max_context_tokens if info else 4096,
            max_output_tokens=info.max_output_tokens if info else 1024,
        )

        if info:
            caps = caps.model_copy(
                update={
                    "streaming": info.supports_streaming,
                    "tool_calling": info.supports_tools,
                    "json_mode": info.supports_json,
                }
            )

        return caps
