"""Cost optimization for LLM provider selection.

Estimates the cost of requests and helps choose the cheapest
provider that satisfies capability and latency requirements.
"""

from __future__ import annotations

import logging
import threading

from app.llm.models import ChatRequest
from app.llm.router_models import (
    CostEstimate,
    ProviderCapabilities,
    RoutingCriteria,
)
from app.llm.router_exceptions import RoutingError

log = logging.getLogger(__name__)


class CostOptimizer:
    """Selects the cheapest provider that meets a request's requirements.

    Attributes:
        _provider_costs: Maps provider_id to (prompt_cost_per_1k, completion_cost_per_1k).
        _lock: Protects ``_provider_costs``.
    """

    def __init__(self) -> None:
        self._provider_costs: dict[str, tuple[float, float]] = {}
        self._lock = threading.RLock()

    def register_provider_costs(
        self,
        provider_id: str,
        prompt_cost_per_1k: float,
        completion_cost_per_1k: float,
    ) -> None:
        """Register cost information for a provider.

        Args:
            provider_id: Provider name.
            prompt_cost_per_1k: Cost per 1000 prompt tokens (USD).
            completion_cost_per_1k: Cost per 1000 completion tokens (USD).
        """
        with self._lock:
            self._provider_costs[provider_id] = (
                max(prompt_cost_per_1k, 0.0),
                max(completion_cost_per_1k, 0.0),
            )
        log.debug(
            "Registered costs for '%s': prompt=$%s, completion=$%s",
            provider_id,
            prompt_cost_per_1k,
            completion_cost_per_1k,
        )

    def unregister_provider(self, provider_id: str) -> None:
        """Remove a provider's cost information."""
        with self._lock:
            self._provider_costs.pop(provider_id, None)

    def get_provider_cost_per_1k_tokens(self, provider_id: str) -> tuple[float, float]:
        """Return ``(prompt_cost_per_1k, completion_cost_per_1k)`` for a provider.

        Returns ``(0.0, 0.0)`` if the provider is not registered.
        """
        with self._lock:
            return self._provider_costs.get(provider_id, (0.0, 0.0))

    def estimate_tokens(self, request: ChatRequest) -> tuple[int, int]:
        """Estimate prompt and completion token counts for a request.

        Uses a simple heuristic: ~4 chars per token for prompt text,
        and ``max_tokens`` or a fraction of ``min_context_length``
        for completion tokens.

        Args:
            request: The chat request.

        Returns:
            ``(prompt_tokens, completion_tokens)``
        """
        prompt_chars = sum(len(msg.content or "") for msg in request.messages)
        for tool in request.tools or []:
            prompt_chars += len(tool.description or "") + len(tool.name)

        if request.tools:
            prompt_chars += sum(
                len(str(p)) for t in request.tools for p in t.parameters.values()
            )

        prompt_tokens = max(prompt_chars // 4, 1)

        max_output = request.max_tokens
        if max_output is None:
            max_output = 1024

        completion_tokens = min(max_output, 4096)

        return prompt_tokens, completion_tokens

    def estimate_cost(
        self,
        provider_id: str,
        request: ChatRequest,
    ) -> CostEstimate:
        """Estimate the cost of fulfilling *request* via *provider_id*.

        Args:
            provider_id: Provider name.
            request: The chat request.

        Returns:
            A :class:`CostEstimate` with breakdown.

        Raises:
            RoutingError: If the provider is not registered.
        """
        with self._lock:
            costs = self._provider_costs.get(provider_id)

        if costs is None:
            raise RoutingError(
                f"No cost information for provider '{provider_id}'",
                module="llm.cost_optimizer",
                context={"provider": provider_id},
            )

        prompt_cost_per_1k, completion_cost_per_1k = costs
        prompt_tokens, completion_tokens = self.estimate_tokens(request)
        total_tokens = prompt_tokens + completion_tokens

        estimated_cost = (prompt_tokens / 1000.0) * prompt_cost_per_1k + (
            completion_tokens / 1000.0
        ) * completion_cost_per_1k

        return CostEstimate(
            provider_id=provider_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_per_1k_prompt=prompt_cost_per_1k,
            cost_per_1k_completion=completion_cost_per_1k,
            estimated_cost=estimated_cost,
        )

    def select_cheapest(
        self,
        request: ChatRequest,
        candidates: list[str],
        criteria: RoutingCriteria,
        capabilities: dict[str, ProviderCapabilities] | None = None,
    ) -> str:
        """Select the cheapest provider that satisfies *criteria*.

        Args:
            request: The chat request.
            candidates: Provider IDs to consider.
            criteria: Routing criteria (cost, latency, capability filters).
            capabilities: Optional capability map for filtering.

        Returns:
            The selected provider ID.

        Raises:
            RoutingError: If no provider satisfies the criteria.
        """
        if not candidates:
            raise RoutingError(
                "No providers available for cost optimization",
                module="llm.cost_optimizer",
            )

        caps = capabilities or {}
        valid_providers = []

        for pid in candidates:
            if pid not in self._provider_costs:
                log.debug("Provider '%s' not cost-registered, skipping", pid)
                continue

            if not self._meets_capability_requirements(pid, criteria, caps):
                continue

            est = self.estimate_cost(pid, request)

            if criteria.max_cost is not None and est.estimated_cost > criteria.max_cost:
                continue

            valid_providers.append((pid, est))

        if not valid_providers:
            raise RoutingError(
                "No providers satisfy cost and capability requirements",
                module="llm.cost_optimizer",
                context={
                    "candidates": candidates,
                    "max_cost": criteria.max_cost,
                },
            )

        valid_providers.sort(key=lambda x: x[1].estimated_cost)
        best = valid_providers[0]
        log.info(
            "Cost optimizer selected '%s' (cost=$%.6f)",
            best[0],
            best[1].estimated_cost,
        )
        return best[0]

    def _meets_capability_requirements(
        self,
        provider_id: str,
        criteria: RoutingCriteria,
        capabilities: dict[str, ProviderCapabilities],
    ) -> bool:
        """Check if a provider meets the routing criteria's capabilities."""
        caps = capabilities.get(provider_id)
        if caps is None:
            return True

        if criteria.requires_streaming and not caps.streaming:
            return False
        if criteria.requires_tool_calling and not caps.tool_calling:
            return False
        if criteria.requires_vision and not caps.vision:
            return False
        if criteria.requires_json_mode and not caps.json_mode:
            return False
        if caps.context_length < criteria.min_context_length:
            return False

        return True

    def get_all_costs(self) -> dict[str, tuple[float, float]]:
        """Return a copy of all provider cost data."""
        with self._lock:
            return dict(self._provider_costs)
