"""LLM Router — the main orchestrator for the Multi-LLM Intelligence Layer.

Selects the best provider for each request based on a weighted scoring
system (health, latency, capability, cost), applies policies, respects
circuit breakers, queues, and caches, and coordinates fallback chains.
"""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator

from app.llm.analytics import LLMAnalytics
from app.llm.base import BaseLLMProvider
from app.llm.capability_matrix import CapabilityMatrix
from app.llm.circuit_breaker import CircuitBreaker
from app.llm.cost_optimizer import CostOptimizer
from app.llm.fallback_manager import FallbackManager
from app.llm.health_monitor import ProviderHealthMonitor
from app.llm.load_balancer import LoadBalancer
from app.llm.models import ChatRequest, ChatResponse, StreamChunk
from app.llm.policy_engine import PolicyEngine
from app.llm.prompt_cache import PromptCache
from app.llm.registry import ProviderRegistry
from app.llm.request_queue import RequestPriority, RequestQueue
from app.llm.router_exceptions import (
    NoHealthyProvidersError,
    RoutingError,
)
from app.llm.router_models import (
    CapabilityFlag,
    ProviderScore,
    RoutingCriteria,
    RoutingDecision,
    RouterStrategy,
)
from app.llm.unified_streamer import StreamHandler, UnifiedStreamer

log = logging.getLogger(__name__)


class LLMRouter:
    """Orchestrates provider selection, execution, and failover.

    The router is the single entry point for LLM requests in the
    system.  It applies routing criteria, weights, policies, circuit
    breakers, the request queue, and the prompt cache before
    delegating to a concrete provider.

    Args:
        registry: The provider registry.
        health_monitor: Health tracking for all providers.
        capability_matrix: Capability advertising per provider.
        cost_optimizer: Cost estimation and cheapest selection.
        load_balancer: Strategy-based provider selection.
        circuit_breakers: Map of provider_id → :class:`CircuitBreaker`.
        fallback_manager: Fallback chain orchestration.
        request_queue: Async queue with concurrency control.
        prompt_cache: Response caching.
        policy_engine: Policy rules evaluated before selection.
        analytics: Usage/cost/health tracking.
        unified_streamer: Streaming normalization.
    """

    def __init__(
        self,
        registry: ProviderRegistry,
        health_monitor: ProviderHealthMonitor,
        capability_matrix: CapabilityMatrix,
        cost_optimizer: CostOptimizer,
        load_balancer: LoadBalancer,
        circuit_breakers: dict[str, CircuitBreaker] | None = None,
        fallback_manager: FallbackManager | None = None,
        request_queue: RequestQueue | None = None,
        prompt_cache: PromptCache | None = None,
        policy_engine: PolicyEngine | None = None,
        analytics: LLMAnalytics | None = None,
        unified_streamer: UnifiedStreamer | None = None,
    ) -> None:
        self._registry = registry
        self._health = health_monitor
        self._capabilities = capability_matrix
        self._cost = cost_optimizer
        self._load_balancer = load_balancer
        self._circuit_breakers = circuit_breakers or {}
        self._fallback = fallback_manager or FallbackManager(
            provider_resolver=self._resolve_provider
        )
        self._queue = request_queue or RequestQueue()
        self._cache = prompt_cache or PromptCache()
        self._policies = policy_engine or PolicyEngine()
        self._analytics = analytics or LLMAnalytics()
        self._streamer = unified_streamer or UnifiedStreamer()

    # ------------------------------------------------------------------
    # Provider resolution
    # ------------------------------------------------------------------

    def _resolve_provider(self, provider_id: str) -> BaseLLMProvider | None:
        """Resolve a provider instance from the registry."""
        return self._registry.get(provider_id)

    def _require_provider(self, provider_id: str) -> BaseLLMProvider:
        """Resolve a provider or raise :class:`RoutingError`."""
        provider = self._registry.get(provider_id)
        if provider is None:
            raise RoutingError(
                f"Provider '{provider_id}' not registered",
                module="llm.llm_router",
                context={"provider": provider_id},
            )
        return provider

    def get_provider(self, provider_id: str) -> BaseLLMProvider | None:
        """Return a registered provider by ID, or ``None``."""
        return self._registry.get(provider_id)

    def register_provider(
        self,
        provider: BaseLLMProvider,
        capabilities: Any | None = None,
        prompt_cost_per_1k: float = 0.0,
        completion_cost_per_1k: float = 0.0,
        overwrite: bool = False,
    ) -> None:
        """Register a provider and its metadata with the router.

        Wires the provider into the registry, capability matrix, health
        monitor, circuit breakers, and cost optimizer.

        Args:
            provider: The provider instance to register.
            capabilities: Optional :class:`ProviderCapabilities`; derived
                from the provider if omitted.
            prompt_cost_per_1k: Cost per 1k prompt tokens.
            completion_cost_per_1k: Cost per 1k completion tokens.
            overwrite: Whether to replace an existing provider.
        """
        self._registry.register(provider, overwrite=overwrite)

        caps = capabilities or CapabilityMatrix.from_provider(provider)
        self._capabilities.register(provider.name, caps)
        self._health.record_success(provider.name, 0.0, 0)
        self._circuit_breakers.setdefault(provider.name, CircuitBreaker())
        self._cost.register_provider_costs(
            provider.name,
            prompt_cost_per_1k,
            completion_cost_per_1k,
        )
        log.info("Registered provider '%s' with router", provider.name)

    def unregister_provider(self, provider_id: str) -> None:
        """Remove a provider and all its metadata from the router."""
        self._registry.unregister(provider_id)
        self._capabilities.unregister(provider_id)
        self._health.reset_provider(provider_id)
        self._circuit_breakers.pop(provider_id, None)
        self._cost.unregister_provider(provider_id)
        self._fallback.clear_chain(provider_id)
        self._cache.invalidate_provider(provider_id)
        log.info("Unregistered provider '%s' from router", provider_id)

    def get_available_providers(self) -> list[str]:
        """Return all provider IDs currently in the registry."""
        return self._registry.names()

    # ------------------------------------------------------------------
    # Criteria derivation
    # ------------------------------------------------------------------

    @staticmethod
    def criteria_from_request(
        request: ChatRequest,
        override: RoutingCriteria | None = None,
    ) -> RoutingCriteria:
        """Derive :class:`RoutingCriteria` from a :class:`ChatRequest`.

        Args:
            request: The chat request.
            override: Optional criteria that take precedence over derived values.

        Returns:
            Criteria combining derived and override values.
        """
        base = RoutingCriteria(
            requires_streaming=request.stream,
            requires_tool_calling=bool(request.tools),
            requires_json_mode=(
                request.response_format is not None
                and request.response_format.type == "json"
            ),
            timeout_s=request.timeout_s,
        )
        if override is None:
            return base
        return override.model_copy(
            update={
                k: v
                for k, v in base.model_dump().items()
                if getattr(override, k) == getattr(RoutingCriteria(), k)
            }
        )

    # ------------------------------------------------------------------
    # Capability filtering
    # ------------------------------------------------------------------

    def _required_flags(self, criteria: RoutingCriteria) -> set[CapabilityFlag]:
        """Return the capability flags required by *criteria*."""
        flags: set[CapabilityFlag] = {CapabilityFlag.CHAT}
        if criteria.requires_streaming:
            flags.add(CapabilityFlag.STREAMING)
        if criteria.requires_tool_calling:
            flags.add(CapabilityFlag.TOOL_CALLING)
        if criteria.requires_vision:
            flags.add(CapabilityFlag.VISION)
        if criteria.requires_json_mode:
            flags.add(CapabilityFlag.JSON)
        return flags

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_providers(
        self,
        candidates: list[str],
        request: ChatRequest,
        criteria: RoutingCriteria,
        weights: dict[str, float] | None = None,
    ) -> list[ProviderScore]:
        """Score all candidate providers using weighted factors.

        Weights default to: health 0.3, latency 0.2, capability 0.3,
        cost 0.2.  Each component is normalized to 0.0-1.0.

        Args:
            candidates: Provider IDs to score.
            request: The chat request.
            criteria: Routing criteria.
            weights: Optional override for component weights.

        Returns:
            Sorted list of :class:`ProviderScore` (best first).
        """
        w = weights or {
            "health": 0.3,
            "latency": 0.2,
            "capability": 0.3,
            "cost": 0.2,
        }
        total_w = sum(max(v, 0.0) for v in w.values())
        if total_w <= 0:
            w = {"health": 0.3, "latency": 0.2, "capability": 0.3, "cost": 0.2}
            total_w = 1.0

        scores: list[ProviderScore] = []
        for pid in candidates:
            health = self._health.get_snapshot(pid)
            caps = self._capabilities.get(pid)
            try:
                cost_est = self._cost.estimate_cost(pid, request)
            except Exception:
                cost_est = None

            health_score = health.success_rate if health else 0.5
            health_score *= 1.0 if health and health.is_healthy else 0.4

            max_latency = criteria.max_latency or 60.0
            latency_score = (
                1.0 - min(health.avg_latency / max_latency, 1.0) if health else 0.5
            )

            capability_score = self._capability_score(caps, criteria)

            max_cost = criteria.max_cost or 0.05
            cost_score = (
                1.0 - min(cost_est.estimated_cost / max_cost, 1.0) if cost_est else 0.5
            )

            breaker = self._circuit_breakers.get(pid)
            circuit_penalty = 0.0
            if breaker is not None and breaker.state.value == "open":
                circuit_penalty = 0.5

            weighted = (
                w["health"] * health_score
                + w["latency"] * latency_score
                + w["capability"] * capability_score
                + w["cost"] * cost_score
            ) / total_w
            weighted -= circuit_penalty

            scores.append(
                ProviderScore(
                    provider_id=pid,
                    total_score=max(min(weighted, 1.0), 0.0),
                    health_score=health_score,
                    latency_score=latency_score,
                    capability_score=capability_score,
                    cost_score=cost_score,
                    circuit_penalty=circuit_penalty,
                    weighted_score=max(min(weighted, 1.0), 0.0),
                )
            )

        scores.sort(key=lambda s: s.weighted_score, reverse=True)
        return scores

    def _capability_score(
        self,
        caps: Any | None,
        criteria: RoutingCriteria,
    ) -> float:
        """Compute a capability match score (0.0-1.0).

        Starts at 1.0 and subtracts for each missing required flag or
        an insufficient context window.
        """
        if caps is None:
            return 0.5

        score = 1.0
        requirements: list[tuple[bool, str]] = [
            (criteria.requires_streaming, "streaming"),
            (criteria.requires_tool_calling, "tool_calling"),
            (criteria.requires_vision, "vision"),
            (criteria.requires_json_mode, "json_mode"),
        ]
        for required, flag in requirements:
            if required and not getattr(caps, flag, False):
                score -= 0.25
        if caps.context_length < criteria.min_context_length:
            score -= 0.5
        return max(score, 0.0)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _apply_strategy(
        self,
        scores: list[ProviderScore],
        candidates: list[str],
        criteria: RoutingCriteria,
        request: ChatRequest,
    ) -> str:
        """Apply the routing strategy to pick the best provider.

        Args:
            scores: Pre-computed provider scores (best first).
            candidates: All candidate provider IDs.
            criteria: Routing criteria.
            request: The chat request.

        Returns:
            The selected provider ID.
        """
        strategy = criteria.strategy

        if strategy == RouterStrategy.FASTEST:
            for s in scores:
                return s.provider_id
            return scores[0].provider_id

        if strategy == RouterStrategy.CHEAPEST:
            try:
                return self._cost.select_cheapest(
                    request,
                    candidates,
                    criteria,
                    self._capabilities.get_all(),
                )
            except RoutingError:
                pass

        if strategy == RouterStrategy.BEST:
            if scores:
                return scores[0].provider_id
            raise RoutingError("No providers available for BEST strategy")

        if strategy == RouterStrategy.CUSTOM:
            if scores:
                return scores[0].provider_id
            raise RoutingError("No providers available for CUSTOM strategy")

        if strategy == RouterStrategy.SMART:
            priority = criteria.priority.value
            if priority == "cost" and scores:
                return min(
                    scores,
                    key=lambda s: (
                        self._cost.estimate_cost(s.provider_id, request).estimated_cost
                    ),
                ).provider_id
            if scores:
                return scores[0].provider_id

        raise RoutingError(
            f"No providers available for strategy '{strategy.value}'",
            module="llm.llm_router",
            context={"strategy": strategy.value},
        )

    def select_provider(
        self,
        request: ChatRequest,
        criteria: RoutingCriteria | None = None,
    ) -> RoutingDecision:
        """Select the best provider for *request*.

        Args:
            request: The chat request.
            criteria: Optional routing criteria override.

        Returns:
            A :class:`RoutingDecision` with the selected provider.

        Raises:
            NoHealthyProvidersError: If no provider can serve the request.
        """
        criteria = criteria or self.criteria_from_request(request)

        registered = self._registry.names()
        if not registered:
            raise NoHealthyProvidersError(
                "No LLM providers are registered",
                module="llm.llm_router",
            )

        caps_map = self._capabilities.get_all()
        context_candidates = [
            pid
            for pid in registered
            if self._capabilities.satisfies_context(pid, criteria.min_context_length)
        ]
        if not context_candidates:
            raise NoHealthyProvidersError(
                f"No providers with context >= {criteria.min_context_length}",
                module="llm.llm_router",
                context={"min_context": criteria.min_context_length},
            )

        required_flags = self._required_flags(criteria)
        compatible = self._capabilities.find_compatible(
            required_flags, criteria.min_context_length
        )
        candidates = [pid for pid in compatible if pid in registered]
        if not candidates:
            candidates = context_candidates

        preferred = [p for p in criteria.preferred_providers if p in candidates]
        if preferred:
            candidates = preferred + [p for p in candidates if p not in preferred]

        health_map = {pid: self._health.get_snapshot(pid) for pid in candidates}

        healthy = [
            pid
            for pid in candidates
            if not self._circuit_breakers.get(pid, None)
            and self._health.is_healthy(pid)
        ]

        circuit_open = [
            pid
            for pid in candidates
            if self._circuit_breakers.get(pid) is not None
            and self._circuit_breakers[pid].state.value == "open"
        ]
        if circuit_open:
            healthy = [pid for pid in healthy if pid not in circuit_open]

        if not healthy:
            healthy = [pid for pid in candidates if self._health.is_healthy(pid)]
        if not healthy:
            healthy = candidates

        allowed = self._policies.filter_providers(
            healthy,
            caps_map,
            health_map,
            {
                "criteria": criteria,
                "max_cost": criteria.max_cost,
            },
        )
        if not allowed:
            raise NoHealthyProvidersError(
                "All providers were filtered by policy",
                module="llm.llm_router",
            )

        scores = self._score_providers(allowed, request, criteria)
        selected = self._apply_strategy(scores, allowed, criteria, request)

        fallback_chain = [selected] + self._fallback.get_fallback_chain(selected)
        fallback_chain = [p for p in fallback_chain if p in allowed]

        decision = RoutingDecision(
            selected_provider=selected,
            strategy=criteria.strategy,
            scores=scores,
            fallback_chain=fallback_chain,
            reason=f"Strategy '{criteria.strategy.value}' selected '{selected}'",
        )
        log.info(
            "Routed request to '%s' (strategy=%s)",
            selected,
            criteria.strategy.value,
        )
        return decision

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _check_circuit(self, provider_id: str) -> None:
        """Raise :class:`RoutingError` if the provider's circuit is open."""
        breaker = self._circuit_breakers.get(provider_id)
        if breaker is not None:
            breaker.check(provider_id)

    def _apply_policies(
        self,
        provider_id: str,
        criteria: RoutingCriteria,
    ) -> None:
        """Apply policy rules for a provider."""
        self._policies.evaluate(
            provider_id,
            self._capabilities.get(provider_id),
            self._health.get_snapshot(provider_id),
            {"criteria": criteria},
        )

    def generate(
        self,
        request: ChatRequest,
        criteria: RoutingCriteria | None = None,
        provider_id: str | None = None,
    ) -> ChatResponse:
        """Generate a complete response, routing through the best provider.

        Args:
            request: The chat request.
            criteria: Optional routing criteria.
            provider_id: Optional explicit provider to use (bypasses routing).

        Returns:
            The provider's response.

        Raises:
            RoutingError: If no provider is available or all fail.
            FallbackExhaustedError: If all fallback providers fail.
        """
        criteria = criteria or self.criteria_from_request(request)

        if provider_id is not None:
            provider = self._resolve_provider(provider_id)
            if provider is None:
                raise RoutingError(
                    f"Provider '{provider_id}' not registered",
                    module="llm.llm_router",
                    context={"provider": provider_id},
                )
            return self._execute_with_tracking(provider_id, provider, request, criteria)

        decision = self.select_provider(request, criteria)
        selected = decision.selected_provider

        return self._execute_with_tracking(
            selected,
            self._require_provider(selected),
            request,
            criteria,
        )

    def _execute_with_tracking(
        self,
        provider_id: str,
        provider: BaseLLMProvider,
        request: ChatRequest,
        criteria: RoutingCriteria,
    ) -> ChatResponse:
        """Execute a provider call, tracking analytics, circuit, and cache."""
        cache_key = self._cache.build_key(request, provider_id)
        cached = self._cache.get(cache_key)
        if cached is not None and not request.stream:
            self._record_event(
                provider_id,
                request,
                cached,
                success=True,
                latency=0.0,
                cached=True,
                criteria=criteria,
            )
            log.info("Served response for '%s' from cache", provider_id)
            return cached

        self._check_circuit(provider_id)
        self._apply_policies(provider_id, criteria)

        start = time.monotonic()
        try:
            response = provider.generate(request)
            latency = time.monotonic() - start
            self._health.record_success(
                provider_id, latency, response.usage.total_tokens
            )
            self._on_success(provider_id)
            self._record_event(
                provider_id,
                request,
                response,
                success=True,
                latency=latency,
                criteria=criteria,
            )
            if not request.stream:
                self._cache.put(cache_key, response)
            return response
        except Exception as exc:
            latency = time.monotonic() - start
            self._health.record_failure(provider_id, latency, exc)
            self._on_failure(provider_id, exc)
            self._record_event(
                provider_id,
                request,
                None,
                success=False,
                latency=latency,
                error=str(exc),
                criteria=criteria,
            )
            raise

    def _record_event(
        self,
        provider_id: str,
        request: ChatRequest,
        response: ChatResponse | None,
        success: bool,
        latency: float,
        cached: bool = False,
        error: str | None = None,
        criteria: RoutingCriteria | None = None,
    ) -> None:
        """Record an analytics event for a request."""
        tokens = response.usage if response else None
        try:
            estimated_cost = (
                self._cost.estimate_cost(provider_id, request).estimated_cost
                if success
                else 0.0
            )
        except Exception:
            estimated_cost = 0.0
        self._analytics.record(
            provider_id=provider_id,
            model=request.model,
            prompt_tokens=tokens.prompt_tokens if tokens else 0,
            completion_tokens=tokens.completion_tokens if tokens else 0,
            total_tokens=tokens.total_tokens if tokens else 0,
            latency=latency,
            first_token_latency=latency,
            success=success,
            error=error,
            estimated_cost=estimated_cost,
            cached=cached,
            strategy=criteria.strategy if criteria else RouterStrategy.SMART,
            request_messages=len(request.messages),
        )

    def _on_success(self, provider_id: str) -> None:
        """Notify the circuit breaker of a success."""
        breaker = self._circuit_breakers.get(provider_id)
        if breaker is not None:
            breaker.on_success()

    def _on_failure(self, provider_id: str, exc: Exception) -> None:
        """Notify the circuit breaker of a failure."""
        breaker = self._circuit_breakers.get(provider_id)
        if breaker is not None:
            breaker.on_failure(exc)

    async def generate_async(
        self,
        request: ChatRequest,
        criteria: RoutingCriteria | None = None,
        provider_id: str | None = None,
    ) -> ChatResponse:
        """Generate a complete response asynchronously.

        Queues the request and applies concurrency/rate-limit controls.
        """
        criteria = criteria or self.criteria_from_request(request)

        if provider_id is not None:
            provider = self._resolve_provider(provider_id)
            if provider is None:
                raise RoutingError(
                    f"Provider '{provider_id}' not registered",
                    module="llm.llm_router",
                    context={"provider": provider_id},
                )
            return await self._execute_async(provider_id, provider, request, criteria)

        decision = self.select_provider(request, criteria)
        selected = decision.selected_provider
        provider = self._require_provider(selected)
        return await self._execute_async(selected, provider, request, criteria)

    async def _execute_async(
        self,
        provider_id: str,
        provider: BaseLLMProvider,
        request: ChatRequest,
        criteria: RoutingCriteria,
    ) -> ChatResponse:
        """Execute an async provider call with queue + tracking."""
        cache_key = self._cache.build_key(request, provider_id)
        cached = self._cache.get(cache_key)
        if cached is not None and not request.stream:
            self._record_event(
                provider_id,
                request,
                cached,
                success=True,
                latency=0.0,
                cached=True,
                criteria=criteria,
            )
            return cached

        self._check_circuit(provider_id)
        self._apply_policies(provider_id, criteria)

        tokens = self._cost.estimate_tokens(request)

        async def _call() -> ChatResponse:
            start = time.monotonic()
            try:
                response = await provider.generate_async(request)
                latency = time.monotonic() - start
                self._health.record_success(
                    provider_id, latency, response.usage.total_tokens
                )
                self._on_success(provider_id)
                self._record_event(
                    provider_id,
                    request,
                    response,
                    success=True,
                    latency=latency,
                    criteria=criteria,
                )
                if not request.stream:
                    self._cache.put(cache_key, response)
                return response
            except Exception as exc:
                latency = time.monotonic() - start
                self._health.record_failure(provider_id, latency, exc)
                self._on_failure(provider_id, exc)
                self._record_event(
                    provider_id,
                    request,
                    None,
                    success=False,
                    latency=latency,
                    error=str(exc),
                    criteria=criteria,
                )
                raise

        return await self._queue.submit(
            provider_id,
            _call(),
            priority=RequestPriority.NORMAL,
            tokens=tokens[0] + tokens[1],
        )

    async def stream(
        self,
        request: ChatRequest,
        criteria: RoutingCriteria | None = None,
        provider_id: str | None = None,
        handler: StreamHandler | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response through the unified streaming layer.

        Args:
            request: The chat request with ``stream`` enabled.
            criteria: Optional routing criteria.
            provider_id: Optional explicit provider.
            handler: Optional per-chunk callback.

        Yields:
            Normalized :class:`StreamChunk` objects.

        Raises:
            RoutingError: If streaming is not possible.
        """
        criteria = criteria or self.criteria_from_request(request)

        if provider_id is not None:
            provider = self._resolve_provider(provider_id)
            if provider is None:
                raise RoutingError(
                    f"Provider '{provider_id}' not registered",
                    module="llm.llm_router",
                    context={"provider": provider_id},
                )
            async for chunk in self._stream_from(provider, request, criteria, handler):
                yield chunk
            return

        decision = self.select_provider(request, criteria)
        provider = self._require_provider(decision.selected_provider)
        async for chunk in self._stream_from(provider, request, criteria, handler):
            yield chunk

    async def _stream_from(
        self,
        provider: BaseLLMProvider,
        request: ChatRequest,
        criteria: RoutingCriteria,
        handler: StreamHandler | None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream from a specific provider with tracking."""
        self._check_circuit(provider.name)
        self._apply_policies(provider.name, criteria)

        start = time.monotonic()
        first_token_latency = 0.0
        first_token = True
        try:
            async for chunk in self._streamer.normalize(provider, request):
                if first_token:
                    first_token_latency = time.monotonic() - start
                    first_token = False
                if handler is not None:
                    handler(chunk)
                yield chunk

            total_latency = time.monotonic() - start
            self._health.record_success(provider.name, total_latency, 0)
            self._on_success(provider.name)
            self._analytics.record(
                provider_id=provider.name,
                model=request.model,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency=total_latency,
                first_token_latency=first_token_latency,
                success=True,
                strategy=criteria.strategy,
                request_messages=len(request.messages),
            )
        except Exception as exc:
            total_latency = time.monotonic() - start
            self._health.record_failure(provider.name, total_latency, exc)
            self._on_failure(provider.name, exc)
            self._analytics.record(
                provider_id=provider.name,
                model=request.model,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency=total_latency,
                first_token_latency=first_token_latency,
                success=False,
                error=str(exc),
                strategy=criteria.strategy,
                request_messages=len(request.messages),
            )
            raise

    async def stream_to_handler(
        self,
        request: ChatRequest,
        handler: StreamHandler,
        criteria: RoutingCriteria | None = None,
        provider_id: str | None = None,
    ) -> ChatResponse | None:
        """Stream a response and pass each chunk to *handler*.

        Returns the accumulated :class:`ChatResponse`.
        """
        provider_id = (
            provider_id or self.select_provider(request, criteria).selected_provider
        )
        provider = self._require_provider(provider_id)
        return await self._streamer.stream_to_handler(provider, request, handler)

    # ------------------------------------------------------------------
    # Analytics / config helpers
    # ------------------------------------------------------------------

    @property
    def analytics(self) -> LLMAnalytics:
        """Return the analytics collector."""
        return self._analytics

    @property
    def health(self) -> ProviderHealthMonitor:
        """Return the health monitor."""
        return self._health

    @property
    def capabilities(self) -> CapabilityMatrix:
        """Return the capability matrix."""
        return self._capabilities

    @property
    def cache(self) -> PromptCache:
        """Return the prompt cache."""
        return self._cache

    @property
    def queue(self) -> RequestQueue:
        """Return the request queue."""
        return self._queue

    @property
    def cost_optimizer(self) -> CostOptimizer:
        """Return the cost optimizer."""
        return self._cost

    @property
    def load_balancer(self) -> LoadBalancer:
        """Return the load balancer."""
        return self._load_balancer

    @property
    def policies(self) -> PolicyEngine:
        """Return the policy engine."""
        return self._policies

    @property
    def fallback(self) -> FallbackManager:
        """Return the fallback manager."""
        return self._fallback
