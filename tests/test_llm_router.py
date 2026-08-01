"""Tests for the Multi-LLM Intelligence Layer (Phase 11)."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Iterator

import pytest

from app.llm.analytics import LLMAnalytics
from app.llm.base import BaseLLMProvider
from app.llm.capability_matrix import CapabilityMatrix
from app.llm.circuit_breaker import CircuitBreaker
from app.llm.cost_optimizer import CostOptimizer
from app.llm.fallback_manager import FallbackManager
from app.llm.health_monitor import ProviderHealthMonitor
from app.llm.llm_router import LLMRouter
from app.llm.load_balancer import LoadBalancer
from app.llm.models import (
    ChatRequest,
    ChatResponse,
    Message,
    ModelInfo,
    Role,
    StreamChunk,
    Usage,
)
from app.llm.policy_engine import (
    MaxCostPolicy,
    PolicyEngine,
    ProviderBlockListPolicy,
)
from app.llm.prompt_cache import PromptCache
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
    RateLimitExceededError,
    RoutingError,
)
from app.llm.router_models import (
    CapabilityFlag,
    CircuitState,
    CostEstimate,
    LoadBalanceStrategy,
    ProviderCapabilities,
    ProviderHealthSnapshot,
    ProviderScore,
    ProviderRanking,
    RouterStrategy,
    RoutingCriteria,
)
from app.llm.unified_streamer import UnifiedStreamer


# ---------------------------------------------------------------------------
# Test double provider
# ---------------------------------------------------------------------------


class _TestProvider(BaseLLMProvider):
    """A configurable test provider."""

    name: str

    def __init__(
        self,
        name: str,
        *,
        model: str = "test-model",
        supports_streaming: bool = True,
        supports_tools: bool = True,
        supports_json: bool = True,
        context_length: int = 8192,
        fail: bool = False,
        delay_s: float = 0.0,
        content: str = "test response",
    ) -> None:
        self.name = name
        self._model = model
        self._supports_streaming = supports_streaming
        self._supports_tools = supports_tools
        self._supports_json = supports_json
        self._context_length = context_length
        self._fail = fail
        self._delay_s = delay_s
        self._content = content

    @property
    def default_model(self) -> str:
        return self._model

    def model_info(self, model: str | None = None) -> ModelInfo:
        return ModelInfo(
            name=model or self._model,
            provider=self.name,
            supports_streaming=self._supports_streaming,
            supports_tools=self._supports_tools,
            supports_json=self._supports_json,
            max_context_tokens=self._context_length,
            max_output_tokens=1024,
            cost_per_1k_tokens=0.001,
        )

    def generate(self, request: ChatRequest) -> ChatResponse:
        if self._delay_s > 0:
            time.sleep(self._delay_s)
        if self._fail:
            raise RuntimeError(f"{self.name} failure")
        return ChatResponse(
            content=self._content,
            finish_reason="stop",
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
            model=self._get_model(request, self._model),
            provider=self.name,
        )

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        if self._fail:
            raise RuntimeError(f"{self.name} failure")
        yield StreamChunk(
            content="hello",
            model=self._get_model(request, self._model),
            provider=self.name,
        )
        yield StreamChunk(
            content=" world",
            finish_reason="stop",
            model=self._get_model(request, self._model),
            provider=self.name,
        )


def _make_request(content: str = "hello", **kwargs: Any) -> ChatRequest:
    return ChatRequest(
        messages=(Message(role=Role.USER, content=content),),
        model="test-model",
        **kwargs,
    )


def _build_router(
    providers: list[_TestProvider],
) -> LLMRouter:
    registry = ProviderRegistry()
    health = ProviderHealthMonitor()
    matrix = CapabilityMatrix()
    cost = CostOptimizer()
    lb = LoadBalancer()
    for p in providers:
        registry.register(p)
        matrix.register(
            p.name,
            ProviderCapabilities(
                chat=True,
                streaming=p._supports_streaming,
                tool_calling=p._supports_tools,
                json_mode=p._supports_json,
                context_length=p._context_length,
            ),
        )
        health.record_success(p.name, 0.01, 15)
        cost.register_provider_costs(p.name, 0.001, 0.002)
    router = LLMRouter(
        registry=registry,
        health_monitor=health,
        capability_matrix=matrix,
        cost_optimizer=cost,
        load_balancer=lb,
    )
    for p in providers:
        router._circuit_breakers[p.name] = CircuitBreaker()
    return router


# ---------------------------------------------------------------------------
# 1. Models
# ---------------------------------------------------------------------------


class TestRouterModels:
    def test_router_strategy_enum(self) -> None:
        assert RouterStrategy.FASTEST.value == "fastest"
        assert RouterStrategy.CHEAPEST.value == "cheapest"
        assert RouterStrategy.BEST.value == "best"
        assert RouterStrategy.SMART.value == "smart"
        assert RouterStrategy.CUSTOM.value == "custom"

    def test_provider_capabilities_defaults(self) -> None:
        caps = ProviderCapabilities()
        assert caps.chat is False
        assert caps.context_length == 4096
        assert caps.max_output_tokens == 1024

    def test_routing_criteria_defaults(self) -> None:
        criteria = RoutingCriteria()
        assert criteria.strategy == RouterStrategy.SMART
        assert criteria.priority.value == "quality"
        assert criteria.min_context_length == 4096
        assert criteria.requires_streaming is False

    def test_routing_criteria_max_cost_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            RoutingCriteria(max_cost=-1.0)

    def test_provider_health_snapshot_fields(self) -> None:
        snap = ProviderHealthSnapshot(
            provider_id="test",
            is_healthy=True,
            circuit_state=CircuitState.CLOSED,
            success_rate=0.9,
            avg_latency=0.1,
        )
        assert snap.provider_id == "test"
        assert snap.success_rate == 0.9

    def test_provider_score(self) -> None:
        score = ProviderScore(
            provider_id="p",
            total_score=0.8,
            health_score=0.9,
            latency_score=0.7,
            capability_score=0.8,
            cost_score=0.6,
            weighted_score=0.75,
        )
        assert score.provider_id == "p"
        assert score.weighted_score == 0.75

    def test_cost_estimate(self) -> None:
        est = CostEstimate(
            provider_id="p",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_per_1k_prompt=0.001,
            cost_per_1k_completion=0.002,
            estimated_cost=0.0002,
        )
        assert est.total_tokens == 150

    def test_analytics_event_immutability_of_key_fields(self) -> None:
        from app.llm.router_models import AnalyticsEvent

        event = AnalyticsEvent(provider_id="p", model="m", success=True)
        assert event.provider_id == "p"
        assert event.cached is False


# ---------------------------------------------------------------------------
# 2. CapabilityMatrix
# ---------------------------------------------------------------------------


class TestCapabilityMatrix:
    def test_register_and_get(self) -> None:
        matrix = CapabilityMatrix()
        caps = ProviderCapabilities(chat=True, vision=True)
        matrix.register("p1", caps)
        assert matrix.get("p1") == caps
        assert matrix.has("p1")
        assert not matrix.has("p2")

    def test_unregister(self) -> None:
        matrix = CapabilityMatrix()
        matrix.register("p1", ProviderCapabilities(chat=True))
        matrix.unregister("p1")
        assert not matrix.has("p1")

    def test_supports_flag(self) -> None:
        matrix = CapabilityMatrix()
        matrix.register(
            "p1",
            ProviderCapabilities(chat=True, vision=True, tool_calling=True),
        )
        assert matrix.supports("p1", CapabilityFlag.CHAT)
        assert matrix.supports("p1", CapabilityFlag.VISION)
        assert not matrix.supports("p1", CapabilityFlag.AUDIO)
        assert not matrix.supports("unknown", CapabilityFlag.CHAT)

    def test_can_handle_all_required(self) -> None:
        matrix = CapabilityMatrix()
        matrix.register(
            "p1",
            ProviderCapabilities(chat=True, vision=True, streaming=True),
        )
        assert matrix.can_handle("p1", {CapabilityFlag.CHAT, CapabilityFlag.VISION})
        assert not matrix.can_handle("p1", {CapabilityFlag.CHAT, CapabilityFlag.AUDIO})

    def test_can_handle_unknown_provider_raises(self) -> None:
        matrix = CapabilityMatrix()
        with pytest.raises(RoutingError):
            matrix.can_handle("ghost", {CapabilityFlag.CHAT})

    def test_satisfies_context(self) -> None:
        matrix = CapabilityMatrix()
        matrix.register("p1", ProviderCapabilities(chat=True, context_length=100000))
        assert matrix.satisfies_context("p1", 100000)
        assert not matrix.satisfies_context("p1", 200000)
        assert not matrix.satisfies_context("ghost", 4096)

    def test_find_compatible(self) -> None:
        matrix = CapabilityMatrix()
        matrix.register(
            "p1", ProviderCapabilities(chat=True, vision=True, context_length=8000)
        )
        matrix.register("p2", ProviderCapabilities(chat=True, context_length=8000))
        matrix.register("p3", ProviderCapabilities(chat=False, context_length=8000))
        result = matrix.find_compatible({CapabilityFlag.CHAT})
        assert result == ["p1", "p2"]
        result2 = matrix.find_compatible({CapabilityFlag.CHAT, CapabilityFlag.VISION})
        assert result2 == ["p1"]

    def test_from_provider(self) -> None:
        provider = _TestProvider("t")
        caps = CapabilityMatrix.from_provider(provider)
        assert caps.context_length == 8192
        assert caps.chat is False  # derived only from ModelInfo flags


# ---------------------------------------------------------------------------
# 3. ProviderHealthMonitor
# ---------------------------------------------------------------------------


class TestProviderHealthMonitor:
    def test_record_success(self) -> None:
        monitor = ProviderHealthMonitor()
        monitor.record_success("p1", 0.1, 15)
        snap = monitor.get_snapshot("p1")
        assert snap.success_count == 1
        assert snap.failure_count == 0
        assert snap.success_rate == 1.0
        assert snap.avg_latency == 0.1
        assert snap.is_healthy

    def test_record_failure(self) -> None:
        monitor = ProviderHealthMonitor()
        monitor.record_failure("p1", 0.1, "boom")
        snap = monitor.get_snapshot("p1")
        assert snap.success_count == 0
        assert snap.failure_count == 1
        assert snap.success_rate == 0.0
        assert snap.last_failure_error == "boom"
        assert not snap.is_healthy

    def test_get_snapshot_unknown_provider(self) -> None:
        monitor = ProviderHealthMonitor()
        snap = monitor.get_snapshot("ghost")
        assert snap.provider_id == "ghost"
        assert snap.success_count == 0

    def test_avg_latency(self) -> None:
        monitor = ProviderHealthMonitor()
        monitor.record_success("p1", 0.1, 10)
        monitor.record_success("p1", 0.3, 20)
        snap = monitor.get_snapshot("p1")
        assert snap.avg_latency == 0.2
        assert snap.min_latency == 0.1
        assert snap.max_latency == 0.3

    def test_requests_per_minute(self) -> None:
        monitor = ProviderHealthMonitor(rpm_window_seconds=60.0)
        for _ in range(10):
            monitor.record_success("p1", 0.01, 15)
        snap = monitor.get_snapshot("p1")
        assert snap.requests_per_minute == 10.0

    def test_healthy_providers_list(self) -> None:
        monitor = ProviderHealthMonitor()
        monitor.record_success("p1", 0.1, 15)
        monitor.record_failure("p2", 0.1, "x")
        healthy = monitor.get_healthy_providers()
        assert "p1" in healthy
        assert "p2" not in healthy

    def test_reset_provider(self) -> None:
        monitor = ProviderHealthMonitor()
        monitor.record_success("p1", 0.1, 15)
        monitor.reset_provider("p1")
        snap = monitor.get_snapshot("p1")
        assert snap.success_count == 0

    def test_circuit_open_sets_health(self) -> None:
        monitor = ProviderHealthMonitor()
        monitor.record_success("p1", 0.1, 15)
        monitor.record_failure(
            "p1", 0.1, "boom", circuit_open_until=time.monotonic() + 60
        )
        snap = monitor.get_snapshot("p1")
        assert snap.circuit_state == CircuitState.OPEN
        assert not snap.is_healthy


# ---------------------------------------------------------------------------
# 4. CircuitBreaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_closed_initial(self) -> None:
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request()

    def test_opens_after_failure_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, timeout=60.0)
        for _ in range(3):
            cb.on_failure(RuntimeError("x"))
        assert cb.state == CircuitState.OPEN
        assert not cb.allow_request()

    def test_does_not_open_before_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=5, timeout=60.0)
        for _ in range(4):
            cb.on_failure(RuntimeError("x"))
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request()

    def test_half_open_transition(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, timeout=0.01, recovery_threshold=2)
        cb.on_failure(RuntimeError("x"))
        cb.on_failure(RuntimeError("x"))
        assert cb.state == CircuitState.OPEN
        time.sleep(0.05)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow_request()

    def test_recovery_closes_circuit(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, timeout=0.01, recovery_threshold=1)
        cb.on_failure(RuntimeError("x"))
        cb.on_failure(RuntimeError("x"))
        time.sleep(0.05)
        assert cb.allow_request()
        cb.on_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_opens_again(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, timeout=0.01, recovery_threshold=2)
        cb.on_failure(RuntimeError("x"))
        cb.on_failure(RuntimeError("x"))
        time.sleep(0.05)
        assert cb.state == CircuitState.HALF_OPEN
        cb.allow_request()
        cb.on_failure(RuntimeError("y"))
        assert cb.state == CircuitState.OPEN

    def test_check_raises_when_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, timeout=60.0)
        cb.on_failure(RuntimeError("x"))
        with pytest.raises(CircuitBreakerError):
            cb.check("provider")

    def test_expected_exception_filter(self) -> None:
        cb = CircuitBreaker(
            failure_threshold=1,
            timeout=60.0,
            expected_exception=ValueError,
        )
        cb.on_failure(RuntimeError("ignored"))
        assert cb.state == CircuitState.CLOSED
        cb.on_failure(ValueError("counts"))
        assert cb.state == CircuitState.OPEN

    def test_reset(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, timeout=60.0)
        cb.on_failure(RuntimeError("x"))
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.consecutive_failures == 0

    def test_consecutive_success_resets_failures(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        cb.on_failure(RuntimeError("x"))
        cb.on_failure(RuntimeError("x"))
        cb.on_success()
        assert cb.consecutive_failures == 0
        assert cb.state == CircuitState.CLOSED


# ---------------------------------------------------------------------------
# 5. RequestQueue
# ---------------------------------------------------------------------------


class TestProviderRateLimiter:
    def test_unlimited_by_default(self) -> None:
        limiter = ProviderRateLimiter()
        assert limiter.can_make_request()
        assert limiter.wait_time() == 0.0

    def test_request_limit_reached(self) -> None:
        limiter = ProviderRateLimiter(requests_per_minute=2)
        assert limiter.can_make_request()
        assert limiter.can_make_request()
        assert not limiter.can_make_request()

    def test_token_limit(self) -> None:
        limiter = ProviderRateLimiter(tokens_per_minute=100)
        assert limiter.can_make_request(60)
        assert not limiter.can_make_request(60)

    def test_wait_time(self) -> None:
        limiter = ProviderRateLimiter(requests_per_minute=1)
        assert limiter.can_make_request()
        wait = limiter.wait_time()
        assert wait > 0

    def test_reset(self) -> None:
        limiter = ProviderRateLimiter(requests_per_minute=1)
        limiter.can_make_request()
        assert not limiter.can_make_request()
        limiter.reset()
        assert limiter.can_make_request()

    def test_get_info(self) -> None:
        limiter = ProviderRateLimiter(requests_per_minute=10)
        limiter.can_make_request()
        info = limiter.get_info()
        assert info.requests_per_minute == 10
        assert info.current_rpm == 1


class TestRequestQueue:
    def test_submit_returns_result(self) -> None:
        async def _run() -> None:
            queue = RequestQueue()

            async def _call() -> str:
                return "done"

            result = await queue.submit("p1", _call())
            assert result == "done"
            await queue.shutdown()

        asyncio.run(_run())

    def test_submit_propagates_exception(self) -> None:
        async def _run() -> None:
            queue = RequestQueue()

            async def _call() -> None:
                raise ValueError("boom")

            with pytest.raises(ValueError, match="boom"):
                await queue.submit("p1", _call())
            await queue.shutdown()

        asyncio.run(_run())

    def test_rate_limit_exceeded(self) -> None:
        async def _run() -> None:
            queue = RequestQueue()
            queue.set_rate_limit("p1", 1, 0)

            async def _call() -> str:
                return "done"

            await queue.submit("p1", _call(), tokens=0)
            with pytest.raises(RateLimitExceededError):
                await queue.submit("p1", _call(), tokens=0)
            await queue.shutdown()

        asyncio.run(_run())

    def test_pending_count(self) -> None:
        async def _run() -> None:
            queue = RequestQueue()

            async def _call() -> str:
                await asyncio.sleep(0.01)
                return "done"

            task = asyncio.create_task(queue.submit("p1", _call()))
            await asyncio.sleep(0)
            count = queue.get_pending_count("p1")
            assert count >= 0
            await task
            await queue.shutdown()

        asyncio.run(_run())

    def test_priority_processing(self) -> None:
        async def _run() -> None:
            queue = RequestQueue()

            order: list[str] = []

            async def _high() -> str:
                order.append("high")
                return "high"

            async def _low() -> str:
                order.append("low")
                return "low"

            low_task = asyncio.create_task(
                queue.submit("p1", _low(), priority=RequestPriority.LOW)
            )
            await asyncio.sleep(0)
            high_task = asyncio.create_task(
                queue.submit("p1", _high(), priority=RequestPriority.HIGH)
            )
            await high_task
            await low_task
            await queue.shutdown()
            assert order[0] == "high"

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# 6. LoadBalancer
# ---------------------------------------------------------------------------


class TestLoadBalancer:
    def test_round_robin(self) -> None:
        lb = LoadBalancer(LoadBalanceStrategy.ROUND_ROBIN)
        first = lb.select(["a", "b", "c"])
        second = lb.select(["a", "b", "c"])
        third = lb.select(["a", "b", "c"])
        assert len({first, second, third}) == 3

    def test_round_robin_empty_raises(self) -> None:
        lb = LoadBalancer()
        with pytest.raises(RoutingError):
            lb.select([])

    def test_weighted_prefers_heavier(self) -> None:
        lb = LoadBalancer(LoadBalanceStrategy.WEIGHTED)
        lb.set_weight("a", 10.0)
        lb.set_weight("b", 0.1)
        picks = {lb.select(["a", "b"]) for _ in range(50)}
        assert "a" in picks

    def test_least_latency(self) -> None:
        lb = LoadBalancer(LoadBalanceStrategy.LEAST_LATENCY)
        snaps = {
            "slow": ProviderHealthSnapshot(
                provider_id="slow",
                is_healthy=True,
                circuit_state=CircuitState.CLOSED,
                avg_latency=2.0,
            ),
            "fast": ProviderHealthSnapshot(
                provider_id="fast",
                is_healthy=True,
                circuit_state=CircuitState.CLOSED,
                avg_latency=0.1,
            ),
        }
        assert lb.select(["slow", "fast"], snaps) == "fast"

    def test_least_error_rate(self) -> None:
        lb = LoadBalancer(LoadBalanceStrategy.LEAST_ERROR_RATE)
        snaps = {
            "erratic": ProviderHealthSnapshot(
                provider_id="erratic",
                is_healthy=True,
                circuit_state=CircuitState.CLOSED,
                success_rate=0.5,
            ),
            "reliable": ProviderHealthSnapshot(
                provider_id="reliable",
                is_healthy=True,
                circuit_state=CircuitState.CLOSED,
                success_rate=0.99,
            ),
        }
        assert lb.select(["erratic", "reliable"], snaps) == "reliable"

    def test_set_strategy(self) -> None:
        lb = LoadBalancer()
        lb.set_strategy(LoadBalanceStrategy.WEIGHTED)
        assert lb.strategy == LoadBalanceStrategy.WEIGHTED

    def test_reset(self) -> None:
        lb = LoadBalancer(LoadBalanceStrategy.WEIGHTED)
        lb.set_weight("a", 5.0)
        lb.reset()
        assert lb.get_weights() == {}


# ---------------------------------------------------------------------------
# 7. PromptCache
# ---------------------------------------------------------------------------


class TestPromptCache:
    def test_get_miss(self) -> None:
        cache = PromptCache()
        key = cache.build_key(_make_request(), "p1")
        assert cache.get(key) is None

    def test_put_and_get_hit(self) -> None:
        cache = PromptCache()
        key = cache.build_key(_make_request(), "p1")
        response = _TestProvider("p1").generate(_make_request())
        cache.put(key, response)
        assert cache.get(key) == response

    def test_same_request_same_key(self) -> None:
        cache = PromptCache()
        k1 = cache.build_key(_make_request(), "p1")
        k2 = cache.build_key(_make_request(), "p1")
        assert k1.hash == k2.hash

    def test_different_request_different_key(self) -> None:
        cache = PromptCache()
        k1 = cache.build_key(_make_request(content="hello"), "p1")
        k2 = cache.build_key(_make_request(content="world"), "p1")
        assert k1.hash != k2.hash

    def test_disable_prevents_caching(self) -> None:
        cache = PromptCache(enabled=False)
        key = cache.build_key(_make_request(), "p1")
        response = _TestProvider("p1").generate(_make_request())
        cache.put(key, response)
        assert cache.get(key) is None

    def test_invalidate(self) -> None:
        cache = PromptCache()
        key = cache.build_key(_make_request(), "p1")
        cache.put(key, "value")
        assert cache.invalidate(key) is True
        assert cache.get(key) is None

    def test_invalidate_provider(self) -> None:
        cache = PromptCache()
        k1 = cache.build_key(_make_request(content="one"), "p1")
        k2 = cache.build_key(_make_request(content="two"), "p1")
        cache.put(k1, "v1")
        cache.put(k2, "v2")
        assert cache.invalidate_provider("p1") == 2

    def test_lru_eviction(self) -> None:
        cache = PromptCache(max_size=2)
        key1 = cache.build_key(_make_request(content="one"), "p1")
        key2 = cache.build_key(_make_request(content="two"), "p1")
        key3 = cache.build_key(_make_request(content="three"), "p1")
        cache.put(key1, "v1")
        cache.put(key2, "v2")
        cache.get(key1)  # touch key1
        cache.put(key3, "v3")  # evicts key2 (LRU)
        assert cache.get(key1) == "v1"
        assert cache.get(key3) == "v3"
        assert cache.get(key2) is None

    def test_ttl_expiry(self) -> None:
        cache = PromptCache(ttl=0.01)
        key = cache.build_key(_make_request(), "p1")
        cache.put(key, "v")
        assert cache.get(key) == "v"
        time.sleep(0.05)
        assert cache.get(key) is None

    def test_stats(self) -> None:
        cache = PromptCache()
        key = cache.build_key(_make_request(), "p1")
        cache.put(key, "v")
        cache.get(key)
        cache.get(cache.build_key(_make_request(content="miss"), "p1"))
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


# ---------------------------------------------------------------------------
# 8. PolicyEngine
# ---------------------------------------------------------------------------


class TestPolicyEngine:
    def test_no_rules_allows_all(self) -> None:
        engine = PolicyEngine()
        assert engine.evaluate("p1")

    def test_max_cost_policy(self) -> None:
        engine = PolicyEngine()
        engine.add_rule(MaxCostPolicy(max_cost=0.01))
        assert engine.evaluate("p1", context={"estimated_cost": 0.005})
        with pytest.raises(PolicyViolationError):
            engine.evaluate("p1", context={"estimated_cost": 0.05})

    def test_block_list_policy(self) -> None:
        engine = PolicyEngine()
        engine.add_rule(ProviderBlockListPolicy(blocked={"bad"}))
        assert engine.evaluate("good")
        with pytest.raises(PolicyViolationError):
            engine.evaluate("bad")

    def test_filter_providers(self) -> None:
        engine = PolicyEngine()
        engine.add_rule(ProviderBlockListPolicy(blocked={"bad"}))
        allowed = engine.filter_providers(["good", "bad", "other"])
        assert allowed == ["good", "other"]

    def test_remove_rule(self) -> None:
        engine = PolicyEngine()
        engine.add_rule(ProviderBlockListPolicy(blocked={"bad"}))
        assert engine.remove_rule("block_list") is True
        assert engine.remove_rule("block_list") is False
        assert engine.evaluate("bad")

    def test_clear_rules(self) -> None:
        engine = PolicyEngine()
        engine.add_rule(ProviderBlockListPolicy(blocked={"bad"}))
        engine.clear_rules()
        assert engine.evaluate("bad")

    def test_rule_names(self) -> None:
        engine = PolicyEngine()
        engine.add_rule(MaxCostPolicy(0.01))
        assert "max_cost" in engine.rule_names


# ---------------------------------------------------------------------------
# 9. CostOptimizer
# ---------------------------------------------------------------------------


class TestCostOptimizer:
    def test_register_and_get_costs(self) -> None:
        opt = CostOptimizer()
        opt.register_provider_costs("p1", 0.001, 0.002)
        assert opt.get_provider_cost_per_1k_tokens("p1") == (0.001, 0.002)
        assert opt.get_provider_cost_per_1k_tokens("ghost") == (0.0, 0.0)

    def test_estimate_cost(self) -> None:
        opt = CostOptimizer()
        opt.register_provider_costs("p1", 1.0, 2.0)
        request = _make_request(content="hello world")
        est = opt.estimate_cost("p1", request)
        assert est.provider_id == "p1"
        assert est.prompt_tokens > 0
        assert est.total_tokens > 0
        assert est.estimated_cost > 0

    def test_estimate_cost_unknown_provider_raises(self) -> None:
        opt = CostOptimizer()
        with pytest.raises(RoutingError):
            opt.estimate_cost("ghost", _make_request())

    def test_estimate_tokens(self) -> None:
        opt = CostOptimizer()
        request = _make_request(content="a" * 100)
        prompt, completion = opt.estimate_tokens(request)
        assert prompt >= 25
        assert completion > 0

    def test_select_cheapest(self) -> None:
        opt = CostOptimizer()
        opt.register_provider_costs("cheap", 0.001, 0.002)
        opt.register_provider_costs("expensive", 10.0, 20.0)
        request = _make_request()
        selected = opt.select_cheapest(
            request, ["expensive", "cheap"], RoutingCriteria()
        )
        assert selected == "cheap"

    def test_select_cheapest_respects_max_cost(self) -> None:
        opt = CostOptimizer()
        opt.register_provider_costs("cheap", 0.001, 0.002)
        opt.register_provider_costs("expensive", 10.0, 20.0)
        request = _make_request()
        criteria = RoutingCriteria(max_cost=0.001)
        with pytest.raises(RoutingError):
            opt.select_cheapest(request, ["expensive"], criteria)

    def test_select_cheapest_no_candidates(self) -> None:
        opt = CostOptimizer()
        with pytest.raises(RoutingError):
            opt.select_cheapest(_make_request(), [], RoutingCriteria())


# ---------------------------------------------------------------------------
# 10. Analytics
# ---------------------------------------------------------------------------


class TestLLMAnalytics:
    def test_record_event(self) -> None:
        analytics = LLMAnalytics()
        analytics.record(
            provider_id="p1",
            model="m1",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency=0.1,
            first_token_latency=0.05,
            success=True,
        )
        assert len(analytics.get_events()) == 1

    def test_success_rate(self) -> None:
        analytics = LLMAnalytics()
        analytics.record("p1", "m", 1, 1, 2, 0.1, 0.1, True)
        analytics.record("p1", "m", 1, 1, 2, 0.1, 0.1, True)
        analytics.record("p1", "m", 1, 1, 2, 0.1, 0.1, False, error="x")
        ranking = analytics.get_provider_ranking("p1")
        assert ranking.success_rate == 2 / 3

    def test_cache_hit_rate(self) -> None:
        analytics = LLMAnalytics()
        analytics.record("p1", "m", 1, 1, 2, 0.1, 0.1, True, cached=True)
        analytics.record("p1", "m", 1, 1, 2, 0.1, 0.1, True, cached=False)
        assert analytics.get_cache_hit_rate() == 0.5

    def test_total_cost(self) -> None:
        analytics = LLMAnalytics()
        analytics.record("p1", "m", 1, 1, 2, 0.1, 0.1, True, estimated_cost=0.01)
        analytics.record("p1", "m", 1, 1, 2, 0.1, 0.1, True, estimated_cost=0.02)
        assert analytics.get_total_cost() == 0.03

    def test_total_tokens(self) -> None:
        analytics = LLMAnalytics()
        analytics.record("p1", "m", 10, 5, 15, 0.1, 0.1, True)
        analytics.record("p1", "m", 20, 10, 30, 0.1, 0.1, True)
        prompt, completion, total = analytics.get_total_tokens()
        assert prompt == 30
        assert completion == 15
        assert total == 45

    def test_provider_ranking_empty(self) -> None:
        analytics = LLMAnalytics()
        ranking = analytics.get_provider_ranking("ghost")
        assert ranking.total_requests == 0

    def test_avg_latency(self) -> None:
        analytics = LLMAnalytics()
        analytics.record("p1", "m", 1, 1, 2, 0.2, 0.1, True)
        analytics.record("p1", "m", 1, 1, 2, 0.4, 0.1, True)
        ranking = analytics.get_provider_ranking("p1")
        assert ranking.avg_latency == 0.3

    def test_all_rankings(self) -> None:
        analytics = LLMAnalytics()
        analytics.record("p1", "m", 1, 1, 2, 0.1, 0.1, True)
        analytics.record("p2", "m", 1, 1, 2, 0.1, 0.1, True)
        rankings = analytics.get_all_rankings()
        assert set(rankings.keys()) == {"p1", "p2"}

    def test_overall_stats(self) -> None:
        analytics = LLMAnalytics()
        analytics.record("p1", "m", 1, 1, 2, 0.1, 0.1, True)
        analytics.record("p1", "m", 1, 1, 2, 0.1, 0.1, False, error="x")
        stats = analytics.get_overall_stats()
        assert stats["total_requests"] == 2
        assert stats["success_rate"] == 0.5

    def test_clear(self) -> None:
        analytics = LLMAnalytics()
        analytics.record("p1", "m", 1, 1, 2, 0.1, 0.1, True)
        analytics.clear()
        assert analytics.get_events() == []

    def test_provider_ranking_fields(self) -> None:
        analytics = LLMAnalytics()
        analytics.record("p1", "m", 1, 1, 2, 0.1, 0.1, True, cached=True)
        ranking = analytics.get_provider_ranking("p1")
        assert isinstance(ranking, ProviderRanking)
        assert ranking.total_requests == 1
        assert ranking.total_tokens == 2


# ---------------------------------------------------------------------------
# 11. UnifiedStreamer
# ---------------------------------------------------------------------------


class TestUnifiedStreamer:
    def test_normalize_yields_chunks(self) -> None:
        async def _run() -> None:
            streamer = UnifiedStreamer()
            provider = _TestProvider("p1")
            chunks = []
            async for chunk in streamer.normalize(provider, _make_request()):
                chunks.append(chunk)
            assert len(chunks) == 2
            assert chunks[0].content == "hello"
            assert chunks[1].content == " world"

        asyncio.run(_run())

    def test_normalize_rejects_non_streaming(self) -> None:
        async def _run() -> None:
            streamer = UnifiedStreamer()
            provider = _TestProvider("p1", supports_streaming=False)
            with pytest.raises(RoutingError):
                async for _ in streamer.normalize(provider, _make_request()):
                    pass

        asyncio.run(_run())

    def test_stream_to_handler(self) -> None:
        async def _run() -> None:
            streamer = UnifiedStreamer()
            provider = _TestProvider("p1")
            received: list[str] = []
            response = await streamer.stream_to_handler(
                provider,
                _make_request(),
                lambda chunk: received.append(chunk.content),
            )
            assert "".join(received) == "hello world"
            assert response is not None
            assert response.content == "hello world"

        asyncio.run(_run())

    def test_stream_propagates_error(self) -> None:
        async def _run() -> None:
            streamer = UnifiedStreamer()
            provider = _TestProvider("p1", fail=True)
            with pytest.raises(RuntimeError, match="failure"):
                async for _ in streamer.normalize(provider, _make_request()):
                    pass

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# 12. FallbackManager
# ---------------------------------------------------------------------------


class TestFallbackManager:
    def test_set_and_get_chain(self) -> None:
        fb = FallbackManager(provider_resolver=lambda x: _TestProvider(x))
        fb.set_fallback_chain("primary", ["backup1", "backup2"])
        assert fb.get_fallback_chain("primary") == ["backup1", "backup2"]

    def test_chain_excludes_primary(self) -> None:
        fb = FallbackManager(provider_resolver=lambda x: _TestProvider(x))
        fb.set_fallback_chain("primary", ["primary", "backup"])
        assert fb.get_fallback_chain("primary") == ["backup"]

    def test_no_chain_returns_empty(self) -> None:
        fb = FallbackManager(provider_resolver=lambda x: _TestProvider(x))
        assert fb.get_fallback_chain("unknown") == []

    def test_primary_succeeds_no_fallback(self) -> None:
        providers = {
            "primary": _TestProvider("primary"),
            "backup": _TestProvider("backup"),
        }
        fb = FallbackManager(provider_resolver=lambda x: providers[x])
        fb.set_fallback_chain("primary", ["backup"])

        async def _run() -> None:
            result = await fb.execute_with_fallback(
                _make_request(), "primary", lambda p, r: p.generate(r)
            )
            assert result.provider == "primary"

        asyncio.run(_run())

    def test_fallback_invoked(self) -> None:
        providers = {
            "primary": _TestProvider("primary", fail=True),
            "backup": _TestProvider("backup"),
        }
        fb = FallbackManager(provider_resolver=lambda x: providers[x])
        fb.set_fallback_chain("primary", ["backup"])

        async def _run() -> None:
            result = await fb.execute_with_fallback(
                _make_request(), "primary", lambda p, r: p.generate(r)
            )
            assert result.provider == "backup"

        asyncio.run(_run())

    def test_all_fail_raises(self) -> None:
        providers = {
            "primary": _TestProvider("primary", fail=True),
            "backup": _TestProvider("backup", fail=True),
        }
        fb = FallbackManager(provider_resolver=lambda x: providers[x])
        fb.set_fallback_chain("primary", ["backup"])

        async def _run() -> None:
            with pytest.raises(FallbackExhaustedError):
                await fb.execute_with_fallback(
                    _make_request(), "primary", lambda p, r: p.generate(r)
                )

        asyncio.run(_run())

    def test_max_attempts_limits_chain(self) -> None:
        providers = {
            "primary": _TestProvider("primary", fail=True),
            "b1": _TestProvider("b1", fail=True),
            "b2": _TestProvider("b2"),
        }
        fb = FallbackManager(
            provider_resolver=lambda x: providers[x],
            max_fallback_attempts=2,
        )
        fb.set_fallback_chain("primary", ["b1", "b2"])

        async def _run() -> None:
            with pytest.raises(FallbackExhaustedError):
                await fb.execute_with_fallback(
                    _make_request(), "primary", lambda p, r: p.generate(r)
                )

        asyncio.run(_run())

    def test_on_success_callback(self) -> None:
        providers = {"primary": _TestProvider("primary")}
        fb = FallbackManager(provider_resolver=lambda x: providers[x])

        async def _run() -> None:
            called: list[str] = []
            await fb.execute_with_fallback(
                _make_request(),
                "primary",
                lambda p, r: p.generate(r),
                on_success=lambda pid, p, result: called.append(pid),
            )
            assert called == ["primary"]

        asyncio.run(_run())

    def test_clear_chain(self) -> None:
        fb = FallbackManager(provider_resolver=lambda x: _TestProvider(x))
        fb.set_fallback_chain("primary", ["backup"])
        fb.clear_chain("primary")
        assert fb.get_fallback_chain("primary") == []

    def test_unresolvable_provider(self) -> None:
        fb = FallbackManager(provider_resolver=lambda x: None)

        async def _run() -> None:
            with pytest.raises(FallbackExhaustedError):
                await fb.execute_with_fallback(
                    _make_request(), "ghost", lambda p, r: p.generate(r)
                )

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# 13. LLMRouter
# ---------------------------------------------------------------------------


class TestLLMRouter:
    def test_register_and_unregister_provider(self) -> None:
        router = _build_router([])
        provider = _TestProvider("new")
        router.register_provider(
            provider,
            prompt_cost_per_1k=0.001,
            completion_cost_per_1k=0.002,
        )
        assert router.get_provider("new") is not None
        assert "new" in router.get_available_providers()
        router.unregister_provider("new")
        assert router.get_provider("new") is None

    def test_select_provider_no_registry(self) -> None:
        router = _build_router([])
        with pytest.raises(NoHealthyProvidersError):
            router.select_provider(_make_request())

    def test_select_provider_with_registry(self) -> None:
        router = _build_router([_TestProvider("p1")])
        decision = router.select_provider(_make_request())
        assert decision.selected_provider == "p1"

    def test_select_provider_context_filter(self) -> None:
        router = _build_router([_TestProvider("small", context_length=1024)])
        criteria = RoutingCriteria(min_context_length=4096)
        with pytest.raises(NoHealthyProvidersError):
            router.select_provider(_make_request(), criteria)

    def test_select_provider_preferred(self) -> None:
        router = _build_router([_TestProvider("a"), _TestProvider("b")])
        criteria = RoutingCriteria(preferred_providers=["b"])
        decision = router.select_provider(_make_request(), criteria)
        assert decision.selected_provider == "b"

    def test_generate_returns_response(self) -> None:
        router = _build_router([_TestProvider("p1")])
        response = router.generate(_make_request())
        assert response.content == "test response"
        assert response.provider == "p1"

    def test_generate_explicit_provider(self) -> None:
        router = _build_router([_TestProvider("p1")])
        response = router.generate(_make_request(), provider_id="p1")
        assert response.provider == "p1"

    def test_generate_unknown_provider_raises(self) -> None:
        router = _build_router([_TestProvider("p1")])
        with pytest.raises(RoutingError):
            router.generate(_make_request(), provider_id="ghost")

    def test_generate_records_analytics(self) -> None:
        router = _build_router([_TestProvider("p1")])
        router.generate(_make_request())
        stats = router.analytics.get_overall_stats()
        assert stats["total_requests"] == 1

    def test_generate_cache_second_call(self) -> None:
        router = _build_router([_TestProvider("p1")])
        request = _make_request()
        router.generate(request)
        router.generate(request)
        stats = router.analytics.get_overall_stats()
        # First call is a real request; second is cached
        assert stats["total_requests"] == 2
        assert router.cache.get_stats()["hits"] == 1

    def test_cached_response_not_double_recorded(self) -> None:
        router = _build_router([_TestProvider("p1")])
        request = _make_request()
        router.generate(request)
        router.generate(request)
        stats = router.analytics.get_overall_stats()
        assert stats["total_requests"] == 2
        assert stats["cache_hit_rate"] == 0.5

    def test_generate_async(self) -> None:
        router = _build_router([_TestProvider("p1")])

        async def _run() -> None:
            response = await router.generate_async(_make_request())
            assert response.content == "test response"

        asyncio.run(_run())

    def test_generate_async_with_queue(self) -> None:
        router = _build_router([_TestProvider("p1")])

        async def _run() -> None:
            results = await asyncio.gather(
                router.generate_async(_make_request()),
                router.generate_async(_make_request()),
            )
            assert len(results) == 2
            await router.queue.shutdown()

        asyncio.run(_run())

    def test_stream_yields_chunks(self) -> None:
        router = _build_router([_TestProvider("p1")])

        async def _run() -> None:
            chunks = []
            async for chunk in router.stream(_make_request()):
                chunks.append(chunk)
            assert len(chunks) == 2
            assert "".join(c.content for c in chunks) == "hello world"

        asyncio.run(_run())

    def test_stream_with_handler(self) -> None:
        router = _build_router([_TestProvider("p1")])

        async def _run() -> None:
            received: list[str] = []
            response = await router.stream_to_handler(
                _make_request(), lambda c: received.append(c.content)
            )
            assert response is not None
            assert response.content == "hello world"

        asyncio.run(_run())

    def test_criteria_from_request(self) -> None:
        criteria = LLMRouter.criteria_from_request(_make_request(stream=True, tools=()))
        assert criteria.requires_streaming is True

    def test_circuit_breaker_blocks_failing_provider(self) -> None:
        router = _build_router([_TestProvider("p1", fail=True)])
        cb = CircuitBreaker(failure_threshold=2, timeout=60.0)
        router._circuit_breakers["p1"] = cb
        with pytest.raises(RuntimeError, match="failure"):
            router.generate(_make_request())
        with pytest.raises(RuntimeError, match="failure"):
            router.generate(_make_request())
        assert cb.state == CircuitState.OPEN

    def test_policy_filters_provider(self) -> None:
        router = _build_router([_TestProvider("p1")])
        router.policies.add_rule(ProviderBlockListPolicy(blocked={"p1"}))
        with pytest.raises(NoHealthyProvidersError):
            router.select_provider(_make_request())

    def test_provider_health_tracked(self) -> None:
        router = _build_router([_TestProvider("p1")])
        before = router.health.get_snapshot("p1").success_count
        router.generate(_make_request())
        after = router.health.get_snapshot("p1").success_count
        assert after == before + 1

    def test_strategy_cheapest_selects_lowest_cost(self) -> None:
        router = _build_router([_TestProvider("cheap"), _TestProvider("expensive")])
        router.cost_optimizer.register_provider_costs("cheap", 0.001, 0.001)
        router.cost_optimizer.register_provider_costs("expensive", 10.0, 10.0)
        criteria = RoutingCriteria(strategy=RouterStrategy.CHEAPEST)
        decision = router.select_provider(_make_request(), criteria)
        assert decision.selected_provider == "cheap"

    def test_strategy_fastest(self) -> None:
        router = _build_router(
            [_TestProvider("slow", delay_s=0.05), _TestProvider("fast")]
        )
        criteria = RoutingCriteria(strategy=RouterStrategy.FASTEST)
        decision = router.select_provider(_make_request(), criteria)
        assert decision.selected_provider == "fast"

    def test_decision_contains_scores(self) -> None:
        router = _build_router([_TestProvider("p1")])
        decision = router.select_provider(_make_request())
        assert len(decision.scores) >= 1
        assert decision.fallback_chain[0] == "p1"

    def test_provider_scoring_weighted(self) -> None:
        router = _build_router([_TestProvider("a", delay_s=0.05), _TestProvider("b")])
        # Give 'a' a failure to degrade its health score
        router.health.record_failure("a", 0.1, "x")
        scores = router._score_providers(["a", "b"], _make_request(), RoutingCriteria())
        assert scores[0].provider_id == "b"
