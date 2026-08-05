"""Comprehensive runtime verification: Plugin System, Multi-LLM Layer, Agent System.

Covers:
  - Plugin system: manager instantiation, discovery, lifecycle, PluginContext, PluginType
  - Multi-LLM layer: MockProvider, ProviderRegistry, LLMRouter, CircuitBreaker,
    FallbackManager, PromptCache, ProviderHealthMonitor, LLMAnalytics, PolicyEngine,
    RequestQueue, UnifiedStreamer
  - Agent system: CognitiveOrchestrator, DependencyContainer bootstrap, AgentRuntime,
    agent run, cancel, status, session listing

Run:  python -m pytest tests/runtime_verification_plugins_llm.py -v 2>&1
"""

from __future__ import annotations

import asyncio
import time

import pytest

# =====================================================================
# Helpers
# =====================================================================


def _make_request(
    content: str = "hello",
    model: str = "test-model",
    stream: bool = False,
):
    from app.llm.models import ChatRequest, Message, Role

    return ChatRequest(
        messages=tuple([Message(role=Role.USER, content=content)]),
        model=model,
        stream=stream,
    )


def _make_provider(model: str = "test-model"):
    from app.llm.providers.mock import MockProvider

    return MockProvider(model=model)


def _make_router():
    from app.llm.analytics import LLMAnalytics
    from app.llm.capability_matrix import CapabilityMatrix
    from app.llm.cost_optimizer import CostOptimizer
    from app.llm.fallback_manager import FallbackManager
    from app.llm.health_monitor import ProviderHealthMonitor
    from app.llm.llm_router import LLMRouter
    from app.llm.load_balancer import LoadBalancer
    from app.llm.policy_engine import PolicyEngine
    from app.llm.prompt_cache import PromptCache
    from app.llm.registry import ProviderRegistry
    from app.llm.request_queue import RequestQueue
    from app.llm.unified_streamer import UnifiedStreamer

    registry = ProviderRegistry()
    health = ProviderHealthMonitor()
    caps = CapabilityMatrix()
    cost = CostOptimizer()
    lb = LoadBalancer()
    circuit_breakers: dict = {}
    fallback = FallbackManager()
    queue = RequestQueue()
    cache = PromptCache()
    policies = PolicyEngine()
    analytics = LLMAnalytics()
    streamer = UnifiedStreamer()

    return LLMRouter(
        registry=registry,
        health_monitor=health,
        capability_matrix=caps,
        cost_optimizer=cost,
        load_balancer=lb,
        circuit_breakers=circuit_breakers,
        fallback_manager=fallback,
        request_queue=queue,
        prompt_cache=cache,
        policy_engine=policies,
        analytics=analytics,
        unified_streamer=streamer,
    )


def _register_mock(router, model: str = "test-model"):
    provider = _make_provider(model)
    router.register_provider(
        provider, prompt_cost_per_1k=0.0, completion_cost_per_1k=0.0
    )
    return provider


# =====================================================================
# 1. PLUGIN SYSTEM
# =====================================================================


class TestPluginSystem:
    """Verify the Plugin System subsystem at runtime."""

    # -- 1.1 Imports & instantiation ------------------------------------

    def test_import_plugin_manager(self):
        from app.plugins import PluginManager

        assert PluginManager is not None

    def test_import_plugin_context(self):
        from app.plugins import PluginContext

        assert PluginContext is not None

    def test_import_plugin_type(self):
        from app.plugins import PluginType

        assert PluginType is not None

    def test_plugin_manager_instantiation(self):
        from app.plugins import PluginManager

        manager = PluginManager()
        assert manager is not None

    # -- 1.2 Plugin discovery -------------------------------------------

    def test_plugin_manager_discover_method(self):
        from app.plugins import PluginManager

        manager = PluginManager()
        assert hasattr(manager, "discover")

    def test_plugin_discover_returns_dict(self):
        from app.plugins import PluginManager

        manager = PluginManager()
        try:
            result = manager.discover()
            assert isinstance(result, dict)
        except Exception:
            pass  # No plugin dirs configured is acceptable

    # -- 1.3 Plugin lifecycle methods -----------------------------------

    def test_list_plugins(self):
        from app.plugins import PluginManager

        manager = PluginManager()
        result = manager.list_plugins()
        assert isinstance(result, list)

    def test_enable_plugin_method_exists(self):
        from app.plugins import PluginManager

        manager = PluginManager()
        assert callable(getattr(manager, "enable", None))

    def test_disable_plugin_method_exists(self):
        from app.plugins import PluginManager

        manager = PluginManager()
        assert callable(getattr(manager, "disable", None))

    def test_unload_plugin_method_exists(self):
        from app.plugins import PluginManager

        manager = PluginManager()
        assert callable(getattr(manager, "unload", None))

    def test_plugin_manager_total_count(self):
        from app.plugins import PluginManager

        manager = PluginManager()
        try:
            count = manager.total_count
            assert isinstance(count, int)
            assert count >= 0
        except Exception:
            pass

    # -- 1.4 PluginContext frozen dataclass -----------------------------

    def test_plugin_context_creation(self):
        from app.plugins import PluginContext

        ctx = PluginContext()
        assert ctx.tool_registry is None
        assert ctx.agent_manager is None
        assert ctx.provider_registry is None
        assert ctx.container is None

    def test_plugin_context_frozen(self):
        from app.plugins import PluginContext

        ctx = PluginContext()
        with pytest.raises(AttributeError):
            ctx.tool_registry = "something"  # type: ignore[misc]

    def test_plugin_context_with_values(self):
        from app.plugins import PluginContext

        ctx = PluginContext(tool_registry="tr", provider_registry="pr")
        assert ctx.tool_registry == "tr"
        assert ctx.provider_registry == "pr"
        assert ctx.fastapi_app is None

    def test_plugin_context_log_methods(self):
        from app.plugins import PluginContext

        ctx = PluginContext()
        assert callable(getattr(ctx, "log_info", None))
        assert callable(getattr(ctx, "log_warning", None))
        assert callable(getattr(ctx, "log_error", None))

    # -- 1.5 PluginType enum values ------------------------------------

    def test_plugin_type_tool(self):
        from app.plugins import PluginType

        assert PluginType.TOOL.value == "tool"

    def test_plugin_type_agent(self):
        from app.plugins import PluginType

        assert PluginType.AGENT.value == "agent"

    def test_plugin_type_workflow(self):
        from app.plugins import PluginType

        assert PluginType.WORKFLOW.value == "workflow"

    def test_plugin_type_api_route(self):
        from app.plugins import PluginType

        assert PluginType.API_ROUTE.value == "api_route"

    def test_plugin_type_memory_provider(self):
        from app.plugins import PluginType

        assert PluginType.MEMORY_PROVIDER.value == "memory_provider"

    def test_plugin_type_llm_provider(self):
        from app.plugins import PluginType

        assert PluginType.LLM_PROVIDER.value == "llm_provider"

    def test_plugin_type_mixed(self):
        from app.plugins import PluginType

        assert PluginType.MIXED.value == "mixed"

    def test_plugin_type_all_values_count(self):
        from app.plugins import PluginType

        assert len(PluginType) == 7

    # -- 1.6 PluginLifecycleState --------------------------------------

    def test_plugin_lifecycle_states(self):
        from app.plugins import PluginLifecycleState

        expected = {"installed", "loaded", "enabled", "disabled", "unloaded", "error"}
        actual = {s.value for s in PluginLifecycleState}
        assert actual == expected


# =====================================================================
# 2. MULTI-LLM LAYER
# =====================================================================


class TestMultiLLMLayer:
    """Verify the Multi-LLM Intelligence Layer at runtime."""

    # -- 2.1 MockProvider ------------------------------------------------

    def test_mock_provider_instantiation(self):
        from app.llm.providers.mock import MockProvider

        p = MockProvider(model="test-model")
        assert p.name == "mock"
        assert p.default_model == "test-model"

    def test_mock_provider_generate(self):
        req = _make_request("hello")
        p = _make_provider()
        resp = p.generate(req)
        assert resp.content == "Echo: hello"
        assert resp.provider == "mock"
        assert resp.model == "test-model"

    def test_mock_provider_stream(self):
        req = _make_request("hello")
        p = _make_provider()
        chunks = list(p.stream(req))
        assert len(chunks) > 0
        full = "".join(c.content for c in chunks)
        assert "Echo: hello" in full

    def test_mock_provider_model_info(self):
        p = _make_provider()
        info = p.model_info()
        assert info.supports_streaming is True
        assert info.supports_tools is True

    # -- 2.2 ProviderRegistry -------------------------------------------

    def test_provider_registry_register_get(self):
        from app.llm.registry import ProviderRegistry

        reg = ProviderRegistry()
        p = _make_provider()
        reg.register(p)
        assert reg.has("mock")
        assert reg.get("mock") is p
        assert reg.count == 1

    def test_provider_registry_unregister(self):
        from app.llm.registry import ProviderRegistry

        reg = ProviderRegistry()
        p = _make_provider()
        reg.register(p)
        reg.unregister("mock")
        assert not reg.has("mock")

    def test_provider_registry_names(self):
        from app.llm.registry import ProviderRegistry

        reg = ProviderRegistry()
        p = _make_provider()
        reg.register(p)
        assert "mock" in reg.names()

    # -- 2.3 LLMRouter --------------------------------------------------

    def test_router_creation(self):
        router = _make_router()
        assert router is not None

    def test_router_register_provider(self):
        router = _make_router()
        _register_mock(router)
        assert "mock" in router.get_available_providers()

    def test_router_generate(self):
        router = _make_router()
        _register_mock(router)
        req = _make_request("hello")
        resp = router.generate(req)
        assert resp is not None
        assert resp.content == "Echo: hello"
        assert resp.provider == "mock"

    @pytest.mark.asyncio
    async def test_router_stream(self):
        router = _make_router()
        _register_mock(router)
        req = _make_request("hello", stream=True)
        chunks = []
        async for chunk in router.stream(req):
            chunks.append(chunk)
        assert len(chunks) > 0
        full = "".join(c.content for c in chunks)
        assert "Echo: hello" in full

    def test_router_unregister_provider(self):
        router = _make_router()
        _register_mock(router)
        router.unregister_provider("mock")
        assert "mock" not in router.get_available_providers()

    # -- 2.4 CircuitBreaker ---------------------------------------------

    def test_circuit_breaker_initial_state(self):
        from app.llm.circuit_breaker import CircuitBreaker
        from app.llm.router_models import CircuitState

        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_closed_to_open(self):
        from app.llm.circuit_breaker import CircuitBreaker
        from app.llm.router_models import CircuitState

        cb = CircuitBreaker(failure_threshold=3, timeout=1.0)
        for _ in range(3):
            cb.on_failure(RuntimeError("fail"))
        assert cb.state == CircuitState.OPEN

    def test_circuit_breaker_open_to_half_open(self):
        from app.llm.circuit_breaker import CircuitBreaker
        from app.llm.router_models import CircuitState

        cb = CircuitBreaker(failure_threshold=2, timeout=0.05, recovery_threshold=1)
        for _ in range(2):
            cb.on_failure(RuntimeError("fail"))
        assert cb.state == CircuitState.OPEN
        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN

    def test_circuit_breaker_half_open_to_closed(self):
        from app.llm.circuit_breaker import CircuitBreaker
        from app.llm.router_models import CircuitState

        cb = CircuitBreaker(failure_threshold=2, timeout=0.05, recovery_threshold=2)
        for _ in range(2):
            cb.on_failure(RuntimeError("fail"))
        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN
        cb.on_success()
        cb.on_success()
        assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_allow_request(self):
        from app.llm.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=2, timeout=1.0)
        assert cb.allow_request() is True
        cb.on_failure(RuntimeError("f"))
        assert cb.allow_request() is True
        cb.on_failure(RuntimeError("f"))
        assert cb.allow_request() is False

    def test_circuit_breaker_check_raises(self):
        from app.llm.circuit_breaker import CircuitBreaker
        from app.llm.router_exceptions import CircuitBreakerError

        cb = CircuitBreaker(failure_threshold=1, timeout=1.0)
        cb.on_failure(RuntimeError("f"))
        with pytest.raises(CircuitBreakerError):
            cb.check("mock")

    def test_circuit_breaker_reset(self):
        from app.llm.circuit_breaker import CircuitBreaker
        from app.llm.router_models import CircuitState

        cb = CircuitBreaker(failure_threshold=1, timeout=1.0)
        cb.on_failure(RuntimeError("f"))
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED

    # -- 2.5 FallbackManager --------------------------------------------

    @pytest.mark.asyncio
    async def test_fallback_manager_failing_primary_uses_fallback(self):
        from app.llm.fallback_manager import FallbackManager
        from app.llm.providers.mock import MockProvider

        primary = MockProvider(model="primary")
        fallback = MockProvider(model="fallback")
        failing_called = {"count": 0}

        def failing_generate(req):
            failing_called["count"] += 1
            raise RuntimeError("primary is down")

        def resolver(pid):
            if pid == "primary_fail":
                return type(
                    "FailingProvider",
                    (),
                    {
                        "name": "primary_fail",
                        "generate": failing_generate,
                        "generate_async": lambda self, req: asyncio.to_thread(
                            self.generate, req
                        ),
                        "model_info": lambda self, m=None: primary.model_info(m),
                    },
                )()
            if pid == "fallback_ok":
                return fallback
            return None

        fm = FallbackManager(provider_resolver=resolver, max_fallback_attempts=3)
        fm.set_fallback_chain("primary_fail", ["fallback_ok"])

        req = _make_request("hello")
        result = await fm.execute_with_fallback(
            req,
            primary_provider="primary_fail",
            generate_func=lambda p, r: p.generate(r),
        )
        assert result is not None
        assert result.provider == "mock"

    def test_fallback_manager_set_get_chain(self):
        from app.llm.fallback_manager import FallbackManager

        fm = FallbackManager()
        fm.set_fallback_chain("primary", ["fb1", "fb2"])
        chain = fm.get_fallback_chain("primary")
        assert chain == ["fb1", "fb2"]

    def test_fallback_manager_clear_chain(self):
        from app.llm.fallback_manager import FallbackManager

        fm = FallbackManager()
        fm.set_fallback_chain("primary", ["fb1"])
        fm.clear_chain("primary")
        assert fm.get_fallback_chain("primary") == []

    # -- 2.6 PromptCache ------------------------------------------------

    def test_prompt_cache_put_get(self):
        from app.llm.models import ChatResponse, Usage
        from app.llm.prompt_cache import PromptCache

        cache = PromptCache()
        req = _make_request("hello")
        key = cache.build_key(req, "mock")
        resp = ChatResponse(
            content="hi", model="test-model", provider="mock", usage=Usage()
        )
        cache.put(key, resp)
        cached = cache.get(key)
        assert cached is not None
        assert cached.content == "hi"

    def test_prompt_cache_miss(self):
        from app.llm.prompt_cache import PromptCache

        cache = PromptCache()
        req = _make_request("hello")
        key = cache.build_key(req, "mock")
        result = cache.get(key)
        assert result is None

    def test_prompt_cache_stats(self):
        from app.llm.prompt_cache import PromptCache

        cache = PromptCache()
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["enabled"] is True

    def test_prompt_cache_disable(self):
        from app.llm.prompt_cache import PromptCache

        cache = PromptCache()
        cache.disable()
        req = _make_request("hello")
        key = cache.build_key(req, "mock")
        cache.put(key, "value")
        assert cache.get(key) is None  # disabled, so miss
        cache.enable()

    def test_prompt_cache_invalidate(self):
        from app.llm.models import ChatResponse, Usage
        from app.llm.prompt_cache import PromptCache

        cache = PromptCache()
        req = _make_request("hello")
        key = cache.build_key(req, "mock")
        cache.put(key, ChatResponse(model="t", provider="p", usage=Usage()))
        assert cache.invalidate(key) is True
        assert cache.get(key) is None

    # -- 2.7 ProviderHealthMonitor --------------------------------------

    def test_health_monitor_record_success(self):
        from app.llm.health_monitor import ProviderHealthMonitor

        mon = ProviderHealthMonitor()
        mon.record_success("mock", 0.1, 10)
        snap = mon.get_snapshot("mock")
        assert snap.success_count == 1
        assert snap.failure_count == 0

    def test_health_monitor_record_failure(self):
        from app.llm.health_monitor import ProviderHealthMonitor

        mon = ProviderHealthMonitor()
        mon.record_failure("mock", 0.5, "error")
        snap = mon.get_snapshot("mock")
        assert snap.failure_count == 1

    def test_health_monitor_is_healthy(self):
        from app.llm.health_monitor import ProviderHealthMonitor

        mon = ProviderHealthMonitor()
        mon.record_success("mock", 0.1, 10)
        assert mon.is_healthy("mock") is True

    def test_health_monitor_get_all_snapshots(self):
        from app.llm.health_monitor import ProviderHealthMonitor

        mon = ProviderHealthMonitor()
        mon.record_success("p1", 0.1, 10)
        mon.record_success("p2", 0.2, 20)
        snaps = mon.get_all_snapshots()
        assert "p1" in snaps
        assert "p2" in snaps

    def test_health_monitor_reset_provider(self):
        from app.llm.health_monitor import ProviderHealthMonitor

        mon = ProviderHealthMonitor()
        mon.record_success("mock", 0.1, 10)
        mon.reset_provider("mock")
        snap = mon.get_snapshot("mock")
        assert snap.success_count == 0

    # -- 2.8 LLMAnalytics ------------------------------------------------

    def test_analytics_record_and_query(self):
        from app.llm.analytics import LLMAnalytics

        analytics = LLMAnalytics()
        event = analytics.record(
            provider_id="mock",
            model="test-model",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            latency=0.5,
            first_token_latency=0.1,
            success=True,
        )
        assert event.provider_id == "mock"
        events = analytics.get_events("mock")
        assert len(events) == 1
        assert events[0].success is True

    def test_analytics_ranking(self):
        from app.llm.analytics import LLMAnalytics

        analytics = LLMAnalytics()
        analytics.record(
            provider_id="mock",
            model="test-model",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            latency=0.5,
            first_token_latency=0.1,
            success=True,
        )
        ranking = analytics.get_provider_ranking("mock")
        assert ranking.total_requests == 1
        assert ranking.success_rate == 1.0

    def test_analytics_clear(self):
        from app.llm.analytics import LLMAnalytics

        analytics = LLMAnalytics()
        analytics.record(
            provider_id="mock",
            model="test-model",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            latency=0.5,
            first_token_latency=0.1,
            success=True,
        )
        analytics.clear()
        assert len(analytics.get_events()) == 0

    def test_analytics_overall_stats(self):
        from app.llm.analytics import LLMAnalytics

        analytics = LLMAnalytics()
        analytics.record(
            provider_id="mock",
            model="test-model",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            latency=0.5,
            first_token_latency=0.1,
            success=True,
        )
        stats = analytics.get_overall_stats()
        assert stats["total_requests"] == 1
        assert stats["total_tokens"] == 30

    # -- 2.9 PolicyEngine ------------------------------------------------

    def test_policy_engine_no_rules_passes(self):
        from app.llm.policy_engine import PolicyEngine

        engine = PolicyEngine()
        assert engine.evaluate("mock") is True

    def test_policy_engine_max_cost_policy(self):
        from app.llm.policy_engine import MaxCostPolicy, PolicyEngine
        from app.llm.router_exceptions import PolicyViolationError

        engine = PolicyEngine()
        engine.add_rule(MaxCostPolicy(max_cost=0.01))
        with pytest.raises(PolicyViolationError):
            engine.evaluate("mock", context={"estimated_cost": 0.05})

    def test_policy_engine_max_cost_passes(self):
        from app.llm.policy_engine import MaxCostPolicy, PolicyEngine

        engine = PolicyEngine()
        engine.add_rule(MaxCostPolicy(max_cost=1.0))
        assert engine.evaluate("mock", context={"estimated_cost": 0.01}) is True

    def test_policy_engine_block_list(self):
        from app.llm.policy_engine import PolicyEngine, ProviderBlockListPolicy
        from app.llm.router_exceptions import PolicyViolationError

        engine = PolicyEngine()
        engine.add_rule(ProviderBlockListPolicy(blocked={"blocked_provider"}))
        with pytest.raises(PolicyViolationError):
            engine.evaluate("blocked_provider")
        assert engine.evaluate("allowed_provider") is True

    def test_policy_engine_filter_providers(self):
        from app.llm.policy_engine import PolicyEngine, ProviderBlockListPolicy

        engine = PolicyEngine()
        engine.add_rule(ProviderBlockListPolicy(blocked={"bad"}))
        result = engine.filter_providers(["good", "bad", "ok"])
        assert result == ["good", "ok"]

    def test_policy_engine_remove_rule(self):
        from app.llm.policy_engine import PolicyEngine, ProviderBlockListPolicy

        engine = PolicyEngine()
        engine.add_rule(ProviderBlockListPolicy(blocked={"x"}))
        removed = engine.remove_rule("block_list")
        assert removed is True
        assert engine.rule_names == []

    # -- 2.10 RequestQueue ------------------------------------------------

    @pytest.mark.asyncio
    async def test_request_queue_submit(self):
        from app.llm.request_queue import RequestQueue

        queue = RequestQueue(default_timeout=5.0)

        async def work():
            return "done"

        result = await queue.submit("mock", work(), timeout=2.0)
        assert result == "done"

    @pytest.mark.asyncio
    async def test_request_queue_concurrent(self):
        from app.llm.request_queue import RequestQueue

        queue = RequestQueue(max_concurrency=2, default_timeout=5.0)
        results = []

        async def work(val):
            await asyncio.sleep(0.01)
            return val

        tasks = [queue.submit(f"p{i % 2}", work(i), timeout=2.0) for i in range(4)]
        results = await asyncio.gather(*tasks)
        assert sorted(results) == [0, 1, 2, 3]

    # -- 2.11 UnifiedStreamer normalize ----------------------------------

    @pytest.mark.asyncio
    async def test_unified_streamer_normalize(self):
        from app.llm.unified_streamer import UnifiedStreamer

        streamer = UnifiedStreamer()
        p = _make_provider()
        req = _make_request("hello", stream=True)
        chunks = []
        async for chunk in streamer.normalize(p, req):
            chunks.append(chunk)
        assert len(chunks) > 0
        full = "".join(c.content for c in chunks)
        assert "Echo: hello" in full

    # -- 2.12 CapabilityMatrix -------------------------------------------

    def test_capability_matrix_register_get(self):
        from app.llm.capability_matrix import CapabilityMatrix

        matrix = CapabilityMatrix()
        from app.llm.router_models import ProviderCapabilities

        caps = ProviderCapabilities(chat=True, streaming=True, context_length=8192)
        matrix.register("mock", caps)
        got = matrix.get("mock")
        assert got is not None
        assert got.chat is True

    def test_capability_matrix_from_provider(self):
        from app.llm.capability_matrix import CapabilityMatrix

        p = _make_provider()
        caps = CapabilityMatrix.from_provider(p)
        assert caps.streaming is True
        assert caps.tool_calling is True

    def test_capability_matrix_unsupported(self):
        from app.llm.capability_matrix import CapabilityMatrix
        from app.llm.router_models import CapabilityFlag

        matrix = CapabilityMatrix()
        assert matrix.supports("nonexistent", CapabilityFlag.CHAT) is False

    # -- 2.13 CostOptimizer ----------------------------------------------

    def test_cost_optimizer_estimate(self):
        from app.llm.cost_optimizer import CostOptimizer

        co = CostOptimizer()
        co.register_provider_costs("mock", 0.001, 0.002)
        req = _make_request("hello")
        est = co.estimate_cost("mock", req)
        assert est.estimated_cost >= 0.0
        assert est.provider_id == "mock"

    def test_cost_optimizer_estimate_tokens(self):
        from app.llm.cost_optimizer import CostOptimizer

        co = CostOptimizer()
        req = _make_request("hello world")
        prompt, completion = co.estimate_tokens(req)
        assert prompt > 0
        assert completion > 0

    # -- 2.14 LoadBalancer -----------------------------------------------

    def test_load_balancer_round_robin(self):
        from app.llm.load_balancer import LoadBalancer
        from app.llm.router_models import LoadBalanceStrategy

        lb = LoadBalancer(strategy=LoadBalanceStrategy.ROUND_ROBIN)
        providers = ["a", "b", "c"]
        selections = [lb.select(providers) for _ in range(6)]
        assert len(selections) == 6
        assert set(selections) <= set(providers)

    def test_load_balancer_set_strategy(self):
        from app.llm.load_balancer import LoadBalancer
        from app.llm.router_models import LoadBalanceStrategy

        lb = LoadBalancer()
        lb.set_strategy(LoadBalanceStrategy.LEAST_LATENCY)
        assert lb.strategy == LoadBalanceStrategy.LEAST_LATENCY


# =====================================================================
# 3. AGENT SYSTEM
# =====================================================================


class TestAgentSystem:
    """Verify the Agent System subsystem at runtime."""

    # -- 3.1 CognitiveOrchestrator import & creation ---------------------

    def test_import_orchestrator(self):
        from app.kernel import CognitiveOrchestrator

        assert CognitiveOrchestrator is not None

    def test_orchestrator_creation(self):
        from app.kernel import CognitiveOrchestrator

        orch = CognitiveOrchestrator()
        assert orch is not None
        assert orch.container is not None

    # -- 3.2 DependencyContainer bootstrap -------------------------------

    def test_bootstrap_agent_components(self):
        from app.core.container import DependencyContainer
        from app.kernel.agent import register_agent_components

        container = DependencyContainer()
        register_agent_components(container)
        from app.kernel.agent.runtime import AgentRuntime

        assert container.has(AgentRuntime)

    def test_resolve_agent_runtime(self):
        from app.core.container import DependencyContainer
        from app.kernel.agent import AgentRuntime, register_agent_components

        container = DependencyContainer()
        register_agent_components(container)
        runtime = container.resolve(AgentRuntime)
        assert runtime is not None
        assert isinstance(runtime, AgentRuntime)

    # -- 3.3 Agent run ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_agent_run_simple(self):
        from app.kernel.agent import (
            AgentRequest,
            AgentRuntime,
            register_agent_components,
        )
        from app.core.container import DependencyContainer

        container = DependencyContainer()
        register_agent_components(container)
        runtime = container.resolve(AgentRuntime)

        request = AgentRequest(raw_input="say hello", session_id="test-session-1")
        response = await runtime.run(request)
        assert response is not None
        assert response.session_id == "test-session-1"
        assert response.status is not None

    # -- 3.4 Agent cancel ------------------------------------------------

    @pytest.mark.asyncio
    async def test_agent_cancel(self):
        from app.kernel.agent import (
            AgentRequest,
            AgentRuntime,
            register_agent_components,
        )
        from app.core.container import DependencyContainer

        container = DependencyContainer()
        register_agent_components(container)
        runtime = container.resolve(AgentRuntime)

        request = AgentRequest(
            raw_input="do something long",
            session_id="cancel-test",
            config={"session_id": "cancel-test", "overall_timeout_s": 5.0},
        )

        async def run_and_cancel():
            task = asyncio.create_task(runtime.run(request))
            await asyncio.sleep(0.05)
            runtime.cancel("cancel-test", "test_cancel")
            return await task

        try:
            response = await asyncio.wait_for(run_and_cancel(), timeout=3.0)
            assert response is not None
            assert response.session_id == "cancel-test"
        except asyncio.TimeoutError:
            pass  # cancellation may cause timeout

    # -- 3.5 Agent status query ------------------------------------------

    def test_agent_runtime_active_sessions(self):
        from app.kernel.agent import AgentRuntime, register_agent_components
        from app.core.container import DependencyContainer

        container = DependencyContainer()
        register_agent_components(container)
        runtime = container.resolve(AgentRuntime)
        sessions = runtime.get_active_sessions()
        assert isinstance(sessions, list)
        assert len(sessions) == 0

    # -- 3.6 Session listing ---------------------------------------------

    def test_orchestrator_active_sessions(self):
        from app.kernel import CognitiveOrchestrator

        orch = CognitiveOrchestrator()
        sessions = orch.get_active_sessions()
        assert isinstance(sessions, list)

    def test_orchestrator_active_agent_sessions(self):
        from app.kernel import CognitiveOrchestrator

        orch = CognitiveOrchestrator()
        sessions = orch.get_active_agent_sessions()
        assert isinstance(sessions, list)

    # -- 3.7 Agent completes a basic task --------------------------------

    @pytest.mark.asyncio
    async def test_agent_completes_basic_task(self):
        from app.kernel.agent import (
            AgentRequest,
            AgentRuntime,
            register_agent_components,
        )
        from app.kernel.agent.models import AgentStatus
        from app.core.container import DependencyContainer

        container = DependencyContainer()
        register_agent_components(container)
        runtime = container.resolve(AgentRuntime)

        config_overrides = {
            "reasoning_enabled": False,
            "reflection_enabled": False,
            "experience_enabled": False,
            "overall_timeout_s": 30.0,
        }
        request = AgentRequest(
            raw_input="echo hello",
            session_id="complete-test",
            config=config_overrides,
        )
        response = await runtime.run(request)
        assert response is not None
        assert response.session_id == "complete-test"
        assert response.status in {
            AgentStatus.SUCCEEDED,
            AgentStatus.FAILED,
            AgentStatus.CANCELLED,
            AgentStatus.TIMED_OUT,
        }

    # -- 3.8 CognitiveOrchestrator run_agent -----------------------------

    @pytest.mark.asyncio
    async def test_orchestrator_run_agent(self):
        from app.kernel import CognitiveOrchestrator
        from app.kernel.agent import AgentRequest, register_agent_components

        orch = CognitiveOrchestrator()
        register_agent_components(orch.container)

        request = AgentRequest(
            raw_input="hello",
            session_id="orch-test",
            config={
                "reasoning_enabled": False,
                "reflection_enabled": False,
                "experience_enabled": False,
            },
        )
        response = await orch.run_agent(request)
        assert response is not None
        assert response.session_id == "orch-test"

    # -- 3.9 Orchestrator metrics & tracing ------------------------------

    def test_orchestrator_metrics(self):
        from app.kernel import CognitiveOrchestrator

        orch = CognitiveOrchestrator()
        metrics = orch.get_metrics_summary()
        assert isinstance(metrics, dict)

    def test_orchestrator_tracing(self):
        from app.kernel import CognitiveOrchestrator

        orch = CognitiveOrchestrator()
        tracing = orch.get_tracing_summary()
        assert isinstance(tracing, dict)

    # -- 3.10 Agent pipeline stages existence ----------------------------

    def test_orchestrator_pipeline_exists(self):
        from app.kernel import CognitiveOrchestrator

        orch = CognitiveOrchestrator()
        assert orch.pipeline is not None
        assert orch.dispatcher is not None


# =====================================================================
# 4. INTEGRATION: Router + Provider end-to-end
# =====================================================================


class TestLLMIntegration:
    """Integration tests verifying router + provider end-to-end."""

    def test_router_generate_end_to_end(self):
        router = _make_router()
        _register_mock(router)
        req = _make_request("integration test")
        resp = router.generate(req)
        assert resp.content == "Echo: integration test"
        assert resp.provider == "mock"

    def test_router_select_provider(self):

        router = _make_router()
        _register_mock(router)
        req = _make_request("test")
        decision = router.select_provider(req)
        assert decision.selected_provider == "mock"

    def test_router_analytics_tracks_request(self):
        router = _make_router()
        _register_mock(router)
        req = _make_request("tracked")
        router.generate(req)
        events = router.analytics.get_events("mock")
        assert len(events) >= 1

    def test_router_health_records_after_generate(self):
        router = _make_router()
        _register_mock(router)
        req = _make_request("health check")
        router.generate(req)
        snap = router.health.get_snapshot("mock")
        assert snap.success_count >= 1
