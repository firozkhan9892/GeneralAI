"""Tests for LifecycleManager."""

from __future__ import annotations

import pytest

from app.core.constants.lifecycle import (
    HOOK_AFTER_CONFIG,
    HOOK_AFTER_PLUGINS,
    HOOK_AFTER_SERVICES,
    HOOK_AFTER_START,
    HOOK_AFTER_STOP,
    HOOK_BEFORE_INIT,
    HOOK_BEFORE_START,
    HOOK_BEFORE_STOP,
    LifecycleStage,
)
from app.core.exceptions.lifecycle import InvalidTransitionError
from app.core.lifecycle import LifecycleManager


class TestLifecycleManager:
    """Suite for lifecycle manager."""

    @pytest.fixture
    def mgr(self) -> LifecycleManager:
        return LifecycleManager()

    @pytest.mark.asyncio
    async def test_initial_stage_is_created(self, mgr: LifecycleManager) -> None:
        assert mgr.current_stage == LifecycleStage.CREATED
        assert mgr.is_running is False

    @pytest.mark.asyncio
    async def test_initialize_reaches_plugins_loaded(
        self, mgr: LifecycleManager
    ) -> None:
        await mgr.initialize()
        assert mgr.current_stage == LifecycleStage.PLUGINS_LOADED
        assert mgr.is_running is False

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, mgr: LifecycleManager) -> None:
        await mgr.initialize()
        await mgr.start()
        assert mgr.current_stage == LifecycleStage.RUNNING
        assert mgr.is_running is True
        await mgr.shutdown()
        assert mgr.current_stage == LifecycleStage.STOPPED
        assert mgr.is_running is False

    @pytest.mark.asyncio
    async def test_invalid_transition_raises(self, mgr: LifecycleManager) -> None:
        with pytest.raises(InvalidTransitionError):
            await mgr._transition(LifecycleStage.RUNNING)

    @pytest.mark.asyncio
    async def test_hooks_execute_in_order(self, mgr: LifecycleManager) -> None:
        order: list[str] = []

        async def hook_a() -> None:
            order.append("a")

        async def hook_b() -> None:
            order.append("b")

        mgr.register_hook(HOOK_BEFORE_INIT, hook_a)
        mgr.register_hook(HOOK_AFTER_CONFIG, hook_b)
        await mgr.initialize()
        assert order == ["a", "b"]

    @pytest.mark.asyncio
    async def test_hook_at_each_stage(self, mgr: LifecycleManager) -> None:
        executed: list[str] = []

        async def _config() -> None:
            executed.append("config")

        async def _services() -> None:
            executed.append("services")

        async def _plugins() -> None:
            executed.append("plugins")

        async def _before_start() -> None:
            executed.append("before_start")

        async def _after_start() -> None:
            executed.append("after_start")

        async def _before_stop() -> None:
            executed.append("before_stop")

        async def _after_stop() -> None:
            executed.append("after_stop")

        mgr.register_hook(HOOK_AFTER_CONFIG, _config)
        mgr.register_hook(HOOK_AFTER_SERVICES, _services)
        mgr.register_hook(HOOK_AFTER_PLUGINS, _plugins)
        mgr.register_hook(HOOK_BEFORE_START, _before_start)
        mgr.register_hook(HOOK_AFTER_START, _after_start)
        mgr.register_hook(HOOK_BEFORE_STOP, _before_stop)
        mgr.register_hook(HOOK_AFTER_STOP, _after_stop)

        await mgr.initialize()
        await mgr.start()
        await mgr.shutdown()

        assert executed == [
            "config",
            "services",
            "plugins",
            "before_start",
            "after_start",
            "before_stop",
            "after_stop",
        ]

    @pytest.mark.asyncio
    async def test_shutdown_from_initialized(self, mgr: LifecycleManager) -> None:
        """Should gracefully shutdown even if start was not called."""
        await mgr.initialize()
        await mgr.shutdown()
        assert mgr.current_stage == LifecycleStage.STOPPED

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self, mgr: LifecycleManager) -> None:
        await mgr.initialize()
        await mgr.shutdown()
        await mgr.shutdown()  # second call should be no-op
        assert mgr.current_stage == LifecycleStage.STOPPED

    def test_register_sync_hook_raises(self, mgr: LifecycleManager) -> None:
        def sync_hook() -> None:
            pass

        with pytest.raises(ValueError, match="async"):
            mgr.register_hook(HOOK_AFTER_CONFIG, sync_hook)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_unregister_hook(self, mgr: LifecycleManager) -> None:
        executed: list[str] = []

        async def hook() -> None:
            executed.append("ran")

        mgr.register_hook(HOOK_AFTER_CONFIG, hook)
        mgr.unregister_hook(HOOK_AFTER_CONFIG, hook)
        await mgr.initialize()
        assert executed == []
