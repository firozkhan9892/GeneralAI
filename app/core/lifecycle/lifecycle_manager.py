"""Application lifecycle manager.

Orchestrates startup → run → shutdown as a deterministic state
machine with extensible hook points at each transition.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Coroutine

from app.core.constants.lifecycle import (
    HOOK_BEFORE_INIT,
    HOOK_AFTER_CONFIG,
    HOOK_AFTER_SERVICES,
    HOOK_AFTER_PLUGINS,
    HOOK_BEFORE_START,
    HOOK_AFTER_START,
    HOOK_BEFORE_STOP,
    HOOK_AFTER_STOP,
    LIFECYCLE_TRANSITIONS,
    SHUTDOWN_TRANSITIONS,
    LifecycleStage,
)
from app.core.exceptions.lifecycle import (
    HookExecutionError,
    InvalidTransitionError,
)

log = logging.getLogger(__name__)

# Type alias for lifecycle hooks
Hook = Callable[[], Coroutine[Any, Any, None]]


class LifecycleManager:
    """Deterministic application lifecycle manager.

    Usage::

        mgr = LifecycleManager()
        mgr.register_hook(HOOK_AFTER_CONFIG, my_async_fn)
        await mgr.run()
    """

    def __init__(self) -> None:
        self._current_stage: LifecycleStage = LifecycleStage.CREATED
        self._hooks: dict[str, list[Hook]] = defaultdict(list)
        self._running = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_stage(self) -> LifecycleStage:
        """Return the current lifecycle stage."""
        return self._current_stage

    @property
    def is_running(self) -> bool:
        """Return ``True`` if the application has reached RUNNING stage."""
        return self._running

    # ------------------------------------------------------------------
    # Lifecycle orchestration
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Execute the full lifecycle: init → start → idle → shutdown.

        Blocks until shutdown is triggered (by :meth:`stop` or
        ``KeyboardInterrupt``).
        """
        try:
            await self.initialize()
            await self.start()
            await self._idle()
        except KeyboardInterrupt:
            log.info("Keyboard interrupt received")
        except Exception:
            log.exception("Unhandled error during lifecycle")
            raise
        finally:
            await self.shutdown()

    async def initialize(self) -> None:
        """Execute the initialisation sequence: config → services → plugins."""
        await self._transition(LifecycleStage.CONFIG_LOADING)
        await self._run_hooks(HOOK_BEFORE_INIT)
        await self._transition(LifecycleStage.CONFIG_LOADED)
        await self._run_hooks(HOOK_AFTER_CONFIG)

        await self._transition(LifecycleStage.SERVICES_INITIALIZING)
        await self._transition(LifecycleStage.SERVICES_INITIALIZED)
        await self._run_hooks(HOOK_AFTER_SERVICES)

        await self._transition(LifecycleStage.PLUGINS_LOADING)
        await self._transition(LifecycleStage.PLUGINS_LOADED)
        await self._run_hooks(HOOK_AFTER_PLUGINS)

    async def start(self) -> None:
        """Transition from initialised to running."""
        await self._transition(LifecycleStage.STARTING)
        await self._run_hooks(HOOK_BEFORE_START)
        await self._transition(LifecycleStage.RUNNING)
        await self._run_hooks(HOOK_AFTER_START)
        self._running = True
        log.info("Application is running")

    async def shutdown(self) -> None:
        """Execute a graceful shutdown."""
        if self._current_stage in (LifecycleStage.STOPPED, LifecycleStage.CREATED):
            return

        await self._run_hooks(HOOK_BEFORE_STOP)
        await self._transition(LifecycleStage.STOPPING)
        await self._transition(LifecycleStage.STOPPED)
        await self._run_hooks(HOOK_AFTER_STOP)
        self._running = False
        log.info("Application stopped")

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def register_hook(self, hook_point: str, hook: Hook) -> None:
        """Register an async callable to run at *hook_point*.

        Args:
            hook_point: One of the ``HOOK_*`` constants from
                :mod:`app.core.constants.lifecycle`.
            hook: Async callable taking no arguments.
        """
        if not asyncio.iscoroutinefunction(hook):
            raise ValueError(f"Hook must be an async function, got {type(hook)}")
        self._hooks[hook_point].append(hook)

    def unregister_hook(self, hook_point: str, hook: Hook) -> None:
        """Remove a previously registered hook."""
        hooks = self._hooks.get(hook_point, [])
        if hook in hooks:
            hooks.remove(hook)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _transition(self, target: LifecycleStage) -> None:
        """Validate and apply a stage transition."""
        allowed = list(LIFECYCLE_TRANSITIONS.get(self._current_stage, []))
        allowed.extend(SHUTDOWN_TRANSITIONS.get(self._current_stage, []))
        if target not in allowed:
            raise InvalidTransitionError(
                f"Cannot transition from '{self._current_stage.value}' "
                f"to '{target.value}'",
                module="lifecycle",
                context={
                    "current": self._current_stage.value,
                    "target": target.value,
                },
            )
        log.debug("Lifecycle: %s → %s", self._current_stage.value, target.value)
        self._current_stage = target

    async def _run_hooks(self, hook_point: str) -> None:
        """Execute all hooks registered at *hook_point* sequentially."""
        for hook in self._hooks.get(hook_point, []):
            try:
                await hook()
            except Exception as exc:
                raise HookExecutionError(
                    f"Hook '{getattr(hook, '__name__', str(hook))}' "
                    f"failed at point '{hook_point}'",
                    module="lifecycle",
                    cause=exc,
                    context={"hook_point": hook_point},
                ) from exc

    async def _idle(self) -> None:
        """Block until :meth:`stop` is called."""
        while self._running:
            await asyncio.sleep(0.5)
