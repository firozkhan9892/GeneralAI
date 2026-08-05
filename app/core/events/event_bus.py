"""Asynchronous event bus for GeneralAI.

Strongly-typed pub/sub event system.  Every event is a Pydantic model
and handlers are async callables.  This is the only communication
channel between modules.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from app.core.constants.events import EVENT_BUS_MAX_HANDLERS_PER_EVENT
from app.core.exceptions.event import (
    HandlerError,
    SubscriptionError,
)
from app.core.interfaces.ievent import Event, EventHandler, IEventBus

log = logging.getLogger(__name__)


class EventBus(IEventBus):
    """Default event bus implementation.

    Thread-safe for subscription management.  Publishing is async and
    fans out to all registered handlers concurrently.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def publish(self, event: Event) -> None:
        """Publish *event* to all subscribed handlers.

        Handlers are awaited concurrently.  If any handler raises,
        a :class:`HandlerError` is logged but other handlers are
        not affected.

        Args:
            event: The event to publish.

        Raises:
            PublishError: If the event is invalid or publishing fails.
        """
        event_type = event.event_type or type(event).__name__
        handlers = self._handlers.get(event_type, []) + self._handlers.get("*", [])

        if not handlers:
            log.debug("No handlers for event type '%s'", event_type)
            return

        tasks = []
        for handler in handlers:
            tasks.append(self._safe_dispatch(handler, event))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for handler, result in zip(handlers, results):
            if isinstance(result, Exception):
                log.error(
                    "Handler %s failed for event %s: %s",
                    getattr(handler, "__name__", str(handler)),
                    event_type,
                    result,
                )

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe *handler* to *event_type* events.

        Use ``"*"`` as the event type to receive all events.

        Args:
            event_type: The event type string to subscribe to.
            handler: An async callable that accepts an :class:`Event`.

        Raises:
            SubscriptionError: If the handler limit is exceeded.
        """
        if not asyncio.iscoroutinefunction(handler):
            raise SubscriptionError(
                f"Handler '{getattr(handler, '__name__', str(handler))}' "
                f"must be an async function",
                module="events",
            )

        handlers = self._handlers[event_type]
        if len(handlers) >= EVENT_BUS_MAX_HANDLERS_PER_EVENT:
            raise SubscriptionError(
                f"Max handlers ({EVENT_BUS_MAX_HANDLERS_PER_EVENT}) "
                f"reached for event type '{event_type}'",
                module="events",
            )

        if handler in handlers:
            log.warning("Handler already subscribed to '%s'", event_type)
            return

        handlers.append(handler)
        log.debug("Handler subscribed to '%s'", event_type)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe *handler* from *event_type*.

        Args:
            event_type: The event type to unsubscribe from.
            handler: The handler to remove.
        """
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)
            log.debug("Handler unsubscribed from '%s'", event_type)

    def emit(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Synchronous convenience: schedule an event publish on the event loop.

        Used by synchronous publishers (e.g. :class:`EventPublisher`) that
        cannot ``await`` the async :meth:`publish` directly.  If no event
        loop is running the event is silently dropped.
        """
        event = Event(
            event_type=event_type,
            source="event_bus.emit",
        )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.publish(event))
        except RuntimeError:
            # No running event loop — best-effort drop.
            log.debug("No running loop; event '%s' dropped", event_type)

    def clear(self) -> None:
        """Remove all handlers."""
        self._handlers.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    async def _safe_dispatch(handler: EventHandler, event: Event) -> None:
        """Invoke *handler* and catch any exception."""
        try:
            await handler(event)
        except Exception as exc:
            raise HandlerError(
                f"Handler '{getattr(handler, '__name__', str(handler))}' "
                f"raised an error",
                module="events",
                cause=exc,
                context={"event_type": event.event_type},
            ) from exc
