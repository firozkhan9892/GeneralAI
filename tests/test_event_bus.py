"""Tests for EventBus."""

from __future__ import annotations

import pytest

from app.core.events import EventBus
from app.core.exceptions.event import SubscriptionError
from app.core.interfaces.ievent import Event


# ------------------------------------------------------------------
# Custom events
# ------------------------------------------------------------------
class OrderCreated(Event):
    event_type: str = "order.created"
    order_id: str = ""


class UserLoggedIn(Event):
    event_type: str = "user.login"
    user_id: str = ""


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------
class TestEventBus:
    """Suite for async event bus."""

    @pytest.fixture
    def bus(self) -> EventBus:
        return EventBus()

    @pytest.mark.asyncio
    async def test_publish_reaches_subscriber(self, bus: EventBus) -> None:
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("order.created", handler)
        event = OrderCreated(order_id="123")
        await bus.publish(event)
        assert len(received) == 1
        e = received[0]
        assert isinstance(e, OrderCreated)
        assert e.order_id == "123"

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, bus: EventBus) -> None:
        results: list[str] = []

        async def handler1(event: Event) -> None:
            results.append("h1")

        async def handler2(event: Event) -> None:
            results.append("h2")

        bus.subscribe("order.created", handler1)
        bus.subscribe("order.created", handler2)
        await bus.publish(OrderCreated(order_id="1"))
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_subscriber_not_called_for_other_events(self, bus: EventBus) -> None:
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("order.created", handler)
        await bus.publish(UserLoggedIn(user_id="abc"))
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_handler(self, bus: EventBus) -> None:
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("order.created", handler)
        bus.unsubscribe("order.created", handler)
        await bus.publish(OrderCreated(order_id="1"))
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_wildcard_subscriber(self, bus: EventBus) -> None:
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("*", handler)
        await bus.publish(OrderCreated(order_id="1"))
        await bus.publish(UserLoggedIn(user_id="abc"))
        assert len(received) == 2

    def test_subscribe_sync_handler_raises(self, bus: EventBus) -> None:
        def sync_handler(event: Event) -> None:
            pass

        with pytest.raises(SubscriptionError):
            bus.subscribe("test", sync_handler)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_handler_error_does_not_block_others(self, bus: EventBus) -> None:
        results: list[str] = []

        async def failing_handler(event: Event) -> None:
            raise RuntimeError("oops")

        async def good_handler(event: Event) -> None:
            results.append("ok")

        bus.subscribe("order.created", failing_handler)
        bus.subscribe("order.created", good_handler)
        await bus.publish(OrderCreated(order_id="1"))
        assert results == ["ok"]

    def test_clear_removes_all_handlers(self, bus: EventBus) -> None:
        async def handler(event: Event) -> None:
            pass

        bus.subscribe("test", handler)
        bus.clear()
        # Should have no handlers
        assert bus._handlers["test"] == []

    @pytest.mark.asyncio
    async def test_duplicate_subscription_warns(self, bus: EventBus) -> None:
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("test", handler)
        bus.subscribe("test", handler)  # second call logs warning, no error
        await bus.publish(Event(event_type="test"))
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_no_handlers_does_not_error(self, bus: EventBus) -> None:
        await bus.publish(Event(event_type="ghost"))
