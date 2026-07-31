"""Event-related interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Coroutine

from pydantic import BaseModel, ConfigDict, Field


class Event(BaseModel):
    """Base class for all events flowing through the event bus.

    Every event carries a unique ID, timestamp, source identifier,
    and an event type string used for routing.
    """

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default_factory=lambda: str(__import__("uuid").uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = Field(default="")
    event_type: str = Field(default="")


# Type alias for async event handlers
EventHandler = Callable[..., Coroutine[Any, Any, None]]


class IEventBus(ABC):
    """Contract for the event bus implementation."""

    @abstractmethod
    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribed handlers."""

    @abstractmethod
    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Register a handler for *event_type*."""

    @abstractmethod
    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Remove a previously registered handler."""
