"""Event bus exceptions."""

from __future__ import annotations

from app.core.exceptions.base import GeneralAIError


class EventError(GeneralAIError):
    """Base for all event bus errors."""


class SubscriptionError(EventError):
    """Raised when subscribing or unsubscribing a handler fails."""


class PublishError(EventError):
    """Raised when publishing an event fails."""


class HandlerError(EventError):
    """Raised when an individual handler raises during event dispatch."""
