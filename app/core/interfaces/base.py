"""Base interface that all modules implement."""

from __future__ import annotations

from abc import ABC, abstractmethod


class IModule(ABC):
    """Base contract for every GeneralAI module.

    Provides common lifecycle hooks that the :class:`LifecycleManager`
    invokes during startup and shutdown.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Perform one-time initialisation.

        Called after all configuration is loaded and the DI container
        is ready.  Implementations should acquire resources, open
        connections, or register event handlers here.
        """

    @abstractmethod
    async def shutdown(self) -> None:
        """Release resources and perform cleanup.

        Called during application shutdown.  Must not raise.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a unique human-readable name for this module."""
