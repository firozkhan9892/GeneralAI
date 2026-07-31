"""Context manager — cross-cutting."""

from __future__ import annotations

import logging

from app.kernel.context.models import (
    CognitiveContext,
    ContextDelta,
    ContextSnapshot,
    ContextSource,
    TokenBudget,
)

log = logging.getLogger(__name__)


class ContextManager:
    """Owns the session's cognitive context across the entire flow."""

    async def build(self, source: ContextSource) -> CognitiveContext:
        """Build a new context from source data.

        Args:
            source: Source data for building context.

        Returns:
            New cognitive context.

        Raises:
            NotImplementedError: Always — placeholder.
        """
        raise NotImplementedError("ContextManager.build not yet implemented")

    async def update(self, context_id: str, delta: ContextDelta) -> CognitiveContext:
        """Apply an incremental update to a context.

        Args:
            context_id: The context identifier.
            delta: The update to apply.

        Returns:
            Updated cognitive context.

        Raises:
            NotImplementedError: Always — placeholder.
        """
        raise NotImplementedError("ContextManager.update not yet implemented")

    async def snapshot(self, context_id: str) -> ContextSnapshot:
        """Create a snapshot of a context.

        Args:
            context_id: The context identifier.

        Returns:
            A frozen snapshot.

        Raises:
            NotImplementedError: Always — placeholder.
        """
        raise NotImplementedError("ContextManager.snapshot not yet implemented")


class ContextBuilder:
    """Builds context incrementally."""

    async def add_message(self, role: str, content: str) -> None:
        """Add a message to the context.

        Raises:
            NotImplementedError: Always — placeholder.
        """
        raise NotImplementedError("ContextBuilder.add_message not yet implemented")

    async def build(self) -> CognitiveContext:
        """Finalize and return the built context.

        Raises:
            NotImplementedError: Always — placeholder.
        """
        raise NotImplementedError("ContextBuilder.build not yet implemented")


class ContextPruner:
    """Prunes context when token budget is exceeded."""

    async def prune(
        self, context: CognitiveContext, budget: TokenBudget
    ) -> CognitiveContext:
        """Prune context to fit within budget.

        Args:
            context: The context to prune.
            budget: Token budget constraints.

        Returns:
            Pruned context.

        Raises:
            NotImplementedError: Always — placeholder.
        """
        raise NotImplementedError("ContextPruner.prune not yet implemented")
