"""Reflection engine — stage 12 of the cognitive pipeline."""

from __future__ import annotations

import logging

from app.kernel.reflection.models import ReflectionRequest, ReflectionReport
from app.kernel.reflection.modes import IReflectionStrategy
from app.kernel.reflection.modes.base import FallbackReflectionStrategy

log = logging.getLogger(__name__)


class ReflectionEngine:
    """Self-evaluation and quality improvement subsystem.

    Evaluates output quality, detects errors, and generates
    refinement suggestions. Independent from the Experience Engine.
    """

    def __init__(self) -> None:
        self._modes: dict[str, IReflectionStrategy] = {}
        self._modes["fallback"] = FallbackReflectionStrategy()

    async def evaluate(self, request: ReflectionRequest) -> ReflectionReport:
        """Evaluate output quality.

        Args:
            request: Reflection request with output and trace.

        Returns:
            Reflection result with scores and errors.
        """
        mode_name = request.mode
        strategy = self._modes.get(mode_name)
        if strategy is None:
            strategy = self._modes.get("fallback")
        if strategy is None:
            log.warning("No reflection strategy available, returning default report")
            return ReflectionReport(
                overall_score=0.5,
                verdict="needs_review",
                token_cost=5,
            )
        report = await strategy.evaluate(request)
        log.info(
            "Reflection complete — mode=%s, score=%.2f, verdict=%s",
            mode_name,
            report.overall_score,
            report.verdict,
        )
        return report

    def register_mode(self, name: str, mode: IReflectionStrategy) -> None:
        """Register a reflection mode."""
        self._modes[name] = mode

    def unregister_mode(self, name: str) -> None:
        """Unregister a reflection mode.

        Args:
            name: Mode name to remove.

        Raises:
            KeyError: If the mode is not registered.
        """
        if name not in self._modes:
            raise KeyError(f"Reflection mode '{name}' is not registered")
        del self._modes[name]
