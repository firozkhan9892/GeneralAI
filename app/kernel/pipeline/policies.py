"""Failure policies for pipeline stage execution.

Defines how the pipeline should react when a stage raises an exception
or returns a non-success result.

Policies:
    - RETRY: Retry the stage up to ``max_retries`` times.
    - ABORT: Stop the entire pipeline immediately.
    - CONTINUE: Skip the failed stage and proceed to the next.
    - FALLBACK: Use a fallback value and continue.

Policies are configurable per-stage and globally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class FailurePolicy(str, Enum):
    """How the pipeline should handle a stage failure."""

    RETRY = "retry"
    ABORT = "abort"
    CONTINUE = "continue"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class StagePolicy:
    """Failure policy configuration for a single pipeline stage."""

    policy: FailurePolicy = FailurePolicy.ABORT
    max_retries: int = 3
    retry_delay_s: float = 0.5
    fallback_factory: Callable[[], Any] | None = field(default=None, repr=False)

    def get_fallback(self) -> Any:
        """Produce a fallback value for this stage.

        Raises:
            RuntimeError: If no fallback factory is configured.
        """
        if self.fallback_factory is None:
            raise RuntimeError("No fallback factory configured for this stage")
        return self.fallback_factory()


@dataclass(frozen=True)
class PolicySet:
    """Global failure policy configuration.

    ``default_policy`` applies to all stages unless overridden.
    ``stage_overrides`` maps stage names to per-stage policies.
    """

    default_policy: StagePolicy = field(
        default_factory=lambda: StagePolicy(policy=FailurePolicy.ABORT, max_retries=3)
    )
    stage_overrides: dict[str, StagePolicy] = field(default_factory=dict)

    def get_policy_for_stage(self, stage_name: str) -> StagePolicy:
        """Return the effective policy for a given stage name."""
        return self.stage_overrides.get(stage_name, self.default_policy)


# ── Predefined policy presets ────────────────────────────────────────────────


def lenient_policy(max_retries: int = 3) -> PolicySet:
    """A policy set that retries and continues on failure."""
    return PolicySet(
        default_policy=StagePolicy(policy=FailurePolicy.RETRY, max_retries=max_retries),
        stage_overrides={},
    )


def strict_policy() -> PolicySet:
    """A policy set that aborts on any failure."""
    return PolicySet(
        default_policy=StagePolicy(policy=FailurePolicy.ABORT, max_retries=0),
        stage_overrides={},
    )


def resilient_policy(max_retries: int = 3) -> PolicySet:
    """A policy set that retries critical stages and continues on non-critical ones."""
    return PolicySet(
        default_policy=StagePolicy(policy=FailurePolicy.RETRY, max_retries=max_retries),
        stage_overrides={
            "response": StagePolicy(policy=FailurePolicy.ABORT, max_retries=0),
            "memory": StagePolicy(policy=FailurePolicy.ABORT, max_retries=0),
        },
    )
