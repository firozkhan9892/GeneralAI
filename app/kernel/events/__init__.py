"""Kernel event definitions and constants.

Re-exports the canonical event constants from ``contracts/events.py``
plus legacy-only constants from ``definitions.py`` for backward
compatibility.
"""

from __future__ import annotations

from app.kernel.contracts.events import (  # noqa: F401 — canonical source
    EVENT_CAPABILITY_RESOLVED,
    EVENT_CAPABILITY_UNAVAILABLE,
    EVENT_DECISION_MADE,
    EVENT_EXECUTION_COMPLETED,
    EVENT_EXECUTION_FAILED,
    EVENT_EXECUTION_STARTED,
    EVENT_EXPERIENCE_RECORDED,
    EVENT_GOAL_COMPLETED,
    EVENT_GOAL_CREATED,
    EVENT_GOAL_DECOMPOSED,
    EVENT_INTENT_IDENTIFIED,
    EVENT_PERCEPTION_COMPLETED,
    EVENT_PERCEPTION_STARTED,
    EVENT_PLAN_CREATED,
    EVENT_PLAN_REVISED,
    EVENT_POLICY_CONFIRMATION_REQUIRED,
    EVENT_POLICY_DENIED,
    EVENT_POLICY_EVALUATED,
    EVENT_REASONING_COMPLETED,
    EVENT_REASONING_STARTED,
    EVENT_REASONING_STEP,
    EVENT_REFLECTION_COMPLETED,
    EVENT_REFLECTION_REFINEMENT,
    EVENT_REFLECTION_STARTED,
    EVENT_STATE_CHANGED,
    EVENT_STATE_ERROR,
)
from app.kernel.events.definitions import (  # noqa: F401 — legacy-only names
    EVENT_DECISION_FAILED,
    EVENT_GOAL_FAILED,
    EVENT_INTENT_CLARIFICATION_REQUESTED,
    EVENT_PLAN_FAILED,
)

__all__ = [k for k in dir() if k.startswith("EVENT_")]
