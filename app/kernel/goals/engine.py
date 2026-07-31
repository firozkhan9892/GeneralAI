"""Goal engine — stage 3 of the cognitive pipeline."""

from __future__ import annotations

import logging
from datetime import datetime
from app.kernel.goals.models import (
    Goal,
    GoalHierarchy,
    GoalPriority,
    GoalStatus,
    GoalType,
)
from app.kernel.intent.models import Intent, IntentType

log = logging.getLogger(__name__)

_INTENT_TO_GOAL_TYPE: dict[IntentType, GoalType] = {
    IntentType.ASK_QUESTION: GoalType.QUESTION,
    IntentType.SOLVE_PROBLEM: GoalType.TASK,
    IntentType.EXECUTE_TASK: GoalType.TASK,
    IntentType.PLAN_PROJECT: GoalType.PROJECT,
    IntentType.LEARN: GoalType.LEARNING,
    IntentType.CREATE_CONTENT: GoalType.TASK,
    IntentType.EXPLORE: GoalType.EXPLORATION,
    IntentType.DEBUG: GoalType.DEBUGGING,
    IntentType.META: GoalType.SYSTEM,
    IntentType.CLARIFY: GoalType.SYSTEM,
    IntentType.UNKNOWN: GoalType.TASK,
}

_INTENT_TO_PRIORITY: dict[IntentType, GoalPriority] = {
    IntentType.ASK_QUESTION: GoalPriority.NORMAL,
    IntentType.SOLVE_PROBLEM: GoalPriority.HIGH,
    IntentType.EXECUTE_TASK: GoalPriority.HIGH,
    IntentType.PLAN_PROJECT: GoalPriority.NORMAL,
    IntentType.LEARN: GoalPriority.LOW,
    IntentType.CREATE_CONTENT: GoalPriority.NORMAL,
    IntentType.EXPLORE: GoalPriority.LOW,
    IntentType.DEBUG: GoalPriority.HIGH,
    IntentType.META: GoalPriority.NORMAL,
    IntentType.CLARIFY: GoalPriority.NORMAL,
    IntentType.UNKNOWN: GoalPriority.NORMAL,
}


def _resolve_goal_type(intent_type: IntentType) -> GoalType:
    return _INTENT_TO_GOAL_TYPE.get(intent_type, GoalType.TASK)


def _resolve_priority(intent_type: IntentType) -> GoalPriority:
    return _INTENT_TO_PRIORITY.get(intent_type, GoalPriority.NORMAL)


def _describe_goal(intent_type: IntentType) -> str:
    descriptions: dict[IntentType, str] = {
        IntentType.ASK_QUESTION: "Answer the user's question",
        IntentType.SOLVE_PROBLEM: "Solve the user's problem",
        IntentType.EXECUTE_TASK: "Execute the requested task",
        IntentType.PLAN_PROJECT: "Plan the project",
        IntentType.LEARN: "Facilitate learning",
        IntentType.CREATE_CONTENT: "Create content",
        IntentType.EXPLORE: "Explore the topic",
        IntentType.DEBUG: "Debug the issue",
        IntentType.META: "Handle system interaction",
        IntentType.CLARIFY: "Clarify user intent",
        IntentType.UNKNOWN: "Process unknown intent",
    }
    return descriptions.get(intent_type, "Process intent")


class GoalEngine:
    """Manages the goal lifecycle.

    Converts structured Intent into a Goal hierarchy and tracks
    goal state through completion.
    """

    def __init__(self) -> None:
        self._active_goals: dict[str, Goal] = {}
        self._next_id: int = 0

    async def resolve(self, intent: Intent) -> GoalHierarchy:
        """Convert intent into a goal hierarchy.

        Args:
            intent: Structured intent from the intent engine.

        Returns:
            Goal hierarchy with root goal and sub-goals.
        """
        root = self._build_goal(intent, parent_id=None)
        children: list[Goal] = []
        all_goals: dict[str, Goal] = {root.id: root}

        for sub_intent in intent.sub_intents:
            child = self._build_goal(sub_intent, parent_id=root.id)
            children.append(child)
            all_goals[child.id] = child

        root = root.model_copy(
            update={
                "sub_goal_ids": tuple(c.id for c in children),
            }
        )
        all_goals[root.id] = root

        hierarchy = GoalHierarchy(
            root=root,
            children=tuple(children),
            all_goals=all_goals,
        )

        self._active_goals[root.id] = root
        for child in children:
            self._active_goals[child.id] = child

        log.info(
            "Goal %s (%s) — priority=%s — %d sub-goal(s)",
            root.id,
            root.goal_type.value,
            root.priority.name,
            len(children),
        )

        return hierarchy

    async def update_progress(self, goal_id: str, progress: float) -> None:
        """Update progress toward a goal.

        Args:
            goal_id: The goal identifier.
            progress: Progress value between 0.0 and 1.0.

        Raises:
            ValueError: If goal_id is unknown.
        """
        if goal_id not in self._active_goals:
            raise ValueError(f"Unknown goal: {goal_id}")

        goal = self._active_goals[goal_id]
        updated = goal.model_copy(
            update={"progress": progress, "updated_at": datetime.utcnow()}
        )
        self._active_goals[goal_id] = updated

        log.debug("Goal %s progress=%.2f", goal_id, progress)

    def _next_goal_id(self) -> str:
        self._next_id += 1
        return f"goal_{self._next_id}"

    def _build_goal(self, intent: Intent, parent_id: str | None) -> Goal:
        goal_type = _resolve_goal_type(intent.primary)
        priority = _resolve_priority(intent.primary)
        description = _describe_goal(intent.primary)
        now = datetime.utcnow()

        return Goal(
            id=self._next_goal_id(),
            parent_id=parent_id,
            description=description,
            goal_type=goal_type,
            intent_type=intent.primary,
            priority=priority,
            status=GoalStatus.PROPOSED,
            created_at=now,
            updated_at=now,
        )
