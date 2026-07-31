"""Experience engine — stage 13 of the cognitive pipeline."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from app.kernel.experience.models import (
    Experience,
    ExperienceQuery,
    Insight,
)

log = logging.getLogger(__name__)


class InMemoryExperienceStore:
    """Deterministic in-memory experience storage."""

    def __init__(self) -> None:
        self._records: dict[str, Experience] = {}
        self._counter: int = 0

    async def save(self, record: Experience) -> str:
        if record.id and record.id in self._records:
            return record.id
        if not record.id:
            self._counter += 1
            exp_id = f"exp_{self._counter}"
            record = record.model_copy(update={"id": exp_id})
        self._records[record.id] = record
        return record.id

    async def query(self, query: ExperienceQuery) -> list[Experience]:
        results = list(self._records.values())

        if query.goal_types is not None:
            gt_set = set(query.goal_types)
            results = [r for r in results if r.goal_type in gt_set]

        if query.skills is not None:
            skill_set = set(query.skills)
            results = [r for r in results if any(s in skill_set for s in r.skills_used)]

        if query.success is not None:
            results = [r for r in results if r.success == query.success]

        if query.timeframe_hours is not None:
            cutoff = datetime.utcnow() - timedelta(hours=query.timeframe_hours)
            results = [r for r in results if r.timestamp >= cutoff]

        results.sort(key=lambda r: r.timestamp, reverse=True)
        return results[: query.limit]

    def get_all(self) -> list[Experience]:
        return list(self._records.values())

    def count(self) -> int:
        return len(self._records)


class ExperienceEngine:
    """Cross-session learning subsystem.

    Records structured outcomes and extracts lessons for future
    sessions. Independent from the Reflection Engine.
    """

    def __init__(self) -> None:
        self._store = InMemoryExperienceStore()

    async def record(self, experience: Experience) -> str:
        """Record a session experience.

        Args:
            experience: The experience record to persist.

        Returns:
            The experience record ID.
        """
        exp_id = await self._store.save(experience)
        log.info(
            "Experience recorded — id=%s, goal=%s",
            exp_id,
            experience.goal_type,
        )
        return exp_id

    async def search(self, query: ExperienceQuery) -> list[Experience]:
        """Retrieve experiences matching a query.

        Args:
            query: Query parameters.

        Returns:
            List of matching experience records.
        """
        return await self._store.query(query)

    async def retrieve(self, query: ExperienceQuery) -> list[Experience]:
        """Alias for search — retained for backward compatibility."""
        return await self.search(query)

    async def summarize(self) -> dict[str, Any]:
        """Produce an aggregated summary of all stored experiences.

        Returns:
            Dict with total_experiences, success/failure counts,
            average_outcome_score, goal_type_counts, common_skills,
            total_lessons, and lesson_category_counts.
        """
        all_exp = self._store.get_all()
        total = len(all_exp)

        if total == 0:
            return {
                "total_experiences": 0,
                "success_count": 0,
                "failure_count": 0,
                "average_outcome_score": 0.0,
                "goal_type_counts": {},
                "common_skills": [],
                "total_lessons": 0,
                "lesson_category_counts": {},
            }

        successes = sum(1 for e in all_exp if e.success)
        avg_score = sum(e.outcome_score for e in all_exp) / total

        goal_counts: dict[str, int] = {}
        skill_counts: dict[str, int] = {}
        lesson_cat_counts: dict[str, int] = {}

        for exp in all_exp:
            gt = exp.goal_type.value
            goal_counts[gt] = goal_counts.get(gt, 0) + 1
            for skill in exp.skills_used:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
            for lesson in exp.lessons:
                lc = lesson.category.value
                lesson_cat_counts[lc] = lesson_cat_counts.get(lc, 0) + 1

        top_skills = sorted(skill_counts.items(), key=lambda x: (-x[1], x[0]))[:5]

        return {
            "total_experiences": total,
            "success_count": successes,
            "failure_count": total - successes,
            "average_outcome_score": round(avg_score, 4),
            "goal_type_counts": dict(sorted(goal_counts.items())),
            "common_skills": [s for s, _ in top_skills],
            "total_lessons": sum(len(e.lessons) for e in all_exp),
            "lesson_category_counts": dict(sorted(lesson_cat_counts.items())),
        }

    async def get_insights(self, goal_type: str) -> list[Insight]:
        """Get aggregated insights for a goal type.

        Args:
            goal_type: The goal type value to get insights for.

        Returns:
            List of Insight objects derived from matching experiences.
        """
        all_exp = [e for e in self._store.get_all() if e.goal_type.value == goal_type]

        if not all_exp:
            return []

        lessons_by_cat: dict[str, list[str]] = {}
        for exp in all_exp:
            for lesson in exp.lessons:
                cat = lesson.category.value
                if cat not in lessons_by_cat:
                    lessons_by_cat[cat] = []
                lessons_by_cat[cat].append(lesson.description)

        insights: list[Insight] = []
        for cat, descriptions in lessons_by_cat.items():
            unique_descs = list(set(descriptions))
            pattern = f"{cat}: {', '.join(unique_descs[:3])}"
            confidence = min(1.0, len(descriptions) / 5.0)
            insights.append(
                Insight(
                    description=f"Observed {len(descriptions)} {cat} lesson(s)",
                    pattern=pattern,
                    confidence=round(confidence, 4),
                    supporting_experience_count=len(all_exp),
                )
            )

        insights.sort(key=lambda i: (-i.confidence, i.description))
        return insights


class ExperienceStore:
    """Interface for experience persistence.

    Implementations may use in-memory storage, a database,
    or a remote service.
    """

    async def save(self, record: Experience) -> str:
        """Persist an experience record.

        Args:
            record: The record to save.

        Returns:
            Record ID.

        Raises:
            NotImplementedError: Subclasses must override.
        """
        raise NotImplementedError("ExperienceStore.save not yet implemented")

    async def query(self, query: ExperienceQuery) -> list[Experience]:
        """Query experience records.

        Args:
            query: Query parameters.

        Returns:
            Matching records.

        Raises:
            NotImplementedError: Subclasses must override.
        """
        raise NotImplementedError("ExperienceStore.query not yet implemented")
