"""Response builder — stage 15 of the cognitive pipeline."""

from __future__ import annotations

import logging
from typing import Any

from app.kernel.response.models import OutputMessage, StreamChunk

log = logging.getLogger(__name__)


class ResponseBuilder:
    """Formats the final output for the caller.

    Consumes pipeline stage outputs (decision, reasoning, policy,
    reflection, experience, intent, goal) and produces a final
    OutputMessage with formatted content and diagnostic metadata.
    """

    def __init__(self, default_format: str = "text") -> None:
        self._default_format = default_format

    async def build(self, context: Any) -> OutputMessage:
        """Build the final output message from pipeline context.

        Args:
            context: PipelineContext (or dict-like) with stage outputs.

        Returns:
            Formatted output message.
        """
        session_id = self._get(context, "session_id", "")
        metadata: dict[str, Any] = {}
        content_lines: list[str] = []
        success = True
        error: str | None = None
        policy_denied = False

        decision = self._get(context, "decision")
        if decision is not None:
            self._add_decision(decision, content_lines, metadata)

        policy = self._get(context, "policy_verdict")
        if policy is not None:
            policy_denied = self._add_policy(policy, content_lines, metadata)
            if policy_denied:
                success = False
                dr = self._get(policy, "denial_reason", None)
                error = dr if dr else "Action denied by policy"

        reasoning = self._get(context, "reasoning_trace")
        if reasoning is not None:
            self._add_reasoning(reasoning, content_lines, metadata)

        goal = self._get(context, "goal_hierarchy")
        if goal is not None:
            self._add_goal(goal, metadata)

        intent = self._get(context, "intent")
        if intent is not None:
            self._add_intent(intent, metadata)

        reflection = self._get(context, "reflection")
        if reflection is not None:
            self._add_reflection(reflection, metadata)

        experience = self._get(context, "experience")
        if experience is not None:
            self._add_experience(experience, metadata)

        percept = self._get(context, "percept")
        if percept is not None:
            nc = self._get(percept, "normalized_content", None)
            if nc:
                metadata["user_input_preview"] = nc[:100]

        plan = self._get(context, "plan")
        if plan is not None:
            pid = self._get(plan, "id", "")
            if pid:
                metadata["plan_id"] = pid

        fmt = self._get(context, "response_format", self._default_format)

        if not content_lines:
            if not success and error:
                content_lines.append(f"Error: {error}")
            elif policy_denied:
                content_lines.append("Request denied by policy")
            else:
                content_lines.append("(no output)")

        content = self._render(content_lines, fmt)

        return OutputMessage(
            content=content,
            format=fmt,
            metadata=dict(sorted(metadata.items())),
            session_id=session_id,
            success=success,
            error=error,
        )

    async def build_chunk(self, chunk_data: Any) -> StreamChunk:
        """Build a streaming chunk.

        Args:
            chunk_data: Dict with content/type/metadata/finished keys,
                        or a plain string.

        Returns:
            Formatted stream chunk.
        """
        if isinstance(chunk_data, dict):
            return StreamChunk(
                content=chunk_data.get("content", ""),
                chunk_type=chunk_data.get("type", "text"),
                metadata=chunk_data.get("metadata", {}),
                finished=chunk_data.get("finished", False),
            )
        text = str(chunk_data) if chunk_data is not None else ""
        return StreamChunk(content=text)

    # ── internal helpers ────────────────────────────────

    @staticmethod
    def _get(obj: Any, name: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    @staticmethod
    def _add_decision(
        decision: Any,
        lines: list[str],
        meta: dict[str, Any],
    ) -> None:
        action = ResponseBuilder._get(decision, "selected_action")
        if action is None:
            return
        action_type = ResponseBuilder._get(action, "action_type", "unknown")
        desc = ResponseBuilder._get(action, "description", "")
        conf = ResponseBuilder._get(action, "confidence", 0.0)
        lines.append(f"Action: {action_type}")
        if desc:
            lines.append(f"Description: {desc}")
        meta["decision_confidence"] = conf
        meta["decision_action"] = action_type

    @staticmethod
    def _add_policy(
        policy: Any,
        lines: list[str],
        meta: dict[str, Any],
    ) -> bool:
        verdict = ResponseBuilder._get(policy, "verdict", None)
        if verdict is not None:
            v = verdict.value if hasattr(verdict, "value") else str(verdict)
            meta["policy_verdict"] = v
            if v == "deny":
                meta["policy_denied"] = True
                return True
            allowed = v == "allow"
            meta["policy_allowed"] = allowed
        return False

    @staticmethod
    def _add_reasoning(
        reasoning: Any,
        lines: list[str],
        meta: dict[str, Any],
    ) -> None:
        conc = ResponseBuilder._get(reasoning, "conclusion", None)
        if conc and str(conc).strip():
            lines.append(f"Conclusion: {conc}")
        steps = ResponseBuilder._get(reasoning, "steps", ())
        meta["reasoning_steps"] = len(steps)
        tc = ResponseBuilder._get(reasoning, "token_cost", 0)
        meta["reasoning_token_cost"] = tc
        strat = ResponseBuilder._get(reasoning, "strategy_used", None)
        if strat is not None:
            meta["reasoning_strategy"] = (
                strat.value if hasattr(strat, "value") else str(strat)
            )

    @staticmethod
    def _add_goal(goal: Any, meta: dict[str, Any]) -> None:
        root = ResponseBuilder._get(goal, "root", None)
        if root is not None:
            gt = ResponseBuilder._get(root, "goal_type", None)
            if gt is not None:
                meta["goal_type"] = gt.value if hasattr(gt, "value") else str(gt)

    @staticmethod
    def _add_intent(intent: Any, meta: dict[str, Any]) -> None:
        primary = ResponseBuilder._get(intent, "primary", None)
        if primary is not None:
            meta["intent_type"] = (
                primary.value if hasattr(primary, "value") else str(primary)
            )
        conf = ResponseBuilder._get(intent, "confidence", None)
        if conf is not None:
            pc = ResponseBuilder._get(conf, "primary", None)
            if pc is not None:
                meta["intent_confidence"] = pc

    @staticmethod
    def _add_reflection(reflection: Any, meta: dict[str, Any]) -> None:
        score = ResponseBuilder._get(reflection, "overall_score", None)
        if score is not None:
            meta["reflection_score"] = score
        verdict = ResponseBuilder._get(reflection, "verdict", None)
        if verdict is not None:
            meta["reflection_verdict"] = verdict
        errors = ResponseBuilder._get(reflection, "errors", ())
        meta["reflection_error_count"] = len(errors)

    @staticmethod
    def _add_experience(experience: Any, meta: dict[str, Any]) -> None:
        if isinstance(experience, dict):
            meta["experience_count"] = experience.get("total_experiences", 0)
        elif hasattr(experience, "get"):
            meta["experience_count"] = experience.get("total_experiences", 0)
        else:
            meta["experience_count"] = 0

    @staticmethod
    def _render(lines: list[str], fmt: str) -> str:
        if fmt == "markdown":
            parts: list[str] = []
            for line in lines:
                if line.startswith("Action:"):
                    parts.append(f"**{line}**")
                elif line.startswith("Conclusion:"):
                    parts.append(f"## {line}")
                elif line.startswith("Description:"):
                    parts.append(f"> {line}")
                elif line.startswith("Error:"):
                    parts.append(f"**{line}**")
                else:
                    parts.append(line)
            return "\n\n".join(parts)
        return "\n".join(lines)
