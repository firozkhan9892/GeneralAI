"""Reflection mode implementations."""

from __future__ import annotations

from app.kernel.reflection.models import (
    ErrorDetail,
    ErrorType,
    Refinement,
    ReflectionReport,
    ReflectionRequest,
    ReflectionScore,
)
from app.kernel.reflection.modes import IReflectionStrategy
from app.kernel.reasoning.models import ReasoningTrace


_CONTRADICTION_PAIRS: list[tuple[str, str]] = [
    ("true", "false"),
    ("yes", "no"),
    ("increase", "decrease"),
    ("positive", "negative"),
    ("good", "bad"),
    ("always", "never"),
    ("accept", "reject"),
    ("correct", "incorrect"),
]


def _get_trace_steps(trace: object) -> tuple:
    if isinstance(trace, ReasoningTrace):
        return trace.steps
    if isinstance(trace, (list, tuple)):
        return tuple(trace)
    return ()


def _get_trace_conclusion(trace: object) -> str | None:
    if isinstance(trace, ReasoningTrace):
        return trace.conclusion
    return None


class BasicReflectionStrategy(IReflectionStrategy):
    """Evaluates basic output presence and step structure."""

    async def evaluate(self, request: ReflectionRequest) -> ReflectionReport:
        output = request.output
        trace = request.trace
        steps = _get_trace_steps(trace)
        conclusion = _get_trace_conclusion(trace)
        num_steps = len(steps)
        has_output = output is not None and (
            not isinstance(output, str) or bool(output.strip())
        )

        errors: list[ErrorDetail] = []
        refinements: list[Refinement] = []
        dimensions: list[ReflectionScore] = []

        if not has_output:
            completeness = 0.0
            errors.append(
                ErrorDetail(
                    type=ErrorType.INCOMPLETE,
                    severity=1.0,
                    description="No output provided",
                    location="request.output",
                )
            )
            refinements.append(
                Refinement(
                    modification="Provide non-empty output for evaluation",
                    priority=1,
                )
            )
        elif num_steps >= 5:
            completeness = 1.0
        elif num_steps >= 3:
            completeness = 0.8
        elif num_steps >= 1:
            completeness = 0.6
            errors.append(
                ErrorDetail(
                    type=ErrorType.INCOMPLETE,
                    severity=0.3,
                    description=f"Only {num_steps} reasoning step(s)",
                    location="trace.steps",
                )
            )
            refinements.append(
                Refinement(
                    modification="Add more reasoning steps",
                    priority=0,
                )
            )
        else:
            completeness = 0.3
            errors.append(
                ErrorDetail(
                    type=ErrorType.INCOMPLETE,
                    severity=0.7,
                    description="No reasoning steps provided",
                    location="trace.steps",
                )
            )
            refinements.append(
                Refinement(
                    modification="Provide reasoning steps for evaluation",
                    priority=1,
                )
            )

        if conclusion and conclusion.strip():
            completeness = min(1.0, completeness + 0.1)

        dimensions.append(ReflectionScore(dimension="completeness", score=completeness))

        if num_steps >= 3:
            structure = 1.0
        elif num_steps >= 1:
            structure = 0.5
        else:
            structure = 0.0

        dimensions.append(ReflectionScore(dimension="structure", score=structure))

        total_weight = sum(d.weight for d in dimensions)
        overall = (
            sum(d.score * d.weight for d in dimensions) / total_weight
            if total_weight > 0
            else 0.0
        )
        verdict = self._verdict(overall, errors)

        return ReflectionReport(
            overall_score=overall,
            dimension_scores=tuple(dimensions),
            errors=tuple(errors),
            refinements=tuple(refinements),
            verdict=verdict,
            token_cost=10,
        )

    @staticmethod
    def _verdict(score: float, errors: list[ErrorDetail]) -> str:
        if len(errors) == 0 and score >= 0.8:
            return "pass"
        if score >= 0.4:
            return "needs_review"
        return "fail"


class FallbackReflectionStrategy(IReflectionStrategy):
    """Minimal fallback — always produces a passing report."""

    async def evaluate(self, request: ReflectionRequest) -> ReflectionReport:
        return ReflectionReport(
            overall_score=1.0,
            dimension_scores=(ReflectionScore(dimension="basic", score=1.0),),
            verdict="pass",
            token_cost=5,
        )


class ConsistencyReflectionStrategy(IReflectionStrategy):
    """Detects contradictions and logical gaps in reasoning steps."""

    async def evaluate(self, request: ReflectionRequest) -> ReflectionReport:
        trace = request.trace
        steps = _get_trace_steps(trace)

        errors: list[ErrorDetail] = []
        refinements: list[Refinement] = []

        content_words: list[str] = []
        for s in steps:
            content_words.append(str(s.content).lower())

        found_pairs: set[str] = set()
        for a, b in _CONTRADICTION_PAIRS:
            has_a = any(a in cw for cw in content_words)
            has_b = any(b in cw for cw in content_words)
            if has_a and has_b:
                key = f"{a}|{b}"
                if key not in found_pairs:
                    found_pairs.add(key)
                    errors.append(
                        ErrorDetail(
                            type=ErrorType.CONTRADICTION,
                            severity=0.6,
                            description=f"Contradictory terms: '{a}' vs '{b}'",
                            location="trace.steps",
                            suggested_fix=f"Resolve contradiction between '{a}' and '{b}'",
                        )
                    )
                    refinements.append(
                        Refinement(
                            modification=f"Resolve contradictory use of '{a}' and '{b}'",
                            priority=2,
                        )
                    )

        for i in range(len(steps) - 1):
            current_words = set(str(steps[i].content).lower().split())
            next_words = set(str(steps[i + 1].content).lower().split())
            overlap = current_words & next_words
            if len(overlap) == 0 and current_words and next_words:
                errors.append(
                    ErrorDetail(
                        type=ErrorType.LOGICAL_GAP,
                        severity=0.4,
                        description=f"Logical gap between step {i} and step {i + 1}",
                        location=f"trace.steps[{i}]",
                        suggested_fix="Add bridging reasoning between steps",
                    )
                )
                refinements.append(
                    Refinement(
                        target_step_id=str(steps[i].id),
                        modification=f"Bridge logical gap to step {i + 1}",
                        priority=1,
                    )
                )

        num_contradictions = sum(1 for e in errors if e.type == ErrorType.CONTRADICTION)
        num_gaps = sum(1 for e in errors if e.type == ErrorType.LOGICAL_GAP)

        consistency = 1.0 - min(0.5, num_contradictions * 0.2)
        coherence = 1.0 - min(0.3, num_gaps * 0.1)

        dimensions = [
            ReflectionScore(dimension="consistency", score=consistency),
            ReflectionScore(dimension="coherence", score=coherence),
        ]

        overall = (consistency * 1.0 + coherence * 1.0) / 2.0

        if overall >= 0.8 and not errors:
            verdict = "pass"
        elif overall >= 0.4:
            verdict = "needs_review"
        else:
            verdict = "fail"

        return ReflectionReport(
            overall_score=overall,
            dimension_scores=tuple(dimensions),
            errors=tuple(errors),
            refinements=tuple(refinements),
            verdict=verdict,
            token_cost=15,
        )


class QualityReflectionStrategy(IReflectionStrategy):
    """Multi-dimensional quality evaluation across correctness, completeness,
    clarity, and relevance."""

    async def evaluate(self, request: ReflectionRequest) -> ReflectionReport:
        output = request.output
        trace = request.trace
        context = request.context

        steps = _get_trace_steps(trace)
        conclusion = _get_trace_conclusion(trace)
        num_steps = len(steps)
        has_output = output is not None and (
            not isinstance(output, str) or bool(output.strip())
        )

        errors: list[ErrorDetail] = []
        refinements: list[Refinement] = []

        if conclusion and conclusion.strip() and has_output:
            correctness = 1.0
        elif conclusion and conclusion.strip():
            correctness = 0.7
        elif has_output:
            correctness = 0.5
            errors.append(
                ErrorDetail(
                    type=ErrorType.LOGICAL_GAP,
                    severity=0.4,
                    description="Output exists but conclusion is missing",
                    location="trace.conclusion",
                )
            )
            refinements.append(
                Refinement(
                    modification="Add a conclusion to the reasoning trace",
                    priority=1,
                )
            )
        else:
            correctness = 0.2
            errors.append(
                ErrorDetail(
                    type=ErrorType.INCOMPLETE,
                    severity=0.8,
                    description="No output or conclusion provided",
                    location="request.output",
                )
            )
            refinements.append(
                Refinement(
                    modification="Provide both output and conclusion",
                    priority=2,
                )
            )

        if num_steps >= 5 and has_output:
            completeness = 1.0
        elif num_steps >= 3:
            completeness = 0.7
        elif num_steps >= 1:
            completeness = 0.4
        else:
            completeness = 0.1

        total_words = sum(len(str(s.content).split()) for s in steps)
        if num_steps > 0 and total_words >= 10:
            clarity = 1.0
        elif num_steps > 0:
            clarity = 0.5
            refinements.append(
                Refinement(
                    modification="Add more detail to reasoning steps",
                    priority=0,
                )
            )
        else:
            clarity = 0.0

        if context:
            context_keys = set(str(k).lower() for k in context)
            all_content = " ".join(str(s.content).lower() for s in steps)
            if has_output and isinstance(output, str):
                all_content += " " + output.lower()
            referenced = sum(1 for k in context_keys if k in all_content)
            relevance = referenced / len(context_keys) if context_keys else 1.0
        else:
            relevance = 1.0

        dimensions = [
            ReflectionScore(dimension="correctness", score=correctness, weight=1.0),
            ReflectionScore(dimension="completeness", score=completeness, weight=0.8),
            ReflectionScore(dimension="clarity", score=clarity, weight=0.6),
            ReflectionScore(dimension="relevance", score=relevance, weight=0.6),
        ]

        weighted_sum = sum(d.score * d.weight for d in dimensions)
        total_weight = sum(d.weight for d in dimensions)
        overall = weighted_sum / total_weight if total_weight > 0 else 0.0

        if overall >= 0.8 and not errors:
            verdict = "pass"
        elif overall >= 0.4:
            verdict = "needs_review"
        else:
            verdict = "fail"

        return ReflectionReport(
            overall_score=overall,
            dimension_scores=tuple(dimensions),
            errors=tuple(errors),
            refinements=tuple(refinements),
            verdict=verdict,
            token_cost=20,
        )
