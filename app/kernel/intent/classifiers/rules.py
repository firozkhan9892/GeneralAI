"""Rule-based intent classifier for the intent pipeline."""

from __future__ import annotations

import re

from app.kernel.intent.models import IntentClassification, IntentConfidence, IntentType
from app.kernel.intent.classifiers import IntentClassifier
from app.kernel.perception.models import Percept


_QUESTION_WORDS = re.compile(
    r"\b(what|why|how|when|where|who|which|whose|whom|does|do|did|"
    r"is|are|was|were|will|would|can|could|shall|should|may|might)\b",
    re.IGNORECASE,
)
_GREETINGS = re.compile(
    r"\b(hi|hello|hey|greetings|good\s*(morning|afternoon|evening|day)|"
    r"howdy|yo|sup|what'?s\s*up)\b",
    re.IGNORECASE,
)
_COMMANDS = re.compile(
    r"\b(run|execute|create|make|build|generate|write|do|show|tell|find|"
    r"list|start|stop|set|get|compute|calculate|translate|summarize|"
    r"explain|define|search|open|close|add|remove|delete|update)\b",
    re.IGNORECASE,
)
_EXPLORE_WORDS = re.compile(
    r"\b(tell\s*me\s*about|explain|what\s*is|describe|define|"
    r"let'?s\s*(talk|discuss|chat)|i\s*wonder|curious|"
    r"can\s*you\s*(tell|explain|help))\b",
    re.IGNORECASE,
)
_SOLVE_WORDS = re.compile(
    r"\b(problem|solve|fix|issue|bug|error|wrong|broken|not\s*working|"
    r"doesn'?t\s*work|failed|crash|debug|troubleshoot|help\s*me\s*(fix|solve))\b",
    re.IGNORECASE,
)
_PLAN_WORDS = re.compile(
    r"\b(plan|project|roadmap|schedule|timeline|milestone|"
    r"strategy|organize|prepare|arrange)\b",
    re.IGNORECASE,
)
_LEARN_WORDS = re.compile(
    r"\b(learn|study|teach|tutorial|guide|how\s*to|understand|"
    r"know\s*about|explain)\b",
    re.IGNORECASE,
)
_CREATE_WORDS = re.compile(
    r"\b(write|compose|draft|create|generate|produce|make\s*a|"
    r"design|develop|build|code|program|script)\b",
    re.IGNORECASE,
)
_DEBUG_WORDS = re.compile(
    r"\b(debug|bug|error|fix|issue|problem|crash|fail|exception|"
    r"trace|log|diagnose|inspect|breakpoint)\b",
    re.IGNORECASE,
)
_META_WORDS = re.compile(
    r"\b(who\s*(are|is)\s*(you|this)|what\s*(are|can)\s*you|"
    r"your\s*(name|purpose|capabilities|features)|"
    r"how\s*(do|does)\s*you\s*work|what\s*can\s*you\s*do)\b",
    re.IGNORECASE,
)
_CLARIFY_WORDS = re.compile(
    r"\b(clarify|elaborate|explain\s*(more|further)|"
    r"what\s*(do|did)\s*you\s*mean|rephrase|"
    r"tell\s*me\s*more|go\s*on|continue)\b",
    re.IGNORECASE,
)


_PATTERNS: list[tuple[IntentType, re.Pattern, float]] = [
    (IntentType.META, _META_WORDS, 0.9),
    (IntentType.CLARIFY, _CLARIFY_WORDS, 0.8),
    (IntentType.ASK_QUESTION, _QUESTION_WORDS, 0.7),
    (IntentType.SOLVE_PROBLEM, _SOLVE_WORDS, 0.8),
    (IntentType.PLAN_PROJECT, _PLAN_WORDS, 0.8),
    (IntentType.LEARN, _LEARN_WORDS, 0.7),
    (IntentType.CREATE_CONTENT, _CREATE_WORDS, 0.7),
    (IntentType.DEBUG, _DEBUG_WORDS, 0.8),
    (IntentType.EXECUTE_TASK, _COMMANDS, 0.5),
    (IntentType.EXPLORE, _EXPLORE_WORDS, 0.7),
]

_GREETING_WEIGHT = 0.6
_HAS_QUESTION_MARK_WEIGHT = 0.3


class RuleBasedClassifier(IntentClassifier):
    """Classifies intent using keyword and pattern matching rules."""

    def __init__(self) -> None:
        self._greeting_pattern = _GREETINGS
        self._patterns = _PATTERNS
        self._name = "RuleBasedClassifier"

    async def classify(self, percept: Percept) -> IntentClassification:
        text = percept.normalized_content
        if not text:
            return IntentClassification(
                primary=IntentType.UNKNOWN,
                confidence=IntentConfidence(primary=0.0, alternatives=()),
            )

        scores: dict[IntentType, float] = {t: 0.0 for t in IntentType}

        if self._greeting_pattern.search(text):
            scores[IntentType.META] = max(scores[IntentType.META], _GREETING_WEIGHT)

        if text.rstrip().endswith("?"):
            for pattern_type, _, weight in self._patterns:
                if pattern_type == IntentType.ASK_QUESTION:
                    scores[IntentType.ASK_QUESTION] += _HAS_QUESTION_MARK_WEIGHT
                    break

        for intent_type, pattern, weight in self._patterns:
            matches = pattern.findall(text)
            if matches:
                score = min(1.0, len(matches) * weight)
                scores[intent_type] = max(scores[intent_type], score)

        total = sum(scores.values())
        if total > 0:
            normalized = {t: round(s / total, 4) for t, s in scores.items()}
        else:
            normalized = {t: 0.0 for t in scores}
            normalized[IntentType.UNKNOWN] = 1.0

        sorted_types = sorted(normalized.items(), key=lambda x: -x[1])
        primary_type, primary_score = sorted_types[0]

        if primary_type == IntentType.UNKNOWN:
            primary_score = 1.0

        alternatives = tuple((it, sc) for it, sc in sorted_types[1:] if sc > 0.0)

        return IntentClassification(
            primary=primary_type,
            confidence=IntentConfidence(
                primary=primary_score,
                alternatives=alternatives,
            ),
            classifier_name=self._name,
        )
