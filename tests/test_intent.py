"""Tests for IntentEngine and RuleBasedClassifier."""

from __future__ import annotations

from pydantic import ValidationError

import pytest

from app.kernel.intent import (
    ClarificationRequest,
    Intent,
    IntentClassification,
    IntentConfidence,
    IntentEngine,
    IntentType,
)
from app.kernel.intent.classifiers import IntentClassifier
from app.kernel.intent.classifiers.rules import RuleBasedClassifier
from app.kernel.perception.models import ModalityType, Percept, RawMessage


def _percept(text: str) -> Percept:
    return Percept(
        raw=RawMessage(content=text),
        normalized_content=text,
        modality=ModalityType.TEXT,
    )


# ── RuleBasedClassifier ───────────────────────────────────────────────────


class TestRuleBasedClassifier:
    """Tests for the rule-based intent classifier."""

    @pytest.fixture
    def classifier(self) -> RuleBasedClassifier:
        return RuleBasedClassifier()

    @pytest.mark.asyncio
    async def test_question(self, classifier: RuleBasedClassifier) -> None:
        result = await classifier.classify(_percept("What is the weather?"))
        assert result.primary == IntentType.ASK_QUESTION
        assert result.classifier_name == "RuleBasedClassifier"

    @pytest.mark.asyncio
    async def test_question_without_mark(self, classifier: RuleBasedClassifier) -> None:
        result = await classifier.classify(_percept("Tell me how this works"))
        assert result.primary == IntentType.ASK_QUESTION

    @pytest.mark.asyncio
    async def test_greeting(self, classifier: RuleBasedClassifier) -> None:
        result = await classifier.classify(_percept("Hello, how are you?"))
        assert result.primary == IntentType.ASK_QUESTION
        assert 0.0 < result.confidence.primary <= 1.0

    @pytest.mark.asyncio
    async def test_simple_greeting(self, classifier: RuleBasedClassifier) -> None:
        result = await classifier.classify(_percept("Hi"))
        assert result.primary == IntentType.META

    @pytest.mark.asyncio
    async def test_command(self, classifier: RuleBasedClassifier) -> None:
        result = await classifier.classify(_percept("Run the deployment script"))
        assert result.primary == IntentType.CREATE_CONTENT
        assert result.confidence.primary > 0.0

    @pytest.mark.asyncio
    async def test_execute_command(self, classifier: RuleBasedClassifier) -> None:
        result = await classifier.classify(_percept("Create a new project"))
        assert result.primary in (
            IntentType.PLAN_PROJECT,
            IntentType.EXECUTE_TASK,
            IntentType.CREATE_CONTENT,
        )

    @pytest.mark.asyncio
    async def test_conversation(self, classifier: RuleBasedClassifier) -> None:
        result = await classifier.classify(_percept("Tell me about yourself"))
        assert result.primary in (IntentType.EXPLORE, IntentType.META)

    @pytest.mark.asyncio
    async def test_information_request(self, classifier: RuleBasedClassifier) -> None:
        result = await classifier.classify(
            _percept("Can you tell me what machine learning is?")
        )
        assert result.primary == IntentType.ASK_QUESTION

    @pytest.mark.asyncio
    async def test_unknown(self, classifier: RuleBasedClassifier) -> None:
        result = await classifier.classify(_percept("xyzzzz"))
        assert result.primary == IntentType.UNKNOWN
        assert result.confidence.primary == 1.0

    @pytest.mark.asyncio
    async def test_empty_input(self, classifier: RuleBasedClassifier) -> None:
        result = await classifier.classify(_percept(""))
        assert result.primary == IntentType.UNKNOWN
        assert result.confidence.primary == 0.0

    @pytest.mark.asyncio
    async def test_confidence_in_range(self, classifier: RuleBasedClassifier) -> None:
        result = await classifier.classify(_percept("What is the weather?"))
        assert 0.0 <= result.confidence.primary <= 1.0
        for _, score in result.confidence.alternatives:
            assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_alternatives_populated(
        self, classifier: RuleBasedClassifier
    ) -> None:
        result = await classifier.classify(_percept("How do I fix this bug?"))
        assert len(result.confidence.alternatives) >= 1

    @pytest.mark.asyncio
    async def test_solve_problem(self, classifier: RuleBasedClassifier) -> None:
        result = await classifier.classify(
            _percept("My code has a bug, help me fix it")
        )
        assert result.primary == IntentType.SOLVE_PROBLEM

    @pytest.mark.asyncio
    async def test_debug(self, classifier: RuleBasedClassifier) -> None:
        result = await classifier.classify(_percept("Debug this error trace"))
        assert result.primary in (IntentType.DEBUG, IntentType.SOLVE_PROBLEM)

    @pytest.mark.asyncio
    async def test_learn(self, classifier: RuleBasedClassifier) -> None:
        result = await classifier.classify(_percept("Teach me about Python"))
        assert result.primary == IntentType.LEARN

    @pytest.mark.asyncio
    async def test_clarify(self, classifier: RuleBasedClassifier) -> None:
        result = await classifier.classify(_percept("What do you mean by that?"))
        assert result.primary in (IntentType.CLARIFY, IntentType.ASK_QUESTION)


# ── IntentEngine ───────────────────────────────────────────────────────────


class TestIntentEngine:
    """Tests for the IntentEngine orchestrator."""

    @pytest.fixture
    def engine(self) -> IntentEngine:
        return IntentEngine()

    @pytest.mark.asyncio
    async def test_init_registers_default(self, engine: IntentEngine) -> None:
        assert len(engine._classifiers) == 1
        assert isinstance(engine._classifiers[0], RuleBasedClassifier)

    @pytest.mark.asyncio
    async def test_understand_returns_intent(self, engine: IntentEngine) -> None:
        percept = _percept("What is the weather?")
        intent = await engine.understand(percept)
        assert isinstance(intent, Intent)
        assert isinstance(intent.primary, IntentType)
        assert isinstance(intent.confidence, IntentConfidence)

    @pytest.mark.asyncio
    async def test_understand_question(self, engine: IntentEngine) -> None:
        intent = await engine.understand(_percept("What is AI?"))
        assert intent.primary == IntentType.ASK_QUESTION

    @pytest.mark.asyncio
    async def test_understand_greeting(self, engine: IntentEngine) -> None:
        intent = await engine.understand(_percept("Hello"))
        assert intent.primary == IntentType.META

    @pytest.mark.asyncio
    async def test_understand_command(self, engine: IntentEngine) -> None:
        intent = await engine.understand(_percept("Run the tests"))
        assert intent.primary == IntentType.EXECUTE_TASK

    @pytest.mark.asyncio
    async def test_understand_unambiguous(self, engine: IntentEngine) -> None:
        intent = await engine.understand(_percept("What is the capital of France?"))
        assert intent.clarification is None

    @pytest.mark.asyncio
    async def test_understand_empty(self, engine: IntentEngine) -> None:
        intent = await engine.understand(_percept(""))
        assert intent.primary == IntentType.UNKNOWN

    @pytest.mark.asyncio
    async def test_confidence_scores(self, engine: IntentEngine) -> None:
        intent = await engine.understand(_percept("How do I deploy this?"))
        assert 0.0 <= intent.confidence.primary <= 1.0
        assert intent.confidence.primary > 0.0

    @pytest.mark.asyncio
    async def test_register_classifier(self, engine: IntentEngine) -> None:
        class MockClassifier(IntentClassifier):
            async def classify(self, percept: Percept) -> IntentClassification:
                return IntentClassification(
                    primary=IntentType.DEBUG,
                    confidence=IntentConfidence(primary=0.95),
                    classifier_name="Mock",
                )

        engine.register_classifier(MockClassifier())
        assert len(engine._classifiers) == 2

    @pytest.mark.asyncio
    async def test_custom_classifier_wins(self, engine: IntentEngine) -> None:
        class HighConfidenceClassifier(IntentClassifier):
            async def classify(self, percept: Percept) -> IntentClassification:
                return IntentClassification(
                    primary=IntentType.DEBUG,
                    confidence=IntentConfidence(primary=1.0),
                    classifier_name="HighConf",
                )

        engine.register_classifier(HighConfidenceClassifier())
        intent = await engine.understand(_percept("Hello"))
        assert intent.primary == IntentType.DEBUG

    @pytest.mark.asyncio
    async def test_unregister_classifier(self, engine: IntentEngine) -> None:
        classifier = engine._classifiers[0]
        engine.unregister_classifier(classifier)
        assert len(engine._classifiers) == 0

    @pytest.mark.asyncio
    async def test_fallback_when_no_classifiers(self, engine: IntentEngine) -> None:
        for c in list(engine._classifiers):
            engine.unregister_classifier(c)
        intent = await engine.understand(_percept("Hello"))
        assert intent.primary == IntentType.UNKNOWN

    @pytest.mark.asyncio
    async def test_intent_timestamp(self, engine: IntentEngine) -> None:
        intent = await engine.understand(_percept("Hello"))
        assert intent.timestamp is not None

    @pytest.mark.asyncio
    async def test_clarification_on_ambiguous(self, engine: IntentEngine) -> None:
        intent = await engine.understand(_percept("xyzzzz"))
        assert intent.primary == IntentType.UNKNOWN
        assert intent.clarification is not None

    @pytest.mark.asyncio
    async def test_clarification_is_clarification_request(
        self, engine: IntentEngine
    ) -> None:
        intent = await engine.understand(_percept("xyzzzz"))
        assert isinstance(intent.clarification, ClarificationRequest)
        assert intent.clarification.freeform_prompt

    @pytest.mark.asyncio
    async def test_intent_is_immutable(self, engine: IntentEngine) -> None:
        intent = await engine.understand(_percept("Hello"))
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            intent.primary = IntentType.UNKNOWN  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_serialization_roundtrip(self, engine: IntentEngine) -> None:
        intent = await engine.understand(_percept("What is AI?"))
        data = intent.model_dump()
        restored = Intent.model_validate(data)
        assert restored.primary == intent.primary
        assert restored.confidence.primary == intent.confidence.primary


# ── Serialization / Model tests ────────────────────────────────────────────


class TestIntentModels:
    """Unit tests for intent domain models."""

    def test_intent_classification_frozen(self) -> None:
        c = IntentClassification(
            primary=IntentType.ASK_QUESTION,
            confidence=IntentConfidence(primary=0.9),
        )
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            c.primary = IntentType.UNKNOWN  # type: ignore[misc]

    def test_intent_confidence_validation(self) -> None:
        with pytest.raises(Exception):
            IntentConfidence(primary=1.5)

    def test_clarification_request_creation(self) -> None:
        cr = ClarificationRequest(
            ambiguity_description="Unclear intent",
            options=("Option A", "Option B"),
            freeform_prompt="What would you like?",
        )
        assert cr.ambiguity_description == "Unclear intent"
        assert len(cr.options) == 2


# ── RuleBasedClassifier edge cases ────────────────────────────────────────


class TestClassifierEdgeCases:
    """Edge case tests for the classifier."""

    @pytest.fixture
    def classifier(self) -> RuleBasedClassifier:
        return RuleBasedClassifier()

    @pytest.mark.asyncio
    async def test_question_mark_boosts_question(
        self, classifier: RuleBasedClassifier
    ) -> None:
        without_q = await classifier.classify(_percept("What is this"))
        with_q = await classifier.classify(_percept("What is this?"))
        assert with_q.confidence.primary >= without_q.confidence.primary

    @pytest.mark.asyncio
    async def test_multiple_question_words(
        self, classifier: RuleBasedClassifier
    ) -> None:
        result = await classifier.classify(
            _percept("Why does this happen and how can I fix it?")
        )
        assert result.primary != IntentType.UNKNOWN

    @pytest.mark.asyncio
    async def test_greeting_with_question(
        self, classifier: RuleBasedClassifier
    ) -> None:
        result = await classifier.classify(_percept("Hello, what is this?"))
        assert result.primary != IntentType.UNKNOWN

    @pytest.mark.asyncio
    async def test_whitespace_only(self, classifier: RuleBasedClassifier) -> None:
        result = await classifier.classify(_percept("   "))
        assert result.primary == IntentType.UNKNOWN
