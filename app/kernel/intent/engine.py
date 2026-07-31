"""Intent engine — stage 2 of the cognitive pipeline."""

from __future__ import annotations

import logging

from app.kernel.intent.models import (
    ClarificationRequest,
    Intent,
    IntentClassification,
    IntentConfidence,
    IntentType,
)
from app.kernel.intent.classifiers import IntentClassifier
from app.kernel.intent.classifiers.rules import RuleBasedClassifier
from app.kernel.perception.models import Percept

log = logging.getLogger(__name__)


class IntentEngine:
    """Classifies user intent from structured percepts.

    Determines what the user wants before any goal or plan is created.
    """

    def __init__(self) -> None:
        self._classifiers: list[IntentClassifier] = []
        self._register_default()

    def _register_default(self) -> None:
        self._classifiers.append(RuleBasedClassifier())

    async def understand(self, percept: Percept) -> Intent:
        """Classify intent from a Percept.

        Args:
            percept: Structured percept from the perception layer.

        Returns:
            Structured intent object.
        """
        classification = await self._classify(percept)
        clarification = self._build_clarification(classification)
        intent = self._build_intent(classification, percept, clarification)
        log.info(
            "Intent %s (confidence=%.2f, alternatives=%d)",
            intent.primary.value,
            intent.confidence.primary,
            len(intent.confidence.alternatives),
        )
        return intent

    def register_classifier(self, classifier: IntentClassifier) -> None:
        """Register an intent classifier.

        Args:
            classifier: Classifier implementation.
        """
        self._classifiers.append(classifier)

    def unregister_classifier(self, classifier: IntentClassifier) -> None:
        """Remove a previously registered classifier.

        Args:
            classifier: Classifier instance to remove.
        """
        self._classifiers = [c for c in self._classifiers if c is not classifier]

    async def _classify(self, percept: Percept) -> IntentClassification:
        if not self._classifiers:
            return IntentClassification(
                primary=IntentType.UNKNOWN,
                confidence=IntentConfidence(primary=0.0, alternatives=()),
            )
        results: list[IntentClassification] = []
        for classifier in self._classifiers:
            try:
                result = await classifier.classify(percept)
                results.append(result)
            except Exception:
                log.exception("Classifier %s failed", classifier.__class__.__name__)

        if not results:
            return IntentClassification(
                primary=IntentType.UNKNOWN,
                confidence=IntentConfidence(primary=0.0, alternatives=()),
            )

        return max(reversed(results), key=lambda r: r.confidence.primary)

    def _build_clarification(
        self, classification: IntentClassification
    ) -> ClarificationRequest | None:
        if classification.primary == IntentType.UNKNOWN:
            return ClarificationRequest(
                ambiguity_description=(
                    "Unable to determine your intent based on the input provided"
                ),
                options=(),
                freeform_prompt="Could you rephrase what you'd like to do?",
            )

        threshold = classification.confidence.ambiguity_threshold
        if classification.confidence.primary >= threshold:
            alternatives = classification.confidence.alternatives
            if len(alternatives) >= 1:
                top_alt = alternatives[0]
                gap = classification.confidence.primary - top_alt[1]
                if gap > 0.15:
                    return None
            else:
                return None

        options = [
            classification.primary.value.replace("_", " ").title(),
        ]
        options.extend(
            alt[0].value.replace("_", " ").title()
            for alt in classification.confidence.alternatives[:3]
        )

        return ClarificationRequest(
            ambiguity_description=(
                f"Uncertain whether you intend to "
                f"{classification.primary.value.replace('_', ' ')} "
                f"(confidence {classification.confidence.primary:.0%})"
            ),
            options=tuple(options),
            freeform_prompt="Could you clarify what you'd like to do?",
        )

    def _build_intent(
        self,
        classification: IntentClassification,
        percept: Percept,
        clarification: ClarificationRequest | None,
    ) -> Intent:
        from datetime import datetime

        return Intent(
            primary=classification.primary,
            confidence=classification.confidence,
            clarification=clarification,
            timestamp=datetime.utcnow(),
        )
