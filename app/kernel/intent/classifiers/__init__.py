"""Intent classifier interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.kernel.intent.models import IntentClassification
from app.kernel.perception.models import Percept


class IntentClassifier(ABC):
    """Classifies user intent from a structured Percept."""

    @abstractmethod
    async def classify(self, percept: Percept) -> IntentClassification:
        """Classify the intent behind *percept*.

        Args:
            percept: Structured percept from the perception layer.

        Returns:
            Intent classification with type and confidence.
        """
