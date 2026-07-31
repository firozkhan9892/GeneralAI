"""Input normalizer interface for the perception pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.kernel.perception.models import Percept, RawMessage


class InputNormalizer(ABC):
    """Normalizes raw input of a specific modality into structured data."""

    @abstractmethod
    async def normalize(self, raw: RawMessage) -> Percept:
        """Normalize *raw* input into a Percept.

        Args:
            raw: Raw input message to normalize.

        Returns:
            Normalized percept for this modality.
        """
