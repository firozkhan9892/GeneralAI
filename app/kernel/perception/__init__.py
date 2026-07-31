"""Perception — stage 1 of the cognitive pipeline."""

from __future__ import annotations

from app.kernel.perception.engine import PerceptionEngine
from app.kernel.perception.models import (
    Entity,
    ModalityType,
    Percept,
    QualityScore,
    RawMessage,
)
from app.kernel.perception.normalizers import InputNormalizer
from app.kernel.perception.normalizers.text import TextNormalizer

__all__ = [
    "Entity",
    "InputNormalizer",
    "ModalityType",
    "Percept",
    "PerceptionEngine",
    "QualityScore",
    "RawMessage",
    "TextNormalizer",
]
