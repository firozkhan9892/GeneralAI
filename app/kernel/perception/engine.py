"""Perception engine — stage 1 of the cognitive pipeline."""

from __future__ import annotations

import logging

from app.kernel.perception.models import ModalityType, Percept, QualityScore, RawMessage
from app.kernel.perception.normalizers import InputNormalizer
from app.kernel.perception.normalizers.text import TextNormalizer

log = logging.getLogger(__name__)


class PerceptionEngine:
    """Transforms raw input into a structured Percept.

    Responsible for modality detection, input normalization,
    entity extraction, and quality assessment.
    """

    def __init__(self) -> None:
        self._normalizers: dict[str, InputNormalizer] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self._normalizers[ModalityType.TEXT.value] = TextNormalizer()

    async def perceive(
        self,
        raw: RawMessage,
        *,
        correlation_id: str = "",
        session_id: str = "",
        context_ref: str | None = None,
    ) -> Percept:
        """Process raw input and produce a Percept.

        Args:
            raw: Raw input message.
            correlation_id: Correlation ID for tracing.
            session_id: Session identifier.
            context_ref: Optional context reference.

        Returns:
            Normalized percept.
        """
        modality = self._detect_modality(raw)
        raw_with_modality = (
            raw
            if raw.modality == modality
            else raw.model_copy(update={"modality": modality})
        )

        normalizer = self._normalizers.get(modality.value)
        if normalizer is not None:
            percept = await normalizer.normalize(raw_with_modality)
        else:
            percept = self._fallback_normalize(raw_with_modality)

        if correlation_id:
            log.debug(
                "Correlation %s — session %s — context %s",
                correlation_id,
                session_id,
                context_ref,
            )

        return percept

    def register_normalizer(self, modality: str, normalizer: InputNormalizer) -> None:
        self._normalizers[modality] = normalizer

    def _detect_modality(self, raw: RawMessage) -> ModalityType:
        if raw.modality != ModalityType.UNKNOWN:
            return raw.modality
        if isinstance(raw.content, bytes):
            try:
                raw.content.decode("utf-8")
                return ModalityType.TEXT
            except (UnicodeDecodeError, UnicodeError):
                return ModalityType.UNKNOWN
        return ModalityType.TEXT

    def _fallback_normalize(self, raw: RawMessage) -> Percept:
        from datetime import datetime

        raw_text = self._decode_content(raw)
        text = raw_text.strip()
        quality = QualityScore()

        return Percept(
            raw=raw,
            modality=raw.modality,
            normalized_content=text,
            quality=quality,
            timestamp=datetime.utcnow(),
        )

    def _decode_content(self, raw: RawMessage) -> str:
        if isinstance(raw.content, bytes):
            return raw.content.decode("utf-8", errors="replace")
        return raw.content
