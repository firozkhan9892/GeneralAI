"""Text input normalizer for the perception pipeline."""

from __future__ import annotations

import re
from datetime import datetime

from app.kernel.perception.models import (
    Entity,
    ModalityType,
    Percept,
    QualityScore,
    RawMessage,
)
from app.kernel.perception.normalizers import InputNormalizer


_URL_PATTERN = re.compile(r"https?://[^\s]+")
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_MENTION_PATTERN = re.compile(r"@\w+")
_HASHTAG_PATTERN = re.compile(r"#\w+")


class TextNormalizer(InputNormalizer):
    """Normalizes raw text input into a structured Percept."""

    async def normalize(self, raw: RawMessage) -> Percept:
        raw_text = self._decode_content(raw)
        normalized = self._normalize_text(raw_text)
        entities = self._extract_entities(raw_text)
        quality = self._compute_quality(normalized)

        return Percept(
            raw=raw,
            modality=ModalityType.TEXT,
            normalized_content=normalized,
            entities=entities,
            quality=quality,
            timestamp=datetime.utcnow(),
        )

    def _decode_content(self, raw: RawMessage) -> str:
        if isinstance(raw.content, bytes):
            return raw.content.decode("utf-8", errors="replace")
        return raw.content

    def _normalize_text(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"[\r\f\v]", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _extract_entities(self, text: str) -> tuple[Entity, ...]:
        entities: list[Entity] = []

        for match in _URL_PATTERN.finditer(text):
            entities.append(
                Entity(
                    type="url",
                    value=match.group(),
                    confidence=1.0,
                    position=(match.start(), match.end()),
                )
            )
        for match in _EMAIL_PATTERN.finditer(text):
            entities.append(
                Entity(
                    type="email",
                    value=match.group(),
                    confidence=1.0,
                    position=(match.start(), match.end()),
                )
            )
        for match in _MENTION_PATTERN.finditer(text):
            entities.append(
                Entity(
                    type="mention",
                    value=match.group(),
                    confidence=0.9,
                    position=(match.start(), match.end()),
                )
            )
        for match in _HASHTAG_PATTERN.finditer(text):
            entities.append(
                Entity(
                    type="hashtag",
                    value=match.group(),
                    confidence=0.9,
                    position=(match.start(), match.end()),
                )
            )
        return tuple(entities)

    def _compute_quality(self, normalized: str) -> QualityScore:
        length = len(normalized)
        word_count = len(normalized.split()) if normalized else 0

        length_sufficiency = min(1.0, length / 100.0)

        if length == 0:
            completeness = 0.0
        elif normalized.rstrip().endswith((".", "!", "?", "...")):
            completeness = 1.0
        elif normalized.rstrip().endswith((",", ";", ":", "-")):
            completeness = 0.5
        else:
            completeness = 0.7

        if word_count == 0:
            coherence = 0.0
        elif word_count == 1:
            coherence = 0.3
        elif word_count <= 3:
            coherence = 0.6
        elif word_count <= 10:
            coherence = 0.9
        else:
            coherence = 1.0

        overall = round(
            0.3 * length_sufficiency + 0.35 * completeness + 0.35 * coherence, 4
        )
        overall = min(1.0, max(0.0, overall))

        return QualityScore(
            overall=overall,
            length_sufficiency=round(length_sufficiency, 4),
            completeness=round(completeness, 4),
            coherence=round(coherence, 4),
        )
