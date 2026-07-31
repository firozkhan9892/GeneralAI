"""Tests for PerceptionEngine and TextNormalizer."""

from __future__ import annotations

from datetime import datetime

import pytest

from app.kernel.perception import PerceptionEngine, TextNormalizer
from app.kernel.perception.models import (
    ModalityType,
    Percept,
    RawMessage,
)
from app.kernel.perception.normalizers import InputNormalizer


class TestTextNormalizer:
    """Unit tests for the TextNormalizer."""

    @pytest.fixture
    def normalizer(self) -> TextNormalizer:
        return TextNormalizer()

    @pytest.mark.asyncio
    async def test_normalize_plain_text(self, normalizer: TextNormalizer) -> None:
        raw = RawMessage(content="Hello, world!")
        percept = await normalizer.normalize(raw)
        assert percept.normalized_content == "Hello, world!"
        assert percept.modality == ModalityType.TEXT
        assert percept.raw is raw
        assert isinstance(percept.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_normalize_strips_whitespace(
        self, normalizer: TextNormalizer
    ) -> None:
        raw = RawMessage(content="  Hello, world!  ")
        percept = await normalizer.normalize(raw)
        assert percept.normalized_content == "Hello, world!"

    @pytest.mark.asyncio
    async def test_normalize_collapses_spaces(self, normalizer: TextNormalizer) -> None:
        raw = RawMessage(content="Hello    world!   How   are   you?")
        percept = await normalizer.normalize(raw)
        assert "  " not in percept.normalized_content
        assert percept.normalized_content == "Hello world! How are you?"

    @pytest.mark.asyncio
    async def test_normalize_line_endings(self, normalizer: TextNormalizer) -> None:
        raw = RawMessage(content="Line1\r\nLine2\rLine3")
        percept = await normalizer.normalize(raw)
        assert percept.normalized_content == "Line1\nLine2\nLine3"

    @pytest.mark.asyncio
    async def test_extract_url(self, normalizer: TextNormalizer) -> None:
        raw = RawMessage(content="Visit https://example.com/path for info")
        percept = await normalizer.normalize(raw)
        urls = [e for e in percept.entities if e.type == "url"]
        assert len(urls) == 1
        assert urls[0].value == "https://example.com/path"
        assert urls[0].confidence == 1.0
        assert urls[0].position == (6, 30)

    @pytest.mark.asyncio
    async def test_extract_email(self, normalizer: TextNormalizer) -> None:
        raw = RawMessage(content="Contact me at user@example.com")
        percept = await normalizer.normalize(raw)
        emails = [e for e in percept.entities if e.type == "email"]
        assert len(emails) == 1
        assert emails[0].value == "user@example.com"
        assert emails[0].confidence == 1.0

    @pytest.mark.asyncio
    async def test_extract_mention(self, normalizer: TextNormalizer) -> None:
        raw = RawMessage(content="Hello @username!")
        percept = await normalizer.normalize(raw)
        mentions = [e for e in percept.entities if e.type == "mention"]
        assert len(mentions) == 1
        assert mentions[0].value == "@username"
        assert mentions[0].confidence == 0.9

    @pytest.mark.asyncio
    async def test_extract_hashtag(self, normalizer: TextNormalizer) -> None:
        raw = RawMessage(content="Trending #AI topic")
        percept = await normalizer.normalize(raw)
        hashtags = [e for e in percept.entities if e.type == "hashtag"]
        assert len(hashtags) == 1
        assert hashtags[0].value == "#AI"
        assert hashtags[0].confidence == 0.9

    @pytest.mark.asyncio
    async def test_extract_multiple_entities(self, normalizer: TextNormalizer) -> None:
        raw = RawMessage(
            content="Check https://github.com and email me@work.com @admin #v2"
        )
        percept = await normalizer.normalize(raw)
        type_counts: dict[str, int] = {}
        for e in percept.entities:
            type_counts[e.type] = type_counts.get(e.type, 0) + 1
        assert type_counts["url"] == 1
        assert type_counts["email"] == 1
        assert type_counts["mention"] >= 1
        assert type_counts["hashtag"] == 1

    @pytest.mark.asyncio
    async def test_decode_bytes(self, normalizer: TextNormalizer) -> None:
        raw = RawMessage(content=b"bytes input")
        percept = await normalizer.normalize(raw)
        assert percept.normalized_content == "bytes input"

    @pytest.mark.asyncio
    async def test_empty_content(self, normalizer: TextNormalizer) -> None:
        raw = RawMessage(content="")
        percept = await normalizer.normalize(raw)
        assert percept.normalized_content == ""
        assert percept.quality.overall < 0.3


class TestTextNormalizerQuality:
    """Quality score computation tests."""

    @pytest.fixture
    def normalizer(self) -> TextNormalizer:
        return TextNormalizer()

    @pytest.mark.asyncio
    async def test_empty_input_quality(self, normalizer: TextNormalizer) -> None:
        raw = RawMessage(content="")
        percept = await normalizer.normalize(raw)
        assert percept.quality.length_sufficiency == 0.0
        assert percept.quality.completeness == 0.0
        assert percept.quality.coherence == 0.0
        assert percept.quality.overall == 0.0

    @pytest.mark.asyncio
    async def test_single_word_quality(self, normalizer: TextNormalizer) -> None:
        raw = RawMessage(content="Hello")
        percept = await normalizer.normalize(raw)
        assert percept.quality.coherence == 0.3

    @pytest.mark.asyncio
    async def test_complete_sentence_quality(self, normalizer: TextNormalizer) -> None:
        raw = RawMessage(content="This is a complete sentence with enough words.")
        percept = await normalizer.normalize(raw)
        assert percept.quality.completeness == 1.0
        assert percept.quality.coherence == 0.9

    @pytest.mark.asyncio
    async def test_incomplete_ending_quality(self, normalizer: TextNormalizer) -> None:
        raw = RawMessage(content="This sentence ends with a comma,")
        percept = await normalizer.normalize(raw)
        assert percept.quality.completeness == 0.5


class TestPerceptionEngine:
    """Tests for the PerceptionEngine orchestrator."""

    @pytest.fixture
    def engine(self) -> PerceptionEngine:
        return PerceptionEngine()

    @pytest.mark.asyncio
    async def test_init_registers_text_normalizer(
        self, engine: PerceptionEngine
    ) -> None:
        assert engine._normalizers.get("text") is not None

    @pytest.mark.asyncio
    async def test_perceive_returns_percept(self, engine: PerceptionEngine) -> None:
        raw = RawMessage(content="Hello world")
        percept = await engine.perceive(raw)
        assert isinstance(percept, Percept)
        assert percept.normalized_content == "Hello world"
        assert percept.modality == ModalityType.TEXT

    @pytest.mark.asyncio
    async def test_perceive_preserves_raw(self, engine: PerceptionEngine) -> None:
        raw = RawMessage(content="test", source="cli")
        percept = await engine.perceive(raw)
        assert percept.raw.source == "cli"
        assert percept.raw.content == "test"

    @pytest.mark.asyncio
    async def test_perceive_uses_declared_modality(
        self, engine: PerceptionEngine
    ) -> None:
        raw = RawMessage(content="<svg>...</svg>", modality=ModalityType.IMAGE)
        percept = await engine.perceive(raw)
        assert percept.modality == ModalityType.IMAGE

    @pytest.mark.asyncio
    async def test_perceive_unknown_modality_fallback(
        self, engine: PerceptionEngine
    ) -> None:
        raw = RawMessage(content="some text", modality=ModalityType.UNKNOWN)
        percept = await engine.perceive(raw)
        assert percept.modality == ModalityType.TEXT

    @pytest.mark.asyncio
    async def test_perceive_bytes_input(self, engine: PerceptionEngine) -> None:
        raw = RawMessage(content=b"bytes input")
        percept = await engine.perceive(raw)
        assert percept.normalized_content == "bytes input"

    @pytest.mark.asyncio
    async def test_perceive_accepts_context_params(
        self, engine: PerceptionEngine
    ) -> None:
        raw = RawMessage(content="test")
        percept = await engine.perceive(
            raw, correlation_id="corr-1", session_id="sess-1", context_ref="ctx-1"
        )
        assert isinstance(percept, Percept)

    @pytest.mark.asyncio
    async def test_perceive_entity_extraction(self, engine: PerceptionEngine) -> None:
        raw = RawMessage(content="Visit https://example.com")
        percept = await engine.perceive(raw)
        assert len(percept.entities) == 1
        assert percept.entities[0].type == "url"

    @pytest.mark.asyncio
    async def test_perceive_quality_score(self, engine: PerceptionEngine) -> None:
        raw = RawMessage(content="A short message.")
        percept = await engine.perceive(raw)
        assert 0.0 <= percept.quality.overall <= 1.0
        assert 0.0 <= percept.quality.length_sufficiency <= 1.0
        assert 0.0 <= percept.quality.completeness <= 1.0
        assert 0.0 <= percept.quality.coherence <= 1.0

    @pytest.mark.asyncio
    async def test_register_normalizer(self, engine: PerceptionEngine) -> None:
        class MockNormalizer(InputNormalizer):
            async def normalize(self, raw: RawMessage) -> Percept:
                return Percept(raw=raw, modality=ModalityType.TEXT)

        engine.register_normalizer("mock", MockNormalizer())
        assert "mock" in engine._normalizers

    @pytest.mark.asyncio
    async def test_custom_normalizer_used(self, engine: PerceptionEngine) -> None:
        class CapsNormalizer(InputNormalizer):
            async def normalize(self, raw: RawMessage) -> Percept:
                text = (
                    raw.content
                    if isinstance(raw.content, str)
                    else raw.content.decode()
                )
                return Percept(
                    raw=raw,
                    modality=ModalityType.TEXT,
                    normalized_content=text.upper(),
                )

        engine.register_normalizer("text", CapsNormalizer())
        raw = RawMessage(content="hello")
        percept = await engine.perceive(raw)
        assert percept.normalized_content == "HELLO"


class TestPerceptionEngineModalityDetection:
    """Modality detection edge cases."""

    @pytest.mark.asyncio
    async def test_explicit_modality_preserved(self) -> None:
        engine = PerceptionEngine()
        raw = RawMessage(content="binary data", modality=ModalityType.SYSTEM_EVENT)
        percept = await engine.perceive(raw)
        assert percept.modality == ModalityType.SYSTEM_EVENT

    @pytest.mark.asyncio
    async def test_text_content_detected(self) -> None:
        engine = PerceptionEngine()
        raw = RawMessage(content="Hello", modality=ModalityType.UNKNOWN)
        percept = await engine.perceive(raw)
        assert percept.modality == ModalityType.TEXT

    @pytest.mark.asyncio
    async def test_bytes_text_detected(self) -> None:
        engine = PerceptionEngine()
        raw = RawMessage(content=b"Hello bytes", modality=ModalityType.UNKNOWN)
        percept = await engine.perceive(raw)
        assert percept.modality == ModalityType.TEXT


class TestPerceptionEngineFallback:
    """Fallback behavior for unregistered modalities."""

    @pytest.mark.asyncio
    async def test_unknown_modality_uses_fallback(self) -> None:
        engine = PerceptionEngine()
        raw = RawMessage(content="test", modality=ModalityType.AUDIO)
        percept = await engine.perceive(raw)
        assert isinstance(percept, Percept)
        assert percept.modality == ModalityType.AUDIO

    @pytest.mark.asyncio
    async def test_fallback_strips_whitespace(self) -> None:
        engine = PerceptionEngine()
        raw = RawMessage(content="  padded  ", modality=ModalityType.TOOL_RESULT)
        percept = await engine.perceive(raw)
        assert percept.normalized_content == "padded"
