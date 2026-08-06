"""Tests for knowledge configuration."""

import pytest

from app.knowledge.config import KnowledgeSettings


def test_default_settings() -> None:
    s = KnowledgeSettings()
    assert s.default_namespace == "default"
    assert s.default_chunk_size >= 100
    assert s.default_chunk_overlap >= 0
    assert s.embedding_cache_size > 0
    assert s.index_workers >= 1
    assert s.keep_versions >= 1


def test_settings_frozen() -> None:
    s = KnowledgeSettings()
    with pytest.raises(Exception):
        s.default_chunk_size = 999  # type: ignore[misc]


def test_settings_custom_values() -> None:
    s = KnowledgeSettings(
        default_chunk_size=500,
        default_chunk_overlap=50,
        embedding_cache_size=1000,
    )
    assert s.default_chunk_size == 500
    assert s.default_chunk_overlap == 50
    assert s.embedding_cache_size == 1000


def test_settings_validation_rejects_small_chunk() -> None:
    with pytest.raises(Exception):
        KnowledgeSettings(default_chunk_size=10)  # ge=100


def test_settings_validation_rejects_negative_overlap() -> None:
    with pytest.raises(Exception):
        KnowledgeSettings(default_chunk_overlap=-1)
