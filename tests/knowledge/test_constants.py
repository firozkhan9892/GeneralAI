"""Tests for knowledge constants."""

from app.knowledge.constants import (
    CHARS_PER_TOKEN,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_NAMESPACE,
    RECURSIVE_SEPARATORS,
)


def test_default_chunk_size_is_positive() -> None:
    assert DEFAULT_CHUNK_SIZE > 0


def test_default_chunk_overlap_is_non_negative() -> None:
    assert DEFAULT_CHUNK_OVERLAP >= 0


def test_overlap_less_than_chunk_size() -> None:
    assert DEFAULT_CHUNK_OVERLAP < DEFAULT_CHUNK_SIZE


def test_chars_per_token_positive() -> None:
    assert CHARS_PER_TOKEN > 0


def test_default_namespace_is_non_empty() -> None:
    assert DEFAULT_NAMESPACE


def test_recursive_separators_is_non_empty_tuple() -> None:
    assert isinstance(RECURSIVE_SEPARATORS, tuple)
    assert len(RECURSIVE_SEPARATORS) >= 2
