"""Tests for embedding providers."""

import math

import pytest

from app.knowledge.embeddings.mock import MockEmbeddingProvider


def test_mock_provider_deterministic() -> None:
    p = MockEmbeddingProvider(dimensions=64)
    v1 = p.embed(["hello"])[0]
    v2 = p.embed(["hello"])[0]
    assert v1 == v2


def test_mock_provider_different_inputs() -> None:
    p = MockEmbeddingProvider(dimensions=64)
    v1 = p.embed(["hello"])[0]
    v2 = p.embed(["world"])[0]
    assert v1 != v2


def test_mock_provider_dimensions() -> None:
    p = MockEmbeddingProvider(dimensions=128)
    assert p.dimensions == 128
    v = p.embed(["test"])[0]
    assert len(v) == 128


def test_mock_provider_unit_vectors() -> None:
    p = MockEmbeddingProvider(dimensions=64)
    v = p.embed(["test"])[0]
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-6


def test_mock_provider_batch() -> None:
    p = MockEmbeddingProvider(dimensions=32)
    vectors = p.embed(["a", "b", "c"])
    assert len(vectors) == 3
    assert all(len(v) == 32 for v in vectors)


def test_mock_provider_model_info() -> None:
    p = MockEmbeddingProvider(dimensions=64)
    info = p.model_info()
    assert info.name == "mock"
    assert info.dimensions == 64
    assert info.model == "mock-v1"


def test_mock_provider_negative_dimensions() -> None:
    with pytest.raises(ValueError):
        MockEmbeddingProvider(dimensions=0)


def test_mock_provider_empty_input() -> None:
    p = MockEmbeddingProvider()
    assert p.embed([]) == []


def test_mock_provider_whitespace_normalisation() -> None:
    p = MockEmbeddingProvider(dimensions=64)
    v1 = p.embed(["  hello  "])[0]
    v2 = p.embed(["hello"])[0]
    assert v1 == v2
