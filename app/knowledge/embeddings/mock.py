"""Deterministic mock embedding provider.

Produces fixed-dimension normalised vectors via a feature-hash of the
input text.  Identical inputs always produce identical outputs — ideal
for unit tests and as the default offline provider.
"""

from __future__ import annotations

import hashlib
import math

from app.knowledge.base import EmbeddingProvider
from app.knowledge.models import EmbeddingModelInfo


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic, dependency-free embedding provider.

    Parameters
    ----------
    dimensions:
        Output vector dimensionality.  Defaults to 128.
    """

    name = "mock"
    model_name = "mock-v1"

    def __init__(self, dimensions: int = 128) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be >= 1")
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic normalised vectors for *texts*."""
        return [self._hash_embed(t) for t in texts]

    def _hash_embed(self, text: str) -> list[float]:
        """Feature-hash *text* into a unit vector of ``self.dimensions``."""
        seed = int(hashlib.sha256(text.strip().encode("utf-8")).hexdigest(), 16)
        # Generate pseudo-random floats from the seed
        rng = _Mulberry32(seed)
        vec = [rng.next_float() for _ in range(self.dimensions)]
        # Normalise to unit vector
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def model_info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            name=self.name,
            provider=self.name,
            dimensions=self.dimensions,
            model=self.model_name,
            max_input_tokens=None,
        )


class _Mulberry32:
    """Simple deterministic 32-bit PRNG (Mulberry32)."""

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        self._state = seed & 0xFFFFFFFF

    def next_float(self) -> float:
        """Return a float in [0, 1)."""
        self._state = (self._state + 0x6D2B79F5) & 0xFFFFFFFF
        z = self._state
        z = ((z ^ (z >> 15)) * (z | 1)) & 0xFFFFFFFF
        z ^= z + ((z ^ (z >> 7)) * (z | 61)) & 0xFFFFFFFF
        z ^= z >> 14
        return (z & 0x7FFFFFFF) / 0x80000000
