"""Sentence-transformers embedding provider.

Uses the ``sentence-transformers`` library (lazy-imported) to produce
real embeddings from a local model.  Falls back to the mock provider
when the dependency is unavailable.
"""

from __future__ import annotations

import math

from app.knowledge.base import EmbeddingProvider
from app.knowledge.exceptions import KnowledgeIngestionError
from app.knowledge.models import EmbeddingModelInfo


def _import_sentence_transformers():  # type: ignore[no-untyped-def]
    """Lazy-import ``sentence_transformers``."""
    try:
        from sentence_transformers import SentenceTransformer as _Model

        return _Model
    except ImportError as exc:
        raise KnowledgeIngestionError(
            "sentence-transformers is required for SentenceTransformerEmbeddingProvider.  "
            "Install it with: pip install sentence-transformers",
            cause=exc,
        ) from exc


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by a ``sentence-transformers`` model.

    Parameters
    ----------
    model_name:
        The HuggingFace model name (e.g. ``"all-MiniLM-L6-v2"``).
    dimensions:
        Expected output dimensionality.  If ``None``, inferred from
        the model after loading.
    """

    name = "sentence_transformer"

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dimensions: int | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = None
        self._dimensions = dimensions

    def _get_model(self):  # type: ignore[no-untyped-def]
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            Model = _import_sentence_transformers()
            self._model = Model(self.model_name)
            if self._dimensions is None:
                self._dimensions = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensionality."""
        if self._dimensions is None:
            self._get_model()
        return self._dimensions or 384  # fallback

    @dimensions.setter
    def dimensions(self, value: int) -> None:  # type: ignore[override]
        self._dimensions = value

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* using the sentence-transformers model."""
        model = self._get_model()
        embeddings = model.encode(texts, show_progress_bar=False)
        # Normalise to unit vectors
        result: list[list[float]] = []
        for vec in embeddings:
            norm = math.sqrt(sum(float(x) * float(x) for x in vec))
            if norm > 0:
                result.append([float(x) / norm for x in vec])
            else:
                result.append([0.0] * self.dimensions)
        return result

    def model_info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            name=self.name,
            provider="sentence_transformers",
            dimensions=self.dimensions,
            model=self.model_name,
            max_input_tokens=None,
        )
