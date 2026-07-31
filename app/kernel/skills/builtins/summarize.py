"""Summarize skill — deterministic text summarization."""

from __future__ import annotations

import re
from typing import Any


async def execute(parameters: dict[str, Any]) -> Any:
    """Summarize text deterministically.

    Extracts key sentences and produces a concise summary.

    Args:
        parameters: Must contain 'text' key.
                    Optional 'max_sentences' (default 3).

    Returns:
        Summary string.
    """
    text = parameters.get("text", "")
    if not text:
        return ""

    max_sentences = parameters.get("max_sentences", 3)

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return text.strip()

    # Score sentences by position (earlier = more important) and length
    scored: list[tuple[float, str]] = []
    for i, sentence in enumerate(sentences):
        position_score = 1.0 - (i / max(len(sentences), 1))
        length = len(sentence.split())
        length_score = min(length / 20.0, 1.0)
        score = position_score * 0.7 + length_score * 0.3
        scored.append((score, sentence))

    # Sort by score descending, take top N
    scored.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [s for _, s in scored[:max_sentences]]

    # Re-sort by original position for readability
    original_positions = {s: i for i, s in enumerate(sentences)}
    top_sentences.sort(key=lambda s: original_positions.get(s, 0))

    return " ".join(top_sentences)
