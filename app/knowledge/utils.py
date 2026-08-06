"""Knowledge subsystem utility functions.

Content hashing, token estimation, and text normalisation helpers
used by loaders and chunkers.
"""

from __future__ import annotations

import hashlib
import re

from app.knowledge.constants import CHARS_PER_TOKEN


def compute_content_hash(text: str) -> str:
    """Return the SHA-256 hex-digest of *text*.

    The text is UTF-8 encoded after stripping leading/trailing
    whitespace so that trivially different representations of the
    same content produce the same hash.
    """
    normalised = text.strip()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def estimate_token_count(text: str) -> int:
    """Estimate the number of tokens in *text*.

    Uses the :data:`CHARS_PER_TOKEN` heuristic (≈4 chars/token for
    English).  For production use with tiktoken, override this via a
    registered provider.
    """
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def normalise_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces and strip."""
    return re.sub(r"\s+", " ", text).strip()


def extract_title_from_content(text: str, max_len: int = 120) -> str:
    """Heuristically extract a title from the first non-empty line."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:max_len]
    return ""
