"""Knowledge subsystem constants.

Default values for chunking, indexing, caching, and namespace
management.  Kept separate from :mod:`config` so they are importable
without pulling in pydantic-settings.
"""

from __future__ import annotations

import typing

# ── Chunking defaults ────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE: typing.Final[int] = 1000
"""Maximum character count per chunk."""

DEFAULT_CHUNK_OVERLAP: typing.Final[int] = 200
"""Character overlap between consecutive chunks."""

DEFAULT_MAX_TOKENS_PER_CHUNK: typing.Final[int | None] = None
"""Optional hard token limit per chunk (None = no limit)."""

# ── Separators for recursive chunking ────────────────────────────────
RECURSIVE_SEPARATORS: typing.Final[tuple[str, ...]] = (
    "\n\n\n",
    "\n\n",
    "\n",
    ". ",
    " ",
)
"""Separator hierarchy from coarsest to finest, tried in order by
:class:`RecursiveChunker`."""

# ── Token estimation ─────────────────────────────────────────────────
CHARS_PER_TOKEN: typing.Final[float] = 4.0
"""Rough heuristic: one token ≈ 4 characters (English)."""

# ── Namespace / collection defaults ──────────────────────────────────
DEFAULT_NAMESPACE: typing.Final[str] = "default"
DEFAULT_COLLECTION_NAME: typing.Final[str] = "default"

# ── Metadata keys propagated from loaders ────────────────────────────
META_KEY_PAGE: typing.Final[str] = "page"
META_KEY_SOURCE_URI: typing.Final[str] = "source_uri"
META_KEY_FORMAT: typing.Final[str] = "format"
META_KEY_TITLE: typing.Final[str] = "title"
META_KEY_HEADING: typing.Final[str] = "heading"
