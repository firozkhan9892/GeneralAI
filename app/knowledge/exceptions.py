"""Knowledge subsystem exception hierarchy.

Every knowledge exception subclasses :class:`KnowledgeError`, which in
turn subclasses :class:`GeneralAIError` so that top-level handlers and
the server can report them uniformly.  The hierarchy is minimal and
extensible: add subclasses here as new failure modes emerge.
"""

from __future__ import annotations

from typing import Any

from app.core.exceptions import GeneralAIError


class KnowledgeError(GeneralAIError):
    """Base error for all knowledge subsystem failures."""

    def __init__(
        self,
        message: str = "",
        *,
        module: str = "knowledge",
        cause: Exception | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Initialise a :class:`KnowledgeError`.

        Args:
            message: Human-readable description of the error.
            module: Name of the module that raised the error
                (defaults to ``knowledge``).
            cause: The original exception that caused this error.
            context: Arbitrary key-value pairs with additional context.
        """
        super().__init__(message, module=module, cause=cause, context=context)


class KnowledgeValidationError(KnowledgeError):
    """Raised when input does not satisfy a knowledge constraint."""


class KnowledgeNamespaceNotFoundError(KnowledgeError):
    """Raised when a namespace does not exist."""


class KnowledgeCollectionNotFoundError(KnowledgeError):
    """Raised when a collection does not exist."""


class KnowledgeDocumentNotFoundError(KnowledgeError):
    """Raised when a document does not exist."""


class KnowledgeChunkNotFoundError(KnowledgeError):
    """Raised when a chunk does not exist."""


class KnowledgeDuplicateError(KnowledgeError):
    """Raised when a unique resource already exists."""


class KnowledgeUnsupportedFormatError(KnowledgeError):
    """Raised when a document format has no registered loader."""


class KnowledgeIngestionError(KnowledgeError):
    """Raised when a document fails to ingest."""


class KnowledgeIndexError(KnowledgeError):
    """Raised when a vector store / index operation fails."""


class KnowledgeVersionError(KnowledgeError):
    """Raised when a document version conflict occurs."""
