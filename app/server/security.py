"""Authentication and rate limiting for the FastAPI server.

- :class:`RateLimiter` is a thread-safe, fixed-window limiter keyed by
  request identity (API key or client IP).
- ``require_api_key`` and ``rate_limit`` are FastAPI dependencies
  applied to every non-public route.
"""

from __future__ import annotations

import logging
import threading
import time

from fastapi import HTTPException, Request, status

from app.server.config import ServerSettings

log = logging.getLogger(__name__)

#: Header name used to carry the shared API key.
API_KEY_HEADER = "X-API-Key"
#: Query parameter fallback for the shared API key.
API_KEY_QUERY = "api_key"


class RateLimiter:
    """Fixed-window rate limiter.

    Each identity tracks a sliding fixed window of ``window_s`` seconds.
    ``consume`` returns ``(allowed, retry_after_s)``.

    Args:
        limit: Maximum requests allowed within a window.
        window_s: Window length in seconds.
    """

    def __init__(self, limit: int, window_s: float = 60.0) -> None:
        self._limit = limit
        self._window_s = window_s
        self._hits: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    def consume(self, identity: str) -> tuple[bool, float]:
        """Attempt to consume one request slot for *identity*.

        Args:
            identity: The request identity (API key or client IP).

        Returns:
            ``(True, 0.0)`` when allowed, otherwise
            ``(False, retry_after_s)``.
        """
        now = time.monotonic()
        with self._lock:
            window_start, count = self._hits.get(identity, (now, 0))
            if now - window_start >= self._window_s:
                window_start, count = now, 0
            count += 1
            self._hits[identity] = (window_start, count)
            if count > self._limit:
                retry_after = max(0.0, window_start + self._window_s - now)
                return False, retry_after
            return True, 0.0

    @property
    def limit(self) -> int:
        """Return the configured request limit per window."""
        return self._limit

    def clear(self) -> None:
        """Drop all tracked identities."""
        with self._lock:
            self._hits.clear()


def _provided_api_key(request: Request) -> str | None:
    """Return the API key supplied by the request, if any."""
    header = request.headers.get(API_KEY_HEADER)
    if header:
        return header
    return request.query_params.get(API_KEY_QUERY)


async def require_api_key(request: Request) -> None:
    """Reject requests that lack the configured API key.

    When ``ServerSettings.api_key`` is ``None`` authentication is
    disabled and every request passes.

    Raises:
        HTTPException: 401 when the key is missing or invalid.
    """
    settings: ServerSettings = request.app.state.settings
    if not settings.api_key:
        return
    if _provided_api_key(request) != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )


async def rate_limit(request: Request) -> None:
    """Enforce the fixed-window rate limit for the current request.

    The identity is the API key when one is presented, otherwise the
    client host.  When rate limiting is disabled the check is a no-op.

    Raises:
        HTTPException: 429 when the limit is exceeded.
    """
    settings: ServerSettings = request.app.state.settings
    if not settings.rate_limit_enabled:
        return

    identity = _provided_api_key(request) or _client_identity(request)
    limiter = request.app.state.rate_limiter
    allowed, retry_after = limiter.consume(identity)
    if not allowed:
        log.warning("Rate limit exceeded for identity %r", identity)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )


def _client_identity(request: Request) -> str:
    """Return a stable identity for an anonymous client."""
    client = request.client
    return client.host if client is not None else "unknown"
