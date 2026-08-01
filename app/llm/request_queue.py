"""Async request queue with concurrency and rate-limit control.

Provides a priority-aware queue that wraps LLM provider calls,
enforcing maximum concurrency and rate-limit awareness per provider.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from typing import Any, Awaitable

from app.llm.router_exceptions import QueueTimeoutError, RateLimitExceededError
from app.llm.router_models import RateLimitInfo

log = logging.getLogger(__name__)


class RequestPriority(IntEnum):
    """Priority levels for queued requests."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class QueuedRequest:
    """A request waiting in the queue.

    Attributes:
        id: Unique identifier for the request.
        priority: Scheduling priority.
        provider_id: Target provider ID.
        request: Opaque request payload (a :class:`ChatRequest`).
        future: Result future.
        submitted_at: Timestamp when the request was submitted.
    """

    id: str
    priority: RequestPriority
    provider_id: str
    request: Any
    future: asyncio.Future
    submitted_at: float

    def __lt__(self, other: QueuedRequest) -> bool:
        """Order queued requests by priority, then submission time."""
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.submitted_at != other.submitted_at:
            return self.submitted_at < other.submitted_at
        return self.id < other.id


class ProviderRateLimiter:
    """Per-provider rate limiter using sliding windows.

    Attributes:
        requests_per_minute: Max requests per minute.
        tokens_per_minute: Max tokens per minute.
        _request_timestamps: Rolling window of request timestamps.
        _token_timestamps: Rolling window of token submission timestamps.
        _lock: Protects internal state.
    """

    def __init__(
        self,
        requests_per_minute: int = 0,
        tokens_per_minute: int = 0,
    ) -> None:
        self.requests_per_minute = max(requests_per_minute, 0)
        self.tokens_per_minute = max(tokens_per_minute, 0)
        self._request_timestamps: list[float] = []
        self._token_events: list[tuple[float, int]] = []
        self._lock = asyncio.Lock()

    def can_make_request(self, tokens: int = 0) -> bool:
        """Return ``True`` if a request with *tokens* would be allowed."""
        now = time.monotonic()
        window = 60.0

        self._request_timestamps = [
            t for t in self._request_timestamps if now - t < window
        ]
        if (
            self.requests_per_minute > 0
            and len(self._request_timestamps) >= self.requests_per_minute
        ):
            return False

        if self.tokens_per_minute > 0 and tokens > 0:
            self._token_events = [
                (t, c) for t, c in self._token_events if now - t < window
            ]
            used_tokens = sum(c for _, c in self._token_events)
            if used_tokens + tokens > self.tokens_per_minute:
                return False

        self._request_timestamps.append(now)
        if tokens > 0 and self.tokens_per_minute > 0:
            self._token_events.append((now, tokens))
        return True

    def get_info(self) -> RateLimitInfo:
        """Return current rate limit info."""
        now = time.monotonic()
        window = 60.0
        recent_requests = [t for t in self._request_timestamps if now - t < window]
        recent_tokens_count = sum(c for t, c in self._token_events if now - t < window)

        reset_at = None
        if self._request_timestamps:
            oldest = min(self._request_timestamps)
            reset_at = (
                datetime.fromtimestamp(oldest + window)
                if (oldest + window - now) > 0
                else datetime.utcnow()
            )

        return RateLimitInfo(
            requests_per_minute=self.requests_per_minute,
            tokens_per_minute=self.tokens_per_minute,
            current_rpm=len(recent_requests),
            current_tpm=recent_tokens_count,
            reset_at=reset_at,
        )

    def wait_time(self, tokens: int = 0) -> float:
        """Return estimated wait time in seconds before a request is allowed."""
        if self.requests_per_minute <= 0 and self.tokens_per_minute <= 0:
            return 0.0
        now = time.monotonic()
        window = 60.0

        if self.requests_per_minute > 0:
            recent = [t for t in self._request_timestamps if now - t < window]
            if len(recent) >= self.requests_per_minute:
                oldest = min(recent)
                return max(oldest + window - now, 0.0)

        if self.tokens_per_minute > 0 and tokens > 0:
            token_recent = [(t, c) for t, c in self._token_events if now - t < window]
            used_tokens = sum(c for _, c in token_recent)
            if used_tokens + tokens > self.tokens_per_minute:
                oldest = min(t for t, _ in token_recent)
                return max(oldest + window - now, 0.0)

        return 0.0

    def reset(self) -> None:
        """Reset all tracked timestamps."""
        self._request_timestamps.clear()
        self._token_events.clear()


class RequestQueue:
    """Async priority queue with per-provider concurrency control.

    Attributes:
        max_concurrency: Max concurrent requests per provider.
        global_max_concurrency: Max total concurrent requests.
        default_timeout: Timeout for waiting in queue.
        _semaphores: Per-provider semaphores.
        _global_semaphore: Global semaphore.
        _queues: Per-provider asyncio priority queues.
        _rate_limiters: Per-provider rate limiters.
        _lock: Protects queue initialization.
        _running: Set of currently processing request IDs.
        _workers: Background worker tasks.
    """

    def __init__(
        self,
        max_concurrency: int = 5,
        global_max_concurrency: int = 20,
        default_timeout: float = 30.0,
    ) -> None:
        self.max_concurrency = max_concurrency
        self.global_max_concurrency = global_max_concurrency
        self.default_timeout = default_timeout

        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._global_semaphore = asyncio.Semaphore(global_max_concurrency)
        self._queues: dict[str, asyncio.PriorityQueue] = {}
        self._rate_limiters: dict[str, ProviderRateLimiter] = {}
        self._lock = asyncio.Lock()
        self._running: set[str] = set()
        self._workers: dict[str, asyncio.Task] = {}
        self._shutdown = False
        self._provider_semaphores: dict[str, asyncio.Semaphore] = {}

    async def _ensure_provider_slot(self, provider_id: str) -> None:
        """Ensure queues and semaphores exist for a provider."""
        async with self._lock:
            if provider_id not in self._queues:
                self._queues[provider_id] = asyncio.PriorityQueue()
                self._provider_semaphores[provider_id] = asyncio.Semaphore(
                    self.max_concurrency
                )
                self._semaphores[provider_id] = asyncio.Semaphore(self.max_concurrency)
                if provider_id not in self._rate_limiters:
                    self._rate_limiters[provider_id] = ProviderRateLimiter()
                self._workers[provider_id] = asyncio.create_task(
                    self._worker(provider_id)
                )

    async def _worker(self, provider_id: str) -> None:
        """Background worker that processes the queue for a provider."""
        while not self._shutdown:
            try:
                priority, item = await asyncio.wait_for(
                    self._queues[provider_id].get(),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            async with self._global_semaphore:
                async with self._provider_semaphores[provider_id]:
                    self._running.add(item.id)
                    try:
                        result = await item.request
                        item.future.set_result(result)
                    except Exception as exc:
                        item.future.set_exception(exc)
                    finally:
                        self._running.discard(item.id)
                        self._queues[provider_id].task_done()

    async def submit(
        self,
        provider_id: str,
        request: Awaitable,
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: float | None = None,
        tokens: int = 0,
    ) -> Any:
        """Submit a request to the queue and await the result.

        Args:
            provider_id: Provider to route to.
            request: Awaitable that performs the actual LLM call.
            priority: Priority level for the request.
            timeout: Queue wait timeout (defaults to ``default_timeout``).
            tokens: Estimated tokens for rate-limit checking.

        Returns:
            The result of the request.

        Raises:
            QueueTimeoutError: If the request times out waiting in the queue.
            RateLimitExceededError: If rate limits prevent submission.
        """
        await self._ensure_provider_slot(provider_id)

        limiter = self._rate_limiters[provider_id]
        if not limiter.can_make_request(tokens):
            wait = limiter.wait_time(tokens)
            if asyncio.iscoroutine(request):
                request.close()
            raise RateLimitExceededError(
                f"Provider '{provider_id}' is rate-limited, "
                f"wait {wait:.1f}s before retrying",
                module="llm.request_queue",
                context={
                    "provider": provider_id,
                    "wait_seconds": wait,
                    "tokens": tokens,
                },
            )

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        item = QueuedRequest(
            id=f"{provider_id}_{time.monotonic_ns()}",
            priority=priority,
            provider_id=provider_id,
            request=request,
            future=future,
            submitted_at=time.monotonic(),
        )

        await self._queues[provider_id].put((priority.value, item))

        try:
            result = await asyncio.wait_for(
                future, timeout=timeout or self.default_timeout
            )
            return result
        except asyncio.TimeoutError as exc:
            raise QueueTimeoutError(
                f"Request to provider '{provider_id}' timed out after "
                f"{timeout or self.default_timeout}s in queue",
                module="llm.request_queue",
                context={
                    "provider": provider_id,
                    "timeout": timeout or self.default_timeout,
                },
            ) from exc

    def get_rate_limiter(self, provider_id: str) -> ProviderRateLimiter | None:
        """Return the rate limiter for a provider, or ``None``."""
        return self._rate_limiters.get(provider_id)

    def get_pending_count(self, provider_id: str | None = None) -> int:
        """Return the number of pending requests (in queue or processing)."""
        if provider_id is None:
            return len(self._running)
        q = self._queues.get(provider_id)
        queue_size = q.qsize() if q else 0
        running = 1 if provider_id in self._running else 0
        return queue_size + running

    async def shutdown(self) -> None:
        """Shut down all worker tasks."""
        self._shutdown = True
        for worker in self._workers.values():
            worker.cancel()
        await asyncio.gather(*self._workers.values(), return_exceptions=True)

    def set_rate_limit(self, provider_id: str, rpm: int, tpm: int) -> None:
        """Configure rate limits for a provider."""
        self._rate_limiters[provider_id] = ProviderRateLimiter(rpm, tpm)
