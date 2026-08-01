"""Analytics for LLM request tracking and provider ranking.

Records every request event, aggregates metrics per provider,
and computes rankings based on success rate, latency, cost,
and usage.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

from app.llm.router_models import (
    AnalyticsEvent,
    ProviderRanking,
    RouterStrategy,
)

log = logging.getLogger(__name__)


class LLMAnalytics:
    """Thread-safe analytics collector for LLM provider metrics.

    Maintains a rolling log of :class:`AnalyticsEvent` records and
    computes aggregate :class:`ProviderRanking` snapshots.

    Attributes:
        _events: List of all recorded events (bounded by ``max_events``).
        _provider_events: Per-provider event lists for fast aggregation.
        _lock: Protects all internal state.
        _max_events: Maximum total events to retain.
    """

    def __init__(self, max_events: int = 10000) -> None:
        self._max_events = max_events
        self._events: list[AnalyticsEvent] = []
        self._provider_events: dict[str, list[AnalyticsEvent]] = defaultdict(list)
        self._lock = threading.RLock()
        self._started_at = time.monotonic()

    def record(
        self,
        provider_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        latency: float,
        first_token_latency: float,
        success: bool,
        error: str | None = None,
        estimated_cost: float = 0.0,
        cached: bool = False,
        strategy: RouterStrategy = RouterStrategy.SMART,
        request_messages: int = 0,
    ) -> AnalyticsEvent:
        """Record a single LLM request event.

        All arguments are stored verbatim for later aggregation.

        Args:
            provider_id: Provider that handled the request.
            model: Model name used.
            prompt_tokens: Prompt token count.
            completion_tokens: Completion token count.
            total_tokens: Total token count.
            latency: Total response time in seconds.
            first_token_latency: Time to first token in seconds.
            success: Whether the request succeeded.
            error: Error message if failed.
            estimated_cost: Cost in USD.
            cached: Whether the response was served from cache.
            strategy: Routing strategy used.
            request_messages: Number of messages in the request.

        Returns:
            The created :class:`AnalyticsEvent`.
        """
        event = AnalyticsEvent(
            provider_id=provider_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency=latency,
            first_token_latency=first_token_latency,
            success=success,
            error=error,
            estimated_cost=estimated_cost,
            cached=cached,
            strategy=strategy,
            request_messages=request_messages,
        )

        with self._lock:
            self._events.append(event)
            self._provider_events[provider_id].append(event)

            if len(self._events) > self._max_events:
                removed = self._events.pop(0)
                if removed.provider_id in self._provider_events:
                    events = self._provider_events[removed.provider_id]
                    if events and events[0] is removed:
                        events.pop(0)

        log.debug(
            "Recorded analytics event for '%s' (success=%s, %d tokens)",
            provider_id,
            success,
            total_tokens,
        )
        return event

    def get_events(
        self,
        provider_id: str | None = None,
        limit: int | None = None,
    ) -> list[AnalyticsEvent]:
        """Return recorded events, optionally filtered by provider.

        Args:
            provider_id: Filter to a specific provider.
            limit: Maximum number of events to return.

        Returns:
            List of events (most recent last).
        """
        with self._lock:
            if provider_id is not None:
                events = self._provider_events.get(provider_id, [])
            else:
                events = self._events

            if limit is not None:
                return list(events[-limit:])
            return list(events)

    def get_provider_ranking(self, provider_id: str) -> ProviderRanking:
        """Compute an aggregate ranking for a single provider.

        Args:
            provider_id: Provider name.

        Returns:
            A :class:`ProviderRanking` with aggregated stats.
        """
        with self._lock:
            events = self._provider_events.get(provider_id, [])

            total_requests = len(events)
            if total_requests == 0:
                return ProviderRanking(
                    provider_id=provider_id,
                    total_requests=0,
                )

            successes = sum(1 for e in events if e.success)
            success_rate = successes / total_requests

            latencies = [e.latency for e in events if e.success]
            avg_latency = (
                round(sum(latencies) / len(latencies), 9) if latencies else 0.0
            )

            total_tokens = sum(e.total_tokens for e in events)
            total_cost = sum(e.estimated_cost for e in events)

            cache_hits = sum(1 for e in events if e.cached)
            cache_hit_rate = cache_hits / total_requests

            first_events = [e for e in events if e.timestamp]
            if first_events:
                now = datetime.utcnow()
                minute_events = [
                    e for e in events if (now - e.timestamp).total_seconds() <= 60.0
                ]
                rpm = (
                    len(minute_events)
                    / max(
                        (now - min(e.timestamp for e in minute_events)).total_seconds()
                        / 60.0,
                        1.0 / 60.0,
                    )
                    if minute_events
                    else 0.0
                )

                minute_tokens = sum(e.total_tokens for e in minute_events)
                tpm = (
                    minute_tokens
                    / max(
                        (now - min(e.timestamp for e in minute_events)).total_seconds()
                        / 60.0,
                        1.0 / 60.0,
                    )
                    if minute_events
                    else 0.0
                )
            else:
                rpm = 0.0
                tpm = 0.0

            last_used = max(e.timestamp for e in events)

            total_time = (
                datetime.utcnow() - min(e.timestamp for e in events)
            ).total_seconds()
            downtime_seconds = sum(e.latency for e in events if not e.success)
            uptime = max(
                1.0 - (downtime_seconds / total_time) if total_time > 0 else 1.0,
                0.0,
            )

            return ProviderRanking(
                provider_id=provider_id,
                total_requests=total_requests,
                success_rate=success_rate,
                avg_latency=avg_latency,
                total_tokens=total_tokens,
                total_cost=total_cost,
                cache_hit_rate=cache_hit_rate,
                requests_per_minute=rpm,
                tokens_per_minute=tpm,
                uptime=uptime,
                last_used=last_used,
            )

    def get_all_rankings(self) -> dict[str, ProviderRanking]:
        """Compute rankings for all tracked providers.

        Returns:
            Dict mapping provider_id to :class:`ProviderRanking`.
        """
        with self._lock:
            return {
                pid: self.get_provider_ranking(pid)
                for pid in self._provider_events.keys()
            }

    def get_total_cost(self) -> float:
        """Return the total estimated cost across all providers."""
        with self._lock:
            return sum(e.estimated_cost for e in self._events)

    def get_total_tokens(self) -> tuple[int, int, int]:
        """Return ``(prompt_tokens, completion_tokens, total_tokens)``."""
        with self._lock:
            prompt = sum(e.prompt_tokens for e in self._events)
            completion = sum(e.completion_tokens for e in self._events)
            return prompt, completion, prompt + completion

    def get_cache_hit_rate(self) -> float:
        """Return the overall cache hit rate (0.0 - 1.0)."""
        with self._lock:
            total = len(self._events)
            if total == 0:
                return 0.0
            hits = sum(1 for e in self._events if e.cached)
            return hits / total

    def get_overall_stats(self) -> dict[str, Any]:
        """Return a summary of all tracked statistics.

        Returns:
            Dict with: total_requests, total_cost, total_tokens,
            cache_hit_rate, success_rate, uptime, provider_count,
            elapsed_seconds.
        """
        with self._lock:
            total = len(self._events)
            successes = sum(1 for e in self._events if e.success)
            return {
                "total_requests": total,
                "total_cost": sum(e.estimated_cost for e in self._events),
                "total_tokens": sum(e.total_tokens for e in self._events),
                "cache_hit_rate": (
                    sum(1 for e in self._events if e.cached) / total
                    if total > 0
                    else 0.0
                ),
                "success_rate": successes / total if total > 0 else 0.0,
                "provider_count": len(self._provider_events),
                "elapsed_seconds": time.monotonic() - self._started_at,
            }

    def clear(self) -> None:
        """Clear all recorded events and reset counters."""
        with self._lock:
            self._events.clear()
            self._provider_events.clear()
            self._started_at = time.monotonic()
