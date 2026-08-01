"""Health monitoring for LLM providers.

Tracks per-provider metrics: latency, success/failure counts, uptime,
RPM, TPM, and exposes health snapshots used by the router and circuit
breaker.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime

from app.llm.router_models import ProviderHealthSnapshot

log = logging.getLogger(__name__)


class ProviderHealthMonitor:
    """Thread-safe tracker of provider health metrics.

    Maintains a rolling window of recent request events per provider
    and computes aggregates (latency, RPM, TPM, success rate, uptime).

    Attributes:
        _events: Rolling window of timestamps for requests per provider.
        _successes: Count of successful requests per provider.
        _failures: Count of failed requests per provider.
        _latencies: Rolling window of latencies per provider.
        _tokens: Rolling window of token counts per provider.
        _last_success: Timestamp of last successful request.
        _last_failure: Timestamp of last failed request.
        _last_failure_error: Error message from last failure.
        _uptime_start: When monitoring started per provider.
        _total_uptime: Accumulated uptime seconds per provider.
        _total_downtime: Accumulated downtime seconds per provider.
        _circuit_open_until: When the circuit breaker is open until per provider.
        _lock: Protects all internal state.
        _window_size: Number of events to retain for rolling stats.
        _rpm_window_seconds: Window for RPM calculation.
    """

    def __init__(
        self,
        window_size: int = 100,
        rpm_window_seconds: float = 60.0,
    ) -> None:
        self._lock = threading.RLock()
        self._window_size = window_size
        self._rpm_window_seconds = rpm_window_seconds

        self._events: dict[str, deque[float]] = {}
        self._successes: dict[str, int] = {}
        self._failures: dict[str, int] = {}
        self._latencies: dict[str, deque[float]] = {}
        self._tokens: dict[str, deque[int]] = {}
        self._last_success: dict[str, datetime | None] = {}
        self._last_failure: dict[str, datetime | None] = {}
        self._last_failure_error: dict[str, str | None] = {}
        self._uptime_start: dict[str, float | None] = {}
        self._total_uptime: dict[str, float] = {}
        self._total_downtime: dict[str, float] = {}
        self._circuit_open_until: dict[str, float | None] = {}
        self._circuit_last_state_change: dict[str, float | None] = {}

    def _ensure_keys(self, provider_id: str) -> None:
        """Initialise all tracking dicts for a provider if needed."""
        if provider_id not in self._successes:
            self._successes[provider_id] = 0
            self._failures[provider_id] = 0
            self._events[provider_id] = deque(maxlen=self._window_size)
            self._latencies[provider_id] = deque(maxlen=self._window_size)
            self._tokens[provider_id] = deque(maxlen=self._window_size)
            self._last_success[provider_id] = None
            self._last_failure[provider_id] = None
            self._last_failure_error[provider_id] = None
            self._uptime_start[provider_id] = time.monotonic()
            self._total_uptime[provider_id] = 0.0
            self._total_downtime[provider_id] = 0.0
            self._circuit_open_until[provider_id] = None
            self._circuit_last_state_change[provider_id] = time.monotonic()

    def record_success(
        self,
        provider_id: str,
        latency: float,
        tokens: int,
    ) -> None:
        """Record a successful request.

        Args:
            provider_id: Provider name.
            latency: Response latency in seconds.
            tokens: Total tokens processed.
        """
        with self._lock:
            self._ensure_keys(provider_id)
            now = time.monotonic()
            now_dt = datetime.utcnow()
            self._events[provider_id].append(now)
            self._latencies[provider_id].append(latency)
            self._tokens[provider_id].append(tokens)
            self._successes[provider_id] += 1
            self._last_success[provider_id] = now_dt

            if self._uptime_start[provider_id] is None:
                self._uptime_start[provider_id] = now

            open_until = self._circuit_open_until[provider_id]
            if open_until is not None and now > open_until:
                self._circuit_open_until[provider_id] = None
                self._circuit_last_state_change[provider_id] = now

    def record_failure(
        self,
        provider_id: str,
        latency: float,
        error: str | Exception | None = None,
        circuit_open_until: float | None = None,
    ) -> None:
        """Record a failed request.

        Args:
            provider_id: Provider name.
            latency: Latency before failure (seconds).
            error: Error message or exception.
            circuit_open_until: If the circuit breaker was opened, the
                Unix timestamp until which it remains open.
        """
        with self._lock:
            self._ensure_keys(provider_id)
            now = time.monotonic()
            now_dt = datetime.utcnow()
            self._events[provider_id].append(now)
            self._failures[provider_id] += 1
            self._last_failure[provider_id] = now_dt
            self._last_failure_error[provider_id] = (
                str(error) if error is not None else "unknown error"
            )

            if self._uptime_start[provider_id] is None:
                self._uptime_start[provider_id] = now
            else:
                current_start = self._uptime_start[provider_id]
                assert current_start is not None
                last_change = self._circuit_last_state_change[provider_id]
                base = last_change if last_change is not None else current_start
                elapsed = now - base
                if self._circuit_open_until[provider_id] is not None:
                    self._total_downtime[provider_id] += elapsed
                else:
                    self._total_uptime[provider_id] += elapsed

            if circuit_open_until is not None:
                self._circuit_open_until[provider_id] = circuit_open_until
                self._circuit_last_state_change[provider_id] = now

    def get_snapshot(self, provider_id: str) -> ProviderHealthSnapshot:
        """Return a health snapshot for *provider_id*.

        Raises:
            KeyError: If the provider has no recorded events.
        """
        with self._lock:
            self._ensure_keys(provider_id)

            total = self._successes[provider_id] + self._failures[provider_id]
            success_rate = self._successes[provider_id] / total if total > 0 else 0.0

            latencies = list(self._latencies[provider_id])
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            min_latency = min(latencies) if latencies else 0.0
            max_latency = max(latencies) if latencies else 0.0

            now_mono = time.monotonic()
            recent_events = [
                t
                for t in self._events[provider_id]
                if now_mono - t <= self._rpm_window_seconds
            ]
            rpm = len(recent_events) / (self._rpm_window_seconds / 60.0)

            recent_tokens = [
                count
                for t, count in zip(
                    self._events[provider_id], self._tokens[provider_id]
                )
                if now_mono - t <= self._rpm_window_seconds
            ]
            tpm = sum(recent_tokens) / (self._rpm_window_seconds / 60.0)

            now = time.monotonic()
            uptime_total = self._total_uptime[provider_id]
            downtime_total = self._total_downtime[provider_id]
            current_open = self._circuit_open_until[provider_id]
            last_change = self._circuit_last_state_change[provider_id] or now
            if current_open is not None and now > current_open:
                self._circuit_open_until[provider_id] = None
                current_open = None

            if current_open is not None:
                downtime_total += now - last_change
            else:
                uptime_total += now - last_change

            total_time = uptime_total + downtime_total
            uptime_pct = uptime_total / total_time if total_time > 0 else 1.0

            from app.llm.router_models import CircuitState

            circuit_state = (
                CircuitState.OPEN if current_open is not None else CircuitState.CLOSED
            )

            return ProviderHealthSnapshot(
                provider_id=provider_id,
                is_healthy=success_rate >= 0.5 and circuit_state == CircuitState.CLOSED,
                circuit_state=circuit_state,
                success_count=self._successes[provider_id],
                failure_count=self._failures[provider_id],
                success_rate=success_rate,
                avg_latency=avg_latency,
                min_latency=min_latency,
                max_latency=max_latency,
                requests_per_minute=rpm,
                tokens_per_minute=tpm,
                uptime=uptime_pct,
                last_success=self._last_success[provider_id],
                last_failure=self._last_failure[provider_id],
                last_failure_error=self._last_failure_error[provider_id],
            )

    def is_healthy(self, provider_id: str) -> bool:
        """Return ``True`` if the provider is currently healthy."""
        snapshot = self.get_snapshot(provider_id)
        return snapshot.is_healthy

    def is_circuit_open(self, provider_id: str) -> bool:
        """Return ``True`` if the circuit breaker is open for the provider."""
        with self._lock:
            self._ensure_keys(provider_id)
            until = self._circuit_open_until[provider_id]
            if until is None:
                return False
            if time.monotonic() > until:
                self._circuit_open_until[provider_id] = None
                return False
            return True

    def get_healthy_providers(self) -> list[str]:
        """Return IDs of all currently healthy providers."""
        with self._lock:
            return [pid for pid in self._successes if self.is_healthy(pid)]

    def get_all_snapshots(self) -> dict[str, ProviderHealthSnapshot]:
        """Return health snapshots for all tracked providers."""
        with self._lock:
            return {pid: self.get_snapshot(pid) for pid in list(self._successes.keys())}

    def reset_provider(self, provider_id: str) -> None:
        """Reset all metrics for a provider.

        Args:
            provider_id: Provider name.
        """
        with self._lock:
            self._successes.pop(provider_id, None)
            self._failures.pop(provider_id, None)
            self._events.pop(provider_id, None)
            self._latencies.pop(provider_id, None)
            self._tokens.pop(provider_id, None)
            self._last_success.pop(provider_id, None)
            self._last_failure.pop(provider_id, None)
            self._last_failure_error.pop(provider_id, None)
            self._uptime_start.pop(provider_id, None)
            self._total_uptime.pop(provider_id, None)
            self._total_downtime.pop(provider_id, None)
            self._circuit_open_until.pop(provider_id, None)
            self._circuit_last_state_change.pop(provider_id, None)

    def clear(self) -> None:
        """Reset all tracked metrics."""
        with self._lock:
            self._successes.clear()
            self._failures.clear()
            self._events.clear()
            self._latencies.clear()
            self._tokens.clear()
            self._last_success.clear()
            self._last_failure.clear()
            self._last_failure_error.clear()
            self._uptime_start.clear()
            self._total_uptime.clear()
            self._total_downtime.clear()
            self._circuit_open_until.clear()
            self._circuit_last_state_change.clear()
