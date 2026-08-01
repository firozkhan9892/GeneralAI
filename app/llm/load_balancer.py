"""Load balancing across LLM providers.

Supports multiple strategies: round-robin, weighted, least-latency,
and least-error-rate.
"""

from __future__ import annotations

import logging
import threading

from app.llm.router_models import LoadBalanceStrategy, ProviderHealthSnapshot
from app.llm.router_exceptions import RoutingError

log = logging.getLogger(__name__)


class LoadBalancer:
    """Thread-safe load balancer with pluggable strategies.

    Attributes:
        _strategy: Current balancing strategy.
        _weights: Provider ID → weight (used by WEIGHTED strategy).
        _counter: Round-robin counter.
        _latencies: Provider ID → list of recent latencies.
        _error_rates: Provider ID → recent error rate.
        _lock: Protects all internal state.
    """

    def __init__(
        self,
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN,
    ) -> None:
        self._strategy = strategy
        self._weights: dict[str, float] = {}
        self._counter: int = 0
        self._latencies: dict[str, list[float]] = {}
        self._error_rates: dict[str, float] = {}
        self._lock = threading.RLock()

    @property
    def strategy(self) -> LoadBalanceStrategy:
        """Return the current balancing strategy."""
        return self._strategy

    def set_strategy(self, strategy: LoadBalanceStrategy) -> None:
        """Change the balancing strategy.

        Args:
            strategy: New strategy to use.
        """
        with self._lock:
            self._strategy = strategy
        log.info("Load balancer strategy set to '%s'", strategy.value)

    def set_weight(self, provider_id: str, weight: float) -> None:
        """Set the weight for a provider (used by WEIGHTED strategy).

        Args:
            provider_id: Provider name.
            weight: Relative weight (higher = more traffic).
        """
        with self._lock:
            self._weights[provider_id] = max(weight, 0.01)

    def update_health(self, snapshot: ProviderHealthSnapshot) -> None:
        """Update with a provider's latest health snapshot.

        Args:
            snapshot: Health data from the monitor.
        """
        with self._lock:
            pid = snapshot.provider_id
            if pid not in self._latencies:
                self._latencies[pid] = []
            if snapshot.avg_latency > 0:
                self._latencies[pid].append(snapshot.avg_latency)
                if len(self._latencies[pid]) > 100:
                    self._latencies[pid] = self._latencies[pid][-100:]
            self._error_rates[pid] = 1.0 - snapshot.success_rate

    def select(
        self,
        providers: list[str],
        health_snapshots: dict[str, ProviderHealthSnapshot] | None = None,
    ) -> str:
        """Select a provider from *providers* based on the current strategy.

        Args:
            providers: Candidate provider IDs.
            health_snapshots: Optional health data for latency-aware strategies.

        Returns:
            The selected provider ID.

        Raises:
            RoutingError: If the provider list is empty.
        """
        if not providers:
            raise RoutingError(
                "No providers available for load balancing",
                module="llm.load_balancer",
            )

        with self._lock:
            if self._strategy == LoadBalanceStrategy.ROUND_ROBIN:
                self._counter = (self._counter + 1) % len(providers)
                return providers[self._counter]

            elif self._strategy == LoadBalanceStrategy.WEIGHTED:
                return self._weighted_select(providers)

            elif self._strategy == LoadBalanceStrategy.LEAST_LATENCY:
                return self._least_latency_select(providers, health_snapshots)

            elif self._strategy == LoadBalanceStrategy.LEAST_ERROR_RATE:
                return self._least_error_select(providers, health_snapshots)

        return providers[0]

    def _weighted_select(self, providers: list[str]) -> str:
        """Select using weighted distribution."""
        total = sum(self._weights.get(p, 1.0) for p in providers)
        if total <= 0:
            return providers[0 % len(providers)]

        r = self._counter % 1000 / 1000.0 * total
        cumulative = 0.0
        for p in providers:
            w = self._weights.get(p, 1.0)
            cumulative += w
            if r <= cumulative:
                return p
        return providers[-1]

    def _least_latency_select(
        self,
        providers: list[str],
        health_snapshots: dict[str, ProviderHealthSnapshot] | None,
    ) -> str:
        """Select the provider with lowest average latency."""
        if not health_snapshots:
            return providers[0]

        best_provider = providers[0]
        best_latency = float("inf")
        for p in providers:
            snap = health_snapshots.get(p)
            if snap and snap.avg_latency < best_latency:
                best_latency = snap.avg_latency
                best_provider = p
        return best_provider

    def _least_error_select(
        self,
        providers: list[str],
        health_snapshots: dict[str, ProviderHealthSnapshot] | None,
    ) -> str:
        """Select the provider with lowest error rate."""
        if not health_snapshots:
            return providers[0]

        best_provider = providers[0]
        best_error_rate = float("inf")
        for p in providers:
            snap = health_snapshots.get(p)
            if snap:
                error_rate = 1.0 - snap.success_rate
                if error_rate < best_error_rate:
                    best_error_rate = error_rate
                    best_provider = p
        return best_provider

    def get_weights(self) -> dict[str, float]:
        """Return a copy of the current weight map."""
        with self._lock:
            return dict(self._weights)

    def reset(self) -> None:
        """Reset all internal counters and stats."""
        with self._lock:
            self._counter = 0
            self._latencies.clear()
            self._error_rates.clear()
            self._weights.clear()
