"""Circuit breaker for LLM provider fault isolation.

Implements the standard three-state circuit breaker pattern:

- ``CLOSED``: Requests flow normally; failures are counted.
- ``OPEN``: Requests are rejected immediately; a timer counts
  down to the recovery window.
- ``HALF_OPEN``: A limited number of probe requests are allowed
  through to test whether the provider has recovered.
"""

from __future__ import annotations

import logging
import threading
import time

from app.llm.router_exceptions import CircuitBreakerError
from app.llm.router_models import CircuitState

log = logging.getLogger(__name__)


class CircuitBreaker:
    """Thread-safe circuit breaker for a single provider.

    Args:
        failure_threshold: Number of consecutive failures before
            transitioning from CLOSED to OPEN.
        timeout: Seconds to stay OPEN before transitioning to
            HALF_OPEN.
        recovery_threshold: Number of successful probe requests
            required in HALF_OPEN to close the circuit.
        expected_exception: Exception type that counts as a failure.
            If ``None``, all exceptions count.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 30.0,
        recovery_threshold: int = 2,
        expected_exception: type[Exception] | None = None,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._timeout = timeout
        self._recovery_threshold = recovery_threshold
        self._expected_exception = expected_exception

        self._state: CircuitState = CircuitState.CLOSED
        self._consecutive_failures: int = 0
        self._consecutive_successes: int = 0
        self._open_until: float = 0.0
        self._half_open_requests: int = 0
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        """Return the current circuit state."""
        with self._lock:
            self._update_state()
            return self._state

    @property
    def consecutive_failures(self) -> int:
        """Return the current consecutive failure count."""
        with self._lock:
            return self._consecutive_failures

    @property
    def consecutive_successes(self) -> int:
        """Return the current consecutive success count."""
        with self._lock:
            return self._consecutive_successes

    def _is_expected_exception(self, exc: Exception) -> bool:
        """Return ``True`` if *exc* matches the expected failure type."""
        if self._expected_exception is None:
            return True
        return isinstance(exc, self._expected_exception)

    def _update_state(self) -> None:
        """Re-evaluate the circuit state based on current time."""
        if self._state == CircuitState.OPEN:
            if time.monotonic() >= self._open_until:
                self._state = CircuitState.HALF_OPEN
                self._consecutive_successes = 0
                self._half_open_requests = 0
                log.info(
                    "Circuit breaker transitioned to HALF_OPEN "
                    "(failures=%d, threshold=%d)",
                    self._consecutive_failures,
                    self._failure_threshold,
                )

    def allow_request(self) -> bool:
        """Return ``True`` if a request is allowed, ``False`` if blocked.

        Updates the internal state, transitioning OPEN → HALF_OPEN
        when the recovery timeout has elapsed.
        """
        with self._lock:
            self._update_state()
            if self._state == CircuitState.OPEN:
                return False
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_requests < self._recovery_threshold:
                    self._half_open_requests += 1
                    return True
                return False
            return True

    def on_success(self) -> None:
        """Record a successful request.

        Transitions the circuit to CLOSED if in HALF_OPEN and the
        recovery threshold of consecutive successes is reached.
        """
        with self._lock:
            self._update_state()
            self._consecutive_failures = 0

            if self._state == CircuitState.HALF_OPEN:
                self._consecutive_successes += 1
                if self._consecutive_successes >= self._recovery_threshold:
                    self._state = CircuitState.CLOSED
                    log.info("Circuit breaker CLOSED after recovery")
            else:
                self._consecutive_successes = 0

    def on_failure(self, exc: Exception | None = None) -> None:
        """Record a failed request.

        Transitions the circuit to OPEN when the failure threshold
        is reached.
        """
        with self._lock:
            self._update_state()

            if exc is not None and not self._is_expected_exception(exc):
                return

            self._consecutive_failures += 1
            self._consecutive_successes = 0

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._open_until = time.monotonic() + self._timeout
                log.warning("Circuit breaker OPEN after HALF_OPEN failure")
            elif (
                self._state == CircuitState.CLOSED
                and self._consecutive_failures >= self._failure_threshold
            ):
                self._state = CircuitState.OPEN
                self._open_until = time.monotonic() + self._timeout
                log.warning(
                    "Circuit breaker OPEN after %d consecutive failures",
                    self._consecutive_failures,
                )

    def get_open_until(self) -> float | None:
        """Return the Unix timestamp when OPEN transitions to HALF_OPEN.

        Returns ``None`` if the circuit is not open.
        """
        with self._lock:
            self._update_state()
            if self._state == CircuitState.OPEN:
                return self._open_until
            return None

    def reset(self) -> None:
        """Reset the circuit breaker to CLOSED and clear counters."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._open_until = 0.0
            self._half_open_requests = 0

    def check(self, provider_id: str) -> None:
        """Raise :class:`CircuitBreakerError` if the circuit is open."""
        if not self.allow_request():
            raise CircuitBreakerError(
                f"Circuit breaker is OPEN for provider '{provider_id}'",
                module="llm.circuit_breaker",
                context={
                    "provider": provider_id,
                    "state": self.state.value,
                    "failures": self.consecutive_failures,
                },
            )
