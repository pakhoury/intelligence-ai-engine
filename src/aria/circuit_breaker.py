"""
Circuit breaker for LLM calls.

States:
- CLOSED: Normal operation, requests pass through.
- OPEN: Too many failures, requests rejected immediately with fallback.
- HALF_OPEN: After recovery timeout, one request allowed through to test.

Prevents cascading failures when the LLM provider is degraded.
"""
import threading
import time
from enum import Enum

from observability import CIRCUIT_BREAKER_STATE, CIRCUIT_BREAKER_TRIPS, logger


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


_GAUGE_VALUE = {
    CircuitState.CLOSED: 0,
    CircuitState.OPEN: 1,
    CircuitState.HALF_OPEN: 2,
}


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit breaker is open and requests are being rejected."""

    def __init__(self, name: str, time_until_retry: float):
        self.name = name
        self.time_until_retry = time_until_retry
        super().__init__(
            f"Circuit breaker '{name}' is OPEN. "
            f"Retry in {time_until_retry:.1f}s."
        )


class CircuitBreaker:
    """
    Thread-safe circuit breaker for protecting external service calls.

    Usage:
        breaker = CircuitBreaker("llm", failure_threshold=5, recovery_timeout=30)

        async def call_llm(prompt):
            breaker.before_call()  # raises CircuitBreakerOpenError if open
            try:
                result = await llm.ainvoke(prompt)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 2,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._lock = threading.Lock()
        CIRCUIT_BREAKER_STATE.labels(name=self.name).set(_GAUGE_VALUE[self._state])

    def _set_state(self, new_state: CircuitState) -> None:
        """Assign state and mirror it to the Prometheus gauge. Caller holds the lock."""
        self._state = new_state
        CIRCUIT_BREAKER_STATE.labels(name=self.name).set(_GAUGE_VALUE[new_state])

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN and self._should_attempt_recovery():
                self._set_state(CircuitState.HALF_OPEN)
                logger.info(
                    f"Circuit breaker '{self.name}' entering HALF_OPEN",
                    extra={"node": "circuit_breaker"},
                )
            return self._state

    def _should_attempt_recovery(self) -> bool:
        if self._last_failure_time is None:
            return True
        return (time.monotonic() - self._last_failure_time) >= self.recovery_timeout

    def before_call(self):
        """Call before making the external request. Raises CircuitBreakerOpenError if open."""
        current_state = self.state
        if current_state == CircuitState.OPEN:
            time_since_failure = time.monotonic() - (self._last_failure_time or 0)
            time_until_retry = max(0, self.recovery_timeout - time_since_failure)
            raise CircuitBreakerOpenError(self.name, time_until_retry)

    def record_success(self):
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._set_state(CircuitState.CLOSED)
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info(
                        f"Circuit breaker '{self.name}' CLOSED (recovered)",
                        extra={"node": "circuit_breaker"},
                    )
            else:
                self._failure_count = 0
                self._success_count = 0

    def record_failure(self):
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._set_state(CircuitState.OPEN)
                self._success_count = 0
                CIRCUIT_BREAKER_TRIPS.labels(name=self.name).inc()
                logger.warning(
                    f"Circuit breaker '{self.name}' OPEN (half-open test failed)",
                    extra={"node": "circuit_breaker"},
                )
            elif self._failure_count >= self.failure_threshold:
                self._set_state(CircuitState.OPEN)
                CIRCUIT_BREAKER_TRIPS.labels(name=self.name).inc()
                logger.warning(
                    f"Circuit breaker '{self.name}' OPEN "
                    f"(threshold {self.failure_threshold} reached)",
                    extra={"node": "circuit_breaker"},
                )

    def reset(self):
        """Manually reset to closed state."""
        with self._lock:
            self._set_state(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None


llm_circuit_breaker = CircuitBreaker(
    name="llm",
    failure_threshold=5,
    recovery_timeout=30.0,
    success_threshold=2,
)
