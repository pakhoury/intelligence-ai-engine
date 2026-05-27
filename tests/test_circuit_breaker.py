"""
Unit tests for the circuit breaker.
"""
import sys
import os
import time
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "rag-system"))

from circuit_breaker import CircuitBreaker, CircuitBreakerOpen, CircuitState


class TestCircuitBreakerClosed:
    def test_starts_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED

    def test_allows_calls_when_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.before_call()  # Should not raise

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_resets_failure_count_on_success(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED


class TestCircuitBreakerOpen:
    def test_opens_at_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_rejects_calls_when_open(self):
        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=60.0)
        for _ in range(3):
            cb.record_failure()
        with pytest.raises(CircuitBreakerOpen) as exc_info:
            cb.before_call()
        assert "test" in str(exc_info.value)

    def test_exception_includes_retry_time(self):
        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=30.0)
        for _ in range(3):
            cb.record_failure()
        with pytest.raises(CircuitBreakerOpen) as exc_info:
            cb.before_call()
        assert exc_info.value.time_until_retry <= 30.0


class TestCircuitBreakerHalfOpen:
    def test_transitions_to_half_open_after_timeout(self):
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_allows_one_call(self):
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        cb.before_call()  # Should not raise in HALF_OPEN

    def test_closes_after_success_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1, success_threshold=2)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_reopens_on_failure_in_half_open(self):
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerReset:
    def test_manual_reset(self):
        cb = CircuitBreaker("test", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        cb.before_call()  # Should not raise


class TestCircuitBreakerIntegration:
    @patch("nodes._ainvoke_llm", side_effect=ConnectionError("LLM down"))
    async def test_circuit_opens_after_repeated_failures(self, mock_ainvoke):
        """Simulates repeated LLM failures opening the circuit."""
        from circuit_breaker import llm_circuit_breaker
        llm_circuit_breaker.reset()

        for _ in range(5):
            llm_circuit_breaker.record_failure()

        assert llm_circuit_breaker.state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerOpen):
            llm_circuit_breaker.before_call()

        llm_circuit_breaker.reset()
