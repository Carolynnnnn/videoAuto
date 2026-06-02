from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from pixelle_snapshot.adapters.contracts import ErrorCategory
from src.utils.logger import get_logger

logger = get_logger("pixelle_reliability")


class RateLimitExceededError(Exception):
    pass


class CircuitOpenError(Exception):
    pass


@dataclass(frozen=True)
class ReliabilityConfig:
    rate_limit_per_second: float = 0.0
    rate_limit_burst: int = 1
    rate_limit_wait_seconds: float = 0.0
    circuit_window_size: int = 100
    circuit_min_requests: int = 50
    circuit_error_rate_threshold: float = 0.6
    circuit_open_seconds: float = 30.0
    circuit_half_open_max_calls: int = 1

    @classmethod
    def from_env(cls) -> "ReliabilityConfig":
        return cls(
            rate_limit_per_second=_read_float_env("PIXELLE_PROVIDER_RATE_LIMIT_PER_SEC", 0.0, minimum=0.0),
            rate_limit_burst=_read_int_env("PIXELLE_PROVIDER_RATE_LIMIT_BURST", 1, minimum=1),
            rate_limit_wait_seconds=_read_float_env("PIXELLE_PROVIDER_RATE_LIMIT_WAIT_SECONDS", 0.0, minimum=0.0),
            circuit_window_size=_read_int_env("PIXELLE_CIRCUIT_WINDOW_SIZE", 100, minimum=1),
            circuit_min_requests=_read_int_env("PIXELLE_CIRCUIT_MIN_REQUESTS", 50, minimum=1),
            circuit_error_rate_threshold=_read_float_env("PIXELLE_CIRCUIT_ERROR_RATE_THRESHOLD", 0.6, minimum=0.0),
            circuit_open_seconds=_read_float_env("PIXELLE_CIRCUIT_OPEN_SECONDS", 30.0, minimum=1.0),
            circuit_half_open_max_calls=_read_int_env("PIXELLE_CIRCUIT_HALF_OPEN_MAX_CALLS", 1, minimum=1),
        )


class TokenBucketRateLimiter:
    def __init__(self, rate_per_second: float, burst: int, *, clock=time.monotonic):
        self._rate_per_second = max(0.0, rate_per_second)
        self._burst = max(1, burst)
        self._clock = clock
        self._lock = threading.Lock()
        self._tokens = float(self._burst)
        self._last_refill = self._clock()

    @property
    def enabled(self) -> bool:
        return self._rate_per_second > 0.0

    def acquire(self, timeout_seconds: float) -> bool:
        if not self.enabled:
            return True

        deadline = self._clock() + max(0.0, timeout_seconds)
        while True:
            with self._lock:
                self._refill_locked()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True

            now = self._clock()
            if now >= deadline:
                return False

            sleep_for = min(0.01, max(0.0, deadline - now))
            if sleep_for <= 0.0:
                return False
            time.sleep(sleep_for)

    def _refill_locked(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last_refill)
        if elapsed <= 0.0:
            return

        self._tokens = min(self._burst, self._tokens + elapsed * self._rate_per_second)
        self._last_refill = now


class ErrorRateCircuitBreaker:
    def __init__(
        self,
        *,
        window_size: int,
        min_requests: int,
        error_rate_threshold: float,
        open_seconds: float,
        half_open_max_calls: int,
        clock=time.monotonic,
    ):
        self._window_size = max(1, window_size)
        self._min_requests = max(1, min_requests)
        self._error_rate_threshold = min(1.0, max(0.0, error_rate_threshold))
        self._open_seconds = max(1.0, open_seconds)
        self._half_open_max_calls = max(1, half_open_max_calls)
        self._clock = clock

        self._lock = threading.Lock()
        self._window: Deque[bool] = deque(maxlen=self._window_size)
        self._state = "closed"
        self._opened_until = 0.0
        self._half_open_calls = 0

    @property
    def state(self) -> str:
        with self._lock:
            self._refresh_state_locked()
            return self._state

    def allow_request(self) -> bool:
        with self._lock:
            self._refresh_state_locked()

            if self._state == "closed":
                return True

            if self._state == "open":
                return False

            if self._state == "half_open":
                if self._half_open_calls < self._half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    def record_success(self) -> None:
        with self._lock:
            self._refresh_state_locked()
            if self._state == "half_open":
                self._state = "closed"
                self._half_open_calls = 0
                self._window.clear()
                logger.info("event=pixelle_circuit_closed reason=half_open_recovery")
                return

            if self._state == "closed":
                self._window.append(False)

    def record_failure(self, *, category: str) -> None:
        if category in (
            ErrorCategory.VALIDATION.value,
            ErrorCategory.MISSING_INPUT.value,
            ErrorCategory.UNSUPPORTED.value,
        ):
            return

        with self._lock:
            self._refresh_state_locked()

            if self._state == "half_open":
                self._state = "open"
                self._opened_until = self._clock() + self._open_seconds
                self._half_open_calls = 0
                logger.warning("event=pixelle_circuit_open reason=half_open_failure category=%s", category)
                return

            if self._state != "closed":
                return

            self._window.append(True)
            if len(self._window) < self._min_requests:
                return

            failures = sum(1 for failed in self._window if failed)
            error_rate = failures / float(len(self._window))
            if error_rate >= self._error_rate_threshold:
                self._state = "open"
                self._opened_until = self._clock() + self._open_seconds
                self._half_open_calls = 0
                logger.warning(
                    "event=pixelle_circuit_open reason=error_rate_window error_rate=%.3f failures=%d window=%d",
                    error_rate,
                    failures,
                    len(self._window),
                )

    def _refresh_state_locked(self) -> None:
        if self._state == "open" and self._clock() >= self._opened_until:
            self._state = "half_open"
            self._half_open_calls = 0
            logger.info("event=pixelle_circuit_half_open")


class PixelleReliabilityControls:
    def __init__(
        self,
        config: ReliabilityConfig,
        *,
        rate_limiter: Optional[TokenBucketRateLimiter] = None,
        circuit_breaker: Optional[ErrorRateCircuitBreaker] = None,
    ):
        self._config = config
        self._rate_limiter = rate_limiter or TokenBucketRateLimiter(
            config.rate_limit_per_second,
            config.rate_limit_burst,
        )
        self._circuit_breaker = circuit_breaker or ErrorRateCircuitBreaker(
            window_size=config.circuit_window_size,
            min_requests=config.circuit_min_requests,
            error_rate_threshold=config.circuit_error_rate_threshold,
            open_seconds=config.circuit_open_seconds,
            half_open_max_calls=config.circuit_half_open_max_calls,
        )

    @classmethod
    def from_env(cls) -> "PixelleReliabilityControls":
        return cls(ReliabilityConfig.from_env())

    def before_provider_call(self, *, capability: str, segment_key: str) -> None:
        if not self._circuit_breaker.allow_request():
            logger.warning(
                "event=pixelle_circuit_open_short_circuit capability=%s segment_key=%s",
                capability,
                segment_key,
            )
            raise CircuitOpenError("Pixelle circuit breaker is open")

        allowed = self._rate_limiter.acquire(self._config.rate_limit_wait_seconds)
        if not allowed:
            logger.warning(
                "event=pixelle_rate_limited capability=%s segment_key=%s limit_per_second=%.3f burst=%d",
                capability,
                segment_key,
                self._config.rate_limit_per_second,
                self._config.rate_limit_burst,
            )
            raise RateLimitExceededError("Pixelle rate limit exceeded")

    def record_success(self) -> None:
        self._circuit_breaker.record_success()

    def record_failure(self, *, category: str) -> None:
        self._circuit_breaker.record_failure(category=category)


def _read_int_env(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def _read_float_env(name: str, default: float, minimum: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(minimum, value)
