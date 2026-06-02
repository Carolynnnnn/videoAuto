"""
Retry utilities: exponential backoff, circuit breaker, timeout handling

Provides stable retry mechanisms for external API calls and long-running operations.
"""
from __future__ import annotations
import time
import functools
import threading
from typing import Callable, TypeVar, Optional, Tuple, Type, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from src.utils.logger import get_logger

logger = get_logger("retry")

T = TypeVar("T")


# ─────────────────────────────────────────────
# Retry Configuration
# ─────────────────────────────────────────────
@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    # Specific exceptions that should NOT be retried
    non_retryable_exceptions: Tuple[Type[BaseException], ...] = (
        KeyboardInterrupt,
        SystemExit,
        ValueError,
    )


DEFAULT_RETRY_CONFIG = RetryConfig()


# ─────────────────────────────────────────────
# Exponential Backoff Calculator
# ─────────────────────────────────────────────
def calculate_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
) -> float:
    """
    Calculate delay with exponential backoff and optional jitter.
    
    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add randomness
    
    Returns:
        Delay in seconds
    """
    import random
    
    delay = base_delay * (exponential_base ** attempt)
    delay = min(delay, max_delay)
    
    if jitter:
        # Add ±25% jitter
        jitter_range = delay * 0.25
        delay = delay + random.uniform(-jitter_range, jitter_range)
        delay = max(0.1, delay)  # Ensure minimum delay
    
    return delay


# ─────────────────────────────────────────────
# Retry Decorator
# ─────────────────────────────────────────────
def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    operation_name: Optional[str] = None,
) -> Callable:
    """
    Decorator for retrying operations with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        jitter: Add randomness to delays
        retryable_exceptions: Tuple of exception types to retry
        on_retry: Callback function called before each retry
        operation_name: Name for logging purposes
    
    Example:
        @with_retry(max_retries=3, base_delay=1.0)
        def fetch_data():
            return api.get("/data")
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            op_name = operation_name or func.__name__
            last_exception: Optional[Exception] = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    # Check if this is a non-retryable exception
                    if isinstance(e, (KeyboardInterrupt, SystemExit)):
                        raise
                    
                    if attempt < max_retries:
                        delay = calculate_backoff(
                            attempt, base_delay, max_delay, exponential_base, jitter
                        )
                        logger.warning(
                            f"[{op_name}] Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        
                        if on_retry:
                            on_retry(e, attempt)
                        
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"[{op_name}] All {max_retries + 1} attempts failed. Last error: {e}"
                        )
            
            # Re-raise the last exception
            if last_exception:
                raise last_exception
            raise RuntimeError(f"[{op_name}] Retry logic error - no exception captured")
        
        return wrapper
    return decorator


# ─────────────────────────────────────────────
# Circuit Breaker
# ─────────────────────────────────────────────
@dataclass
class CircuitBreakerState:
    """Internal state for circuit breaker."""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "closed"  # closed, open, half-open
    lock: threading.Lock = field(default_factory=threading.Lock)


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Prevents repeated calls to a failing service, allowing it time to recover.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests fail fast
    - HALF-OPEN: Testing if service recovered
    
    Example:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        
        @breaker
        def call_external_api():
            return api.get("/data")
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
        name: str = "circuit_breaker",
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.name = name
        self._state = CircuitBreakerState()
        self._half_open_calls = 0
    
    @property
    def state(self) -> str:
        return self._state.state
    
    @property
    def is_open(self) -> bool:
        return self._state.state == "open"
    
    def _should_allow_request(self) -> bool:
        """Check if request should be allowed based on circuit state."""
        with self._state.lock:
            if self._state.state == "closed":
                return True
            
            if self._state.state == "open":
                # Check if recovery timeout has passed
                if self._state.last_failure_time:
                    time_since_failure = datetime.now() - self._state.last_failure_time
                    if time_since_failure > timedelta(seconds=self.recovery_timeout):
                        self._state.state = "half-open"
                        self._half_open_calls = 0
                        logger.info(f"[{self.name}] Circuit breaker transitioning to half-open")
                        return True
                return False
            
            if self._state.state == "half-open":
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            
            return False
    
    def _record_success(self):
        """Record successful call."""
        with self._state.lock:
            if self._state.state == "half-open":
                # Service recovered, close the circuit
                self._state.state = "closed"
                self._state.failure_count = 0
                logger.info(f"[{self.name}] Circuit breaker closed - service recovered")
            elif self._state.state == "closed":
                # Reset failure count on success
                self._state.failure_count = 0
    
    def _record_failure(self):
        """Record failed call."""
        with self._state.lock:
            self._state.failure_count += 1
            self._state.last_failure_time = datetime.now()
            
            if self._state.state == "half-open":
                # Service still failing, open the circuit
                self._state.state = "open"
                logger.warning(f"[{self.name}] Circuit breaker opened - service still failing")
            elif self._state.state == "closed":
                if self._state.failure_count >= self.failure_threshold:
                    self._state.state = "open"
                    logger.warning(
                        f"[{self.name}] Circuit breaker opened after {self._state.failure_count} failures"
                    )
    
    def reset(self):
        """Manually reset the circuit breaker to closed state."""
        with self._state.lock:
            self._state.state = "closed"
            self._state.failure_count = 0
            self._state.last_failure_time = None
            self._half_open_calls = 0
            logger.info(f"[{self.name}] Circuit breaker manually reset")
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator interface for circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if not self._should_allow_request():
                raise CircuitBreakerOpenError(
                    f"[{self.name}] Circuit breaker is open - failing fast"
                )
            
            try:
                result = func(*args, **kwargs)
                self._record_success()
                return result
            except Exception as e:
                self._record_failure()
                raise
        
        return wrapper


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and request is rejected."""
    pass


# ─────────────────────────────────────────────
# Timeout Decorator
# ─────────────────────────────────────────────
class TimeoutError(Exception):
    """Raised when operation times out."""
    pass


def with_timeout(
    seconds: float,
    operation_name: Optional[str] = None,
) -> Callable:
    """
    Decorator to add timeout to a function.
    
    Note: Uses threading, so may not interrupt blocking I/O operations.
    For I/O-bound operations, prefer using library-level timeouts.
    
    Args:
        seconds: Timeout in seconds
        operation_name: Name for logging
    
    Example:
        @with_timeout(30)
        def long_operation():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            import concurrent.futures
            
            op_name = operation_name or func.__name__
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except concurrent.futures.TimeoutError:
                    logger.error(f"[{op_name}] Operation timed out after {seconds}s")
                    raise TimeoutError(f"[{op_name}] Operation timed out after {seconds}s")
        
        return wrapper
    return decorator


# ─────────────────────────────────────────────
# Combined Retry + Circuit Breaker
# ─────────────────────────────────────────────
def resilient_call(
    func: Callable[..., T],
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    circuit_breaker: Optional[CircuitBreaker] = None,
    timeout: Optional[float] = None,
    operation_name: Optional[str] = None,
    **kwargs,
) -> T:
    """
    Execute a function with retry, circuit breaker, and timeout protection.
    
    Args:
        func: Function to execute
        *args: Positional arguments for func
        max_retries: Maximum retry attempts
        base_delay: Base delay for exponential backoff
        circuit_breaker: Optional circuit breaker instance
        timeout: Optional timeout in seconds
        operation_name: Name for logging
        **kwargs: Keyword arguments for func
    
    Returns:
        Result of func
    
    Example:
        result = resilient_call(
            api.get,
            "/endpoint",
            max_retries=3,
            timeout=30,
            operation_name="fetch_data"
        )
    """
    op_name = operation_name or getattr(func, "__name__", "unknown")
    
    import concurrent.futures as cf
    
    if circuit_breaker and circuit_breaker.is_open:
        if not circuit_breaker._should_allow_request():
            raise CircuitBreakerOpenError(f"[{op_name}] Circuit breaker is open")
    
    last_exception: Optional[Exception] = None
    
    for attempt in range(max_retries + 1):
        try:
            if timeout:
                with cf.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func, *args, **kwargs)
                    result = future.result(timeout=timeout)
            else:
                result = func(*args, **kwargs)
            
            if circuit_breaker:
                circuit_breaker._record_success()
            
            return result
            
        except cf.TimeoutError:
            last_exception = TimeoutError(f"[{op_name}] Timed out after {timeout}s")
            logger.warning(f"[{op_name}] Attempt {attempt + 1} timed out")
            
        except (KeyboardInterrupt, SystemExit):
            raise
            
        except Exception as e:
            last_exception = e
            
            if circuit_breaker:
                circuit_breaker._record_failure()
            
            if attempt < max_retries:
                delay = calculate_backoff(attempt, base_delay)
                logger.warning(
                    f"[{op_name}] Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(f"[{op_name}] All attempts failed. Last error: {e}")
    
    if last_exception:
        raise last_exception
    raise RuntimeError(f"[{op_name}] Unexpected retry logic error")
