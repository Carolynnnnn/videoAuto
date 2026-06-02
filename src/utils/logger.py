"""
日志工具：统一日志格式，支持文件 + 控制台双输出，增强调试支持
"""
from __future__ import annotations
import logging
import sys
import time
import functools
import contextvars
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Callable, TypeVar

T = TypeVar("T")

_correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)


class CorrelationIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = _correlation_id.get() or "-"
        return True


def set_correlation_id(cid: str) -> None:
    _correlation_id.set(cid)


def get_correlation_id() -> Optional[str]:
    return _correlation_id.get()


def get_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.addFilter(CorrelationIdFilter())
    
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] [%(correlation_id)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(
            Path(log_dir) / f"build_{ts}.log",
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def log_performance(
    logger: Optional[logging.Logger] = None,
    operation_name: Optional[str] = None,
    log_args: bool = False,
) -> Callable:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            _logger = logger or logging.getLogger(func.__module__)
            op_name = operation_name or func.__name__
            
            start_time = time.perf_counter()
            
            if log_args:
                _logger.debug(f"[PERF] {op_name} starting with args={args[:2]}... kwargs_keys={list(kwargs.keys())}")
            else:
                _logger.debug(f"[PERF] {op_name} starting")
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                _logger.info(f"[PERF] {op_name} completed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                _logger.error(f"[PERF] {op_name} failed after {elapsed:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator


class LogContext:
    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        **context_data: Any,
    ):
        self.logger = logger
        self.operation = operation
        self.context_data = context_data
        self.start_time: Optional[float] = None
    
    def __enter__(self) -> "LogContext":
        self.start_time = time.perf_counter()
        ctx_str = " ".join(f"{k}={v}" for k, v in self.context_data.items())
        self.logger.debug(f"[CTX] {self.operation} started {ctx_str}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed = time.perf_counter() - (self.start_time or 0)
        if exc_type is None:
            self.logger.debug(f"[CTX] {self.operation} completed in {elapsed:.3f}s")
        else:
            self.logger.error(
                f"[CTX] {self.operation} failed after {elapsed:.3f}s: {exc_type.__name__}: {exc_val}"
            )
        return False


def create_debug_logger(name: str, log_dir: str) -> logging.Logger:
    return get_logger(name, log_dir=log_dir, level=logging.DEBUG)
