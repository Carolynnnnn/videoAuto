"""
Pixelle Observability Schema - Correlation IDs and Structured Metrics

This module provides:
1. CorrelationContext - Thread-safe correlation ID propagation
2. StructuredLogRecord - Machine-parseable log events with required keys
3. MetricsExtractor - Contract for extracting latency/error/cost counters

Design Principles:
- Keep provider internals from leaking beyond adapter boundary
- Structured, parseable, machine-friendly schema
- Required keys: request_id, segment_key, capability, status
- Preserves existing logger namespaces and troubleshooting compatibility
"""
from __future__ import annotations

import contextvars
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Protocol

from src.utils.logger import get_logger, set_correlation_id, get_correlation_id

logger = get_logger("pixelle_observability")


# ─────────────────────────────────────────────
# Correlation Context
# ─────────────────────────────────────────────
_workflow_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "workflow_id", default=None
)


def generate_request_id() -> str:
    """Generate a unique request ID for tracing."""
    return f"req_{uuid.uuid4().hex[:16]}"


def generate_workflow_id() -> str:
    """Generate a workflow-level ID for batch operations."""
    return f"wf_{uuid.uuid4().hex[:12]}"


def set_workflow_id(wf_id: str) -> None:
    """Set the current workflow ID in context."""
    _workflow_id.set(wf_id)


def get_workflow_id() -> Optional[str]:
    """Get the current workflow ID from context."""
    return _workflow_id.get()


@dataclass
class CorrelationContext:
    """
    Thread-safe correlation context for tracing provider lifecycle events.
    
    Propagates request_id, segment_key, workflow_id, and capability
    through the adapter call stack.
    """
    request_id: str
    segment_key: str
    capability: Optional[str] = None
    workflow_id: Optional[str] = None
    
    # Timestamps for latency calculation
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @classmethod
    def create(
        cls,
        segment_key: str,
        capability: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> "CorrelationContext":
        """Create a new correlation context with auto-generated request_id."""
        ctx = cls(
            request_id=generate_request_id(),
            segment_key=segment_key,
            capability=capability,
            workflow_id=workflow_id or get_workflow_id(),
        )
        # Set correlation_id in logger context
        set_correlation_id(ctx.request_id)
        return ctx
    
    def to_dict(self) -> Dict[str, Any]:
        """Export context as dictionary for logging/metrics."""
        return {
            "request_id": self.request_id,
            "segment_key": self.segment_key,
            "capability": self.capability,
            "workflow_id": self.workflow_id,
            "created_at": self.created_at.isoformat(),
        }


# ─────────────────────────────────────────────
# Structured Log Events
# ─────────────────────────────────────────────
class EventType(str, Enum):
    """Event types for provider lifecycle logging."""
    ADAPTER_START = "ADAPTER_START"
    ADAPTER_SUCCESS = "ADAPTER_SUCCESS"
    ADAPTER_ERROR = "ADAPTER_ERROR"
    ADAPTER_RETRY = "ADAPTER_RETRY"
    ADAPTER_FALLBACK = "ADAPTER_FALLBACK"
    
    PROVIDER_SUBMIT = "PROVIDER_SUBMIT"
    PROVIDER_POLL = "PROVIDER_POLL"
    PROVIDER_FETCH = "PROVIDER_FETCH"
    PROVIDER_CANCEL = "PROVIDER_CANCEL"
    
    METRICS_EMIT = "METRICS_EMIT"


class StatusType(str, Enum):
    """Status types for structured log records."""
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    FALLBACK = "FALLBACK"


@dataclass
class StructuredLogRecord:
    """
    Machine-parseable log event with required correlation fields.
    
    Required keys (always present):
    - request_id: Unique identifier for this request
    - segment_key: Segment being processed
    - capability: Pixelle capability (may be None before routing)
    - status: Current status of the operation
    
    Optional keys:
    - workflow_id: Batch workflow identifier
    - event_type: Type of lifecycle event
    - duration_ms: Elapsed time in milliseconds
    - error_category: Error classification (if failure)
    - reason_code: Specific failure reason (if failure)
    - attempt: Retry attempt number
    - metadata: Additional context
    """
    # Required fields (contract)
    request_id: str
    segment_key: str
    capability: Optional[str]
    status: StatusType
    
    # Event classification
    event_type: EventType = EventType.ADAPTER_START
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Optional context
    workflow_id: Optional[str] = None
    duration_ms: Optional[float] = None
    error_category: Optional[str] = None
    reason_code: Optional[str] = None
    attempt: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_context(
        cls,
        ctx: CorrelationContext,
        event_type: EventType,
        status: StatusType,
        **kwargs,
    ) -> "StructuredLogRecord":
        """Create a log record from correlation context."""
        return cls(
            request_id=ctx.request_id,
            segment_key=ctx.segment_key,
            capability=ctx.capability,
            workflow_id=ctx.workflow_id,
            event_type=event_type,
            status=status,
            **kwargs,
        )
    
    def with_duration(self, start_time: float) -> "StructuredLogRecord":
        """Set duration from start time."""
        self.duration_ms = (time.perf_counter() - start_time) * 1000
        return self
    
    def with_error(
        self,
        error_category: str,
        reason_code: str,
    ) -> "StructuredLogRecord":
        """Set error details."""
        self.error_category = error_category
        self.reason_code = reason_code
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary for structured logging."""
        result = {
            "request_id": self.request_id,
            "segment_key": self.segment_key,
            "capability": self.capability,
            "status": self.status.value,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.workflow_id:
            result["workflow_id"] = self.workflow_id
        if self.duration_ms is not None:
            result["duration_ms"] = round(self.duration_ms, 2)
        if self.error_category:
            result["error_category"] = self.error_category
        if self.reason_code:
            result["reason_code"] = self.reason_code
        if self.attempt is not None:
            result["attempt"] = self.attempt
        if self.metadata:
            result["metadata"] = self.metadata
        return result
    
    def to_json(self) -> str:
        """Export as JSON string for log output."""
        return json.dumps(self.to_dict(), default=str)


def log_structured(record: StructuredLogRecord) -> None:
    """
    Emit a structured log record through the standard logger.
    
    Preserves existing logger namespaces while adding structured payload.
    """
    level_map = {
        StatusType.STARTED: logger.info,
        StatusType.SUCCESS: logger.info,
        StatusType.FAILURE: logger.error,
        StatusType.RETRY: logger.warning,
        StatusType.FALLBACK: logger.warning,
    }
    log_fn = level_map.get(record.status, logger.info)
    log_fn(f"[{record.event_type.value}] {record.to_json()}")


# ─────────────────────────────────────────────
# Metrics Extraction Contract
# ─────────────────────────────────────────────
@dataclass
class ProviderMetrics:
    """
    Minimal metrics extraction contract for provider lifecycle events.
    
    Counters:
    - request_count: Total requests
    - success_count: Successful completions
    - error_count: Failed requests
    - retry_count: Retry attempts
    - fallback_count: Fallback activations
    
    Latencies (milliseconds):
    - total_latency_ms: End-to-end duration
    - submit_latency_ms: Time to submit job
    - poll_latency_ms: Time spent polling
    - fetch_latency_ms: Time to fetch result
    
    Cost:
    - cost_usd: Provider-reported cost (if available)
    """
    # Correlation
    request_id: str
    segment_key: str
    capability: str
    workflow_id: Optional[str] = None
    
    # Counters
    request_count: int = 1
    success_count: int = 0
    error_count: int = 0
    retry_count: int = 0
    fallback_count: int = 0
    
    # Latencies (ms)
    total_latency_ms: float = 0.0
    submit_latency_ms: float = 0.0
    poll_latency_ms: float = 0.0
    fetch_latency_ms: float = 0.0
    
    # Cost
    cost_usd: Optional[float] = None
    
    # Error details (for error_count > 0)
    last_error_category: Optional[str] = None
    last_error_code: Optional[str] = None
    
    def mark_success(self, total_latency_ms: float) -> "ProviderMetrics":
        """Mark operation as successful."""
        self.success_count = 1
        self.total_latency_ms = total_latency_ms
        return self
    
    def mark_failure(
        self,
        error_category: str,
        error_code: str,
        total_latency_ms: float,
    ) -> "ProviderMetrics":
        """Mark operation as failed."""
        self.error_count = 1
        self.last_error_category = error_category
        self.last_error_code = error_code
        self.total_latency_ms = total_latency_ms
        return self
    
    def mark_retry(self, attempt: int) -> "ProviderMetrics":
        """Increment retry counter."""
        self.retry_count = attempt
        return self
    
    def mark_fallback(self) -> "ProviderMetrics":
        """Mark fallback activation."""
        self.fallback_count = 1
        return self
    
    def set_cost(self, cost_usd: float) -> "ProviderMetrics":
        """Set provider-reported cost."""
        self.cost_usd = cost_usd
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary for metrics emission."""
        result = {
            "request_id": self.request_id,
            "segment_key": self.segment_key,
            "capability": self.capability,
            "request_count": self.request_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "retry_count": self.retry_count,
            "fallback_count": self.fallback_count,
            "total_latency_ms": round(self.total_latency_ms, 2),
        }
        if self.workflow_id:
            result["workflow_id"] = self.workflow_id
        if self.submit_latency_ms > 0:
            result["submit_latency_ms"] = round(self.submit_latency_ms, 2)
        if self.poll_latency_ms > 0:
            result["poll_latency_ms"] = round(self.poll_latency_ms, 2)
        if self.fetch_latency_ms > 0:
            result["fetch_latency_ms"] = round(self.fetch_latency_ms, 2)
        if self.cost_usd is not None:
            result["cost_usd"] = self.cost_usd
        if self.last_error_category:
            result["last_error_category"] = self.last_error_category
        if self.last_error_code:
            result["last_error_code"] = self.last_error_code
        return result


class MetricsEmitter(Protocol):
    """Protocol for metrics emission backends."""
    
    def emit(self, metrics: ProviderMetrics) -> None:
        """Emit metrics to the configured backend."""
        ...


class LoggingMetricsEmitter:
    """Emit metrics as structured log records."""
    
    def emit(self, metrics: ProviderMetrics) -> None:
        """Emit metrics through structured logging."""
        record = StructuredLogRecord(
            request_id=metrics.request_id,
            segment_key=metrics.segment_key,
            capability=metrics.capability,
            workflow_id=metrics.workflow_id,
            event_type=EventType.METRICS_EMIT,
            status=StatusType.SUCCESS if metrics.success_count > 0 else StatusType.FAILURE,
            duration_ms=metrics.total_latency_ms,
            error_category=metrics.last_error_category,
            reason_code=metrics.last_error_code,
            metadata={
                "counters": {
                    "success": metrics.success_count,
                    "error": metrics.error_count,
                    "retry": metrics.retry_count,
                    "fallback": metrics.fallback_count,
                },
                "latencies": {
                    "total_ms": metrics.total_latency_ms,
                    "submit_ms": metrics.submit_latency_ms,
                    "poll_ms": metrics.poll_latency_ms,
                    "fetch_ms": metrics.fetch_latency_ms,
                },
                "cost_usd": metrics.cost_usd,
            },
        )
        log_structured(record)


# Singleton emitter instance
_metrics_emitter: Optional[MetricsEmitter] = None


def get_metrics_emitter() -> MetricsEmitter:
    """Get the configured metrics emitter (default: logging)."""
    global _metrics_emitter
    if _metrics_emitter is None:
        _metrics_emitter = LoggingMetricsEmitter()
    return _metrics_emitter


def set_metrics_emitter(emitter: MetricsEmitter) -> None:
    """Configure a custom metrics emitter."""
    global _metrics_emitter
    _metrics_emitter = emitter


def emit_metrics(metrics: ProviderMetrics) -> None:
    """Emit metrics through the configured emitter."""
    get_metrics_emitter().emit(metrics)


# ─────────────────────────────────────────────
# Adapter Response Metadata Extension
# ─────────────────────────────────────────────
OBSERVABILITY_METADATA_KEYS = {
    "request_id",
    "segment_key",
    "capability",
    "workflow_id",
    "total_latency_ms",
    "submit_latency_ms",
    "poll_latency_ms",
    "fetch_latency_ms",
    "cost_usd",
    "attempt",
    "error_category",
    "reason_code",
}


def extract_observability_metadata(
    response_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract observability-relevant keys from adapter response metadata.
    
    Filters to allowlisted keys to prevent provider-specific data leakage.
    """
    return {
        k: v for k, v in response_metadata.items()
        if k in OBSERVABILITY_METADATA_KEYS
    }


def inject_correlation_metadata(
    ctx: CorrelationContext,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Inject correlation context into response metadata.
    
    Merges correlation IDs without overwriting existing keys.
    """
    result = metadata.copy()
    result.setdefault("request_id", ctx.request_id)
    result.setdefault("segment_key", ctx.segment_key)
    result.setdefault("capability", ctx.capability)
    if ctx.workflow_id:
        result.setdefault("workflow_id", ctx.workflow_id)
    return result


# ─────────────────────────────────────────────
# Metrics Extraction from Adapter Response
# ─────────────────────────────────────────────
def extract_metrics_from_response(
    ctx: CorrelationContext,
    response_metadata: Dict[str, Any],
    success: bool,
    start_time: float,
    error_category: Optional[str] = None,
    error_code: Optional[str] = None,
    attempt: int = 1,
) -> ProviderMetrics:
    """
    Extract ProviderMetrics from adapter response metadata.
    
    Combines correlation context with response data to build
    a complete metrics record.
    """
    total_latency_ms = (time.perf_counter() - start_time) * 1000
    
    metrics = ProviderMetrics(
        request_id=ctx.request_id,
        segment_key=ctx.segment_key,
        capability=ctx.capability or "unknown",
        workflow_id=ctx.workflow_id,
    )
    
    # Set counters based on outcome
    if success:
        metrics.mark_success(total_latency_ms)
    else:
        metrics.mark_failure(
            error_category=error_category or "UNKNOWN",
            error_code=error_code or "UNKNOWN",
            total_latency_ms=total_latency_ms,
        )
    
    # Extract retry count
    if attempt > 1:
        metrics.mark_retry(attempt - 1)
    
    # Extract latency breakdown from response metadata
    if "submit_latency_ms" in response_metadata:
        metrics.submit_latency_ms = response_metadata["submit_latency_ms"]
    if "poll_latency_ms" in response_metadata:
        metrics.poll_latency_ms = response_metadata["poll_latency_ms"]
    if "fetch_latency_ms" in response_metadata:
        metrics.fetch_latency_ms = response_metadata["fetch_latency_ms"]
    
    # Extract cost if available
    if "cost_usd" in response_metadata:
        metrics.set_cost(response_metadata["cost_usd"])
    
    return metrics
