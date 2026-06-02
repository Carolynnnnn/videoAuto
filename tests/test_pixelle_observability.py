"""
Task 5 Tests: Correlation IDs and Structured Observability Schema

Tests verify:
1. CorrelationContext creation and thread-safe propagation
2. StructuredLogRecord required keys (request_id, segment_key, capability, status)
3. ProviderMetrics extraction contract for latency/error/cost counters
4. Metrics emission through structured logging
5. Metadata injection and extraction without provider data leakage
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.steps.pixelle_observability import (
    # Correlation
    CorrelationContext,
    generate_request_id,
    generate_workflow_id,
    set_workflow_id,
    get_workflow_id,
    # Structured Logging
    EventType,
    StatusType,
    StructuredLogRecord,
    log_structured,
    # Metrics
    ProviderMetrics,
    MetricsEmitter,
    LoggingMetricsEmitter,
    get_metrics_emitter,
    set_metrics_emitter,
    emit_metrics,
    # Metadata Helpers
    OBSERVABILITY_METADATA_KEYS,
    extract_observability_metadata,
    inject_correlation_metadata,
    extract_metrics_from_response,
)


class TestCorrelationContext:
    """Tests for correlation ID generation and propagation."""
    
    def test_generate_request_id_format(self):
        """Request IDs should follow req_<16-hex-chars> format."""
        req_id = generate_request_id()
        assert req_id.startswith("req_")
        assert len(req_id) == 20  # req_ + 16 hex chars
        assert all(c in "0123456789abcdef" for c in req_id[4:])
    
    def test_generate_request_id_uniqueness(self):
        """Request IDs should be unique."""
        ids = {generate_request_id() for _ in range(100)}
        assert len(ids) == 100
    
    def test_generate_workflow_id_format(self):
        """Workflow IDs should follow wf_<12-hex-chars> format."""
        wf_id = generate_workflow_id()
        assert wf_id.startswith("wf_")
        assert len(wf_id) == 15  # wf_ + 12 hex chars
    
    def test_workflow_id_context_propagation(self):
        """Workflow ID should propagate through context."""
        wf_id = generate_workflow_id()
        set_workflow_id(wf_id)
        assert get_workflow_id() == wf_id
    
    def test_correlation_context_create_required_fields(self):
        """CorrelationContext.create() should set all required fields."""
        ctx = CorrelationContext.create(
            segment_key="seg_abc123",
            capability="digital_human",
        )
        
        # Required fields present
        assert ctx.request_id is not None
        assert ctx.request_id.startswith("req_")
        assert ctx.segment_key == "seg_abc123"
        assert ctx.capability == "digital_human"
        assert ctx.created_at is not None
    
    def test_correlation_context_inherits_workflow_id(self):
        """CorrelationContext should inherit workflow_id from context."""
        wf_id = generate_workflow_id()
        set_workflow_id(wf_id)
        
        ctx = CorrelationContext.create(segment_key="seg_test")
        assert ctx.workflow_id == wf_id
    
    def test_correlation_context_to_dict_all_keys(self):
        """to_dict() should include all correlation fields."""
        ctx = CorrelationContext.create(
            segment_key="seg_key_1",
            capability="i2v",
            workflow_id="wf_test123",
        )
        
        d = ctx.to_dict()
        assert "request_id" in d
        assert d["segment_key"] == "seg_key_1"
        assert d["capability"] == "i2v"
        assert d["workflow_id"] == "wf_test123"
        assert "created_at" in d


class TestStructuredLogRecord:
    """Tests for structured log records with required keys."""
    
    def test_required_keys_present(self):
        """StructuredLogRecord must include request_id, segment_key, capability, status."""
        record = StructuredLogRecord(
            request_id="req_test123",
            segment_key="seg_abc",
            capability="digital_human",
            status=StatusType.STARTED,
        )
        
        d = record.to_dict()
        
        # Required keys contract
        assert "request_id" in d
        assert "segment_key" in d
        assert "capability" in d
        assert "status" in d
        
        assert d["request_id"] == "req_test123"
        assert d["segment_key"] == "seg_abc"
        assert d["capability"] == "digital_human"
        assert d["status"] == "STARTED"
    
    def test_from_context_creates_complete_record(self):
        """from_context() should create record with all context fields."""
        ctx = CorrelationContext.create(
            segment_key="seg_from_ctx",
            capability="action_transfer",
            workflow_id="wf_ctx_test",
        )
        
        record = StructuredLogRecord.from_context(
            ctx,
            event_type=EventType.ADAPTER_START,
            status=StatusType.STARTED,
        )
        
        assert record.request_id == ctx.request_id
        assert record.segment_key == ctx.segment_key
        assert record.capability == ctx.capability
        assert record.workflow_id == ctx.workflow_id
    
    def test_with_duration_sets_ms(self):
        """with_duration() should calculate elapsed milliseconds."""
        record = StructuredLogRecord(
            request_id="req_dur",
            segment_key="seg_dur",
            capability="i2v",
            status=StatusType.SUCCESS,
        )
        
        start = time.perf_counter() - 0.1  # 100ms ago
        record.with_duration(start)
        
        assert record.duration_ms is not None
        assert record.duration_ms >= 100  # At least 100ms
    
    def test_with_error_sets_category_and_code(self):
        """with_error() should set error details."""
        record = StructuredLogRecord(
            request_id="req_err",
            segment_key="seg_err",
            capability="digital_human",
            status=StatusType.FAILURE,
        )
        
        record.with_error(
            error_category="VALIDATION",
            reason_code="PIXELLE_REQUEST_BUILD_FAILED",
        )
        
        assert record.error_category == "VALIDATION"
        assert record.reason_code == "PIXELLE_REQUEST_BUILD_FAILED"
    
    def test_to_json_parseable(self):
        """to_json() should produce valid JSON."""
        record = StructuredLogRecord(
            request_id="req_json",
            segment_key="seg_json",
            capability="i2v",
            status=StatusType.SUCCESS,
            duration_ms=150.5,
            metadata={"key": "value"},
        )
        
        json_str = record.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["request_id"] == "req_json"
        assert parsed["segment_key"] == "seg_json"
        assert parsed["capability"] == "i2v"
        assert parsed["status"] == "SUCCESS"
        assert parsed["duration_ms"] == 150.5
    
    def test_capability_can_be_none_before_routing(self):
        """capability may be None before routing decision."""
        record = StructuredLogRecord(
            request_id="req_none_cap",
            segment_key="seg_pre_route",
            capability=None,
            status=StatusType.STARTED,
        )
        
        d = record.to_dict()
        assert "capability" in d
        assert d["capability"] is None


class TestProviderMetrics:
    """Tests for metrics extraction contract."""
    
    def test_required_fields_present(self):
        """ProviderMetrics must include correlation fields."""
        metrics = ProviderMetrics(
            request_id="req_met",
            segment_key="seg_met",
            capability="digital_human",
        )
        
        d = metrics.to_dict()
        assert d["request_id"] == "req_met"
        assert d["segment_key"] == "seg_met"
        assert d["capability"] == "digital_human"
    
    def test_counter_fields(self):
        """Metrics should include counter fields."""
        metrics = ProviderMetrics(
            request_id="req_cnt",
            segment_key="seg_cnt",
            capability="i2v",
        )
        
        d = metrics.to_dict()
        assert d["request_count"] == 1
        assert d["success_count"] == 0
        assert d["error_count"] == 0
        assert d["retry_count"] == 0
        assert d["fallback_count"] == 0
    
    def test_mark_success_sets_counters_and_latency(self):
        """mark_success() should set success counter and total latency."""
        metrics = ProviderMetrics(
            request_id="req_suc",
            segment_key="seg_suc",
            capability="digital_human",
        )
        
        metrics.mark_success(total_latency_ms=500.0)
        
        assert metrics.success_count == 1
        assert metrics.total_latency_ms == 500.0
        assert metrics.error_count == 0
    
    def test_mark_failure_sets_error_details(self):
        """mark_failure() should set error counter and details."""
        metrics = ProviderMetrics(
            request_id="req_fail",
            segment_key="seg_fail",
            capability="i2v",
        )
        
        metrics.mark_failure(
            error_category="TIMEOUT",
            error_code="PIXELLE_TIMEOUT",
            total_latency_ms=30000.0,
        )
        
        assert metrics.error_count == 1
        assert metrics.last_error_category == "TIMEOUT"
        assert metrics.last_error_code == "PIXELLE_TIMEOUT"
        assert metrics.total_latency_ms == 30000.0
    
    def test_mark_retry_increments_counter(self):
        """mark_retry() should set retry count."""
        metrics = ProviderMetrics(
            request_id="req_retry",
            segment_key="seg_retry",
            capability="action_transfer",
        )
        
        metrics.mark_retry(attempt=3)
        
        assert metrics.retry_count == 3
    
    def test_mark_fallback(self):
        """mark_fallback() should set fallback counter."""
        metrics = ProviderMetrics(
            request_id="req_fb",
            segment_key="seg_fb",
            capability="digital_human",
        )
        
        metrics.mark_fallback()
        
        assert metrics.fallback_count == 1
    
    def test_set_cost(self):
        """set_cost() should set provider cost."""
        metrics = ProviderMetrics(
            request_id="req_cost",
            segment_key="seg_cost",
            capability="digital_human",
        )
        
        metrics.set_cost(cost_usd=0.05)
        
        assert metrics.cost_usd == 0.05
        d = metrics.to_dict()
        assert d["cost_usd"] == 0.05
    
    def test_latency_breakdown_fields(self):
        """Metrics should include latency breakdown when set."""
        metrics = ProviderMetrics(
            request_id="req_lat",
            segment_key="seg_lat",
            capability="i2v",
            submit_latency_ms=100.0,
            poll_latency_ms=200.0,
            fetch_latency_ms=50.0,
            total_latency_ms=350.0,
        )
        
        d = metrics.to_dict()
        assert d["submit_latency_ms"] == 100.0
        assert d["poll_latency_ms"] == 200.0
        assert d["fetch_latency_ms"] == 50.0
        assert d["total_latency_ms"] == 350.0


class TestMetricsEmission:
    """Tests for metrics emission through structured logging."""
    
    def test_logging_metrics_emitter(self):
        """LoggingMetricsEmitter should emit through structured logging."""
        emitter = LoggingMetricsEmitter()
        metrics = ProviderMetrics(
            request_id="req_emit",
            segment_key="seg_emit",
            capability="digital_human",
        )
        metrics.mark_success(total_latency_ms=200.0)
        
        with patch("src.steps.pixelle_observability.log_structured") as mock_log:
            emitter.emit(metrics)
            mock_log.assert_called_once()
            record = mock_log.call_args[0][0]
            assert record.request_id == "req_emit"
            assert record.event_type == EventType.METRICS_EMIT
    
    def test_emit_metrics_uses_configured_emitter(self):
        """emit_metrics() should use the configured emitter."""
        mock_emitter = MagicMock(spec=MetricsEmitter)
        set_metrics_emitter(mock_emitter)
        
        metrics = ProviderMetrics(
            request_id="req_cfg",
            segment_key="seg_cfg",
            capability="i2v",
        )
        
        emit_metrics(metrics)
        mock_emitter.emit.assert_called_once_with(metrics)
        
        # Reset to default
        set_metrics_emitter(LoggingMetricsEmitter())


class TestMetadataExtraction:
    """Tests for metadata extraction and injection."""
    
    def test_extract_observability_metadata_filters_keys(self):
        """extract_observability_metadata() should filter to allowlist."""
        full_metadata = {
            "request_id": "req_ext",
            "segment_key": "seg_ext",
            "capability": "digital_human",
            "workflow_id": "wf_ext",
            "total_latency_ms": 500.0,
            "cost_usd": 0.03,
            # Provider-specific keys that should be filtered
            "provider_internal_id": "abc123",
            "raw_response": {"data": "sensitive"},
            "api_version": "v2",
        }
        
        filtered = extract_observability_metadata(full_metadata)
        
        # Allowed keys present
        assert "request_id" in filtered
        assert "segment_key" in filtered
        assert "capability" in filtered
        assert "workflow_id" in filtered
        assert "total_latency_ms" in filtered
        assert "cost_usd" in filtered
        
        # Provider-specific keys filtered out
        assert "provider_internal_id" not in filtered
        assert "raw_response" not in filtered
        assert "api_version" not in filtered
    
    def test_inject_correlation_metadata_merges_context(self):
        """inject_correlation_metadata() should merge context without overwriting."""
        ctx = CorrelationContext.create(
            segment_key="seg_inj",
            capability="i2v",
            workflow_id="wf_inj",
        )
        
        existing_metadata = {
            "existing_key": "value",
            "capability": "should_not_overwrite",  # Existing value
        }
        
        result = inject_correlation_metadata(ctx, existing_metadata)
        
        # Correlation fields injected
        assert result["request_id"] == ctx.request_id
        assert result["segment_key"] == ctx.segment_key
        assert result["workflow_id"] == ctx.workflow_id
        
        # Existing values preserved
        assert result["existing_key"] == "value"
        assert result["capability"] == "should_not_overwrite"  # Not overwritten


class TestExtractMetricsFromResponse:
    """Tests for extracting metrics from adapter response."""
    
    def test_success_case(self):
        """Should extract metrics for successful response."""
        ctx = CorrelationContext.create(
            segment_key="seg_resp_ok",
            capability="digital_human",
            workflow_id="wf_resp",
        )
        
        response_metadata = {
            "submit_latency_ms": 50.0,
            "poll_latency_ms": 100.0,
            "fetch_latency_ms": 30.0,
            "cost_usd": 0.02,
        }
        
        start_time = time.perf_counter() - 0.2  # 200ms ago
        
        metrics = extract_metrics_from_response(
            ctx=ctx,
            response_metadata=response_metadata,
            success=True,
            start_time=start_time,
        )
        
        assert metrics.request_id == ctx.request_id
        assert metrics.segment_key == ctx.segment_key
        assert metrics.capability == "digital_human"
        assert metrics.workflow_id == ctx.workflow_id
        assert metrics.success_count == 1
        assert metrics.error_count == 0
        assert metrics.total_latency_ms >= 200  # At least 200ms
        assert metrics.submit_latency_ms == 50.0
        assert metrics.poll_latency_ms == 100.0
        assert metrics.fetch_latency_ms == 30.0
        assert metrics.cost_usd == 0.02
    
    def test_failure_case(self):
        """Should extract metrics for failed response."""
        ctx = CorrelationContext.create(
            segment_key="seg_resp_fail",
            capability="i2v",
        )
        
        start_time = time.perf_counter() - 0.5  # 500ms ago
        
        metrics = extract_metrics_from_response(
            ctx=ctx,
            response_metadata={},
            success=False,
            start_time=start_time,
            error_category="VALIDATION",
            error_code="PIXELLE_REQUEST_BUILD_FAILED",
        )
        
        assert metrics.success_count == 0
        assert metrics.error_count == 1
        assert metrics.last_error_category == "VALIDATION"
        assert metrics.last_error_code == "PIXELLE_REQUEST_BUILD_FAILED"
        assert metrics.total_latency_ms >= 500
    
    def test_retry_tracking(self):
        """Should track retry attempts."""
        ctx = CorrelationContext.create(
            segment_key="seg_retry_track",
            capability="action_transfer",
        )
        
        metrics = extract_metrics_from_response(
            ctx=ctx,
            response_metadata={},
            success=True,
            start_time=time.perf_counter(),
            attempt=3,
        )
        
        assert metrics.retry_count == 2  # attempt - 1


class TestObservabilityMetadataKeys:
    """Tests for observability metadata allowlist."""
    
    def test_required_keys_in_allowlist(self):
        """Required keys must be in the allowlist."""
        required = {"request_id", "segment_key", "capability", "workflow_id"}
        assert required.issubset(OBSERVABILITY_METADATA_KEYS)
    
    def test_latency_keys_in_allowlist(self):
        """Latency keys must be in the allowlist."""
        latency_keys = {
            "total_latency_ms",
            "submit_latency_ms",
            "poll_latency_ms",
            "fetch_latency_ms",
        }
        assert latency_keys.issubset(OBSERVABILITY_METADATA_KEYS)
    
    def test_cost_key_in_allowlist(self):
        """Cost key must be in the allowlist."""
        assert "cost_usd" in OBSERVABILITY_METADATA_KEYS
    
    def test_error_keys_in_allowlist(self):
        """Error-related keys must be in the allowlist."""
        error_keys = {"error_category", "reason_code"}
        assert error_keys.issubset(OBSERVABILITY_METADATA_KEYS)


class TestEventTypes:
    """Tests for event type enumeration."""
    
    def test_adapter_lifecycle_events(self):
        """Adapter lifecycle events should be defined."""
        assert EventType.ADAPTER_START.value == "ADAPTER_START"
        assert EventType.ADAPTER_SUCCESS.value == "ADAPTER_SUCCESS"
        assert EventType.ADAPTER_ERROR.value == "ADAPTER_ERROR"
        assert EventType.ADAPTER_RETRY.value == "ADAPTER_RETRY"
        assert EventType.ADAPTER_FALLBACK.value == "ADAPTER_FALLBACK"
    
    def test_provider_lifecycle_events(self):
        """Provider lifecycle events should be defined."""
        assert EventType.PROVIDER_SUBMIT.value == "PROVIDER_SUBMIT"
        assert EventType.PROVIDER_POLL.value == "PROVIDER_POLL"
        assert EventType.PROVIDER_FETCH.value == "PROVIDER_FETCH"
        assert EventType.PROVIDER_CANCEL.value == "PROVIDER_CANCEL"
    
    def test_metrics_emit_event(self):
        """Metrics emit event should be defined."""
        assert EventType.METRICS_EMIT.value == "METRICS_EMIT"


class TestStatusTypes:
    """Tests for status type enumeration."""
    
    def test_all_status_types_defined(self):
        """All status types should be defined."""
        assert StatusType.STARTED.value == "STARTED"
        assert StatusType.SUCCESS.value == "SUCCESS"
        assert StatusType.FAILURE.value == "FAILURE"
        assert StatusType.RETRY.value == "RETRY"
        assert StatusType.FALLBACK.value == "FALLBACK"
