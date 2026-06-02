"""
Tests for I2V Adapter - Image-to-Video conversion.

Tests cover:
- Non-test mode: real provider client submit->wait->fetch flow
- Test mode: deterministic behavior preservation
- Extension validation (unchanged)
- Success only when fetched output exists
- Typed error mapping to AdapterError/ErrorCategory
"""
import os
import tempfile
from unittest.mock import Mock, MagicMock, patch

import pytest

from pixelle_snapshot.adapters.i2v import I2VAdapter
from pixelle_snapshot.adapters.contracts import (
    I2VRequest,
    I2VResponse,
    ValidationError,
    ExecutionError,
    TimeoutError,
    ErrorCategory,
    ProviderJobStatus,
    ProviderSubmitResult,
    ProviderPollResult,
    ProviderFetchResult,
)


@pytest.fixture
def temp_dirs():
    """Create temporary project structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = tmpdir
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create dummy input image
        input_path = os.path.join(tmpdir, "test.png")
        with open(input_path, "wb") as f:
            f.write(b"PNG_DUMMY")
        
        yield {
            "project_root": project_root,
            "output_dir": output_dir,
            "input_path": input_path,
        }


@pytest.fixture
def base_request(temp_dirs):
    """Base I2V request for testing."""
    return I2VRequest(
        segment_key="seg_001",
        segment_text="Test segment",
        segment_duration=5.0,
        project_root=temp_dirs["project_root"],
        output_dir=temp_dirs["output_dir"],
        input_image_path=temp_dirs["input_path"],
        motion_type="kenburns",
        motion_direction="in",
        motion_speed=1.0,
        enhance_image=False,
        add_ambient_motion=True,
        loop_seamless=False,
        target_duration=None,
        target_fps=30,
        timeout_seconds=10.0,
    )


# ─────────────────────────────────────────────
# Extension Validation Tests (Unchanged)
# ─────────────────────────────────────────────
def test_validate_supported_extensions(base_request, temp_dirs):
    """Supported image extensions pass validation."""
    adapter = I2VAdapter()
    
    for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"]:
        img_path = os.path.join(temp_dirs["project_root"], f"test{ext}")
        with open(img_path, "wb") as f:
            f.write(b"DUMMY")
        
        request = I2VRequest(
            **{**base_request.__dict__, "input_image_path": img_path}
        )
        adapter.validate(request)  # Should not raise


def test_validate_unsupported_extension(base_request, temp_dirs):
    """Unsupported image extensions raise ValidationError."""
    adapter = I2VAdapter()
    
    bad_path = os.path.join(temp_dirs["project_root"], "test.tiff")
    with open(bad_path, "wb") as f:
        f.write(b"DUMMY")
    
    request = I2VRequest(
        **{**base_request.__dict__, "input_image_path": bad_path}
    )
    
    with pytest.raises(ValidationError) as exc_info:
        adapter.validate(request)
    
    assert ".tiff" in str(exc_info.value)
    assert exc_info.value.category == ErrorCategory.VALIDATION


# ─────────────────────────────────────────────
# Test Mode Tests (Deterministic Behavior)
# ─────────────────────────────────────────────
@patch("pixelle_snapshot.adapters.i2v.test_doubles.is_test_mode_enabled", return_value=True)
@patch("pixelle_snapshot.adapters.i2v.test_doubles.create_i2v_test_output")
def test_test_mode_deterministic_output(mock_create, mock_test_mode, base_request):
    """Test mode uses deterministic test doubles."""
    mock_output = Mock()
    mock_output.output_path = "/fake/output.mp4"
    mock_output.video_duration = 5.0
    mock_output.video_resolution = "1080x1920"
    mock_output.video_fps = 30
    mock_output.to_metadata.return_value = {"test": True}
    mock_create.return_value = mock_output
    
    adapter = I2VAdapter()
    response = adapter.invoke(base_request)
    
    assert response.success is True
    assert response.segment_key == "seg_001"
    assert response.output_path == "/fake/output.mp4"
    assert response.video_duration == 5.0
    assert response.video_resolution == "1080x1920"
    assert response.video_fps == 30
    assert response.motion_applied == "kenburns"
    assert response.enhancement_applied is False
    assert "test" in response.metadata


# ─────────────────────────────────────────────
# Non-Test Mode: Real Provider Client Flow
# ─────────────────────────────────────────────
@patch("pixelle_snapshot.adapters.i2v.test_doubles.is_test_mode_enabled", return_value=False)
def test_provider_client_submit_wait_fetch_success(mock_test_mode, base_request, temp_dirs):
    """Non-test mode executes submit->wait_for_completion->fetch."""
    mock_client = Mock()
    
    # Setup mock responses
    mock_client.submit.return_value = ProviderSubmitResult(
        job_id="job_123",
        status=ProviderJobStatus.SUBMITTED,
        metadata={"provider_job_id": "job_123"},
    )
    
    mock_client.wait_for_completion.return_value = ProviderPollResult(
        job_id="job_123",
        status=ProviderJobStatus.SUCCEEDED,
        metadata={"run_seconds": 12.5},
    )
    
    output_path = os.path.join(temp_dirs["output_dir"], "i2v_seg_001.mp4")
    with open(output_path, "wb") as f:
        f.write(b"MP4_VIDEO_DATA")
    
    mock_client.fetch.return_value = ProviderFetchResult(
        job_id="job_123",
        output_path=output_path,
        metadata={"artifact_bytes": 14},
    )
    
    adapter = I2VAdapter(provider_client=mock_client)
    response = adapter.invoke(base_request)
    
    # Verify client calls
    mock_client.submit.assert_called_once()
    mock_client.wait_for_completion.assert_called_once_with(
        job_id="job_123",
        timeout_seconds=10.0,
        cancel_on_timeout=True,
    )
    mock_client.fetch.assert_called_once_with(
        job_id="job_123",
        output_dir=temp_dirs["output_dir"],
    )
    
    # Verify response
    assert response.success is True
    assert response.segment_key == "seg_001"
    assert response.output_path == output_path
    assert response.output_path is not None
    assert os.path.exists(response.output_path)
    assert response.metadata["provider_job_id"] == "job_123"
    assert response.metadata["run_seconds"] == 12.5


@patch("pixelle_snapshot.adapters.i2v.test_doubles.is_test_mode_enabled", return_value=False)
def test_provider_client_submit_success_idempotency_key(mock_test_mode, base_request):
    """Submit includes computed idempotency key."""
    mock_client = Mock()
    mock_client.submit.return_value = ProviderSubmitResult(
        job_id="job_456",
        status=ProviderJobStatus.SUBMITTED,
    )
    mock_client.wait_for_completion.side_effect = TimeoutError(
        message="Timeout",
        timeout_seconds=10.0,
    )
    
    adapter = I2VAdapter(provider_client=mock_client)
    response = adapter.invoke(base_request)
    
    # Verify idempotency key passed to submit
    call_kwargs = mock_client.submit.call_args.kwargs
    assert call_kwargs["capability"] == "i2v"
    assert call_kwargs["request"] == base_request
    assert "idempotency_key" in call_kwargs
    assert isinstance(call_kwargs["idempotency_key"], str)
    assert len(call_kwargs["idempotency_key"]) == 32  # SHA256[:32]


@patch("pixelle_snapshot.adapters.i2v.test_doubles.is_test_mode_enabled", return_value=False)
def test_provider_client_fetch_missing_output_fails(mock_test_mode, base_request, temp_dirs):
    """Success only when fetched output path exists."""
    mock_client = Mock()
    mock_client.submit.return_value = ProviderSubmitResult(
        job_id="job_789",
        status=ProviderJobStatus.SUBMITTED,
    )
    mock_client.wait_for_completion.return_value = ProviderPollResult(
        job_id="job_789",
        status=ProviderJobStatus.SUCCEEDED,
    )
    
    # Fetch returns path but file doesn't exist
    nonexistent_path = os.path.join(temp_dirs["output_dir"], "missing.mp4")
    mock_client.fetch.return_value = ProviderFetchResult(
        job_id="job_789",
        output_path=nonexistent_path,
    )
    
    adapter = I2VAdapter(provider_client=mock_client)
    response = adapter.invoke(base_request)
    
    # Should fail because output doesn't exist
    assert response.success is False
    assert response.error is not None
    assert response.error.category == ErrorCategory.EXECUTION
    assert "output file not found" in response.error.message.lower()


# ─────────────────────────────────────────────
# Error Mapping Tests
# ─────────────────────────────────────────────
@patch("pixelle_snapshot.adapters.i2v.test_doubles.is_test_mode_enabled", return_value=False)
def test_provider_client_timeout_error_mapping(mock_test_mode, base_request):
    """TimeoutError from wait_for_completion mapped correctly."""
    mock_client = Mock()
    mock_client.submit.return_value = ProviderSubmitResult(
        job_id="job_timeout",
        status=ProviderJobStatus.SUBMITTED,
    )
    mock_client.wait_for_completion.side_effect = TimeoutError(
        message="Provider job timeout after 10s",
        timeout_seconds=10.0,
        job_id="job_timeout",
    )
    
    adapter = I2VAdapter(provider_client=mock_client)
    response = adapter.invoke(base_request)
    
    assert response.success is False
    assert response.error is not None
    assert response.error.category == ErrorCategory.TIMEOUT
    assert "timeout" in response.error.message.lower()


@patch("pixelle_snapshot.adapters.i2v.test_doubles.is_test_mode_enabled", return_value=False)
def test_provider_client_execution_error_mapping(mock_test_mode, base_request):
    """ExecutionError from provider client mapped correctly."""
    mock_client = Mock()
    mock_client.submit.side_effect = ExecutionError(
        message="Provider submit failed",
        provider_status=500,
    )
    
    adapter = I2VAdapter(provider_client=mock_client)
    response = adapter.invoke(base_request)
    
    assert response.success is False
    assert response.error is not None
    assert response.error.category == ErrorCategory.EXECUTION
    assert "failed" in response.error.message.lower()


@patch("pixelle_snapshot.adapters.i2v.test_doubles.is_test_mode_enabled", return_value=False)
def test_provider_client_poll_failed_status(mock_test_mode, base_request, temp_dirs):
    """Poll returns FAILED status - should map to ExecutionError."""
    mock_client = Mock()
    mock_client.submit.return_value = ProviderSubmitResult(
        job_id="job_fail",
        status=ProviderJobStatus.SUBMITTED,
    )
    mock_client.wait_for_completion.return_value = ProviderPollResult(
        job_id="job_fail",
        status=ProviderJobStatus.FAILED,
        metadata={"error": "Provider processing failed"},
    )
    
    adapter = I2VAdapter(provider_client=mock_client)
    response = adapter.invoke(base_request)
    
    # wait_for_completion returns FAILED status - adapter should detect and fail
    assert response.success is False
    assert response.error is not None
    assert response.error.category == ErrorCategory.PROVIDER


# ─────────────────────────────────────────────
# Integration Test
# ─────────────────────────────────────────────
@patch("pixelle_snapshot.adapters.i2v.test_doubles.is_test_mode_enabled", return_value=False)
def test_full_integration_real_provider_flow(mock_test_mode, base_request, temp_dirs):
    """Full end-to-end flow with real provider client lifecycle."""
    mock_client = Mock()
    
    # Simulate full lifecycle
    mock_client.submit.return_value = ProviderSubmitResult(
        job_id="job_integration",
        status=ProviderJobStatus.SUBMITTED,
        metadata={"request_id": "req_123"},
    )
    
    mock_client.wait_for_completion.return_value = ProviderPollResult(
        job_id="job_integration",
        status=ProviderJobStatus.SUCCEEDED,
        metadata={"run_seconds": 8.2, "queued_seconds": 1.1},
    )
    
    output_path = os.path.join(temp_dirs["output_dir"], "i2v_seg_001.mp4")
    with open(output_path, "wb") as f:
        f.write(b"COMPLETE_VIDEO_DATA")
    
    mock_client.fetch.return_value = ProviderFetchResult(
        job_id="job_integration",
        output_path=output_path,
        metadata={"artifact_bytes": 19, "cost_usd": 0.05},
    )
    
    adapter = I2VAdapter(provider_client=mock_client)
    response = adapter.invoke(base_request)
    
    # Verify full success
    assert response.success is True
    assert response.segment_key == "seg_001"
    assert response.output_path == output_path
    assert response.output_path is not None
    assert os.path.exists(response.output_path)
    
    # Verify metadata includes provider info
    assert response.metadata["provider_job_id"] == "job_integration"
    assert response.metadata["run_seconds"] == 8.2
    assert response.metadata["queued_seconds"] == 1.1
    assert response.metadata["artifact_bytes"] == 19
    assert response.metadata["cost_usd"] == 0.05
    
    # Verify execution time recorded
    assert response.execution_time_seconds > 0
