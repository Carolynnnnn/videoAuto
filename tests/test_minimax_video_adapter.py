"""
Tests for Minimax Video Adapter - Text/Image-to-Video generation.

Tests cover:
- Test mode: deterministic behavior preservation
- Non-test mode: real provider client submit->poll->fetch flow
- Success only when fetched output exists
- Typed error mapping to AdapterError/ErrorCategory
- Idempotency key computation
"""
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from pixelle_snapshot.adapters.minimax_video import MinimaxVideoAdapter
from pixelle_snapshot.adapters.contracts import (
    MinimaxVideoRequest,
    MinimaxVideoResponse,
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
        
        # Create dummy input image (optional for minimax)
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
    """Base Minimax video request for testing."""
    return MinimaxVideoRequest(
        segment_key="seg_001",
        segment_text="Test segment text",
        segment_duration=5.0,
        project_root=temp_dirs["project_root"],
        output_dir=temp_dirs["output_dir"],
        input_image_path=None,  # Text-to-video mode
        model="video-01",
        target_duration=None,
        target_fps=30,
        timeout_seconds=300.0,
    )


# ─────────────────────────────────────────────
# Test Mode Tests (Deterministic Behavior)
# ─────────────────────────────────────────────
@patch("pixelle_snapshot.adapters.minimax_video.test_doubles.is_test_mode_enabled", return_value=True)
@patch("pixelle_snapshot.adapters.minimax_video.test_doubles.create_deterministic_video_file")
def test_minimax_video_adapter_happy(mock_create, mock_test_mode, base_request, temp_dirs):
    """Test mode uses deterministic test doubles - happy path."""
    output_filename = "minimax_video_seg_001.test.mp4"
    expected_output = os.path.join(temp_dirs["output_dir"], output_filename)
    
    # Create the deterministic video file
    with open(expected_output, "wb") as f:
        f.write(b"DETERMINISTIC_VIDEO_DATA")
    
    adapter = MinimaxVideoAdapter()
    response = adapter.invoke(base_request)
    
    # Verify test mode was called
    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["output_path"] == expected_output
    assert call_kwargs["segment_key"] == "seg_001"
    assert call_kwargs["workflow_type"] == "minimax_video"
    assert call_kwargs["duration"] == 5.0
    
    # Verify response structure
    assert response.success is True
    assert response.segment_key == "seg_001"
    assert response.output_path == expected_output
    assert response.video_duration == 5.0
    assert response.video_resolution == "1080x1920"
    assert response.video_fps == 30
    assert response.model_used == "video-01"
    assert response.task_id == "test-task-id"
    
    # Verify metadata
    assert response.metadata["test_mode"] is True
    assert response.metadata["deterministic"] is True
    assert response.metadata["capability"] == "minimax_video"


@patch("pixelle_snapshot.adapters.minimax_video.test_doubles.is_test_mode_enabled", return_value=True)
@patch("pixelle_snapshot.adapters.minimax_video.test_doubles.create_deterministic_video_file")
def test_minimax_video_adapter_happy_with_image_input(mock_create, mock_test_mode, base_request, temp_dirs):
    """Test mode with input image - image-to-video mode."""
    request = MinimaxVideoRequest(
        **{**base_request.__dict__, "input_image_path": temp_dirs["input_path"]}
    )
    
    output_filename = "minimax_video_seg_001.test.mp4"
    expected_output = os.path.join(temp_dirs["output_dir"], output_filename)
    
    with open(expected_output, "wb") as f:
        f.write(b"DETERMINISTIC_VIDEO_DATA")
    
    adapter = MinimaxVideoAdapter()
    response = adapter.invoke(request)
    
    assert response.success is True
    assert response.segment_key == "seg_001"
    assert response.output_path == expected_output
    assert response.metadata["test_mode"] is True


@patch("pixelle_snapshot.adapters.minimax_video.test_doubles.is_test_mode_enabled", return_value=True)
@patch("pixelle_snapshot.adapters.minimax_video.test_doubles.create_deterministic_video_file")
def test_minimax_video_adapter_happy_with_target_duration(mock_create, mock_test_mode, base_request, temp_dirs):
    """Test mode with custom target_duration."""
    request = MinimaxVideoRequest(
        **{**base_request.__dict__, "target_duration": 8.0}
    )
    
    output_filename = "minimax_video_seg_001.test.mp4"
    expected_output = os.path.join(temp_dirs["output_dir"], output_filename)
    
    with open(expected_output, "wb") as f:
        f.write(b"DETERMINISTIC_VIDEO_DATA")
    
    adapter = MinimaxVideoAdapter()
    response = adapter.invoke(request)
    
    # Should use target_duration instead of segment_duration
    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["duration"] == 8.0
    
    assert response.success is True
    assert response.video_duration == 8.0


# ─────────────────────────────────────────────
# Non-Test Mode: Real Provider Client Flow
# ─────────────────────────────────────────────
@patch("pixelle_snapshot.adapters.minimax_video.test_doubles.is_test_mode_enabled", return_value=False)
def test_provider_client_submit_poll_fetch_success(mock_test_mode, base_request, temp_dirs):
    """Non-test mode executes submit->poll->fetch."""
    mock_client = Mock()
    
    # Setup mock responses
    mock_client.submit.return_value = ProviderSubmitResult(
        job_id="job_minimax_123",
        status=ProviderJobStatus.SUBMITTED,
        metadata={"task_id": "task_minimax_123"},
    )
    
    # Simulate polling loop
    mock_client.poll.side_effect = [
        ProviderPollResult(
            job_id="job_minimax_123",
            status=ProviderJobStatus.RUNNING,
            metadata={},
        ),
        ProviderPollResult(
            job_id="job_minimax_123",
            status=ProviderJobStatus.SUCCEEDED,
            metadata={"run_seconds": 45.2},
        ),
    ]
    
    output_path = os.path.join(temp_dirs["output_dir"], "minimax_video_seg_001.mp4")
    with open(output_path, "wb") as f:
        f.write(b"MINIMAX_VIDEO_DATA")
    
    mock_client.fetch.return_value = ProviderFetchResult(
        job_id="job_minimax_123",
        output_path=output_path,
        metadata={"artifact_bytes": 18},
    )
    
    adapter = MinimaxVideoAdapter(provider_client=mock_client)
    response = adapter.invoke(base_request)
    
    # Verify client calls
    mock_client.submit.assert_called_once()
    assert mock_client.poll.call_count >= 2  # At least one running, one succeeded
    # Minimax adapter calls fetch with positional args: fetch(job_id, output_dir)
    mock_client.fetch.assert_called_once_with("job_minimax_123", temp_dirs["output_dir"])
    
    # Verify response
    assert response.success is True
    assert response.segment_key == "seg_001"
    assert response.output_path == output_path
    assert response.output_path is not None
    assert os.path.exists(response.output_path)
    assert response.metadata["provider_job_id"] == "job_minimax_123"
    assert response.metadata["run_seconds"] == 45.2
    assert response.task_id == "task_minimax_123"


@patch("pixelle_snapshot.adapters.minimax_video.test_doubles.is_test_mode_enabled", return_value=False)
def test_provider_client_submit_includes_idempotency_key(mock_test_mode, base_request):
    """Submit includes computed idempotency key."""
    mock_client = Mock()
    mock_client.submit.return_value = ProviderSubmitResult(
        job_id="job_456",
        status=ProviderJobStatus.SUBMITTED,
    )
    mock_client.poll.return_value = ProviderPollResult(
        job_id="job_456",
        status=ProviderJobStatus.FAILED,
        metadata={"error": "Provider error"},
    )
    
    adapter = MinimaxVideoAdapter(provider_client=mock_client)
    response = adapter.invoke(base_request)
    
    # Verify idempotency key passed to submit
    call_kwargs = mock_client.submit.call_args.kwargs
    assert call_kwargs["capability"] == "video"
    assert call_kwargs["request"] == base_request
    assert "idempotency_key" in call_kwargs
    assert isinstance(call_kwargs["idempotency_key"], str)
    assert len(call_kwargs["idempotency_key"]) == 32  # SHA256[:32]


@patch("pixelle_snapshot.adapters.minimax_video.test_doubles.is_test_mode_enabled", return_value=False)
def test_provider_client_fetch_missing_output_fails(mock_test_mode, base_request, temp_dirs):
    """Success only when fetched output path exists."""
    mock_client = Mock()
    mock_client.submit.return_value = ProviderSubmitResult(
        job_id="job_789",
        status=ProviderJobStatus.SUBMITTED,
    )
    mock_client.poll.return_value = ProviderPollResult(
        job_id="job_789",
        status=ProviderJobStatus.SUCCEEDED,
    )
    
    # Fetch returns path but file doesn't exist
    nonexistent_path = os.path.join(temp_dirs["output_dir"], "missing.mp4")
    mock_client.fetch.return_value = ProviderFetchResult(
        job_id="job_789",
        output_path=nonexistent_path,
    )
    
    adapter = MinimaxVideoAdapter(provider_client=mock_client)
    response = adapter.invoke(base_request)
    
    # Should fail because output doesn't exist
    assert response.success is False
    assert response.error is not None
    assert response.error.category == ErrorCategory.EXECUTION
    assert "output file not found" in response.error.message.lower()


# ─────────────────────────────────────────────
# Error Mapping Tests
# ─────────────────────────────────────────────
@patch("pixelle_snapshot.adapters.minimax_video.test_doubles.is_test_mode_enabled", return_value=False)
def test_provider_client_poll_failed_status(mock_test_mode, base_request):
    """Poll returns FAILED status - should map to ExecutionError."""
    mock_client = Mock()
    mock_client.submit.return_value = ProviderSubmitResult(
        job_id="job_fail",
        status=ProviderJobStatus.SUBMITTED,
    )
    mock_client.poll.return_value = ProviderPollResult(
        job_id="job_fail",
        status=ProviderJobStatus.FAILED,
        metadata={"error": "Provider processing failed"},
    )
    
    adapter = MinimaxVideoAdapter(provider_client=mock_client)
    response = adapter.invoke(base_request)
    
    assert response.success is False
    assert response.error is not None
    assert response.error.category == ErrorCategory.PROVIDER
    assert "failed" in response.error.message.lower()


@patch("pixelle_snapshot.adapters.minimax_video.test_doubles.is_test_mode_enabled", return_value=False)
def test_provider_client_poll_canceled_status(mock_test_mode, base_request):
    """Poll returns CANCELED status - should map to ExecutionError."""
    mock_client = Mock()
    mock_client.submit.return_value = ProviderSubmitResult(
        job_id="job_cancel",
        status=ProviderJobStatus.SUBMITTED,
    )
    mock_client.poll.return_value = ProviderPollResult(
        job_id="job_cancel",
        status=ProviderJobStatus.CANCELED,
        metadata={},
    )
    
    adapter = MinimaxVideoAdapter(provider_client=mock_client)
    response = adapter.invoke(base_request)
    
    assert response.success is False
    assert response.error is not None
    assert response.error.category == ErrorCategory.EXECUTION
    assert "canceled" in response.error.message.lower()


@patch("pixelle_snapshot.adapters.minimax_video.test_doubles.is_test_mode_enabled", return_value=False)
def test_provider_client_execution_error_mapping(mock_test_mode, base_request):
    """ExecutionError from provider client mapped correctly."""
    mock_client = Mock()
    mock_client.submit.side_effect = ExecutionError(
        message="Provider submit failed",
        provider_status=500,
    )
    
    adapter = MinimaxVideoAdapter(provider_client=mock_client)
    response = adapter.invoke(base_request)
    
    assert response.success is False
    assert response.error is not None
    assert response.error.category == ErrorCategory.EXECUTION
    assert "failed" in response.error.message.lower()


# ─────────────────────────────────────────────
# Vendor Template Registration Tests
# ─────────────────────────────────────────────
def test_vendor_template_registration_import():
    """Template vendor module can be imported."""
    from src.integrations.template_vendor import (
        TemplateVendorClient,
        TemplateVendorConfig,
        TemplateVendorTransport,
        MockTemplateVendorTransport,
    )
    
    assert TemplateVendorClient is not None
    assert TemplateVendorConfig is not None
    assert TemplateVendorTransport is not None
    assert MockTemplateVendorTransport is not None


def test_vendor_template_registration_config_from_env():
    """Template vendor config loads from environment."""
    from src.integrations.template_vendor import TemplateVendorConfig
    
    config = TemplateVendorConfig.from_env()
    
    assert config is not None
    assert isinstance(config.api_key, str)
    assert isinstance(config.base_url, str)
    assert isinstance(config.timeout, float)


def test_vendor_template_registration_client_instantiation():
    """Template vendor client can be instantiated."""
    from src.integrations.template_vendor import TemplateVendorClient, TemplateVendorConfig
    
    config = TemplateVendorConfig()
    client = TemplateVendorClient(config=config)
    
    assert client is not None
    assert client.config == config
    assert client.is_available() is False


def test_vendor_template_registration_mock_transport():
    """Template vendor mock transport works correctly."""
    from src.integrations.template_vendor import MockTemplateVendorTransport
    
    transport = MockTemplateVendorTransport()
    response = transport.post("/test/endpoint", {"key": "value"})
    
    assert response.status_code == 200
    assert response.data.get("mock") is True
    assert response.data.get("endpoint") == "/test/endpoint"
    assert response.data.get("call_count") == 1


def test_vendor_template_registration_minimax_in_registry():
    """Minimax video adapter registered in capability list."""
    from pixelle_snapshot.adapters import list_capabilities
    
    capabilities = list_capabilities()
    
    assert "minimax_video" in capabilities
    assert "digital_human" in capabilities
    assert "i2v" in capabilities
    assert "action_transfer" in capabilities


def test_vendor_template_registration_minimax_available():
    """Minimax video adapter available with test vendor preference."""
    from pixelle_snapshot.adapters import is_capability_available
    
    available = is_capability_available("minimax_video", vendor_preference="test")
    
    assert available is True


def test_vendor_template_registration_adapter_via_get_adapter():
    """Minimax video adapter retrieved via registry."""
    from pixelle_snapshot.adapters import get_adapter
    
    adapter = get_adapter("minimax_video")
    
    assert adapter is not None
    assert adapter.capability_name == "minimax_video"


def test_vendor_template_registration_contracts_exported():
    """Minimax video contracts exported from adapters module."""
    from pixelle_snapshot.adapters import MinimaxVideoRequest, MinimaxVideoResponse
    
    assert MinimaxVideoRequest is not None
    assert MinimaxVideoResponse is not None


# ─────────────────────────────────────────────
# Coverage Target Tests (71% → 80%)
# ─────────────────────────────────────────────
@patch("pixelle_snapshot.adapters.minimax_video.test_doubles.is_test_mode_enabled", return_value=False)
def test_lazy_provider_client_instantiation(mock_test_mode, base_request):
    """Adapter instantiates provider client lazily when not injected."""
    adapter = MinimaxVideoAdapter(provider_client=None)
    
    # Client should be None initially
    assert adapter._provider_client is None
    
    # Mock the real client import to avoid dependency
    mock_client = Mock()
    mock_client.submit.return_value = ProviderSubmitResult(
        job_id="job_lazy",
        status=ProviderJobStatus.SUBMITTED,
    )
    mock_client.poll.return_value = ProviderPollResult(
        job_id="job_lazy",
        status=ProviderJobStatus.FAILED,
        metadata={"error": "test error"},
    )
    
    with patch("pixelle_snapshot.adapters.minimax_video.MinimaxVideoAdapter._get_provider_client", return_value=mock_client):
        response = adapter.invoke(base_request)
    
    # Should have attempted provider execution
    assert response.success is False


@patch("pixelle_snapshot.adapters.minimax_video.test_doubles.is_test_mode_enabled", return_value=False)
def test_provider_config_error_fallback_to_mvp_placeholder(mock_test_mode, base_request, temp_dirs):
    """ProviderConfigError triggers fallback to MVP placeholder mode."""
    from pixelle_snapshot.config_loader import ProviderConfigError
    
    # Trigger ProviderConfigError by making _get_provider_client fail
    adapter = MinimaxVideoAdapter(provider_client=None)
    
    def mock_get_client():
        raise ProviderConfigError("Missing API key")
    
    # Mock ffmpeg availability check
    with patch.object(adapter, "_get_provider_client", side_effect=mock_get_client):
        with patch("shutil.which", return_value=None):
            response = adapter.invoke(base_request)
    
    # Should succeed with MVP placeholder
    assert response.success is True
    assert response.task_id == "mvp-placeholder"
    assert response.metadata["mvp_placeholder"] is True
    assert response.output_path is not None
    assert os.path.exists(response.output_path)


@patch("pixelle_snapshot.adapters.minimax_video.test_doubles.is_test_mode_enabled", return_value=False)
def test_mvp_placeholder_with_ffmpeg_available(mock_test_mode, base_request, temp_dirs):
    """MVP placeholder uses ffmpeg if available."""
    from pixelle_snapshot.config_loader import ProviderConfigError
    
    adapter = MinimaxVideoAdapter(provider_client=None)
    
    def mock_get_client():
        raise ProviderConfigError("Missing API key")
    
    # Mock ffmpeg being available but failing
    with patch.object(adapter, "_get_provider_client", side_effect=mock_get_client):
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("subprocess.run", side_effect=Exception("ffmpeg failed")):
                response = adapter.invoke(base_request)
    
    # Should still succeed with file placeholder even if ffmpeg fails
    assert response.success is True
    assert response.task_id == "mvp-placeholder"
    assert os.path.exists(response.output_path)


@patch("pixelle_snapshot.adapters.minimax_video.test_doubles.is_test_mode_enabled", return_value=False)
def test_adapter_error_is_re_raised(mock_test_mode, base_request):
    """AdapterError subclasses are re-raised from _execute_via_provider."""
    mock_client = Mock()
    mock_client.submit.side_effect = ValidationError(
        message="Invalid request parameters",
        details={"field": "segment_text"},
    )
    
    adapter = MinimaxVideoAdapter(provider_client=mock_client)
    response = adapter.invoke(base_request)
    
    # ValidationError should be caught by base adapter and returned as failed response
    assert response.success is False
    assert response.error is not None
    assert response.error.category == ErrorCategory.VALIDATION


@patch("pixelle_snapshot.adapters.minimax_video.test_doubles.is_test_mode_enabled", return_value=False)
def test_unexpected_exception_wrapped_as_execution_error(mock_test_mode, base_request):
    """Unexpected exceptions wrapped as ExecutionError in _execute_via_provider."""
    mock_client = Mock()
    mock_client.submit.side_effect = RuntimeError("Unexpected failure")
    
    adapter = MinimaxVideoAdapter(provider_client=mock_client)
    response = adapter.invoke(base_request)
    
    # RuntimeError should be wrapped as ExecutionError and returned as failed response
    assert response.success is False
    assert response.error is not None
    # Base adapter normalizes all exceptions to ExecutionError
    assert response.error.category == ErrorCategory.EXECUTION


@patch("pixelle_snapshot.adapters.minimax_video.test_doubles.is_test_mode_enabled", return_value=False)
def test_is_available_with_valid_config(mock_test_mode):
    """is_available returns True when API key configured."""
    mock_config = Mock()
    mock_config.api_key = "valid-api-key"
    
    with patch("pixelle_snapshot.adapters.minimax_video.test_doubles.is_test_mode_enabled", return_value=False):
        with patch("src.integrations.minimax.client.MinimaxConfig.from_env", return_value=mock_config):
            adapter = MinimaxVideoAdapter()
            assert adapter.is_available() is True


@patch("pixelle_snapshot.adapters.minimax_video.test_doubles.is_test_mode_enabled", return_value=False)
def test_is_available_without_config(mock_test_mode):
    """is_available returns False when config fails."""
    with patch("src.integrations.minimax.client.MinimaxConfig.from_env", side_effect=Exception("No config")):
        adapter = MinimaxVideoAdapter()
        assert adapter.is_available() is False


@patch("pixelle_snapshot.adapters.minimax_video.test_doubles.is_test_mode_enabled", return_value=True)
def test_is_available_in_test_mode(mock_test_mode):
    """is_available returns True in test mode."""
    adapter = MinimaxVideoAdapter()
    assert adapter.is_available() is True
