"""
Tests for Minimax Unified Client

Validates submit/poll/fetch/cancel lifecycle using mock transport.
Tests both happy path and auth failure scenarios.

Test function names for selectors:
- Functions with 'happy' or 'success' in name: Happy path success scenarios
- Functions with 'auth_failure' in name: Authentication failure handling
"""
import os
import pytest
import tempfile
from dataclasses import dataclass

from src.integrations.minimax import (
    MinimaxUnifiedClient,
    MinimaxConfig,
)
from src.integrations.minimax.transport import (
    MockMinimaxTransportFactory,
)
from pixelle_snapshot.adapters.contracts import (
    AdapterRequest,
    ProviderJobStatus,
    ExecutionError,
)


# Minimal AdapterRequest for testing (not a test class)
@dataclass
class AdapterRequestFixture(AdapterRequest):
    """Fixture adapter request with minimal fields for testing."""
    segment_key: str = "test-segment"
    segment_text: str = "Test narration"
    segment_duration: float = 5.0
    project_root: str = "/tmp/test"
    output_dir: str = "/tmp/test/output"
    timeout_seconds: float = 30.0


def test_minimax_client_happy_voice_submit_success():
    """Test voice endpoint submit returns SUCCEEDED immediately."""
    mock_transport = MockMinimaxTransportFactory.create_voice_success()
    config = MinimaxConfig(api_key="test-key")
    client = MinimaxUnifiedClient(config=config, transport=mock_transport)
    
    request = AdapterRequestFixture()
    result = client.submit("voice", request)
    
    assert result.job_id.startswith("voice-")
    assert result.status == ProviderJobStatus.SUCCEEDED
    assert "audio_url" in result.metadata


def test_minimax_client_happy_voice_poll_returns_completed():
    """Test voice endpoint poll returns SUCCEEDED without API call."""
    mock_transport = MockMinimaxTransportFactory.create_voice_success()
    config = MinimaxConfig(api_key="test-key")
    client = MinimaxUnifiedClient(config=config, transport=mock_transport)
    
    request = AdapterRequestFixture()
    submit_result = client.submit("voice", request)
    poll_result = client.poll(submit_result.job_id)
    
    assert poll_result.status == ProviderJobStatus.SUCCEEDED


def test_minimax_client_happy_voice_fetch_downloads_audio():
    """Test voice endpoint fetch downloads audio file to output_dir."""
    mock_transport = MockMinimaxTransportFactory.create_voice_success()
    config = MinimaxConfig(api_key="test-key")
    client = MinimaxUnifiedClient(config=config, transport=mock_transport)
    
    request = AdapterRequestFixture()
    submit_result = client.submit("voice", request)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        fetch_result = client.fetch(submit_result.job_id, tmpdir)
        
        assert fetch_result.output_path.endswith(".mp3")
        assert os.path.exists(fetch_result.output_path)
        
        # Verify content
        with open(fetch_result.output_path, "rb") as f:
            content = f.read()
        assert content == b"MOCK_AUDIO_CONTENT_MP3"


def test_minimax_client_happy_video_async_lifecycle():
    """Test video endpoint full async lifecycle: submit -> poll -> fetch."""
    mock_transport = MockMinimaxTransportFactory.create_video_async_success()
    config = MinimaxConfig(api_key="test-key")
    client = MinimaxUnifiedClient(config=config, transport=mock_transport)
    
    request = AdapterRequestFixture()
    
    # Submit
    submit_result = client.submit("video", request)
    assert submit_result.job_id.startswith("video-")
    assert submit_result.status == ProviderJobStatus.SUBMITTED
    
    # Poll
    poll_result = client.poll(submit_result.job_id)
    assert poll_result.status == ProviderJobStatus.SUCCEEDED
    
    # Fetch
    with tempfile.TemporaryDirectory() as tmpdir:
        fetch_result = client.fetch(submit_result.job_id, tmpdir)
        
        assert fetch_result.output_path.endswith(".mp4")
        assert os.path.exists(fetch_result.output_path)
        
        # Verify content
        with open(fetch_result.output_path, "rb") as f:
            content = f.read()
        assert content == b"MOCK_VIDEO_CONTENT_MP4"


def test_minimax_client_auth_failure_voice_submit():
    """Test voice endpoint submit raises ExecutionError on auth failure."""
    mock_transport = MockMinimaxTransportFactory.create_auth_failure()
    config = MinimaxConfig(api_key="invalid-key")
    client = MinimaxUnifiedClient(config=config, transport=mock_transport)
    
    request = AdapterRequestFixture()
    
    with pytest.raises(ExecutionError) as exc_info:
        client.submit("voice", request)
    
    assert "Invalid API Key" in str(exc_info.value)


def test_minimax_client_auth_failure_video_submit():
    """Test video endpoint submit raises ExecutionError on auth failure."""
    mock_transport = MockMinimaxTransportFactory.create_auth_failure()
    config = MinimaxConfig(api_key="invalid-key")
    client = MinimaxUnifiedClient(config=config, transport=mock_transport)
    
    request = AdapterRequestFixture()
    
    with pytest.raises(ExecutionError) as exc_info:
        client.submit("video", request)
    
    assert "Invalid API Key" in str(exc_info.value)


def test_minimax_client_happy_cancel_returns_no_op():
    """Test cancel operation returns no-op result (Minimax doesn't support cancellation)."""
    mock_transport = MockMinimaxTransportFactory.create_voice_success()
    config = MinimaxConfig(api_key="test-key")
    client = MinimaxUnifiedClient(config=config, transport=mock_transport)
    
    request = AdapterRequestFixture()
    submit_result = client.submit("voice", request)
    cancel_result = client.cancel(submit_result.job_id)
    
    assert cancel_result.canceled is False
    assert "does not support cancellation" in cancel_result.metadata.get("reason", "")


def test_minimax_client_happy_fetch_idempotent():
    """Test fetch can be called multiple times and returns cached result."""
    mock_transport = MockMinimaxTransportFactory.create_voice_success()
    config = MinimaxConfig(api_key="test-key")
    client = MinimaxUnifiedClient(config=config, transport=mock_transport)
    
    request = AdapterRequestFixture()
    submit_result = client.submit("voice", request)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # First fetch
        fetch_result_1 = client.fetch(submit_result.job_id, tmpdir)
        
        # Second fetch (should return cached path)
        fetch_result_2 = client.fetch(submit_result.job_id, tmpdir)
        
        assert fetch_result_1.output_path == fetch_result_2.output_path


# ============================================================================
# TTS Contract Regression Tests (T2A v2 data.audio)
# These tests ensure parser doesn't regress to legacy audio_file.audio_url
# ============================================================================

def test_minimax_client_voice_hex_mode_decodes_audio_bytes():
    """
    Regression: Voice fetch decodes hex-encoded data.audio to raw bytes.
    
    This test fails if parser regresses to expecting URL in audio_file.audio_url.
    The T2A v2 default output_format=hex returns hex-encoded audio data.
    """
    expected_audio = b"HEX_MODE_AUDIO_REGRESSION_TEST"
    mock_transport = MockMinimaxTransportFactory.create_voice_success(
        audio_content=expected_audio,
        use_hex_mode=True,
    )
    config = MinimaxConfig(api_key="test-key")
    client = MinimaxUnifiedClient(config=config, transport=mock_transport)
    
    request = AdapterRequestFixture()
    submit_result = client.submit("voice", request)
    
    # Verify submit succeeded and metadata contains data.audio (stored as audio_url)
    assert submit_result.status == ProviderJobStatus.SUCCEEDED
    assert "audio_url" in submit_result.metadata
    # Critical: audio_url must be hex string, not URL
    audio_payload = submit_result.metadata["audio_url"]
    assert not audio_payload.startswith("http"), \
        "Regression: hex mode should NOT return URL in audio_url metadata"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        fetch_result = client.fetch(submit_result.job_id, tmpdir)
        
        assert os.path.exists(fetch_result.output_path)
        with open(fetch_result.output_path, "rb") as f:
            content = f.read()
        
        # Core regression check: content must match original bytes after hex decode
        assert content == expected_audio, \
            f"Regression: hex decode failed. Got {len(content)} bytes, expected {len(expected_audio)}"


def test_minimax_client_voice_url_mode_downloads_from_url():
    """
    Regression: Voice fetch downloads audio when data.audio is URL.
    
    When output_format=url, T2A v2 returns a download URL in data.audio.
    This test ensures dual-mode fetch correctly detects and handles URLs.
    """
    expected_audio = b"URL_MODE_AUDIO_REGRESSION_TEST"
    mock_transport = MockMinimaxTransportFactory.create_voice_success(
        audio_content=expected_audio,
        use_hex_mode=False,  # URL mode
    )
    config = MinimaxConfig(api_key="test-key")
    client = MinimaxUnifiedClient(config=config, transport=mock_transport)
    
    request = AdapterRequestFixture()
    submit_result = client.submit("voice", request)
    
    # Verify submit succeeded and metadata contains URL
    assert submit_result.status == ProviderJobStatus.SUCCEEDED
    assert "audio_url" in submit_result.metadata
    audio_payload = submit_result.metadata["audio_url"]
    assert audio_payload.startswith("http"), \
        "URL mode should return HTTP(S) URL in audio_url metadata"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        fetch_result = client.fetch(submit_result.job_id, tmpdir)
        
        assert os.path.exists(fetch_result.output_path)
        with open(fetch_result.output_path, "rb") as f:
            content = f.read()
        
        # Core regression check: content must match downloaded bytes
        assert content == expected_audio, \
            f"Regression: URL download failed. Got {len(content)} bytes, expected {len(expected_audio)}"


def test_minimax_client_voice_null_data_raises_execution_error():
    """
    Regression: Parser rejects response with null/missing data object.
    
    Official T2A v2 docs: "The returned data object may be null, so a null check is required."
    This test ensures parser doesn't assume data always exists.
    """
    from src.integrations.minimax.transport import MockMinimaxTransport, TransportResponse
    
    mock_transport = MockMinimaxTransport()
    mock_transport.add_response("/v1/t2a_v2", TransportResponse(
        status_code=200,
        payload={
            "extra_info": {"request_id": "test-null-data"},
            "trace_id": "trace-null-data",
            "base_resp": {
                "status_code": 0,
                "status_msg": "success",
            },
            # data field intentionally missing
        },
    ))
    
    config = MinimaxConfig(api_key="test-key")
    client = MinimaxUnifiedClient(config=config, transport=mock_transport)
    request = AdapterRequestFixture()
    
    with pytest.raises(ExecutionError) as exc_info:
        client.submit("voice", request)
    
    # Verify error message indicates missing data
    error_msg = str(exc_info.value)
    assert "data" in error_msg.lower(), \
        f"Regression: error should mention missing 'data' object. Got: {error_msg}"


def test_minimax_client_voice_empty_audio_raises_execution_error():
    """
    Regression: Parser rejects response where data.audio is empty.
    
    Even if data object exists, audio field must be non-empty string.
    """
    from src.integrations.minimax.transport import MockMinimaxTransport, TransportResponse
    
    mock_transport = MockMinimaxTransport()
    mock_transport.add_response("/v1/t2a_v2", TransportResponse(
        status_code=200,
        payload={
            "data": {
                "audio": "",  # Empty audio
                "status": 2,
            },
            "extra_info": {"request_id": "test-empty-audio"},
            "trace_id": "trace-empty-audio",
            "base_resp": {
                "status_code": 0,
                "status_msg": "success",
            },
        },
    ))
    
    config = MinimaxConfig(api_key="test-key")
    client = MinimaxUnifiedClient(config=config, transport=mock_transport)
    request = AdapterRequestFixture()
    
    with pytest.raises(ExecutionError) as exc_info:
        client.submit("voice", request)
    
    error_msg = str(exc_info.value)
    assert "audio" in error_msg.lower(), \
        f"Regression: error should mention missing 'audio' field. Got: {error_msg}"


def test_minimax_client_voice_provider_error_maps_status_code():
    """
    Regression: Provider-level errors (base_resp.status_code != 0) raise ExecutionError.
    
    T2A v2 returns HTTP 200 even on provider errors, distinguished by base_resp.status_code.
    This test ensures we don't assume HTTP 200 = success.
    """
    from src.integrations.minimax.transport import MockMinimaxTransport, TransportResponse
    
    mock_transport = MockMinimaxTransport()
    mock_transport.add_response("/v1/t2a_v2", TransportResponse(
        status_code=200,  # HTTP success
        payload={
            "data": None,
            "trace_id": "trace-provider-error",
            "base_resp": {
                "status_code": 1002,  # Rate limit error code
                "status_msg": "Rate limit exceeded",
            },
        },
    ))
    
    config = MinimaxConfig(api_key="test-key")
    client = MinimaxUnifiedClient(config=config, transport=mock_transport)
    request = AdapterRequestFixture()
    
    with pytest.raises(ExecutionError) as exc_info:
        client.submit("voice", request)
    
    error = exc_info.value
    # Verify error captures provider status code
    assert "Rate limit" in str(error), \
        f"Regression: error should contain provider message. Got: {error}"
    # Verify details dict contains provider_status_code
    assert error.details.get("provider_status_code") == 1002, \
        f"Regression: ExecutionError should capture provider_status_code. Got details: {error.details}"


def test_minimax_client_voice_submit_captures_trace_metadata():
    """
    Regression: Successful submit captures trace_id and request_id in metadata.
    
    These fields are essential for debugging and tracing API calls.
    """
    mock_transport = MockMinimaxTransportFactory.create_voice_success()
    config = MinimaxConfig(api_key="test-key")
    client = MinimaxUnifiedClient(config=config, transport=mock_transport)
    
    request = AdapterRequestFixture()
    submit_result = client.submit("voice", request)
    
    # Verify traceability metadata
    assert "trace_id" in submit_result.metadata, \
        "Regression: submit should capture trace_id in metadata"
    assert "request_id" in submit_result.metadata, \
        "Regression: submit should capture request_id in metadata"
    assert "provider_status_code" in submit_result.metadata, \
        "Regression: submit should capture provider_status_code (0 on success)"
    assert submit_result.metadata["provider_status_code"] == 0


def test_minimax_client_voice_invalid_hex_raises_execution_error():
    """
    Regression: Fetch rejects payloads that are neither valid URL nor valid hex.
    
    Dual-mode fetch must detect malformed data.audio gracefully.
    """
    from src.integrations.minimax.transport import MockMinimaxTransport, TransportResponse
    
    # Create mock with invalid hex (odd length, invalid chars)
    mock_transport = MockMinimaxTransport()
    mock_transport.add_response("/v1/t2a_v2", TransportResponse(
        status_code=200,
        payload={
            "data": {
                "audio": "invalid_hex_XYZ!@#",  # Not valid hex, not URL
                "status": 2,
            },
            "extra_info": {"request_id": "test-invalid"},
            "trace_id": "trace-invalid",
            "base_resp": {
                "status_code": 0,
                "status_msg": "success",
            },
        },
    ))
    
    config = MinimaxConfig(api_key="test-key")
    client = MinimaxUnifiedClient(config=config, transport=mock_transport)
    request = AdapterRequestFixture()
    
    # Submit succeeds (parser only checks non-empty)
    submit_result = client.submit("voice", request)
    assert submit_result.status == ProviderJobStatus.SUCCEEDED
    
    # Fetch should fail with clear error
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ExecutionError) as exc_info:
            client.fetch(submit_result.job_id, tmpdir)
        
        error_msg = str(exc_info.value)
        assert "url" in error_msg.lower() or "hex" in error_msg.lower(), \
            f"Regression: error should mention URL/hex validation. Got: {error_msg}"
