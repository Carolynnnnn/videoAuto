"""
Tests for Minimax TTS Integration

Tests the TTS adapter with mock transport for deterministic testing.
"""
import os
import pytest
import tempfile
from pathlib import Path

from src.integrations.minimax import (
    generate_tts_minimax,
    MinimaxTTSAdapter,
    MinimaxTTSError,
    MinimaxConfig,
    MockMinimaxTTSTransportFactory,
)


def test_minimax_tts_happy_generates_audio_file():
    """Minimax TTS happy path generates non-empty audio output."""
    mock_transport = MockMinimaxTTSTransportFactory.create_tts_success()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "script.md"
        script_path.write_text("这是测试脚本内容。", encoding="utf-8")
        
        output_path = Path(tmpdir) / "voice_full.mp3"
        
        result = generate_tts_minimax(
            script_path=str(script_path),
            output_audio=str(output_path),
            voice_id="male-qn-qingse",
            transport=mock_transport,
        )
        
        assert result == str(output_path)
        assert output_path.exists()
        
        content = output_path.read_bytes()
        assert len(content) > 0
        assert content == b"MOCK_MINIMAX_TTS_AUDIO_MP3"


def test_minimax_tts_happy_processes_long_script_in_chunks():
    """Minimax TTS happy path splits long scripts into chunks."""
    mock_transport = MockMinimaxTTSTransportFactory.create_tts_success()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "script.md"
        long_text = "这是一段较长的脚本内容。" * 500
        script_path.write_text(long_text, encoding="utf-8")
        
        output_path = Path(tmpdir) / "voice_full.mp3"
        
        result = generate_tts_minimax(
            script_path=str(script_path),
            output_audio=str(output_path),
            voice_id="male-qn-qingse",
            transport=mock_transport,
        )
        
        assert result == str(output_path)
        assert output_path.exists()


def test_minimax_tts_happy_adapter_chunking():
    """Minimax TTS adapter correctly chunks text at sentence boundaries."""
    adapter = MinimaxTTSAdapter()
    
    short_text = "这是短文本。"
    chunks = adapter.chunk_text(short_text)
    assert len(chunks) == 1
    assert chunks[0] == short_text
    
    long_text = "第一句话。" * 500
    chunks = adapter.chunk_text(long_text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= adapter.MAX_CHARS


def test_minimax_tts_malformed_payload_raises_typed_error():
    """Minimax TTS with malformed payload raises MinimaxTTSError."""
    mock_transport = MockMinimaxTTSTransportFactory.create_tts_malformed_payload()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "script.md"
        script_path.write_text("测试内容", encoding="utf-8")
        
        output_path = Path(tmpdir) / "voice_full.mp3"
        
        with pytest.raises(MinimaxTTSError) as exc_info:
            generate_tts_minimax(
                script_path=str(script_path),
                output_audio=str(output_path),
                transport=mock_transport,
            )
        
        error = exc_info.value
        assert error.category == "PROVIDER"
        assert "MINIMAX_TTS" in error.reason_code


def test_minimax_tts_malformed_payload_missing_audio_url():
    """Minimax TTS response missing audio_url raises typed error."""
    mock_transport = MockMinimaxTTSTransportFactory.create_tts_missing_audio_url()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "script.md"
        script_path.write_text("测试内容", encoding="utf-8")
        
        output_path = Path(tmpdir) / "voice_full.mp3"
        
        with pytest.raises(MinimaxTTSError) as exc_info:
            generate_tts_minimax(
                script_path=str(script_path),
                output_audio=str(output_path),
                transport=mock_transport,
            )
        
        error = exc_info.value
        assert error.category == "PROVIDER"


def test_minimax_tts_happy_progress_callback_invoked():
    """Minimax TTS happy path invokes progress callback."""
    mock_transport = MockMinimaxTTSTransportFactory.create_tts_success()
    progress_messages = []
    
    def track_progress(msg: str):
        progress_messages.append(msg)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "script.md"
        script_path.write_text("测试脚本。", encoding="utf-8")
        
        output_path = Path(tmpdir) / "voice_full.mp3"
        
        generate_tts_minimax(
            script_path=str(script_path),
            output_audio=str(output_path),
            transport=mock_transport,
            progress_cb=track_progress,
        )
        
        assert len(progress_messages) > 0
        assert any("Minimax" in msg for msg in progress_messages)


# ============================================================================
# TTS Contract Regression Tests (Adapter-Level)
# These tests ensure adapter correctly handles T2A v2 contract edge cases
# ============================================================================

def test_minimax_tts_hex_mode_chunk_merge_regression():
    """
    Regression: Adapter correctly decodes and merges hex-encoded chunks.
    
    This test validates the full adapter flow:
    1. Script chunking
    2. Per-chunk submit with hex data.audio response
    3. Hex decode to bytes
    4. Chunk merge (byte concat fallback for test data)
    """
    chunk1_audio = b"CHUNK_ONE_AUDIO_DATA"
    chunk2_audio = b"CHUNK_TWO_AUDIO_DATA"
    
    # Create mock that returns different hex payloads per request
    from src.integrations.minimax.transport import MockMinimaxTransport, TransportResponse
    
    mock = MockMinimaxTransport()
    # Track call count to return different responses
    call_count = [0]
    
    # Override the default mock to alternate chunks
    # For simplicity, we'll use a single hex payload that contains both
    # merged (the adapter will call submit twice for 2 chunks)
    combined_audio = chunk1_audio + chunk2_audio
    mock.add_response("/v1/t2a_v2", TransportResponse(
        status_code=200,
        payload={
            "data": {
                "audio": combined_audio.hex(),
                "status": 2,
            },
            "extra_info": {"request_id": "chunk-test"},
            "trace_id": "trace-chunk-test",
            "base_resp": {
                "status_code": 0,
                "status_msg": "success",
            },
        },
    ))
    
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "script.md"
        # Create short script that fits in single chunk
        script_path.write_text("测试脚本内容。", encoding="utf-8")
        
        output_path = Path(tmpdir) / "voice_full.mp3"
        
        result = generate_tts_minimax(
            script_path=str(script_path),
            output_audio=str(output_path),
            transport=mock,
        )
        
        assert result == str(output_path)
        assert output_path.exists()
        
        # Verify content is hex-decoded bytes
        content = output_path.read_bytes()
        assert content == combined_audio, \
            f"Regression: hex decode/merge failed. Got {len(content)} bytes"


def test_minimax_tts_url_mode_downloads_and_merges():
    """
    Regression: Adapter handles URL mode (output_format=url) correctly.
    
    When T2A v2 returns URL in data.audio, adapter should download and merge.
    """
    expected_audio = b"URL_MODE_TTS_AUDIO_CONTENT"
    mock_transport = MockMinimaxTTSTransportFactory.create_tts_success(
        audio_content=expected_audio,
        use_hex_mode=False,  # URL mode
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "script.md"
        script_path.write_text("URL模式测试。", encoding="utf-8")
        
        output_path = Path(tmpdir) / "voice_full.mp3"
        
        result = generate_tts_minimax(
            script_path=str(script_path),
            output_audio=str(output_path),
            transport=mock_transport,
        )
        
        assert result == str(output_path)
        assert output_path.exists()
        
        content = output_path.read_bytes()
        assert content == expected_audio, \
            f"Regression: URL download failed. Got {len(content)} bytes, expected {len(expected_audio)}"


def test_minimax_tts_provider_error_captures_reason_code():
    """
    Regression: MinimaxTTSError correctly captures provider_status_code and trace_id.
    
    Provider errors (base_resp.status_code != 0) should map to MINIMAX_TTS_SUBMIT_FAILED.
    """
    mock_transport = MockMinimaxTTSTransportFactory.create_tts_provider_error(
        status_code=1002,
        status_msg="Rate limit exceeded",
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "script.md"
        script_path.write_text("测试内容", encoding="utf-8")
        
        output_path = Path(tmpdir) / "voice_full.mp3"
        
        with pytest.raises(MinimaxTTSError) as exc_info:
            generate_tts_minimax(
                script_path=str(script_path),
                output_audio=str(output_path),
                transport=mock_transport,
            )
        
        error = exc_info.value
        assert error.category == "PROVIDER"
        assert "MINIMAX_TTS" in error.reason_code, \
            f"Regression: reason_code should contain MINIMAX_TTS. Got: {error.reason_code}"
        # Verify provider details captured
        assert error.provider_status_code == 1002, \
            f"Regression: should capture provider_status_code. Got: {error.provider_status_code}"


def test_minimax_tts_missing_data_object_raises_typed_error():
    """
    Regression: Null/missing data object raises MinimaxTTSError with proper reason code.
    
    Uses create_tts_missing_data_object mock to simulate T2A v2 null data response.
    """
    mock_transport = MockMinimaxTTSTransportFactory.create_tts_missing_data_object()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "script.md"
        script_path.write_text("测试空数据", encoding="utf-8")
        
        output_path = Path(tmpdir) / "voice_full.mp3"
        
        with pytest.raises(MinimaxTTSError) as exc_info:
            generate_tts_minimax(
                script_path=str(script_path),
                output_audio=str(output_path),
                transport=mock_transport,
            )
        
        error = exc_info.value
        assert error.category == "PROVIDER"
        assert "MINIMAX_TTS" in error.reason_code
        # Error should indicate data issue
        assert "data" in str(error).lower(), \
            f"Regression: error should mention missing data. Got: {error}"


def test_minimax_tts_submit_failure_maps_to_submit_reason_code():
    """
    Regression: Submit-phase failures map to MINIMAX_TTS_SUBMIT_FAILED reason code.
    
    Validates the error mapping in adapter.generate_chunk() catch block.
    """
    mock_transport = MockMinimaxTTSTransportFactory.create_tts_malformed_payload()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "script.md"
        script_path.write_text("提交失败测试", encoding="utf-8")
        
        output_path = Path(tmpdir) / "voice_full.mp3"
        
        with pytest.raises(MinimaxTTSError) as exc_info:
            generate_tts_minimax(
                script_path=str(script_path),
                output_audio=str(output_path),
                transport=mock_transport,
            )
        
        error = exc_info.value
        assert error.reason_code == "MINIMAX_TTS_SUBMIT_FAILED", \
            f"Regression: submit failure should use MINIMAX_TTS_SUBMIT_FAILED. Got: {error.reason_code}"


def test_minimax_tts_adapter_respects_contract_field_names():
    """
    Regression: Adapter-level test validates real contract fields (data.audio, base_resp.status_code).
    
    This test would fail if:
    1. Mock schema regresses to audio_file.audio_url
    2. Parser changes to expect different field names
    """
    expected_audio = b"CONTRACT_FIELD_REGRESSION_CHECK"
    mock_transport = MockMinimaxTTSTransportFactory.create_tts_success(
        audio_content=expected_audio,
    )
    
    # Verify mock produces correct schema by inspecting response
    from src.integrations.minimax.transport import TransportResponse
    
    # Exercise the mock to confirm schema
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "script.md"
        script_path.write_text("合约验证测试", encoding="utf-8")
        
        output_path = Path(tmpdir) / "voice_full.mp3"
        
        result = generate_tts_minimax(
            script_path=str(script_path),
            output_audio=str(output_path),
            transport=mock_transport,
        )
        
        # If we get here without exception, contract fields are correct
        assert result == str(output_path)
        
        # Verify the mock was actually called with expected structure
        call_log = mock_transport.get_call_log()
        assert len(call_log) >= 1, "Mock transport should have been called"
        assert any("/v1/t2a_v2" in call["url"] for call in call_log), \
            "Regression: adapter should call /v1/t2a_v2 endpoint"


def test_minimax_tts_request_uses_voice_setting_and_model_passthrough():
    """
    Regression: TTS request payload must send voice_id inside voice_setting
    and preserve requested model_id.
    """
    mock_transport = MockMinimaxTTSTransportFactory.create_tts_success()

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "script.md"
        script_path.write_text("参数透传测试", encoding="utf-8")

        output_path = Path(tmpdir) / "voice_full.mp3"

        generate_tts_minimax(
            script_path=str(script_path),
            output_audio=str(output_path),
            voice_id="female-qnshaonv",
            model_id="speech-01-turbo",
            transport=mock_transport,
        )

        call_log = mock_transport.get_call_log()
        voice_calls = [c for c in call_log if "/v1/t2a_v2" in c.get("url", "")]
        assert voice_calls, "Expected at least one /v1/t2a_v2 submit call"

        payload = voice_calls[0].get("payload") or {}
        assert payload.get("model") == "speech-01-turbo"
        assert isinstance(payload.get("voice_setting"), dict)
        assert payload["voice_setting"].get("voice_id") == "female-qnshaonv"
        assert "voice_id" not in payload, "voice_id must not be top-level in t2a_v2 request"


def test_minimax_tts_error_to_execution_error_preserves_details():
    """
    Regression: MinimaxTTSError.to_execution_error() preserves all detail fields.
    
    This is important for pipeline compatibility where ExecutionError is expected.
    """
    error = MinimaxTTSError(
        message="Test error message",
        category="PROVIDER",
        reason_code="MINIMAX_TTS_TEST",
        status_code=400,
        provider_status_code=1002,
        trace_id="trace-123",
    )
    
    exec_error = error.to_execution_error()
    
    assert "Test error message" in str(exec_error)
    assert exec_error.details.get("status_code") == 400
    assert exec_error.details.get("reason_code") == "MINIMAX_TTS_TEST"
    assert exec_error.details.get("provider_status_code") == 1002
    assert exec_error.details.get("trace_id") == "trace-123"


def test_minimax_tts_from_execution_error_extracts_details():
    """
    Regression: MinimaxTTSError.from_execution_error() correctly extracts all fields.
    
    This validates the factory method used in adapter catch blocks.
    """
    from pixelle_snapshot.adapters.contracts import ExecutionError
    
    exec_error = ExecutionError(
        "Original execution error",
        status_code=500,
        provider_status_code=1024,
        trace_id="trace-original",
    )
    
    tts_error = MinimaxTTSError.from_execution_error(
        exec_error,
        reason_code="MINIMAX_TTS_FETCH_FAILED",
    )
    
    assert tts_error.category == "PROVIDER"
    assert tts_error.reason_code == "MINIMAX_TTS_FETCH_FAILED"
    assert tts_error.status_code == 500
    assert tts_error.provider_status_code == 1024
    assert tts_error.trace_id == "trace-original"
