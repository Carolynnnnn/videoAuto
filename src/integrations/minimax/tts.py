"""
Minimax TTS Adapter

Provides TTS (Text-to-Speech) interface for the PDF pipeline.
Wraps MinimaxUnifiedClient voice endpoint for compatibility with
existing step_pdf.py patterns (chunking, merge, output artifact).

Key Components:
- MinimaxTTSAdapter: High-level adapter for step_pdf.py integration
- MinimaxTTSError: Typed error for provider failures (maps to ExecutionError)
- generate_tts_minimax: Function matching elevenlabs signature

Usage:
    from src.integrations.minimax.tts import generate_tts_minimax
    
    output_path = generate_tts_minimax(
        script_path="input/script.md",
        output_audio="input/voice_full.mp3",
        voice_id="male-qn-qingse",
        progress_cb=lambda msg: print(msg),
    )
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from src.utils.logger import get_logger
from src.core.api_config import MINIMAX_VOICES
from src.integrations.minimax.client import (
    MinimaxUnifiedClient,
    MinimaxConfig,
)
from src.integrations.minimax.transport import (
    MinimaxTransport,
    MockMinimaxTransport,
    TransportResponse,
)
from pixelle_snapshot.adapters.contracts import (
    AdapterRequest,
    ExecutionError,
    ErrorCategory,
)

logger = get_logger("minimax_tts")


@dataclass
class MinimaxTTSError(Exception):
    """
    Typed error for Minimax TTS failures.
    
    Maps to ExecutionError/PROVIDER category for typed failure handling.
    Captures both HTTP status and provider-specific error codes for debugging.
    """
    message: str
    category: str = "PROVIDER"
    reason_code: str = "MINIMAX_TTS_ERROR"
    status_code: Optional[int] = None
    provider_status_code: Optional[int] = None
    trace_id: Optional[str] = None
    
    def __str__(self) -> str:
        return f"[{self.category}] {self.message}"
    
    def to_execution_error(self) -> ExecutionError:
        """Convert to ExecutionError for pipeline compatibility."""
        return ExecutionError(
            self.message,
            status_code=self.status_code,
            reason_code=self.reason_code,
            provider_status_code=self.provider_status_code,
            trace_id=self.trace_id,
        )
    
    @classmethod
    def from_execution_error(
        cls,
        e: ExecutionError,
        reason_code: str = "MINIMAX_TTS_ERROR",
    ) -> "MinimaxTTSError":
        """Create MinimaxTTSError from an ExecutionError with full detail extraction."""
        return cls(
            message=str(e),
            status_code=e.details.get("status_code"),
            provider_status_code=e.details.get("provider_status_code"),
            trace_id=e.details.get("trace_id"),
            reason_code=reason_code,
        )


@dataclass  
class MinimaxTTSRequest(AdapterRequest):
    """Request wrapper for TTS chunk processing."""
    segment_key: str = "tts-chunk"
    segment_text: str = ""
    segment_duration: float = 0.0
    project_root: str = "/tmp"
    output_dir: str = "/tmp/output"
    timeout_seconds: float = 300.0


class MinimaxTTSAdapter:
    """
    High-level TTS adapter using Minimax voice endpoint.
    
    Handles chunking, submit/fetch lifecycle, and merging.
    Compatible with step_pdf.py patterns.
    """
    
    MAX_CHARS = 2000  # Minimax recommended limit per request
    
    def __init__(
        self,
        config: Optional[MinimaxConfig] = None,
        transport: Optional[MinimaxTransport] = None,
    ):
        """
        Initialize adapter.
        
        Args:
            config: Minimax config (defaults to from_env)
            transport: Transport layer (defaults to production urllib)
        """
        self.config = config or MinimaxConfig.from_env()
        self.client = MinimaxUnifiedClient(config=self.config, transport=transport)
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks respecting MAX_CHARS limit.
        
        Splits on sentence boundaries (。！？\n) to preserve natural speech.
        """
        if len(text) <= self.MAX_CHARS:
            return [text]
        
        sentences = re.split(r"([。！？\n]+)", text)
        chunks = []
        current = ""
        
        for s in sentences:
            if len(current) + len(s) <= self.MAX_CHARS:
                current += s
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = s
        
        if current.strip():
            chunks.append(current.strip())
        
        return chunks
    
    def generate_chunk(
        self,
        text: str,
        output_path: str,
        voice_id: str = "male-qn-qingse",
        model_id: str = "speech-01-hd",
    ) -> str:
        """
        Generate TTS for a single chunk.
        
        Args:
            text: Text to synthesize
            output_path: Path for output MP3
            voice_id: Minimax voice ID
        
        Returns:
            Path to generated audio file
        
        Raises:
            MinimaxTTSError: On API failure
        """
        request = MinimaxTTSRequest(
            segment_key=f"tts-{hash(text) % 10000:04d}",
            segment_text=text,
            segment_duration=0.0,  # Not used for voice
            project_root=str(Path(output_path).parent.parent),
            output_dir=str(Path(output_path).parent),
            metadata={
                "voice_id": voice_id,
                "model_id": model_id,
            },
        )
        
        try:
            submit_result = self.client.submit("voice", request)
        except ExecutionError as e:
            raise MinimaxTTSError.from_execution_error(
                e, reason_code="MINIMAX_TTS_SUBMIT_FAILED"
            )
        
        # Voice endpoint completes immediately - fetch output
        try:
            output_dir = str(Path(output_path).parent)
            fetch_result = self.client.fetch(submit_result.job_id, output_dir)
        except ExecutionError as e:
            raise MinimaxTTSError.from_execution_error(
                e, reason_code="MINIMAX_TTS_FETCH_FAILED"
            )
        
        # Rename to desired output path if different
        if fetch_result.output_path != output_path:
            import shutil
            shutil.move(fetch_result.output_path, output_path)
        
        return output_path
    
    def merge_chunks(
        self,
        chunk_paths: List[str],
        output_path: str,
    ) -> str:
        """
        Merge multiple audio chunks using ffmpeg.
        
        Falls back to byte concatenation if ffmpeg fails (e.g., with mock data).
        
        Args:
            chunk_paths: List of MP3 file paths
            output_path: Final merged output path
        
        Returns:
            Path to merged audio file
        
        Raises:
            MinimaxTTSError: On merge failure
        """
        if len(chunk_paths) == 1:
            import shutil
            shutil.move(chunk_paths[0], output_path)
            return output_path
        
        # Try ffmpeg concat first (production path for real MP3s)
        list_path = str(Path(output_path).parent / "_minimax_tts_list.txt")
        try:
            with open(list_path, "w") as f:
                for p in chunk_paths:
                    f.write(f"file '{Path(p).resolve()}'\n")
            
            # Run ffmpeg concat
            cmd = (
                f'ffmpeg -y -f concat -safe 0 -i "{list_path}" '
                f'-c:a libmp3lame -b:a 192k "{output_path}" -loglevel error'
            )
            
            result = subprocess.run(cmd, shell=True, capture_output=True)
            
            if result.returncode == 0:
                # Success - cleanup and return
                for p in chunk_paths:
                    Path(p).unlink(missing_ok=True)
                Path(list_path).unlink(missing_ok=True)
                return output_path
            
            # ffmpeg failed - fall through to byte concatenation
            logger.warning(f"ffmpeg merge failed, using byte concatenation fallback")
        
        except Exception as e:
            logger.warning(f"ffmpeg merge error: {e}, using byte concatenation fallback")
        
        finally:
            # Cleanup list file
            Path(list_path).unlink(missing_ok=True)
        
        # Fallback: deterministic byte concatenation (for tests with mock data)
        try:
            merged_content = b""
            for p in chunk_paths:
                merged_content += Path(p).read_bytes()
                Path(p).unlink(missing_ok=True)
            
            Path(output_path).write_bytes(merged_content)
            return output_path
        
        except Exception as e:
            raise MinimaxTTSError(
                message=f"Chunk merge failed (ffmpeg and fallback): {e}",
                reason_code="MINIMAX_TTS_MERGE_FAILED",
            )
    
    def generate(
        self,
        script_path: str,
        output_audio: str,
        voice_id: str = "male-qn-qingse",
        model_id: str = "speech-01-hd",
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Full TTS pipeline: read script, chunk, generate, merge.
        
        Args:
            script_path: Path to script text file
            output_audio: Path for final output audio
            voice_id: Minimax voice ID
            progress_cb: Optional progress callback
        
        Returns:
            Path to generated audio file
        
        Raises:
            MinimaxTTSError: On any failure
        """
        logger.info(f"Minimax TTS: voice_id={voice_id}, model={model_id}")
        if progress_cb:
            progress_cb("正在使用 Minimax 生成高质量语音...")
        
        # Read and clean script
        script_text = Path(script_path).read_text(encoding="utf-8").strip()
        script_text = re.sub(r"^#+\s+", "", script_text, flags=re.MULTILINE)
        script_text = re.sub(r"\*+", "", script_text).strip()
        
        logger.info(f"  脚本长度: {len(script_text)} 字符")
        
        # Chunk text
        chunks = self.chunk_text(script_text)
        logger.info(f"  分为 {len(chunks)} 段生成 TTS")
        
        Path(output_audio).parent.mkdir(parents=True, exist_ok=True)
        audio_parts = []
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            
            if progress_cb:
                progress_cb(f"Minimax TTS 第 {i+1}/{len(chunks)} 段...")
            
            tmp_path = str(Path(output_audio).parent / f"_tts_mm_{i}.mp3")
            
            try:
                self.generate_chunk(chunk, tmp_path, voice_id, model_id)
                audio_parts.append(tmp_path)
                logger.info(f"  TTS 段 {i+1}/{len(chunks)} 完成")
            except MinimaxTTSError as e:
                logger.error(f"  Minimax TTS 段 {i+1} 失败: {e}")
                # Cleanup partial files
                for p in audio_parts:
                    Path(p).unlink(missing_ok=True)
                raise
        
        if not audio_parts:
            raise MinimaxTTSError(
                message="Minimax TTS 生成失败：无音频输出",
                reason_code="MINIMAX_TTS_NO_OUTPUT",
            )
        
        # Merge chunks
        self.merge_chunks(audio_parts, output_audio)
        
        logger.info(f"Minimax TTS 语音已生成: {output_audio}")
        if progress_cb:
            progress_cb("语音生成完成")
        
        return output_audio


def generate_tts_minimax(
    script_path: str,
    output_audio: str,
    voice_id: str = "male-qn-qingse",
    model_id: str = "speech-01-hd",
    api_key: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
    transport: Optional[MinimaxTransport] = None,
) -> str:
    """
    Generate TTS using Minimax API.
    
    Drop-in replacement for generate_tts_elevenlabs with matching signature.
    
    Args:
        script_path: Path to script text file
        output_audio: Path for output audio file
        voice_id: Minimax voice ID (default: male-qn-qingse)
        model_id: Model ID (default: speech-01-hd)
        api_key: Optional API key (defaults to MINIMAX_API_KEY env var)
        progress_cb: Optional progress callback
        transport: Optional transport (for testing with mocks)
    
    Returns:
        Path to generated audio file
    
    Raises:
        MinimaxTTSError: On provider failure (typed error)
    """
    config = None
    if api_key:
        config = MinimaxConfig(api_key=api_key)
    
    adapter = MinimaxTTSAdapter(config=config, transport=transport)
    
    return adapter.generate(
        script_path=script_path,
        output_audio=output_audio,
        voice_id=voice_id,
        model_id=model_id,
        progress_cb=progress_cb,
    )


# Mock transport factory for TTS testing
class MockMinimaxTTSTransportFactory:
    """Factory for creating pre-configured mock transports for TTS tests."""
    
    @staticmethod
    def create_tts_success(
        audio_content: bytes = b"MOCK_MINIMAX_TTS_AUDIO_MP3",
        request_id: str = "tts-test-request-id",
        use_hex_mode: bool = True,
    ) -> MockMinimaxTransport:
        """
        Create mock transport for successful TTS response.
        
        Uses official T2A v2 contract with data.audio field.
        
        Args:
            audio_content: Raw audio bytes to return
            request_id: Request ID for tracing
            use_hex_mode: If True, data.audio is hex string; if False, data.audio is URL
        
        Returns:
            Mock transport configured for TTS success
        """
        mock = MockMinimaxTransport()
        
        if use_hex_mode:
            # Default mode: data.audio is hex-encoded audio bytes
            audio_hex = audio_content.hex()
            mock.add_response("/v1/t2a_v2", TransportResponse(
                status_code=200,
                payload={
                    "data": {
                        "audio": audio_hex,
                        "status": 2,  # 2 = completed
                    },
                    "extra_info": {
                        "request_id": request_id,
                        "audio_length": len(audio_content),
                        "audio_size": len(audio_content),
                        "audio_format": "mp3",
                    },
                    "trace_id": f"trace-{request_id}",
                    "base_resp": {
                        "status_code": 0,
                        "status_msg": "success",
                    },
                },
            ))
        else:
            # URL mode: data.audio is a download URL
            audio_url = "https://cdn.minimax.io/audio/tts-test.mp3"
            mock.add_response("/v1/t2a_v2", TransportResponse(
                status_code=200,
                payload={
                    "data": {
                        "audio": audio_url,
                        "status": 2,
                    },
                    "extra_info": {
                        "request_id": request_id,
                        "audio_length": len(audio_content),
                        "audio_size": len(audio_content),
                        "audio_format": "mp3",
                    },
                    "trace_id": f"trace-{request_id}",
                    "base_resp": {
                        "status_code": 0,
                        "status_msg": "success",
                    },
                },
            ))
            
            # Mock audio download for URL mode
            mock.add_response(audio_url, TransportResponse(
                status_code=200,
                payload={},
                content=audio_content,
                headers={"Content-Type": "audio/mpeg"},
            ))
        
        return mock
    
    @staticmethod
    def create_tts_malformed_payload() -> MockMinimaxTransport:
        """Create mock transport for malformed payload error."""
        mock = MockMinimaxTransport()
        
        mock.add_response("/v1/t2a_v2", TransportResponse(
            status_code=400,
            payload={
                "base_resp": {
                    "status_code": 1001,
                    "status_msg": "Invalid request: malformed payload",
                },
            },
        ))
        
        return mock
    
    @staticmethod
    def create_tts_missing_audio_url() -> MockMinimaxTransport:
        """Create mock transport for response missing data.audio field."""
        mock = MockMinimaxTransport()
        
        mock.add_response("/v1/t2a_v2", TransportResponse(
            status_code=200,
            payload={
                "data": {
                    "status": 2,
                },
                "extra_info": {
                    "request_id": "test-req",
                },
                "trace_id": "trace-test-req",
                "base_resp": {
                    "status_code": 0,
                    "status_msg": "success",
                },
            },
        ))
        
        return mock
    
    @staticmethod
    def create_tts_missing_data_object() -> MockMinimaxTransport:
        """Create mock transport for response with null/missing data object."""
        mock = MockMinimaxTransport()
        
        mock.add_response("/v1/t2a_v2", TransportResponse(
            status_code=200,
            payload={
                "extra_info": {
                    "request_id": "test-req",
                },
                "trace_id": "trace-test-req",
                "base_resp": {
                    "status_code": 0,
                    "status_msg": "success",
                },
            },
        ))
        
        return mock
    
    @staticmethod
    def create_tts_provider_error(
        status_code: int = 1002,
        status_msg: str = "Rate limit exceeded",
    ) -> MockMinimaxTransport:
        """Create mock transport for provider-level error (base_resp.status_code != 0)."""
        mock = MockMinimaxTransport()
        
        mock.add_response("/v1/t2a_v2", TransportResponse(
            status_code=200,
            payload={
                "data": None,
                "trace_id": "trace-error-test",
                "base_resp": {
                    "status_code": status_code,
                    "status_msg": status_msg,
                },
            },
        ))
        
        return mock
