"""
Minimax Unified Client

Provides canonical submit/poll/fetch/cancel interface for both voice (T2A sync)
and video (async) endpoints, normalized to ProviderLifecycleClient protocol.

Key Types:
- MinimaxEndpointType: Enum for voice/video workflows
- MinimaxConfig: Environment-based configuration
- MinimaxUnifiedClient: Implements ProviderLifecycleClient protocol

Design:
- Voice endpoint (T2A) returns completed job in submit phase
- Video endpoint follows async lifecycle with polling
- All paths normalized to job_id, status, output_path structure
"""
from __future__ import annotations

import os
from urllib.parse import urlparse, urlencode
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pixelle_snapshot.adapters.contracts import (
    AdapterRequest,
    ProviderJobStatus,
    ProviderSubmitResult,
    ProviderPollResult,
    ProviderFetchResult,
    ProviderCancelResult,
    ExecutionError,
)
from src.integrations.minimax.transport import (
    MinimaxTransport,
    UrllibMinimaxTransport,
)


class MinimaxEndpointType(str, Enum):
    """Minimax endpoint types."""
    VOICE = "voice"
    VIDEO = "video"


@dataclass
class MinimaxConfig:
    """Configuration for Minimax API client."""
    api_key: str
    base_url: str = "https://api.minimaxi.com"
    group_id: str = ""
    timeout_seconds: float = 300.0
    
    @classmethod
    def from_env(cls) -> "MinimaxConfig":
        """Load config from environment variables."""
        api_key = os.getenv("MINIMAX_API_KEY", "")
        base_url = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com")
        group_id = os.getenv("MINIMAX_GROUP_ID", "")
        timeout_str = os.getenv("MINIMAX_TIMEOUT_SECONDS", "300.0")
        
        try:
            timeout_seconds = float(timeout_str)
        except ValueError:
            timeout_seconds = 300.0
        
        return cls(
            api_key=api_key,
            base_url=base_url,
            group_id=group_id,
            timeout_seconds=timeout_seconds,
        )


_SUPPORTED_VOICE_MODELS = {
    "speech-2.8-hd",
    "speech-2.8-turbo",
    "speech-2.6-hd",
    "speech-2.6-turbo",
    "speech-02-hd",
    "speech-02-turbo",
    "speech-01-hd",
    "speech-01-turbo",
}


class MinimaxUnifiedClient:
    """
    Unified client for Minimax voice and video endpoints.
    
    Implements ProviderLifecycleClient protocol with submit/poll/fetch/cancel.
    Supports both sync (voice) and async (video) workflows transparently.
    
    Usage:
        config = MinimaxConfig.from_env()
        client = MinimaxUnifiedClient(config=config)
        
        # Submit job
        result = client.submit("voice", request)
        
        # Poll status (no-op for voice, returns completed)
        poll_result = client.poll(result.job_id)
        
        # Fetch output (no-op for voice if already fetched)
        fetch_result = client.fetch(result.job_id, output_dir)
    """
    
    def __init__(
        self,
        config: Optional[MinimaxConfig] = None,
        transport: Optional[MinimaxTransport] = None,
    ):
        """
        Initialize client.
        
        Args:
            config: Minimax configuration (defaults to from_env)
            transport: Transport layer (defaults to urllib production transport)
        """
        self.config = config or MinimaxConfig.from_env()
        self.transport = transport or UrllibMinimaxTransport()
        
        # Internal state for tracking jobs
        self._jobs: dict[str, dict] = {}
    
    def submit(
        self,
        capability: str,
        request: AdapterRequest,
        *,
        idempotency_key: Optional[str] = None,
    ) -> ProviderSubmitResult:
        """
        Submit job to Minimax API.
        
        For voice endpoint: returns completed job immediately
        For video endpoint: returns submitted job with task_id
        
        Args:
            capability: "voice" or "video"
            request: Adapter request with segment metadata
            idempotency_key: Optional idempotency key (unused for Minimax)
        
        Returns:
            ProviderSubmitResult with job_id and status
        
        Raises:
            ExecutionError: On API failure
        """
        endpoint_type = MinimaxEndpointType(capability)
        
        if endpoint_type == MinimaxEndpointType.VOICE:
            return self._submit_voice(request, idempotency_key)
        elif endpoint_type == MinimaxEndpointType.VIDEO:
            return self._submit_video(request, idempotency_key)
        else:
            raise ExecutionError(f"Unsupported capability: {capability}")
    
    def _submit_voice(
        self,
        request: AdapterRequest,
        idempotency_key: Optional[str],
    ) -> ProviderSubmitResult:
        """Submit voice (T2A sync) job."""
        url = f"{self.config.base_url}/v1/t2a_v2"
        if self.config.group_id:
            url = f"{url}?{urlencode({'GroupId': self.config.group_id})}"

        metadata = request.metadata if isinstance(request.metadata, dict) else {}
        requested_model = str(metadata.get("model_id", "")).strip()
        if requested_model in _SUPPORTED_VOICE_MODELS:
            model = requested_model
        elif requested_model in {"speech-01", ""}:
            model = "speech-01-hd"
        else:
            model = "speech-01-hd"

        requested_voice_id = str(metadata.get("voice_id", "")).strip()
        voice_id = requested_voice_id or "male-qn-qingse"

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "text": request.segment_text,
            "model": model,
            "voice_setting": {
                "voice_id": voice_id,
            },
        }
        
        resp = self.transport.request(
            method="POST",
            url=url,
            headers=headers,
            payload=payload,
            timeout_seconds=self.config.timeout_seconds,
        )

        base_resp = resp.payload.get("base_resp") or {}
        provider_status_code = base_resp.get("status_code", 0)
        provider_status_msg = base_resp.get("status_msg", "Unknown error")
        trace_id = resp.payload.get("trace_id")
        request_id = resp.payload.get("extra_info", {}).get("request_id")

        if resp.status_code != 200:
            raise ExecutionError(
                f"Minimax voice submit failed: {provider_status_msg}",
                status_code=resp.status_code,
                provider_status_code=provider_status_code,
                trace_id=trace_id,
                request_id=request_id,
            )

        if provider_status_code != 0:
            raise ExecutionError(
                f"Minimax voice submit failed: {provider_status_msg}",
                status_code=resp.status_code,
                provider_status_code=provider_status_code,
                trace_id=trace_id,
                request_id=request_id,
            )

        data = resp.payload.get("data")
        if not isinstance(data, dict):
            raise ExecutionError(
                "Minimax voice response missing data object",
                status_code=resp.status_code,
                provider_status_code=provider_status_code,
                trace_id=trace_id,
                request_id=request_id,
            )

        audio = data.get("audio")
        if not isinstance(audio, str) or not audio:
            raise ExecutionError(
                "Minimax voice response missing data.audio",
                status_code=resp.status_code,
                provider_status_code=provider_status_code,
                trace_id=trace_id,
                request_id=request_id,
            )

        request_id = request_id or trace_id or "unknown"

        # Store job state (voice completes immediately)
        job_id = f"voice-{request_id}"
        self._jobs[job_id] = {
            "endpoint_type": "voice",
            "status": ProviderJobStatus.SUCCEEDED,
            "audio_url": audio,
            "output_path": None,
        }

        return ProviderSubmitResult(
            job_id=job_id,
            status=ProviderJobStatus.SUCCEEDED,
            metadata={
                "request_id": request_id,
                "trace_id": trace_id,
                "audio_url": audio,
                "provider_status_code": provider_status_code,
            },
        )
    
    def _submit_video(
        self,
        request: AdapterRequest,
        idempotency_key: Optional[str],
    ) -> ProviderSubmitResult:
        """Submit video (async) job."""
        url = f"{self.config.base_url}/v1/video_generation"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": request.segment_text,
            "model": "video-01",
        }
        
        resp = self.transport.request(
            method="POST",
            url=url,
            headers=headers,
            payload=payload,
            timeout_seconds=self.config.timeout_seconds,
        )
        
        if resp.status_code != 200:
            error_msg = resp.payload.get("base_resp", {}).get("status_msg", "Unknown error")
            raise ExecutionError(f"Minimax video submit failed: {error_msg}", status_code=resp.status_code)
        
        task_id = resp.payload.get("task_id", "")
        if not task_id:
            raise ExecutionError("Minimax video response missing task_id")
        
        # Store job state
        job_id = f"video-{task_id}"
        self._jobs[job_id] = {
            "endpoint_type": "video",
            "status": ProviderJobStatus.SUBMITTED,
            "task_id": task_id,
            "file_id": None,
            "output_path": None,
        }
        
        return ProviderSubmitResult(
            job_id=job_id,
            status=ProviderJobStatus.SUBMITTED,
            metadata={"task_id": task_id},
        )
    
    def poll(self, job_id: str) -> ProviderPollResult:
        """
        Poll job status.
        
        For voice: returns SUCCEEDED immediately (no polling needed)
        For video: queries task status endpoint
        
        Args:
            job_id: Job ID from submit
        
        Returns:
            ProviderPollResult with current status
        
        Raises:
            ExecutionError: On unknown job_id or API failure
        """
        if job_id not in self._jobs:
            raise ExecutionError(f"Unknown job_id: {job_id}")
        
        job_state = self._jobs[job_id]
        endpoint_type = job_state["endpoint_type"]
        
        if endpoint_type == "voice":
            # Voice jobs complete immediately
            return ProviderPollResult(
                job_id=job_id,
                status=ProviderJobStatus.SUCCEEDED,
                metadata={},
            )
        elif endpoint_type == "video":
            return self._poll_video(job_id, job_state)
        else:
            raise ExecutionError(f"Unknown endpoint type: {endpoint_type}")
    
    def _poll_video(self, job_id: str, job_state: dict) -> ProviderPollResult:
        """Poll video job status."""
        task_id = job_state["task_id"]
        url = f"{self.config.base_url}/v1/query/video_generation?task_id={task_id}"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
        }
        
        resp = self.transport.request(
            method="GET",
            url=url,
            headers=headers,
            payload=None,
            timeout_seconds=self.config.timeout_seconds,
        )
        
        if resp.status_code != 200:
            error_msg = resp.payload.get("base_resp", {}).get("status_msg", "Unknown error")
            raise ExecutionError(f"Minimax video poll failed: {error_msg}", status_code=resp.status_code)
        
        # Map Minimax status to ProviderJobStatus
        raw_status = resp.payload.get("status", "Unknown")
        if raw_status == "Success":
            status = ProviderJobStatus.SUCCEEDED
            job_state["file_id"] = resp.payload.get("file_id", "")
        elif raw_status == "Failed":
            status = ProviderJobStatus.FAILED
        elif raw_status == "Processing":
            status = ProviderJobStatus.RUNNING
        else:
            status = ProviderJobStatus.QUEUED
        
        job_state["status"] = status
        
        return ProviderPollResult(
            job_id=job_id,
            status=status,
            metadata={"raw_status": raw_status},
        )
    
    def fetch(self, job_id: str, output_dir: str) -> ProviderFetchResult:
        """
        Fetch output artifact and download to output_dir.
        
        For voice: downloads audio file from audio_url
        For video: retrieves file_id info and downloads video
        
        Args:
            job_id: Job ID from submit
            output_dir: Directory to save output file
        
        Returns:
            ProviderFetchResult with local output_path
        
        Raises:
            ExecutionError: On unknown job_id, incomplete job, or download failure
        """
        if job_id not in self._jobs:
            raise ExecutionError(f"Unknown job_id: {job_id}")
        
        job_state = self._jobs[job_id]
        
        # Check if already fetched
        if job_state.get("output_path"):
            return ProviderFetchResult(
                job_id=job_id,
                output_path=job_state["output_path"],
                metadata={},
            )
        
        endpoint_type = job_state["endpoint_type"]
        
        if endpoint_type == "voice":
            output_path = self._fetch_voice(job_id, job_state, output_dir)
        elif endpoint_type == "video":
            output_path = self._fetch_video(job_id, job_state, output_dir)
        else:
            raise ExecutionError(f"Unknown endpoint type: {endpoint_type}")
        
        job_state["output_path"] = output_path
        
        return ProviderFetchResult(
            job_id=job_id,
            output_path=output_path,
            metadata={},
        )
    
    def _fetch_voice(self, job_id: str, job_state: dict, output_dir: str) -> str:
        """Fetch voice audio file."""
        audio_payload = job_state.get("audio_payload")
        if audio_payload is None:
            audio_payload = job_state.get("audio_url")

        if not isinstance(audio_payload, str) or not audio_payload.strip():
            raise ExecutionError(
                "Voice job missing audio payload; expected data.audio URL or hex string"
            )

        audio_payload = audio_payload.strip()
        if self._is_http_url(audio_payload):
            resp = self.transport.request(
                method="GET",
                url=audio_payload,
                headers={},
                payload=None,
                timeout_seconds=self.config.timeout_seconds,
            )

            if resp.status_code != 200:
                raise ExecutionError(
                    f"Failed to download voice audio URL: HTTP {resp.status_code}"
                )
            audio_bytes = resp.content
        else:
            if not self._looks_like_hex_payload(audio_payload):
                raise ExecutionError(
                    "Voice audio payload is neither a valid HTTP(S) URL nor a valid hex string"
                )

            try:
                audio_bytes = bytes.fromhex(audio_payload)
            except ValueError as exc:
                raise ExecutionError(
                    "Voice audio payload hex decode failed; payload must be even-length hexadecimal"
                ) from exc

            if not audio_bytes:
                raise ExecutionError(
                    "Voice audio payload decoded to empty bytes; verify provider returned non-empty data.audio"
                )

        output_path = self._build_output_path(output_dir, f"{job_id}.mp3")
        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        return output_path

    def _is_http_url(self, value: str) -> bool:
        if not isinstance(value, str) or " " in value:
            return False
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def _looks_like_hex_payload(self, value: str) -> bool:
        if not value or len(value) % 2 != 0:
            return False
        hex_chars = set("0123456789abcdefABCDEF")
        return all(ch in hex_chars for ch in value)

    def _build_output_path(self, output_dir: str, output_filename: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        output_dir_abs = os.path.abspath(output_dir)
        output_path = os.path.abspath(os.path.join(output_dir_abs, output_filename))
        if os.path.commonpath([output_dir_abs, output_path]) != output_dir_abs:
            raise ExecutionError(
                f"Resolved output path escapes output_dir: {output_path}"
            )
        return output_path
    
    def _fetch_video(self, job_id: str, job_state: dict, output_dir: str) -> str:
        """Fetch video file."""
        file_id = job_state.get("file_id", "")
        if not file_id:
            raise ExecutionError("Video job not completed or missing file_id")
        
        # Retrieve file info
        url = f"{self.config.base_url}/v1/files/retrieve?file_id={file_id}"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
        }
        
        resp = self.transport.request(
            method="GET",
            url=url,
            headers=headers,
            payload=None,
            timeout_seconds=self.config.timeout_seconds,
        )
        
        if resp.status_code != 200:
            error_msg = resp.payload.get("base_resp", {}).get("status_msg", "Unknown error")
            raise ExecutionError(f"Minimax file retrieve failed: {error_msg}", status_code=resp.status_code)
        
        download_url = resp.payload.get("file", {}).get("download_url", "")
        if not download_url:
            raise ExecutionError("File retrieve response missing download_url")
        
        # Download video
        download_resp = self.transport.request(
            method="GET",
            url=download_url,
            headers={},
            payload=None,
            timeout_seconds=self.config.timeout_seconds,
        )
        
        if download_resp.status_code != 200:
            raise ExecutionError(f"Failed to download video: HTTP {download_resp.status_code}")
        
        # Save to output_dir
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{job_id}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, "wb") as f:
            f.write(download_resp.content)
        
        return output_path
    
    def cancel(self, job_id: str) -> ProviderCancelResult:
        """
        Cancel job.
        
        Note: Minimax API does not support cancellation. This is a no-op
        that returns current job status.
        
        Args:
            job_id: Job ID to cancel
        
        Returns:
            ProviderCancelResult with canceled=False
        """
        if job_id not in self._jobs:
            raise ExecutionError(f"Unknown job_id: {job_id}")
        
        job_state = self._jobs[job_id]
        current_status = job_state.get("status", ProviderJobStatus.QUEUED)
        
        return ProviderCancelResult(
            job_id=job_id,
            canceled=False,
            status=current_status,
            metadata={"reason": "Minimax API does not support cancellation"},
        )
