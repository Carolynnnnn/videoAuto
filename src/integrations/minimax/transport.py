"""
Minimax Transport Layer

Provides transport abstraction for HTTP communication with Minimax API.
Includes both production transport (urllib-based) and mock transport
for deterministic testing.

Transport Protocol:
- request() method for all HTTP operations
- Returns TransportResponse with status_code, payload, content, headers
- Handles timeouts and connection errors
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Protocol


@dataclass
class TransportResponse:
    """Response from transport layer."""
    status_code: int
    payload: Dict[str, Any] = field(default_factory=dict)
    content: bytes = b""
    headers: Dict[str, str] = field(default_factory=dict)


class MinimaxTransport(Protocol):
    """Protocol for Minimax HTTP transport."""
    
    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        payload: Optional[Dict[str, Any]],
        timeout_seconds: float,
    ) -> TransportResponse:
        """Execute HTTP request and return response."""
        ...


class UrllibMinimaxTransport:
    """Production transport using urllib."""
    
    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        payload: Optional[Dict[str, Any]],
        timeout_seconds: float,
    ) -> TransportResponse:
        """Execute HTTP request via urllib."""
        body: Optional[bytes] = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url=url, data=body, method=method)
        for key, value in headers.items():
            req.add_header(key, value)

        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                content = resp.read()
                parsed_payload: Dict[str, Any] = {}
                if content:
                    try:
                        parsed_payload = json.loads(content.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        parsed_payload = {}
                return TransportResponse(
                    status_code=resp.getcode() or 200,
                    payload=parsed_payload,
                    content=content,
                    headers=dict(resp.headers.items()),
                )
        except urllib.error.HTTPError as exc:
            content = exc.read() if hasattr(exc, "read") else b""
            parsed_payload_err: Dict[str, Any] = {}
            if content:
                try:
                    parsed_payload_err = json.loads(content.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    parsed_payload_err = {}
            return TransportResponse(
                status_code=exc.code,
                payload=parsed_payload_err,
                content=content,
                headers=dict(getattr(exc, "headers", {}).items()),
            )
        except urllib.error.URLError as exc:
            # Network/connection errors - return synthetic error response
            reason = str(getattr(exc, "reason", "Connection failed"))
            return TransportResponse(
                status_code=0,
                payload={"error": reason, "error_type": "connection_error"},
                content=b"",
                headers={},
            )


@dataclass
class MockMinimaxTransport:
    """
    Deterministic mock transport for testing.
    
    Provides configurable responses for different URL patterns to enable
    unit testing without network dependency.
    
    Usage:
        mock = MockMinimaxTransport()
        mock.add_response("/v1/t2a_v2", TransportResponse(
            status_code=200,
            payload={"audio_url": "https://..."}
        ))
        
        # Use in client
        client = MinimaxUnifiedClient(transport=mock)
    """
    
    _responses: Dict[str, TransportResponse] = field(default_factory=dict)
    _call_log: list = field(default_factory=list)
    _default_response: Optional[TransportResponse] = None
    
    def add_response(self, url_pattern: str, response: TransportResponse) -> None:
        """Add a mock response for URL pattern (path suffix match)."""
        self._responses[url_pattern] = response
    
    def set_default_response(self, response: TransportResponse) -> None:
        """Set default response for unmatched URLs."""
        self._default_response = response
    
    def get_call_log(self) -> list:
        """Get list of all requests made through this transport."""
        return list(self._call_log)
    
    def clear_call_log(self) -> None:
        """Clear the call log."""
        self._call_log.clear()
    
    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        payload: Optional[Dict[str, Any]],
        timeout_seconds: float,
    ) -> TransportResponse:
        """Return mock response matching URL pattern."""
        # Log the call
        self._call_log.append({
            "method": method,
            "url": url,
            "headers": dict(headers),
            "payload": payload,
            "timeout_seconds": timeout_seconds,
        })
        
        # Find matching response by URL pattern (check if pattern is substring)
        for pattern, response in self._responses.items():
            if pattern in url:
                return response
        
        # Return default or 404
        if self._default_response:
            return self._default_response
        
        return TransportResponse(
            status_code=404,
            payload={"error": f"No mock response for {url}"},
            content=b"",
            headers={},
        )


# Factory for creating mock transport with common test scenarios
class MockMinimaxTransportFactory:
    """Factory for creating pre-configured mock transports."""
    
    @staticmethod
    def create_voice_success(
        audio_content: bytes = b"MOCK_AUDIO_CONTENT_MP3",
        request_id: str = "test-request-id",
        use_hex_mode: bool = True,
    ) -> MockMinimaxTransport:
        """Create mock transport for successful voice (T2A v2) response using data.audio contract."""
        mock = MockMinimaxTransport()
        
        if use_hex_mode:
            audio_hex = audio_content.hex()
            mock.add_response("/v1/t2a_v2", TransportResponse(
                status_code=200,
                payload={
                    "data": {
                        "audio": audio_hex,
                        "status": 2,
                    },
                    "extra_info": {
                        "request_id": request_id,
                        "audio_size": len(audio_content),
                    },
                    "trace_id": f"trace-{request_id}",
                    "base_resp": {
                        "status_code": 0,
                        "status_msg": "success",
                    },
                },
            ))
        else:
            audio_url = "https://cdn.minimax.io/audio/test.mp3"
            mock.add_response("/v1/t2a_v2", TransportResponse(
                status_code=200,
                payload={
                    "data": {
                        "audio": audio_url,
                        "status": 2,
                    },
                    "extra_info": {
                        "request_id": request_id,
                        "audio_size": len(audio_content),
                    },
                    "trace_id": f"trace-{request_id}",
                    "base_resp": {
                        "status_code": 0,
                        "status_msg": "success",
                    },
                },
            ))
            mock.add_response(audio_url, TransportResponse(
                status_code=200,
                payload={},
                content=audio_content,
                headers={"Content-Type": "audio/mpeg"},
            ))
        return mock
    
    @staticmethod
    def create_video_async_success(
        task_id: str = "test-task-id",
        file_id: str = "test-file-id",
        video_url: str = "https://cdn.minimax.io/video/test.mp4",
    ) -> MockMinimaxTransport:
        """Create mock transport for successful video (async) lifecycle."""
        mock = MockMinimaxTransport()
        
        # Submit response
        mock.add_response("/v1/video_generation", TransportResponse(
            status_code=200,
            payload={
                "task_id": task_id,
                "base_resp": {"status_code": 0, "status_msg": "success"},
            },
        ))
        
        # Poll response (completed)
        mock.add_response(f"/v1/query/video_generation?task_id={task_id}", TransportResponse(
            status_code=200,
            payload={
                "task_id": task_id,
                "status": "Success",
                "file_id": file_id,
                "base_resp": {"status_code": 0, "status_msg": "success"},
            },
        ))
        
        # File retrieval info
        mock.add_response(f"/v1/files/retrieve?file_id={file_id}", TransportResponse(
            status_code=200,
            payload={
                "file": {
                    "file_id": file_id,
                    "download_url": video_url,
                },
                "base_resp": {"status_code": 0, "status_msg": "success"},
            },
        ))
        
        # Video download
        mock.add_response(video_url, TransportResponse(
            status_code=200,
            payload={},
            content=b"MOCK_VIDEO_CONTENT_MP4",
            headers={"Content-Type": "video/mp4"},
        ))
        
        return mock
    
    @staticmethod
    def create_auth_failure() -> MockMinimaxTransport:
        """Create mock transport for authentication failure."""
        mock = MockMinimaxTransport()
        mock.set_default_response(TransportResponse(
            status_code=401,
            payload={
                "base_resp": {
                    "status_code": 1002,
                    "status_msg": "Invalid API Key",
                },
            },
        ))
        return mock
    
    @staticmethod
    def create_rate_limit() -> MockMinimaxTransport:
        """Create mock transport for rate limit response."""
        mock = MockMinimaxTransport()
        mock.set_default_response(TransportResponse(
            status_code=429,
            payload={
                "base_resp": {
                    "status_code": 1007,
                    "status_msg": "Rate limit exceeded",
                },
            },
        ))
        return mock
