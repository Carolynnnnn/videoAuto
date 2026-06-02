import itertools
from pathlib import Path

pytest = __import__("pytest")

from pixelle_snapshot.adapters.contracts import (
    DigitalHumanRequest,
    ErrorCategory,
    ProviderJobStatus,
    TimeoutError,
)
from pixelle_snapshot.adapters.provider_client import (
    PixelleProviderClient,
    TransportResponse,
)
from pixelle_snapshot.config_loader import ProviderConfig


class FakeTransport:
    def __init__(self) -> None:
        self.calls = []
        self._scripted = {}

    def queue(self, method: str, url: str, responses):
        self._scripted[(method, url)] = iter(responses)

    def request(self, *, method, url, headers, payload, timeout_seconds):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        key = (method, url)
        sequence = self._scripted.get(key)
        if sequence is None:
            return TransportResponse(status_code=404, payload={"error": "not found"})
        return next(sequence)


def _client(transport: FakeTransport, *, clock=None, sleeper=None) -> PixelleProviderClient:
    cfg = ProviderConfig(
        provider_url="https://provider.example",
        provider_api_key="test-key",
        timeout_seconds=0.5,
        test_mode=False,
    )
    return PixelleProviderClient(
        config=cfg,
        transport=transport,
        poll_interval_seconds=0.01,
        clock=clock or (lambda: 0.0),
        sleeper=sleeper or (lambda _: None),
    )


def test_submit_maps_queued_status_and_idempotency_header(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.queue(
        "POST",
        "https://provider.example/jobs",
        [
            TransportResponse(
                status_code=200,
                payload={"job_id": "job-1", "status": "queued", "queued_seconds": 1.2, "opaque": "x"},
            )
        ],
    )
    client = _client(transport)

    request = DigitalHumanRequest(
        segment_key="seg#1",
        segment_text="hello",
        segment_duration=2.0,
        project_root=str(tmp_path),
        output_dir=str(tmp_path / "generated"),
        avatar_id="avatar-1",
        voice_id="voice-1",
    )

    submit_result = client.submit("digital_human", request, idempotency_key="idem-123")

    assert submit_result.job_id == "job-1"
    assert submit_result.status == ProviderJobStatus.QUEUED
    assert submit_result.metadata == {"queued_seconds": 1.2}
    assert transport.calls[0]["headers"]["Idempotency-Key"] == "idem-123"


def test_wait_for_completion_reaches_succeeded_after_queue_and_running() -> None:
    transport = FakeTransport()
    transport.queue(
        "GET",
        "https://provider.example/jobs/job-2",
        [
            TransportResponse(status_code=200, payload={"status": "queued"}),
            TransportResponse(status_code=200, payload={"status": "running"}),
            TransportResponse(status_code=200, payload={"status": "succeeded", "run_seconds": 3.1}),
        ],
    )

    ticks = itertools.count(start=0)
    client = _client(transport, clock=lambda: float(next(ticks)), sleeper=lambda _: None)

    result = client.wait_for_completion("job-2", timeout_seconds=10.0)

    assert result.status == ProviderJobStatus.SUCCEEDED
    assert result.metadata["run_seconds"] == 3.1
    assert len(transport.calls) == 3


def test_wait_for_completion_timeout_cancels_job() -> None:
    transport = FakeTransport()
    transport.queue(
        "GET",
        "https://provider.example/jobs/job-3",
        [
            TransportResponse(status_code=200, payload={"status": "queued"}),
            TransportResponse(status_code=200, payload={"status": "running"}),
            TransportResponse(status_code=200, payload={"status": "running"}),
        ],
    )
    transport.queue(
        "POST",
        "https://provider.example/jobs/job-3/cancel",
        [TransportResponse(status_code=200, payload={"status": "canceled", "canceled": True})],
    )

    times = iter([0.0, 0.2, 0.4, 0.6])
    client = _client(transport, clock=lambda: next(times), sleeper=lambda _: None)

    with pytest.raises(TimeoutError):
        client.wait_for_completion("job-3", timeout_seconds=0.5, cancel_on_timeout=True)

    assert any(
        call["method"] == "POST" and call["url"].endswith("/jobs/job-3/cancel")
        for call in transport.calls
    )


def test_wait_for_completion_returns_failed_terminal_state() -> None:
    transport = FakeTransport()
    transport.queue(
        "GET",
        "https://provider.example/jobs/job-4",
        [TransportResponse(status_code=200, payload={"status": "failed", "attempt": 2})],
    )
    client = _client(transport)

    result = client.wait_for_completion("job-4", timeout_seconds=10.0)

    assert result.status == ProviderJobStatus.FAILED
    assert result.metadata["attempt"] == 2


def test_fetch_downloads_artifact_to_output_dir(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.queue(
        "GET",
        "https://provider.example/jobs/job-5/artifact",
        [
            TransportResponse(
                status_code=200,
                payload={
                    "artifact_url": "https://cdn.example/output-5.mp4",
                    "cost_usd": 0.12,
                    "duration": 2.5,
                },
            )
        ],
    )
    transport.queue(
        "GET",
        "https://cdn.example/output-5.mp4",
        [TransportResponse(status_code=200, content=b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")],
    )

    client = _client(transport)
    result = client.fetch("job-5", str(tmp_path))

    assert Path(result.output_path).exists()
    assert Path(result.output_path).read_bytes() == b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    assert result.metadata["artifact_bytes"] == len(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
    assert result.metadata["artifact_format"] == "mp4"
    assert result.metadata["artifact_duration"] == 2.5
    assert result.metadata["cost_usd"] == 0.12


def test_fetch_rejects_corrupted_artifact_and_removes_file(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.queue(
        "GET",
        "https://provider.example/jobs/job-bad/artifact",
        [TransportResponse(status_code=200, payload={"artifact_url": "https://cdn.example/output-bad.mp4"})],
    )
    transport.queue(
        "GET",
        "https://cdn.example/output-bad.mp4",
        [TransportResponse(status_code=200, content=b"not-a-video")],
    )

    client = _client(transport)
    with pytest.raises(Exception) as exc:
        client.fetch("job-bad", str(tmp_path))

    assert getattr(exc.value, "category", None) == ErrorCategory.EXECUTION
    assert getattr(exc.value, "details", {}).get("reason_code") == "PIXELLE_ARTIFACT_CORRUPTED"
    assert not (tmp_path / "output-bad.mp4").exists()


def test_fetch_sanitizes_artifact_path_to_output_dir(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.queue(
        "GET",
        "https://provider.example/jobs/job-unsafe/artifact",
        [
            TransportResponse(
                status_code=200,
                payload={"artifact_url": "https://cdn.example/../../escape.mp4", "duration": 1.0},
            )
        ],
    )
    transport.queue(
        "GET",
        "https://cdn.example/../../escape.mp4",
        [TransportResponse(status_code=200, content=b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")],
    )

    client = _client(transport)
    result = client.fetch("job-unsafe", str(tmp_path))

    assert str(tmp_path) in str(Path(result.output_path).resolve())
    assert Path(result.output_path).name == "escape.mp4"


def test_cancel_maps_result() -> None:
    transport = FakeTransport()
    transport.queue(
        "POST",
        "https://provider.example/jobs/job-6/cancel",
        [TransportResponse(status_code=200, payload={"status": "cancelled", "canceled": True, "request_id": "req-77"})],
    )
    client = _client(transport)

    result = client.cancel("job-6")

    assert result.canceled is True
    assert result.status == ProviderJobStatus.CANCELED
    assert result.metadata["request_id"] == "req-77"


def test_unknown_status_raises_provider_category_error() -> None:
    transport = FakeTransport()
    transport.queue(
        "GET",
        "https://provider.example/jobs/job-7",
        [TransportResponse(status_code=200, payload={"status": "mystery"})],
    )
    client = _client(transport)

    with pytest.raises(Exception) as exc:
        client.poll("job-7")

    assert isinstance(exc.value, Exception)
    assert getattr(exc.value, "category", None) == ErrorCategory.PROVIDER
