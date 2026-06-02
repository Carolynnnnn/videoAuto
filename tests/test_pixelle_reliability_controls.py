import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pixelle_snapshot.adapters.contracts import AdapterError, ErrorCategory
from src.core.models import AudioRef, Segment, VisualPlan
from src.steps.pixelle_reliability_controls import (
    ErrorRateCircuitBreaker,
    PixelleReliabilityControls,
    ReliabilityConfig,
    TokenBucketRateLimiter,
)
from src.steps.step4_assets import resolve_asset_for_segment


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_segment() -> Segment:
    text = "Reliability control segment"
    content_key = Segment.compute_content_key(text)
    return Segment(
        segment_key=Segment.compute_segment_key(content_key, 1),
        content_key=content_key,
        index=1,
        start=0.0,
        end=4.0,
        duration=4.0,
        text=text,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=4.0),
        visual_plan=VisualPlan(type="pixelle_digital_human", pixelle_workflow="digital_human"),
        plan_hash="reliabilityhash1234",
    )


def test_token_bucket_rate_limiter_enforces_configured_ceiling() -> None:
    clock = FakeClock()
    limiter = TokenBucketRateLimiter(rate_per_second=2.0, burst=2, clock=clock)

    assert limiter.acquire(timeout_seconds=0.0) is True
    assert limiter.acquire(timeout_seconds=0.0) is True
    assert limiter.acquire(timeout_seconds=0.0) is False

    clock.advance(0.5)
    assert limiter.acquire(timeout_seconds=0.0) is True


def test_circuit_breaker_opens_on_sustained_error_rate_window() -> None:
    clock = FakeClock()
    breaker = ErrorRateCircuitBreaker(
        window_size=4,
        min_requests=4,
        error_rate_threshold=0.75,
        open_seconds=5.0,
        half_open_max_calls=1,
        clock=clock,
    )

    breaker.record_failure(category=ErrorCategory.PROVIDER.value)
    breaker.record_failure(category=ErrorCategory.PROVIDER.value)
    breaker.record_success()
    breaker.record_failure(category=ErrorCategory.TIMEOUT.value)

    assert breaker.state == "open"
    assert breaker.allow_request() is False

    clock.advance(5.0)
    assert breaker.allow_request() is True
    breaker.record_success()
    assert breaker.state == "closed"


def test_step4_short_circuits_when_circuit_is_open(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    calls = {"count": 0}

    class FailingAdapter:
        def invoke(self, request):
            calls["count"] += 1
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(category=ErrorCategory.PROVIDER, message="provider failed"),
                },
            )()

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FailingAdapter())

    def fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template")
        return str(path)

    monkeypatch.setattr("src.steps.step4_assets.generate_template_asset", fake_template)

    controls = PixelleReliabilityControls(
        ReliabilityConfig(
            rate_limit_per_second=0.0,
            rate_limit_burst=1,
            rate_limit_wait_seconds=0.0,
            circuit_window_size=4,
            circuit_min_requests=1,
            circuit_error_rate_threshold=1.0,
            circuit_open_seconds=999.0,
            circuit_half_open_max_calls=1,
        )
    )
    controls.record_failure(category=ErrorCategory.PROVIDER.value)
    monkeypatch.setattr("src.steps.step4_assets._pixelle_reliability_controls", controls)

    seg = _make_segment()
    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
    )

    ref = resolved.asset_refs[0]
    assert calls["count"] == 0
    assert ref.kind == "template"
    assert ref.fallback_reason_code == "PIXELLE_CIRCUIT_OPEN"
    assert ref.fallback_error_category == "PROVIDER"
    assert ref.fallback_diagnostic is not None
    assert ref.fallback_diagnostic["reason_code"] == "PIXELLE_CIRCUIT_OPEN"


def test_step4_rate_limit_throttles_provider_invocation(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    calls = {"count": 0}

    class FakeAdapter:
        def invoke(self, request):
            calls["count"] += 1
            output_path = Path(request.output_dir) / f"pixelle_{request.segment_key}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"pixelle-video")
            return type("Resp", (), {"success": True, "output_path": str(output_path), "error": None})()

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FakeAdapter())

    def fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template")
        return str(path)

    monkeypatch.setattr("src.steps.step4_assets.generate_template_asset", fake_template)

    clock = FakeClock()
    controls = PixelleReliabilityControls(
        ReliabilityConfig(
            rate_limit_per_second=1.0,
            rate_limit_burst=1,
            rate_limit_wait_seconds=0.0,
            circuit_window_size=10,
            circuit_min_requests=5,
            circuit_error_rate_threshold=0.8,
            circuit_open_seconds=5.0,
            circuit_half_open_max_calls=1,
        ),
        rate_limiter=TokenBucketRateLimiter(rate_per_second=1.0, burst=1, clock=clock),
    )
    monkeypatch.setattr("src.steps.step4_assets._pixelle_reliability_controls", controls)

    first = resolve_asset_for_segment(
        segment=_make_segment(),
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
    )
    assert first.asset_refs[0].kind == "pixelle_video"

    second_seg = _make_segment()
    second_seg.segment_key = f"{second_seg.segment_key}-2"
    second_seg.content_key = Segment.compute_content_key("second call")
    second = resolve_asset_for_segment(
        segment=second_seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
    )

    ref = second.asset_refs[0]
    assert calls["count"] == 1
    assert ref.kind == "template"
    assert ref.fallback_reason_code == "PIXELLE_RATE_LIMITED"
    assert ref.fallback_error_category == "RESOURCE"
