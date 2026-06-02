import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import pytest

from src.core.models import AudioRef, Segment, VisualPlan
from src.steps.pixelle_quota_accounting import (
    QuotaAccounting,
    QuotaConfig,
    QuotaEnforcement,
    QuotaExceededError,
    create_quota_diagnostic,
    get_quota_enforcement,
    reset_quota_enforcement,
    set_quota_enforcement,
)
from src.steps.step4_assets import resolve_asset_for_segment


def _make_segment(index: int = 1) -> Segment:
    text = f"Quota test segment {index}"
    content_key = Segment.compute_content_key(text)
    return Segment(
        segment_key=Segment.compute_segment_key(content_key, index),
        content_key=content_key,
        index=index,
        start=0.0,
        end=4.0,
        duration=4.0,
        text=text,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=4.0),
        visual_plan=VisualPlan(type="pixelle_digital_human", pixelle_workflow="digital_human"),
        plan_hash=f"quotahash{index}",
    )


@pytest.fixture(autouse=True)
def reset_quota_singleton():
    reset_quota_enforcement()
    yield
    reset_quota_enforcement()


class TestQuotaConfig:
    def test_from_env_defaults(self, monkeypatch):
        monkeypatch.delenv("PIXELLE_QUOTA_ENABLED", raising=False)
        monkeypatch.delenv("PIXELLE_TEST_MODE", raising=False)
        monkeypatch.delenv("PIXELLE_QUOTA_MAX_REQUESTS_PER_BUILD", raising=False)
        monkeypatch.delenv("PIXELLE_QUOTA_MAX_COST_USD_PER_BUILD", raising=False)
        monkeypatch.delenv("PIXELLE_QUOTA_MAX_COST_USD_PER_REQUEST", raising=False)

        config = QuotaConfig.from_env()

        assert config.enabled is False
        assert config.test_mode is False
        assert config.max_requests_per_build == 0
        assert config.max_cost_usd_per_build == 0.0
        assert config.max_cost_usd_per_request == 0.0

    def test_from_env_enabled(self, monkeypatch):
        monkeypatch.setenv("PIXELLE_QUOTA_ENABLED", "1")
        monkeypatch.setenv("PIXELLE_QUOTA_MAX_REQUESTS_PER_BUILD", "10")
        monkeypatch.setenv("PIXELLE_QUOTA_MAX_COST_USD_PER_BUILD", "5.50")
        monkeypatch.setenv("PIXELLE_QUOTA_MAX_COST_USD_PER_REQUEST", "0.25")

        config = QuotaConfig.from_env()

        assert config.enabled is True
        assert config.max_requests_per_build == 10
        assert config.max_cost_usd_per_build == 5.50
        assert config.max_cost_usd_per_request == 0.25

    def test_is_enforcement_active_when_enabled_and_not_test_mode(self, monkeypatch):
        config = QuotaConfig(enabled=True, test_mode=False)
        assert config.is_enforcement_active is True

    def test_is_enforcement_active_disabled_in_test_mode(self, monkeypatch):
        config = QuotaConfig(enabled=True, test_mode=True)
        assert config.is_enforcement_active is False

    def test_is_enforcement_active_disabled_when_not_enabled(self):
        config = QuotaConfig(enabled=False, test_mode=False)
        assert config.is_enforcement_active is False


class TestQuotaAccounting:
    def test_record_usage_tracks_requests(self):
        config = QuotaConfig(enabled=True, max_requests_per_build=10)
        accounting = QuotaAccounting(config)

        accounting.record_usage(
            request_id="req-001",
            segment_key="seg#1",
            capability="digital_human",
            cost_usd=0.10,
        )
        accounting.record_usage(
            request_id="req-002",
            segment_key="seg#2",
            capability="i2v",
            cost_usd=0.15,
        )

        snapshot = accounting.get_snapshot()
        assert snapshot.total_requests == 2
        assert abs(snapshot.total_cost_usd - 0.25) < 1e-6

    def test_record_usage_with_build_id(self):
        config = QuotaConfig(enabled=True)
        accounting = QuotaAccounting(config)

        accounting.record_usage(
            request_id="req-001",
            segment_key="seg#1",
            capability="digital_human",
            cost_usd=0.10,
            build_id="build-abc",
        )
        accounting.record_usage(
            request_id="req-002",
            segment_key="seg#2",
            capability="i2v",
            cost_usd=0.20,
            build_id="build-def",
        )

        snapshot_abc = accounting.get_snapshot(build_id="build-abc")
        assert snapshot_abc.total_requests == 1
        assert abs(snapshot_abc.total_cost_usd - 0.10) < 1e-6

        snapshot_def = accounting.get_snapshot(build_id="build-def")
        assert snapshot_def.total_requests == 1
        assert abs(snapshot_def.total_cost_usd - 0.20) < 1e-6

    def test_reset_build_clears_usage(self):
        config = QuotaConfig(enabled=True)
        accounting = QuotaAccounting(config)

        accounting.record_usage(
            request_id="req-001",
            segment_key="seg#1",
            capability="digital_human",
            build_id="build-123",
        )
        assert accounting.get_snapshot(build_id="build-123").total_requests == 1

        accounting.reset_build("build-123")
        assert accounting.get_snapshot(build_id="build-123").total_requests == 0

    def test_snapshot_remaining_calculations(self):
        config = QuotaConfig(
            enabled=True,
            max_requests_per_build=5,
            max_cost_usd_per_build=1.0,
        )
        accounting = QuotaAccounting(config)

        accounting.record_usage(
            request_id="req-001",
            segment_key="seg#1",
            capability="digital_human",
            cost_usd=0.30,
        )

        snapshot = accounting.get_snapshot()
        assert snapshot.requests_remaining == 4
        assert snapshot.cost_usd_remaining is not None
        assert abs(snapshot.cost_usd_remaining - 0.70) < 1e-6


class TestQuotaEnforcement:
    def test_check_passes_when_within_quota(self):
        config = QuotaConfig(
            enabled=True,
            max_requests_per_build=10,
            max_cost_usd_per_build=5.0,
            max_cost_usd_per_request=1.0,
        )
        enforcement = QuotaEnforcement(config=config)

        enforcement.check_before_request(
            segment_key="seg#1",
            capability="digital_human",
            estimated_cost_usd=0.50,
        )

    def test_check_raises_when_request_count_exceeded(self):
        config = QuotaConfig(
            enabled=True,
            max_requests_per_build=2,
        )
        enforcement = QuotaEnforcement(config=config)

        enforcement.accounting.record_usage(
            request_id="req-001",
            segment_key="seg#1",
            capability="digital_human",
        )
        enforcement.accounting.record_usage(
            request_id="req-002",
            segment_key="seg#2",
            capability="digital_human",
        )

        with pytest.raises(QuotaExceededError) as exc_info:
            enforcement.check_before_request(
                segment_key="seg#3",
                capability="digital_human",
            )

        assert exc_info.value.reason_code == "PIXELLE_QUOTA_EXCEEDED"
        assert exc_info.value.category == "RESOURCE"
        assert exc_info.value.current_value == 2.0
        assert exc_info.value.limit_value == 2.0

    def test_check_raises_when_build_budget_exceeded(self):
        config = QuotaConfig(
            enabled=True,
            max_cost_usd_per_build=0.50,
        )
        enforcement = QuotaEnforcement(config=config)

        enforcement.accounting.record_usage(
            request_id="req-001",
            segment_key="seg#1",
            capability="digital_human",
            cost_usd=0.40,
        )

        with pytest.raises(QuotaExceededError) as exc_info:
            enforcement.check_before_request(
                segment_key="seg#2",
                capability="digital_human",
                estimated_cost_usd=0.20,
            )

        assert exc_info.value.reason_code == "PIXELLE_BUDGET_EXCEEDED"
        assert exc_info.value.category == "RESOURCE"

    def test_check_raises_when_per_request_cost_exceeded(self):
        config = QuotaConfig(
            enabled=True,
            max_cost_usd_per_request=0.10,
        )
        enforcement = QuotaEnforcement(config=config)

        with pytest.raises(QuotaExceededError) as exc_info:
            enforcement.check_before_request(
                segment_key="seg#1",
                capability="digital_human",
                estimated_cost_usd=0.50,
            )

        assert exc_info.value.reason_code == "PIXELLE_REQUEST_COST_EXCEEDED"
        assert exc_info.value.category == "RESOURCE"
        assert exc_info.value.current_value == 0.50
        assert exc_info.value.limit_value == 0.10

    def test_check_bypassed_in_test_mode(self, monkeypatch):
        config = QuotaConfig(
            enabled=True,
            test_mode=True,
            max_requests_per_build=1,
        )
        enforcement = QuotaEnforcement(config=config)

        enforcement.accounting.record_usage(
            request_id="req-001",
            segment_key="seg#1",
            capability="digital_human",
        )

        enforcement.check_before_request(
            segment_key="seg#2",
            capability="digital_human",
        )


class TestCreateQuotaDiagnostic:
    def test_creates_valid_diagnostic_dict(self):
        error = QuotaExceededError(
            reason_code="PIXELLE_QUOTA_EXCEEDED",
            category="RESOURCE",
            guidance="Build quota exceeded.",
            current_value=10.0,
            limit_value=10.0,
        )

        diagnostic = create_quota_diagnostic(error)

        assert diagnostic["category"] == "RESOURCE"
        assert diagnostic["reason_code"] == "PIXELLE_QUOTA_EXCEEDED"
        assert diagnostic["guidance"] == "Build quota exceeded."
        assert diagnostic["retryable"] is False
        assert diagnostic["fallback_hint"] == "Pipeline will use next fallback source."
        assert diagnostic["quota_details"]["current_value"] == 10.0
        assert diagnostic["quota_details"]["limit_value"] == 10.0


class TestStep4QuotaIntegration:
    def test_happy_path_within_quota(self, monkeypatch, tmp_path: Path):
        project_root = tmp_path
        generated_dir = tmp_path / "assets" / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)

        config = QuotaConfig(enabled=True, max_requests_per_build=10)
        enforcement = QuotaEnforcement(config=config)
        set_quota_enforcement(enforcement)

        class FakeAdapter:
            def invoke(self, request):
                output_path = Path(request.output_dir) / f"pixelle_{request.segment_key}.mp4"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"pixelle-video")
                return type(
                    "Resp",
                    (),
                    {
                        "success": True,
                        "output_path": str(output_path),
                        "error": None,
                        "metadata": {"cost_usd": 0.10},
                    },
                )()

        monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
        monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FakeAdapter())

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

        assert resolved.asset_refs[0].kind == "pixelle_video"
        assert resolved.asset_refs[0].fallback_reason_code is None

        snapshot = enforcement.accounting.get_snapshot()
        assert snapshot.total_requests == 1

    def test_request_count_exceeded_triggers_fallback(self, monkeypatch, tmp_path: Path):
        project_root = tmp_path
        generated_dir = tmp_path / "assets" / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)

        config = QuotaConfig(enabled=True, max_requests_per_build=1)
        enforcement = QuotaEnforcement(config=config)
        enforcement.accounting.record_usage(
            request_id="prior-req",
            segment_key="seg#0",
            capability="digital_human",
        )
        set_quota_enforcement(enforcement)

        def fake_template(output_path: str, width: int, height: int, text: str):
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"template")
            return str(path)

        monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
        monkeypatch.setattr("src.steps.step4_assets.generate_template_asset", fake_template)

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
        assert ref.kind == "template"
        assert ref.fallback_reason_code == "PIXELLE_QUOTA_EXCEEDED"
        assert ref.fallback_error_category == "RESOURCE"
        assert ref.fallback_diagnostic is not None
        assert ref.fallback_diagnostic["reason_code"] == "PIXELLE_QUOTA_EXCEEDED"
        assert ref.fallback_diagnostic["quota_details"]["current_value"] == 1.0
        assert ref.fallback_diagnostic["quota_details"]["limit_value"] == 1.0

    def test_build_budget_exceeded_triggers_fallback(self, monkeypatch, tmp_path: Path):
        project_root = tmp_path
        generated_dir = tmp_path / "assets" / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)

        config = QuotaConfig(enabled=True, max_cost_usd_per_build=0.50)
        enforcement = QuotaEnforcement(config=config)
        enforcement.accounting.record_usage(
            request_id="prior-req",
            segment_key="seg#0",
            capability="digital_human",
            cost_usd=0.50,
        )
        set_quota_enforcement(enforcement)

        def fake_template(output_path: str, width: int, height: int, text: str):
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"template")
            return str(path)

        monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
        monkeypatch.setattr("src.steps.step4_assets.generate_template_asset", fake_template)

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
        assert ref.kind == "template"
        assert ref.fallback_reason_code == "PIXELLE_BUDGET_EXCEEDED"
        assert ref.fallback_error_category == "RESOURCE"
        assert ref.fallback_diagnostic is not None

    def test_test_mode_bypasses_quota_enforcement(self, monkeypatch, tmp_path: Path):
        project_root = tmp_path
        generated_dir = tmp_path / "assets" / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)

        config = QuotaConfig(enabled=True, test_mode=True, max_requests_per_build=1)
        enforcement = QuotaEnforcement(config=config)
        enforcement.accounting.record_usage(
            request_id="prior-req",
            segment_key="seg#0",
            capability="digital_human",
        )
        set_quota_enforcement(enforcement)

        class FakeAdapter:
            def invoke(self, request):
                output_path = Path(request.output_dir) / f"pixelle_{request.segment_key}.mp4"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"pixelle-video")
                return type(
                    "Resp",
                    (),
                    {
                        "success": True,
                        "output_path": str(output_path),
                        "error": None,
                        "metadata": {"test_mode": True},
                    },
                )()

        monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
        monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FakeAdapter())

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

        assert resolved.asset_refs[0].kind == "pixelle_video"
        assert resolved.asset_refs[0].fallback_reason_code is None

    def test_quota_failure_does_not_block_fallback_chain(self, monkeypatch, tmp_path: Path):
        project_root = tmp_path
        generated_dir = tmp_path / "assets" / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)

        config = QuotaConfig(enabled=True, max_requests_per_build=1)
        enforcement = QuotaEnforcement(config=config)
        enforcement.accounting.record_usage(
            request_id="prior-req",
            segment_key="seg#0",
            capability="digital_human",
        )
        set_quota_enforcement(enforcement)

        def fake_template(output_path: str, width: int, height: int, text: str):
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"template-fallback")
            return str(path)

        monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
        monkeypatch.setattr("src.steps.step4_assets.generate_template_asset", fake_template)

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

        assert resolved.asset_refs[0].kind == "template"
        assert resolved.asset_refs[0].fallback_reason_code == "PIXELLE_QUOTA_EXCEEDED"
        assert Path(resolved.asset_refs[0].path).exists()

    def test_usage_recorded_after_successful_provider_call(self, monkeypatch, tmp_path: Path):
        project_root = tmp_path
        generated_dir = tmp_path / "assets" / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)

        config = QuotaConfig(enabled=True, max_requests_per_build=100)
        enforcement = QuotaEnforcement(config=config)
        set_quota_enforcement(enforcement)

        class FakeAdapter:
            def invoke(self, request):
                output_path = Path(request.output_dir) / f"pixelle_{request.segment_key}.mp4"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"pixelle-video")
                return type(
                    "Resp",
                    (),
                    {
                        "success": True,
                        "output_path": str(output_path),
                        "error": None,
                        "metadata": {"cost_usd": 0.25},
                    },
                )()

        monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
        monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FakeAdapter())

        seg = _make_segment()
        resolve_asset_for_segment(
            segment=seg,
            project_root=str(project_root),
            generated_dir=str(generated_dir),
            library_dir=str(tmp_path / "assets" / "library"),
            pexels_api_key="",
            enable_pexels_video=False,
            enable_pexels_photo=False,
            enable_ai_image=False,
        )

        snapshot = enforcement.accounting.get_snapshot()
        assert snapshot.total_requests == 1
        assert abs(snapshot.total_cost_usd - 0.25) < 1e-6
