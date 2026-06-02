import sys
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

pytest = __import__("pytest")

from pixelle_snapshot import test_doubles
from pixelle_snapshot.adapters import (
    ActionTransferAdapter,
    ActionTransferRequest,
    DigitalHumanAdapter,
    DigitalHumanRequest,
    I2VAdapter,
    I2VRequest,
)
from pixelle_snapshot.adapters.contracts import (
    ProviderFetchResult,
    ProviderJobStatus,
    ProviderPollResult,
    ProviderSubmitResult,
)
from pixelle_snapshot.config_loader import ProviderConfigError, load_provider_config
from src.core.models import AudioRef, GlobalStyle, Manifest, Segment, VisualPlan
from src.steps.step4_assets import run_step4
from src.workbench.apply import apply_pixelle_selections
from src.workbench.state import init_workbench, load_session, save_session


PixelleWorkflow = Literal["digital_human", "i2v", "action_transfer"]


@pytest.fixture(autouse=True)
def _deterministic_pixelle_mode(request):
    if "provider_mode_env" in request.fixturenames or "provider_mode_missing_api_key" in request.fixturenames:
        test_doubles.disable_test_mode()
        try:
            yield
        finally:
            test_doubles.disable_test_mode()
        return

    test_doubles.enable_test_mode()
    try:
        yield
    finally:
        test_doubles.disable_test_mode()


@pytest.fixture
def provider_mode_env(monkeypatch):
    monkeypatch.setenv("PIXELLE_TEST_MODE", "0")
    monkeypatch.setenv("PIXELLE_PROVIDER_URL", "https://staging.provider.invalid")
    monkeypatch.setenv("PIXELLE_PROVIDER_API_KEY", "staging-api-key")


@pytest.fixture
def provider_mode_missing_api_key(monkeypatch):
    monkeypatch.setenv("PIXELLE_TEST_MODE", "0")
    monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
    monkeypatch.setenv("PIXELLE_PROVIDER_URL", "https://staging.provider.invalid")
    monkeypatch.delenv("PIXELLE_PROVIDER_API_KEY", raising=False)


class _FakeProviderClient:
    def __init__(self, capability: PixelleWorkflow):
        self.capability = capability

    def submit(self, capability, request, idempotency_key=None):
        return ProviderSubmitResult(
            job_id=f"{self.capability}-job-001",
            status=ProviderJobStatus.SUBMITTED,
            metadata={"request_id": f"req-{self.capability}", "attempt": 1},
        )

    def wait_for_completion(self, job_id, timeout_seconds=None, cancel_on_timeout=True):
        metadata = {
            "duration": 2.4,
            "resolution": "1080x1920",
            "run_seconds": 1.2,
            "total_seconds": 1.4,
        }
        if self.capability == "i2v":
            metadata["fps"] = 30
        return ProviderPollResult(job_id=job_id, status=ProviderJobStatus.SUCCEEDED, metadata=metadata)

    def fetch(self, job_id, output_dir):
        output_path = Path(output_dir) / f"{job_id}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"provider-{self.capability}".encode("utf-8"))
        return ProviderFetchResult(
            job_id=job_id,
            output_path=str(output_path),
            metadata={"artifact_bytes": output_path.stat().st_size, "artifact_format": "mp4", "artifact_duration": 2.4},
        )


def _build_provider_backed_adapter(workflow: PixelleWorkflow, **kwargs):
    client = _FakeProviderClient(workflow)
    if workflow == "digital_human":
        return DigitalHumanAdapter(provider_client=client)
    if workflow == "i2v":
        return I2VAdapter(provider_client=client)
    if workflow == "action_transfer":
        return ActionTransferAdapter(provider_client=client)
    raise AssertionError(f"unsupported workflow: {workflow}")


def _make_segment(text: str, index: int = 1) -> Segment:
    content_key = Segment.compute_content_key(text)
    return Segment(
        segment_key=Segment.compute_segment_key(content_key, 1),
        content_key=content_key,
        index=index,
        start=float(index - 1),
        end=float(index),
        duration=1.0,
        text=text,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=1.0),
        visual_plan=VisualPlan(type="template"),
        plan_hash=f"plan-hash-{index}",
    )


def _seed_project(project_root: Path, segment: Segment) -> Path:
    (project_root / "build").mkdir(parents=True, exist_ok=True)
    manifest = Manifest(
        project_id="pixelle-workbench-step4-int",
        build_id="integration-build-1",
        global_style=GlobalStyle(),
        segments=[segment],
    )
    manifest_path = project_root / "build" / "manifest.json"
    manifest.save(str(manifest_path))
    return manifest_path


def _apply_workbench_settings(
    project_root: Path,
    *,
    default_workflow: PixelleWorkflow,
    overrides: dict[str, str | None],
):
    paths = init_workbench(project_root)
    session = load_session(paths)
    session.pixelle_default_workflow = default_workflow
    session.pixelle_segment_overrides = overrides
    save_session(paths, session)
    apply_result = apply_pixelle_selections(paths)
    return apply_result, Manifest.load(str(project_root / "build" / "manifest.json"))


@pytest.mark.workbench_apply_vendor_continuity
@pytest.mark.parametrize("workflow", ["digital_human", "i2v", "action_transfer"])
def test_workbench_apply_to_step4_happy_path_for_all_pixelle_workflows(
    workflow: PixelleWorkflow,
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
    segment = _make_segment(text=f"happy path {workflow}")
    project_root = tmp_path / f"project_happy_{workflow}"
    manifest_path = _seed_project(project_root, segment)

    apply_result, applied_manifest = _apply_workbench_settings(
        project_root,
        default_workflow=workflow,
        overrides={},
    )

    assert apply_result.manifest_path == str(manifest_path)
    assert apply_result.applied_default_workflow == workflow
    assert apply_result.applied_override_count == 0
    assert applied_manifest.pixelle_default_workflow == workflow
    assert applied_manifest.pixelle_segment_overrides == {}

    updated_manifest = run_step4(
        manifest=applied_manifest,
        output_manifest=str(project_root / "build" / f"manifest_step4_happy_{workflow}.json"),
        project_root=str(project_root),
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
    )

    resolved_segment = updated_manifest.segments[0]
    assert resolved_segment.visual_plan is not None
    assert resolved_segment.asset_refs

    ref = resolved_segment.asset_refs[0]
    assert ref.kind == "pixelle_video"
    assert ref.path
    assert Path(ref.path).exists()
    assert ref.fallback_reason_code is None
    assert ref.fallback_error_category is None
    assert ref.fallback_diagnostic is None
    assert resolved_segment.visual_plan.asset_path == ref.path

    resolved_target_name = Path(ref.path).resolve().name
    assert resolved_target_name.endswith(".test.mp4")
    assert workflow in resolved_target_name


@pytest.mark.parametrize("workflow", ["digital_human", "i2v", "action_transfer"])
def test_workbench_apply_to_step4_provider_path_happy_for_all_pixelle_workflows(
    workflow: PixelleWorkflow,
    tmp_path: Path,
    monkeypatch,
    provider_mode_env,
):
    segment = _make_segment(text=f"provider path {workflow}")
    project_root = tmp_path / f"project_provider_{workflow}"
    _seed_project(project_root, segment)

    _, applied_manifest = _apply_workbench_settings(
        project_root,
        default_workflow=workflow,
        overrides={},
    )

    monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: _build_provider_backed_adapter(name))

    updated_manifest = run_step4(
        manifest=applied_manifest,
        output_manifest=str(project_root / "build" / f"manifest_step4_provider_{workflow}.json"),
        project_root=str(project_root),
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
    )

    resolved_segment = updated_manifest.segments[0]
    ref = resolved_segment.asset_refs[0]
    assert ref.kind == "pixelle_video"
    assert ref.path
    assert Path(ref.path).exists()
    assert ref.fallback_reason_code is None
    assert ref.fallback_error_category is None
    assert ref.fallback_diagnostic is None

    resolved_name = Path(ref.path).resolve().name
    assert resolved_name.endswith(".mp4")
    assert ".test.mp4" not in resolved_name
    assert workflow in resolved_name


@pytest.mark.parametrize("workflow", ["digital_human", "i2v", "action_transfer"])
def test_provider_adapter_happy_path_includes_expected_metadata_keys(
    workflow: PixelleWorkflow,
    tmp_path: Path,
    provider_mode_env,
):
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    request: Any
    if workflow == "digital_human":
        request = DigitalHumanRequest(
            segment_key=f"{workflow}-segment#1",
            segment_text="provider metadata integration",
            segment_duration=2.4,
            project_root=str(tmp_path),
            output_dir=str(output_dir),
            avatar_id="avatar-a",
            voice_id="voice-b",
        )
    elif workflow == "i2v":
        request = I2VRequest(
            segment_key=f"{workflow}-segment#1",
            segment_text="provider metadata integration",
            segment_duration=2.4,
            project_root=str(tmp_path),
            output_dir=str(output_dir),
            input_image_path=str(tmp_path / "input" / "img.png"),
        )
    else:
        request = ActionTransferRequest(
            segment_key=f"{workflow}-segment#1",
            segment_text="provider metadata integration",
            segment_duration=2.4,
            project_root=str(tmp_path),
            output_dir=str(output_dir),
            reference_video_path=str(tmp_path / "input" / "ref.mp4"),
            target_image_path=str(tmp_path / "input" / "target.png"),
        )

    adapter: Any = _build_provider_backed_adapter(workflow)
    response = adapter.invoke(request)

    assert response.success is True
    assert response.output_path
    assert Path(response.output_path).exists()

    for key in [
        "provider_job_id",
        "provider_status",
        "request_id",
        "attempt",
        "run_seconds",
        "artifact_bytes",
        "artifact_format",
        "artifact_duration",
        "capability",
    ]:
        assert key in response.metadata

    assert response.metadata["provider_status"] == "SUCCEEDED"
    assert response.metadata["capability"] == workflow


def test_provider_mode_missing_api_key_fails_fast_with_explicit_config_error(
    provider_mode_missing_api_key,
):
    with pytest.raises(ProviderConfigError) as exc_info:
        load_provider_config()

    error_msg = str(exc_info.value)
    assert "PIXELLE_PROVIDER_API_KEY" in error_msg
    assert "ACTION REQUIRED" in error_msg
    assert "PIXELLE_TEST_MODE=1" in error_msg


@pytest.mark.parametrize("workflow", ["digital_human", "i2v", "action_transfer"])
def test_workbench_apply_to_step4_failure_fallback_with_error_fields_for_all_pixelle_workflows(
    workflow: PixelleWorkflow,
    tmp_path: Path,
    monkeypatch,
):
    segment = _make_segment(text=f"failure path {workflow}")
    project_root = tmp_path / f"project_failure_{workflow}"
    _seed_project(project_root, segment)

    apply_result, applied_manifest = _apply_workbench_settings(
        project_root,
        default_workflow="digital_human",
        overrides={segment.segment_key: workflow},
    )

    assert apply_result.applied_default_workflow == "digital_human"
    assert apply_result.applied_override_count == 1
    assert applied_manifest.pixelle_segment_overrides == {segment.segment_key: workflow}

    def fake_build_pixelle_request(
        capability: str,
        segment: Segment,
        project_root: str,
        output_dir: str,
        continuity_directive=None,
    ):
        common = {
            "segment_key": segment.segment_key,
            "segment_text": segment.text,
            "segment_duration": segment.duration,
            "project_root": project_root,
            "output_dir": output_dir,
        }
        if capability == "digital_human":
            return DigitalHumanRequest(**common)
        if capability == "i2v":
            return I2VRequest(**common, input_image_path=str(Path(project_root) / "bad_input.psd"))
        if capability == "action_transfer":
            return ActionTransferRequest(
                **common,
                reference_video_path=str(Path(project_root) / "bad_reference.wmv"),
                target_image_path=str(Path(project_root) / "target.png"),
            )
        raise AssertionError(f"unexpected capability: {capability}")

    monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
    monkeypatch.setattr("src.steps.step4_assets._build_pixelle_request", fake_build_pixelle_request)

    updated_manifest = run_step4(
        manifest=applied_manifest,
        output_manifest=str(project_root / "build" / f"manifest_step4_failure_{workflow}.json"),
        project_root=str(project_root),
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
    )

    resolved_segment = updated_manifest.segments[0]
    assert resolved_segment.visual_plan is not None
    assert resolved_segment.asset_refs

    ref = resolved_segment.asset_refs[0]
    
    # In test mode with bad inputs, all workflows fall back to template
    assert ref.kind == "template"
    assert ref.path
    assert Path(ref.path).exists()
    assert ref.path.endswith(".png")
    assert ref.fallback_reason_code == "PIXELLE_REQUEST_BUILD_FAILED"
    assert ref.fallback_error_category in ["VALIDATION", "CONFIGURATION"]
    assert ref.fallback_diagnostic is not None
    assert ref.fallback_diagnostic["reason_code"] == "PIXELLE_REQUEST_BUILD_FAILED"
    assert ref.fallback_diagnostic["category"] in ["VALIDATION", "CONFIGURATION"]
    assert ref.fallback_diagnostic["retryable"] is False
    assert ref.fallback_diagnostic["guidance"]
    assert ref.fallback_diagnostic["fallback_hint"]
    assert resolved_segment.visual_plan.asset_path == ref.path
