import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.models import AssetRef, AudioRef, GlobalStyle, Manifest, MotionConfig, RenderRef, Segment, VisualPlan
from src.steps.step3_visual_plan import run_step3
from src.steps.step4_assets import run_step4
from src.steps.step5_render import run_step5


VisualPlanType = Literal[
    "pdf_chart",
    "ui_mock",
    "broll",
    "ai_image",
    "ai_video_short",
    "kinetic_text",
    "template",
    "pixelle_digital_human",
    "pixelle_i2v",
    "pixelle_action_transfer",
]
PixelleWorkflow = Literal["digital_human", "i2v", "action_transfer"]


def _make_segment(
    index: int,
    text: str,
    vp_type: VisualPlanType,
    pixelle_workflow: PixelleWorkflow | None = None,
) -> Segment:
    content_key = Segment.compute_content_key(text)
    segment_key = Segment.compute_segment_key(content_key, 1)
    return Segment(
        segment_key=segment_key,
        content_key=content_key,
        index=index,
        start=float(index - 1),
        end=float(index),
        duration=1.0,
        text=text,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=1.0),
        visual_plan=VisualPlan(
            type=vp_type,
            pixelle_workflow=pixelle_workflow,
            prompt=f"old-prompt-{index}",
            motion=MotionConfig(preset="static", speed=1.0),
        ),
        plan_hash=f"old-plan-{index}",
        asset_refs=[AssetRef(kind="template", path=f"/tmp/asset-{index}.png", asset_hash=f"old-asset-{index}")],
        render_ref=RenderRef(segment_video_path=f"/tmp/segment-{index}.mp4", render_hash=f"old-render-{index}", status="ok"),
    )


def _make_manifest(*segments: Segment) -> Manifest:
    return Manifest(project_id="targeted-test", global_style=GlobalStyle(), segments=list(segments))


def test_step3_targeted_pixelle_only_selected_processed(monkeypatch, tmp_path: Path):
    seg1 = _make_segment(1, "pixelle one", "pixelle_digital_human", "digital_human")
    seg2 = _make_segment(2, "pixelle two", "pixelle_i2v", "i2v")
    manifest = _make_manifest(seg1, seg2)
    called_keys: list[str] = []

    def fake_generate(segment, global_style_asset_fields, prev_text, next_text, cache_dir, llm_model):
        called_keys.append(segment.segment_key)
        return (
            VisualPlan(
                type=segment.visual_plan.type if segment.visual_plan else "template",
                pixelle_workflow=segment.visual_plan.pixelle_workflow if segment.visual_plan else None,
                prompt=f"new-prompt-{segment.segment_key}",
                motion=MotionConfig(preset="static", speed=1.0),
            ),
            f"new-plan-{segment.index}",
        )

    monkeypatch.setattr("src.steps.step3_visual_plan.generate_visual_plan_for_segment", fake_generate)

    run_step3(
        manifest=manifest,
        output_manifest=str(tmp_path / "manifest_step3.json"),
        target_segment_keys=[seg1.segment_key],
    )

    assert called_keys == [seg1.segment_key]
    assert seg1.visual_plan is not None
    assert seg1.visual_plan.prompt.startswith("new-prompt-")
    assert seg2.visual_plan is not None
    assert seg2.visual_plan.prompt == "old-prompt-2"


def test_step4_targeted_pixelle_only_selected_processed(monkeypatch, tmp_path: Path):
    seg1 = _make_segment(1, "pixelle one", "pixelle_action_transfer", "action_transfer")
    seg2 = _make_segment(2, "pixelle two", "pixelle_i2v", "i2v")
    manifest = _make_manifest(seg1, seg2)
    called_keys: list[str] = []

    def fake_resolve(*, segment, **kwargs):
        called_keys.append(segment.segment_key)
        out = tmp_path / "assets" / "generated" / f"{segment.content_key}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"pixelle")
        segment.asset_refs = [AssetRef(kind="pixelle_video", path=str(out), asset_hash="new-asset")]
        if segment.visual_plan is not None:
            segment.visual_plan.asset_path = str(out)
        return segment

    monkeypatch.setattr("src.steps.step4_assets.resolve_asset_for_segment", fake_resolve)

    run_step4(
        manifest=manifest,
        output_manifest=str(tmp_path / "manifest_step4.json"),
        project_root=str(tmp_path),
        target_segment_keys=[seg1.segment_key],
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
    )

    assert called_keys == [seg1.segment_key]
    assert seg1.asset_refs[0].kind == "pixelle_video"
    assert seg2.asset_refs[0].kind == "template"


def test_step5_targeted_only_selected_rerender(monkeypatch, tmp_path: Path):
    seg1 = _make_segment(1, "target render", "pixelle_digital_human", "digital_human")
    seg2 = _make_segment(2, "untouched render", "broll")
    manifest = _make_manifest(seg1, seg2)
    called_keys: list[str] = []

    def fake_render(segment, output_path, style, max_retries):
        called_keys.append(segment.segment_key)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"rendered")
        return True

    monkeypatch.setattr("src.steps.step5_render.render_segment", fake_render)

    original_seg2_path = seg2.render_ref.segment_video_path
    run_step5(
        manifest=manifest,
        output_manifest=str(tmp_path / "manifest_step5.json"),
        segments_dir=str(tmp_path / "render" / "segments"),
        target_segment_keys=[seg1.segment_key],
    )

    assert called_keys == [seg1.segment_key]
    assert seg1.render_ref.status == "ok"
    assert seg2.render_ref.segment_video_path == original_seg2_path
    assert seg2.render_ref.render_hash == "old-render-2"


def test_step4_targeted_non_pixelle_flow_unchanged(monkeypatch, tmp_path: Path):
    seg1 = _make_segment(1, "legacy one", "broll")
    seg2 = _make_segment(2, "legacy two", "template")
    manifest = _make_manifest(seg1, seg2)
    called_keys: list[str] = []

    def fake_resolve(*, segment, **kwargs):
        called_keys.append(segment.segment_key)
        out = tmp_path / "assets" / "generated" / f"legacy-{segment.content_key}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"legacy")
        segment.asset_refs = [AssetRef(kind="template", path=str(out), asset_hash="legacy-asset")]
        if segment.visual_plan is not None:
            segment.visual_plan.asset_path = str(out)
        return segment

    monkeypatch.setattr("src.steps.step4_assets.resolve_asset_for_segment", fake_resolve)

    run_step4(
        manifest=manifest,
        output_manifest=str(tmp_path / "manifest_step4_legacy.json"),
        project_root=str(tmp_path),
        target_segment_keys=[seg2.segment_key],
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
    )

    assert called_keys == [seg2.segment_key]
    assert seg1.asset_refs[0].path == "/tmp/asset-1.png"
    assert seg2.asset_refs[0].path.endswith(".png")


def test_step3_invalid_target_keys_abort_without_writes(tmp_path: Path):
    seg = _make_segment(1, "single", "pixelle_digital_human", "digital_human")
    manifest = _make_manifest(seg)
    output_manifest = tmp_path / "step3_manifest.json"

    with _expect_value_error("Invalid target_segment_keys for Step3"):
        run_step3(
            manifest=manifest,
            output_manifest=str(output_manifest),
            target_segment_keys=["missing#1"],
        )

    assert not output_manifest.exists()
    assert not (tmp_path / "render" / "segments").exists()
    assert not (tmp_path / "assets" / "generated").exists()


def test_step4_invalid_target_keys_abort_without_writes(tmp_path: Path):
    seg = _make_segment(1, "single", "pixelle_digital_human", "digital_human")
    manifest = _make_manifest(seg)
    output_manifest = tmp_path / "step4_manifest.json"

    with _expect_value_error("Invalid target_segment_keys for Step4"):
        run_step4(
            manifest=manifest,
            output_manifest=str(output_manifest),
            project_root=str(tmp_path),
            target_segment_keys=["missing#1"],
        )

    assert not output_manifest.exists()
    assert not (tmp_path / "render" / "segments").exists()
    assert not (tmp_path / "assets" / "generated").exists()


def test_step5_invalid_target_keys_abort_without_writes(tmp_path: Path):
    seg = _make_segment(1, "single", "pixelle_digital_human", "digital_human")
    manifest = _make_manifest(seg)
    output_manifest = tmp_path / "step5_manifest.json"

    with _expect_value_error("Invalid target_segment_keys for Step5"):
        run_step5(
            manifest=manifest,
            output_manifest=str(output_manifest),
            segments_dir=str(tmp_path / "render" / "segments"),
            target_segment_keys=["missing#1"],
        )

    assert not output_manifest.exists()
    assert not (tmp_path / "render" / "segments").exists()
    assert not (tmp_path / "assets" / "generated").exists()


class _expect_value_error:
    def __init__(self, message: str):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f"Expected ValueError containing: {self.message}")
        if exc_type is not ValueError:
            return False
        if self.message not in str(exc):
            raise AssertionError(f"Expected '{self.message}' in '{exc}'")
        return True
