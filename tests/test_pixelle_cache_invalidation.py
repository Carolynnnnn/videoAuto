import sys
from pathlib import Path
from typing import Literal, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

pytest = __import__("pytest")
from src.core.models import AudioRef, Segment, VisualPlan
from src.steps.step4_assets import (
    _compute_effective_cache_hash,
    _resolve_effective_pixelle_workflow,
    resolve_asset_for_segment,
)

VisualPlanType = Literal[
    "pdf_chart", "ui_mock", "broll", "ai_image", "ai_video_short",
    "kinetic_text", "template", "pixelle_digital_human", "pixelle_i2v", "pixelle_action_transfer"
]
PixelleWorkflow = Literal["digital_human", "i2v", "action_transfer"]


def _make_segment(
    workflow: Optional[PixelleWorkflow] = None,
    plan_type: VisualPlanType = "template",
    plan_hash: str = "basehash1234",
) -> Segment:
    text = "Cache invalidation test segment"
    content_key = Segment.compute_content_key(text)
    if workflow:
        vp = VisualPlan(type=plan_type, pixelle_workflow=workflow)
    else:
        vp = VisualPlan(type=plan_type)
    return Segment(
        segment_key=Segment.compute_segment_key(content_key, 1),
        content_key=content_key,
        index=1,
        start=0.0,
        end=3.0,
        duration=3.0,
        text=text,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=3.0),
        visual_plan=vp,
        plan_hash=plan_hash,
    )


class TestEffectiveWorkflowResolution:
    def test_explicit_plan_workflow_takes_precedence(self):
        seg = _make_segment(workflow="digital_human")
        effective = _resolve_effective_pixelle_workflow(
            seg,
            manifest_default="i2v",
            manifest_overrides={seg.segment_key: "action_transfer"},
        )
        assert effective == "digital_human"

    def test_segment_override_takes_precedence_over_default(self):
        seg = _make_segment(workflow=None)
        effective = _resolve_effective_pixelle_workflow(
            seg,
            manifest_default="i2v",
            manifest_overrides={seg.segment_key: "action_transfer"},
        )
        assert effective == "action_transfer"

    def test_manifest_default_used_when_no_override(self):
        seg = _make_segment(workflow=None)
        effective = _resolve_effective_pixelle_workflow(
            seg,
            manifest_default="i2v",
            manifest_overrides={},
        )
        assert effective == "i2v"

    def test_type_inference_when_no_manifest_settings(self):
        seg = _make_segment(workflow=None, plan_type="pixelle_i2v")
        effective = _resolve_effective_pixelle_workflow(
            seg,
            manifest_default=None,
            manifest_overrides=None,
        )
        assert effective == "i2v"

    def test_none_when_non_pixelle_and_no_settings(self):
        seg = _make_segment(workflow=None, plan_type="template")
        effective = _resolve_effective_pixelle_workflow(
            seg,
            manifest_default=None,
            manifest_overrides=None,
        )
        assert effective is None


class TestCacheHashComputation:
    def test_same_hash_when_no_workflow(self):
        plan_hash = "abc123"
        h1 = _compute_effective_cache_hash(plan_hash, None)
        h2 = _compute_effective_cache_hash(plan_hash, None)
        assert h1 == h2 == plan_hash

    def test_different_hash_with_workflow(self):
        plan_hash = "abc123"
        h_none = _compute_effective_cache_hash(plan_hash, None)
        h_dh = _compute_effective_cache_hash(plan_hash, "digital_human")
        assert h_none != h_dh

    def test_different_workflows_produce_different_hashes(self):
        plan_hash = "abc123"
        h_dh = _compute_effective_cache_hash(plan_hash, "digital_human")
        h_i2v = _compute_effective_cache_hash(plan_hash, "i2v")
        h_at = _compute_effective_cache_hash(plan_hash, "action_transfer")
        assert len({h_dh, h_i2v, h_at}) == 3

    def test_same_workflow_produces_same_hash(self):
        plan_hash = "abc123"
        h1 = _compute_effective_cache_hash(plan_hash, "digital_human")
        h2 = _compute_effective_cache_hash(plan_hash, "digital_human")
        assert h1 == h2


class TestCacheHitWithSameSettings:
    def test_cache_hit_when_settings_unchanged(self, tmp_path: Path):
        project_root = tmp_path
        generated_dir = tmp_path / "assets" / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)

        seg = _make_segment(workflow=None, plan_hash="samehash")
        effective_workflow = "digital_human"
        cache_hash = _compute_effective_cache_hash("samehash", effective_workflow)

        cached_file = generated_dir / f"{seg.content_key}_{cache_hash}.png"
        cached_file.write_bytes(b"x" * 2048)

        resolved = resolve_asset_for_segment(
            segment=seg,
            project_root=str(project_root),
            generated_dir=str(generated_dir),
            library_dir=str(tmp_path / "library"),
            pixelle_default_workflow="digital_human",
            pixelle_segment_overrides={},
        )

        assert resolved.asset_refs[0].kind == "cached"
        assert cache_hash in resolved.asset_refs[0].path


class TestCacheInvalidationOnWorkflowChange:
    def test_cache_miss_when_workflow_changes(self, tmp_path: Path):
        project_root = tmp_path
        generated_dir = tmp_path / "assets" / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)

        seg = _make_segment(workflow=None, plan_hash="samehash")
        old_cache_hash = _compute_effective_cache_hash("samehash", "digital_human")

        cached_file = generated_dir / f"{seg.content_key}_{old_cache_hash}.png"
        cached_file.write_bytes(b"x" * 2048)

        resolved = resolve_asset_for_segment(
            segment=seg,
            project_root=str(project_root),
            generated_dir=str(generated_dir),
            library_dir=str(tmp_path / "library"),
            pixelle_default_workflow="i2v",
            pixelle_segment_overrides={},
        )

        assert resolved.asset_refs[0].kind != "cached"

    def test_cache_miss_when_segment_override_changes(self, tmp_path: Path):
        project_root = tmp_path
        generated_dir = tmp_path / "assets" / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)

        seg = _make_segment(workflow=None, plan_hash="overridehash")
        old_cache_hash = _compute_effective_cache_hash("overridehash", "digital_human")

        cached_file = generated_dir / f"{seg.content_key}_{old_cache_hash}.png"
        cached_file.write_bytes(b"x" * 2048)

        resolved = resolve_asset_for_segment(
            segment=seg,
            project_root=str(project_root),
            generated_dir=str(generated_dir),
            library_dir=str(tmp_path / "library"),
            pixelle_default_workflow="digital_human",
            pixelle_segment_overrides={seg.segment_key: "action_transfer"},
        )

        assert resolved.asset_refs[0].kind != "cached"


class TestLegacyCachingUnchanged:
    def test_non_pixelle_cache_hit_uses_plan_hash_only(self, tmp_path: Path):
        project_root = tmp_path
        generated_dir = tmp_path / "assets" / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)

        seg = _make_segment(workflow=None, plan_type="template", plan_hash="legacyhash")

        cached_file = generated_dir / f"{seg.content_key}_legacyhash.png"
        cached_file.write_bytes(b"x" * 2048)

        resolved = resolve_asset_for_segment(
            segment=seg,
            project_root=str(project_root),
            generated_dir=str(generated_dir),
            library_dir=str(tmp_path / "library"),
            pixelle_default_workflow=None,
            pixelle_segment_overrides={},
        )

        assert resolved.asset_refs[0].kind == "cached"
        assert "legacyhash" in resolved.asset_refs[0].path

    def test_non_pixelle_unaffected_by_manifest_pixelle_settings(self, tmp_path: Path):
        project_root = tmp_path
        generated_dir = tmp_path / "assets" / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)

        seg = _make_segment(workflow=None, plan_type="template", plan_hash="nonpixelle")

        cached_file = generated_dir / f"{seg.content_key}_nonpixelle.png"
        cached_file.write_bytes(b"x" * 2048)

        resolved = resolve_asset_for_segment(
            segment=seg,
            project_root=str(project_root),
            generated_dir=str(generated_dir),
            library_dir=str(tmp_path / "library"),
            pixelle_default_workflow=None,
            pixelle_segment_overrides={"other_segment": "digital_human"},
        )

        assert resolved.asset_refs[0].kind == "cached"


class TestManifestWorkflowActuallyInvoked:
    def test_manifest_default_workflow_invoked_not_visual_plan(self, tmp_path: Path, monkeypatch):
        project_root = tmp_path
        generated_dir = tmp_path / "assets" / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)

        seg = _make_segment(workflow=None, plan_type="template", plan_hash="invoketest")

        invoked_capabilities = []

        def fake_resolve_pixelle_asset(segment, project_root, generated_dir, effective_capability=None, vendor_preference=None, continuity_directive=None):
            invoked_capabilities.append(effective_capability)
            return None, None, "FAKE_FAIL", "EXECUTION"

        from src.steps import step4_assets
        monkeypatch.setattr(step4_assets, "_resolve_pixelle_asset", fake_resolve_pixelle_asset)

        resolve_asset_for_segment(
            segment=seg,
            project_root=str(project_root),
            generated_dir=str(generated_dir),
            library_dir=str(tmp_path / "library"),
            pixelle_default_workflow="i2v",
            pixelle_segment_overrides={},
        )

        assert invoked_capabilities == ["i2v"]

    def test_segment_override_workflow_invoked_over_default(self, tmp_path: Path, monkeypatch):
        project_root = tmp_path
        generated_dir = tmp_path / "assets" / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)

        seg = _make_segment(workflow=None, plan_type="template", plan_hash="overridetest")

        invoked_capabilities = []

        def fake_resolve_pixelle_asset(segment, project_root, generated_dir, effective_capability=None, vendor_preference=None, continuity_directive=None):
            invoked_capabilities.append(effective_capability)
            return None, None, "FAKE_FAIL", "EXECUTION"

        from src.steps import step4_assets
        monkeypatch.setattr(step4_assets, "_resolve_pixelle_asset", fake_resolve_pixelle_asset)

        resolve_asset_for_segment(
            segment=seg,
            project_root=str(project_root),
            generated_dir=str(generated_dir),
            library_dir=str(tmp_path / "library"),
            pixelle_default_workflow="digital_human",
            pixelle_segment_overrides={seg.segment_key: "action_transfer"},
        )

        assert invoked_capabilities == ["action_transfer"]

    def test_visual_plan_workflow_invoked_when_set(self, tmp_path: Path, monkeypatch):
        project_root = tmp_path
        generated_dir = tmp_path / "assets" / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)

        seg = _make_segment(workflow="digital_human", plan_type="template", plan_hash="plantest")

        invoked_capabilities = []

        def fake_resolve_pixelle_asset(segment, project_root, generated_dir, effective_capability=None, vendor_preference=None, continuity_directive=None):
            invoked_capabilities.append(effective_capability)
            return None, None, "FAKE_FAIL", "EXECUTION"

        from src.steps import step4_assets
        monkeypatch.setattr(step4_assets, "_resolve_pixelle_asset", fake_resolve_pixelle_asset)

        resolve_asset_for_segment(
            segment=seg,
            project_root=str(project_root),
            generated_dir=str(generated_dir),
            library_dir=str(tmp_path / "library"),
            pixelle_default_workflow="i2v",
            pixelle_segment_overrides={seg.segment_key: "action_transfer"},
        )

        assert invoked_capabilities == ["digital_human"]
