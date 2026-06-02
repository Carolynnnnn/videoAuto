import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pixelle_snapshot.adapters.contracts import AdapterError, ErrorCategory
from src.core.models import AssetRef, AudioRef, Manifest, Segment, VisualPlan
from src.steps.step4_assets import (
    build_top6_ai_allocation_map,
    compute_segment_semantic_priority_score,
    resolve_asset_for_segment,
    run_step4,
)


def _make_segment() -> Segment:
    text = "Pixelle routing segment"
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
        plan_hash="pixellehash1234",
    )


def test_step4_pixelle_selected_route_happy_path(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    routed_capabilities: list[str] = []

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
                },
            )()

    def fake_get_adapter(name: str, **kwargs):
        routed_capabilities.append(name)
        return FakeAdapter()

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", fake_get_adapter)

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

    assert routed_capabilities == ["digital_human"]
    assert resolved.asset_refs[0].kind == "pixelle_video"
    assert resolved.asset_refs[0].fallback_reason_code is None
    assert resolved.asset_refs[0].fallback_error_category is None
    assert resolved.visual_plan is not None
    assert resolved.visual_plan.asset_path is not None
    assert resolved.visual_plan.asset_path.endswith(".mp4")


def test_step4_pixelle_failure_records_reason_and_falls_back(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    class FailingAdapter:
        def invoke(self, request):
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(
                        category=ErrorCategory.PROVIDER,
                        message="provider failed",
                    ),
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
    assert resolved.asset_refs[0].fallback_reason_code == "PIXELLE_INVOCATION_FAILED"
    assert resolved.asset_refs[0].fallback_error_category == "PROVIDER"
    assert resolved.visual_plan is not None
    assert resolved.visual_plan.asset_path is not None


def test_step4_route_diagnostics_success_ai_preferred(monkeypatch, tmp_path: Path, caplog):
    """Route diagnostics include mode, provider, route for successful AI path."""
    import logging
    caplog.set_level(logging.INFO)
    
    monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

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
        material_mode="ai_preferred",
    )

    assert resolved.asset_refs[0].kind == "pixelle_video"
    
    log_messages = [rec.message for rec in caplog.records if "Route diagnostic" in rec.message]
    assert any("route=pixelle_video" in msg for msg in log_messages)
    assert any("mode=ai_preferred" in msg for msg in log_messages)
    assert any("SUCCESS" in msg for msg in log_messages)
    assert any("FINAL" in msg for msg in log_messages)
    assert any("selected=pixelle_video" in msg for msg in log_messages)


def test_step4_route_diagnostics_error_provider_failure(monkeypatch, tmp_path: Path, caplog):
    """Route diagnostics include reason_code and error_category on provider failure."""
    import logging
    caplog.set_level(logging.INFO)
    
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    class FailingAdapter:
        def invoke(self, request):
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(
                        category=ErrorCategory.TIMEOUT,
                        message="timeout",
                    ),
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
        material_mode="auto",
    )

    assert resolved.asset_refs[0].kind == "template"
    assert resolved.asset_refs[0].fallback_reason_code is not None
    assert resolved.asset_refs[0].fallback_error_category is not None
    
    log_messages = [rec.message for rec in caplog.records if "Route diagnostic" in rec.message]
    assert any("EXHAUSTED" in msg for msg in log_messages)
    assert any("reason=" in msg and "category=" in msg for msg in log_messages)
    assert any("mode=auto" in msg for msg in log_messages)
    assert any("FINAL" in msg for msg in log_messages)
    assert any("selected=template" in msg for msg in log_messages)
    assert resolved.visual_plan is not None
    assert resolved.visual_plan.asset_path is not None
    assert resolved.visual_plan.asset_path.endswith(".png")


# ════════════════════════════════════════════════════════════════════════════
# Strict ai_only mode tests - blocks Pexels routes, returns ai_only_exhausted
# ════════════════════════════════════════════════════════════════════════════


def test_step4_ai_only_blocks_pexels_video_route(monkeypatch, tmp_path: Path):
    """ai_only mode: Pexels video route is blocked even when Pixelle fails.
    
    Instead of falling back to Pexels video, ai_only mode should return
    ai_only_exhausted with explicit diagnostic.
    """
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    
    pexels_cache = tmp_path / "assets" / "pexels_cache" / "videos"
    pexels_cache.mkdir(parents=True, exist_ok=True)
    pexels_video_path = pexels_cache / "fallback_video.mp4"
    pexels_video_path.write_bytes(b"pexels-video-content" * 1000)

    class FailingPixelleAdapter:
        def invoke(self, request):
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(
                        category=ErrorCategory.PROVIDER,
                        message="Pixelle provider failed",
                    ),
                },
            )()

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FailingPixelleAdapter())

    pexels_called = []
    def fake_pexels_video(api_key, query, output_dir, min_duration):
        pexels_called.append("pexels_video")
        return str(pexels_video_path)
    monkeypatch.setattr("src.steps.step4_assets.fetch_pexels_video", fake_pexels_video)

    seg = _make_segment_with_keywords()
    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="test-pexels-key",
        enable_pexels_video=True,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="ai_only",
    )

    assert pexels_called == [], "Pexels video should not be called in ai_only mode"
    assert resolved.asset_refs[0].kind == "ai_only_exhausted"
    assert resolved.asset_refs[0].fallback_reason_code == "AI_ONLY_ROUTES_EXHAUSTED"
    assert resolved.asset_refs[0].fallback_error_category == "POLICY"
    
    diagnostic = resolved.asset_refs[0].fallback_diagnostic
    assert diagnostic is not None
    assert diagnostic.get("reason_code") == "AI_ONLY_ROUTES_EXHAUSTED"
    assert diagnostic.get("category") == "POLICY"
    assert "ai_only" in diagnostic.get("guidance", "").lower()


def test_step4_ai_only_blocks_pexels_photo_route(monkeypatch, tmp_path: Path):
    """ai_only mode: Pexels photo route is blocked even when Pixelle fails.
    
    Instead of falling back to Pexels photo, ai_only mode should return
    ai_only_exhausted with explicit diagnostic.
    """
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    class FailingPixelleAdapter:
        def invoke(self, request):
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(
                        category=ErrorCategory.PROVIDER,
                        message="Pixelle provider failed",
                    ),
                },
            )()

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FailingPixelleAdapter())

    pexels_called = []
    def fake_pexels_photo(api_key, query, output_path, width, height):
        pexels_called.append("pexels_photo")
        Path(output_path).write_bytes(b"fake-pexels-photo")
        return output_path
    monkeypatch.setattr("src.steps.step4_assets.fetch_pexels_photo", fake_pexels_photo)

    seg = _make_segment_with_keywords()
    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="test-pexels-key",
        enable_pexels_video=False,
        enable_pexels_photo=True,
        enable_ai_image=False,
        material_mode="ai_only",
    )

    assert pexels_called == [], "Pexels photo should not be called in ai_only mode"
    assert resolved.asset_refs[0].kind == "ai_only_exhausted"
    assert resolved.asset_refs[0].fallback_reason_code == "AI_ONLY_ROUTES_EXHAUSTED"
    assert resolved.asset_refs[0].fallback_error_category == "POLICY"


def test_step4_ai_only_returns_exhausted_not_template_on_all_failures(monkeypatch, tmp_path: Path):
    """ai_only mode: When all AI routes fail, returns ai_only_exhausted NOT template.
    
    This is the critical strict guarantee - ai_only should NEVER silently
    degrade to template as a successful output.
    """
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    class FailingPixelleAdapter:
        def invoke(self, request):
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(
                        category=ErrorCategory.PROVIDER,
                        message="All AI providers exhausted",
                    ),
                },
            )()

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FailingPixelleAdapter())

    # Mock template to track if it gets called (it should NOT be called in ai_only)
    template_called = []
    def fake_template(output_path: str, width: int, height: int, text: str):
        template_called.append("template")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template")
        return str(path)
    monkeypatch.setattr("src.steps.step4_assets.generate_template_asset", fake_template)

    # Mock AI image to also fail
    def fake_ai_image(*args, **kwargs):
        return None
    monkeypatch.setattr("src.steps.step4_assets.generate_ai_image", fake_ai_image)

    seg = _make_segment_with_keywords()
    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=True,  # AI image enabled but will fail
        material_mode="ai_only",
    )

    # Template should NOT have been called
    assert template_called == [], "Template should not be called in ai_only mode"
    
    # Result should be ai_only_exhausted with explicit failure
    assert resolved.asset_refs[0].kind == "ai_only_exhausted"
    assert resolved.asset_refs[0].path == ""  # No valid path
    assert resolved.asset_refs[0].fallback_reason_code == "AI_ONLY_ROUTES_EXHAUSTED"
    assert resolved.asset_refs[0].fallback_error_category == "POLICY"
    
    # Diagnostic should include original failure info
    diagnostic = resolved.asset_refs[0].fallback_diagnostic
    assert diagnostic is not None
    assert diagnostic.get("retryable") is False
    assert diagnostic.get("fallback_hint") is None  # No fallback available


def test_step4_ai_only_diagnostic_includes_original_failure(monkeypatch, tmp_path: Path):
    """ai_only mode: Diagnostic includes original Pixelle failure details."""
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    class FailingPixelleAdapter:
        def invoke(self, request):
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(
                        category=ErrorCategory.RESOURCE,
                        message="Resource limit exceeded - try again later",
                    ),
                },
            )()

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FailingPixelleAdapter())

    seg = _make_segment_with_keywords()
    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="ai_only",
    )

    assert resolved.asset_refs[0].kind == "ai_only_exhausted"
    
    # Diagnostic should include the original Pixelle failure
    diagnostic = resolved.asset_refs[0].fallback_diagnostic
    assert diagnostic is not None
    assert "original_failure" in diagnostic
    original = diagnostic["original_failure"]
    assert original.get("category") == "RESOURCE"
    assert "resource limit" in original.get("guidance", "").lower()




def test_step4_ai_only_cache_rejects_non_ai_cached_asset(tmp_path: Path):
    """ai_only mode: cached non-AI assets must not be silently accepted."""
    from src.steps import step4_assets

    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    # Build a segment with no Pixelle workflow and no AI image enabled.
    text_content = "ai_only cache reject non-ai"
    content_key = Segment.compute_content_key(text_content)
    plan_hash = "cachehash_non_ai"
    seg = Segment(
        segment_key=Segment.compute_segment_key(content_key, 1),
        content_key=content_key,
        index=1,
        start=0.0,
        end=4.0,
        duration=4.0,
        text=text_content,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=4.0),
        visual_plan=VisualPlan(type="template", keywords=["demo"], prompt=""),
        plan_hash=plan_hash,
    )

    cache_path = Path(step4_assets._asset_cache_path(str(generated_dir), content_key, plan_hash, "png"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"X" * 2048)

    # Mark this cached asset as coming from a non-AI route (template).
    meta_path = Path(step4_assets._asset_cache_meta_path(str(cache_path)))
    meta_path.write_text(
        '{"kind": "template", "material_mode": "auto", "is_ai_generated": false}',
        encoding="utf-8",
    )

    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="ai_only",
    )

    # Cache must be ignored; with no AI workflow configured, ai_only fails fast on precondition.
    assert resolved.asset_refs[0].kind == "ai_only_exhausted"
    assert resolved.asset_refs[0].fallback_reason_code == "AI_ONLY_MISSING_WORKFLOW"
    assert resolved.asset_refs[0].fallback_error_category == "CONFIG"


def test_step4_ai_only_cache_allows_ai_cached_asset(tmp_path: Path):
    """ai_only mode: cached AI assets remain reusable."""
    from src.steps import step4_assets

    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    text_content = "ai_only cache allow ai"
    content_key = Segment.compute_content_key(text_content)
    plan_hash = "cachehash_ai"
    seg = Segment(
        segment_key=Segment.compute_segment_key(content_key, 1),
        content_key=content_key,
        index=1,
        start=0.0,
        end=4.0,
        duration=4.0,
        text=text_content,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=4.0),
        visual_plan=VisualPlan(type="template", keywords=["demo"], prompt=""),
        plan_hash=plan_hash,
    )

    cache_path = Path(step4_assets._asset_cache_path(str(generated_dir), content_key, plan_hash, "png"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"Y" * 2048)

    # Mark this cached asset as AI-generated.
    meta_path = Path(step4_assets._asset_cache_meta_path(str(cache_path)))
    meta_path.write_text(
        '{"kind": "ai_image", "material_mode": "auto", "is_ai_generated": true}',
        encoding="utf-8",
    )

    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="ai_only",
    )

    assert resolved.asset_refs[0].kind == "cached"
    assert resolved.asset_refs[0].path.endswith(f"{content_key}_{plan_hash}.png")

def test_step4_ai_preferred_still_allows_pexels_fallback(monkeypatch, tmp_path: Path):
    """ai_preferred mode: Pexels fallback is still allowed (contrast with ai_only).
    
    This test ensures we didn't break ai_preferred mode while implementing
    strict ai_only behavior.
    """
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    
    pexels_cache = tmp_path / "assets" / "pexels_cache" / "videos"
    pexels_cache.mkdir(parents=True, exist_ok=True)
    pexels_video_path = pexels_cache / "fallback_video.mp4"
    pexels_video_path.write_bytes(b"pexels-video-content" * 1000)

    class FailingPixelleAdapter:
        def invoke(self, request):
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(
                        category=ErrorCategory.PROVIDER,
                        message="Pixelle provider failed",
                    ),
                },
            )()

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FailingPixelleAdapter())

    pexels_called = []
    def fake_pexels_video(keywords, visual_type, segment_duration, download_dir, cache_dir, api_key, aspect_ratio="9:16"):
        pexels_called.append("pexels_video")
        return str(pexels_video_path)
    monkeypatch.setattr("src.steps.step4_assets.fetch_pexels_video", fake_pexels_video)

    seg = _make_segment_with_keywords()
    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="test-pexels-key",
        enable_pexels_video=True,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="ai_preferred",
    )

    assert pexels_called == ["pexels_video"], "Pexels should be allowed in ai_preferred mode"
    assert resolved.asset_refs[0].kind == "pexels_video"


def test_step4_auto_mode_still_allows_pexels_fallback(monkeypatch, tmp_path: Path):
    """auto mode: Pexels fallback is still allowed (contrast with ai_only).
    
    This test ensures we didn't break auto mode while implementing
    strict ai_only behavior.
    """
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    
    pexels_cache = tmp_path / "assets" / "pexels_cache" / "videos"
    pexels_cache.mkdir(parents=True, exist_ok=True)
    pexels_video_path = pexels_cache / "fallback_video.mp4"
    pexels_video_path.write_bytes(b"pexels-video-content" * 1000)

    class FailingPixelleAdapter:
        def invoke(self, request):
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(
                        category=ErrorCategory.PROVIDER,
                        message="Pixelle provider failed",
                    ),
                },
            )()

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FailingPixelleAdapter())

    pexels_called = []
    def fake_pexels_video(keywords, visual_type, segment_duration, download_dir, cache_dir, api_key, aspect_ratio="9:16"):
        pexels_called.append("pexels_video")
        return str(pexels_video_path)
    monkeypatch.setattr("src.steps.step4_assets.fetch_pexels_video", fake_pexels_video)

    seg = _make_segment_with_keywords()
    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="test-pexels-key",
        enable_pexels_video=True,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="auto",
    )

    assert pexels_called == ["pexels_video"], "Pexels should be allowed in auto mode"
    assert resolved.asset_refs[0].kind == "pexels_video"


# ─────────────────────────────────────────────
# Task 7: ai_preferred Route Branch Tests
# ─────────────────────────────────────────────


def test_ai_preferred_attempts_pixelle_before_pexels(monkeypatch, tmp_path: Path):
    """ai_preferred mode attempts Pixelle route BEFORE Pexels routes."""
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    
    route_attempts: list[str] = []

    class FakePixelleAdapter:
        def invoke(self, request):
            route_attempts.append("pixelle")
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
                },
            )()

    def fake_fetch_pexels_video(*args, **kwargs):
        route_attempts.append("pexels_video")
        return None

    def fake_fetch_pexels_photo(*args, **kwargs):
        route_attempts.append("pexels_photo")
        return None

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FakePixelleAdapter())
    monkeypatch.setattr("src.steps.step4_assets.fetch_pexels_video", fake_fetch_pexels_video)
    monkeypatch.setattr("src.steps.step4_assets.fetch_pexels_photo", fake_fetch_pexels_photo)

    seg = _make_segment()
    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="test_key",
        enable_pexels_video=True,
        enable_pexels_photo=True,
        enable_ai_image=False,
        material_mode="ai_preferred",
    )

    assert resolved.asset_refs[0].kind == "pixelle_video"
    assert route_attempts == ["pixelle"]


def _make_segment_with_keywords() -> Segment:
    text = "Pixelle routing segment with keywords"
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
        visual_plan=VisualPlan(
            type="pixelle_digital_human",
            pixelle_workflow="digital_human",
            keywords=["technology", "digital"],
        ),
        plan_hash="pixellekwdshash1234",
    )


def test_ai_preferred_pixelle_failure_falls_back_to_pexels_video(monkeypatch, tmp_path: Path):
    """ai_preferred mode: On Pixelle failure, falls back to Pexels video (non-AI fallback)."""
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    pexels_cache = tmp_path / "assets" / "pexels_cache" / "videos"
    pexels_cache.mkdir(parents=True, exist_ok=True)
    pexels_video_path = pexels_cache / "fallback_video.mp4"
    pexels_video_path.write_bytes(b"pexels-video-content" * 1000)

    class FailingPixelleAdapter:
        def invoke(self, request):
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(
                        category=ErrorCategory.PROVIDER,
                        message="Pixelle provider failed",
                    ),
                },
            )()

    def fake_fetch_pexels_video(*args, **kwargs):
        return str(pexels_video_path)

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FailingPixelleAdapter())
    monkeypatch.setattr("src.steps.step4_assets.fetch_pexels_video", fake_fetch_pexels_video)

    seg = _make_segment_with_keywords()
    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="test_key",
        enable_pexels_video=True,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="ai_preferred",
    )

    assert resolved.asset_refs[0].kind == "pexels_video"
    assert resolved.asset_refs[0].fallback_reason_code == "PIXELLE_INVOCATION_FAILED"
    assert resolved.asset_refs[0].fallback_error_category == "PROVIDER"


def test_ai_preferred_pixelle_failure_falls_back_to_pexels_photo(monkeypatch, tmp_path: Path):
    """ai_preferred mode: On Pixelle failure and no video, falls back to Pexels photo."""
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    pexels_photo_path = tmp_path / "assets" / "pexels_cache" / "photos" / "fallback_photo.jpg"
    pexels_photo_path.parent.mkdir(parents=True, exist_ok=True)
    pexels_photo_path.write_bytes(b"pexels-photo-content" * 100)

    class FailingPixelleAdapter:
        def invoke(self, request):
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(
                        category=ErrorCategory.TIMEOUT,
                        message="Pixelle timed out",
                    ),
                },
            )()

    def fake_fetch_pexels_video(*args, **kwargs):
        return None

    def fake_fetch_pexels_photo(*args, **kwargs):
        return str(pexels_photo_path)

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FailingPixelleAdapter())
    monkeypatch.setattr("src.steps.step4_assets.fetch_pexels_video", fake_fetch_pexels_video)
    monkeypatch.setattr("src.steps.step4_assets.fetch_pexels_photo", fake_fetch_pexels_photo)

    seg = _make_segment_with_keywords()
    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="test_key",
        enable_pexels_video=True,
        enable_pexels_photo=True,
        enable_ai_image=False,
        material_mode="ai_preferred",
    )

    assert resolved.asset_refs[0].kind == "pexels_photo"
    assert resolved.asset_refs[0].fallback_reason_code == "PIXELLE_INVOCATION_FAILED"
    assert resolved.asset_refs[0].fallback_error_category == "TIMEOUT"


def test_ai_preferred_fallback_preserves_diagnostic_fields(monkeypatch, tmp_path: Path):
    """ai_preferred mode: Fallback preserves full diagnostic with reason_code/category."""
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    class FailingPixelleAdapter:
        def invoke(self, request):
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(
                        category=ErrorCategory.EXECUTION,
                        message="Artifact corrupted",
                        details={"reason_code": "PIXELLE_ARTIFACT_CORRUPTED"},
                    ),
                },
            )()

    def fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template")
        return str(path)

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FailingPixelleAdapter())
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
        material_mode="ai_preferred",
    )

    ref = resolved.asset_refs[0]
    assert ref.kind == "template"
    assert ref.fallback_reason_code == "PIXELLE_ARTIFACT_CORRUPTED"
    assert ref.fallback_error_category == "EXECUTION"
    assert ref.fallback_diagnostic is not None
    assert ref.fallback_diagnostic["category"] == "EXECUTION"
    assert ref.fallback_diagnostic["reason_code"] == "PIXELLE_ARTIFACT_CORRUPTED"


def test_ai_preferred_no_pixelle_workflow_tries_pexels_first(monkeypatch, tmp_path: Path):
    """ai_preferred mode: With no Pixelle workflow, Pexels routes are attempted first."""
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    pexels_cache = tmp_path / "assets" / "pexels_cache" / "videos"
    pexels_cache.mkdir(parents=True, exist_ok=True)
    pexels_video_path = pexels_cache / "no_pixelle_video.mp4"
    pexels_video_path.write_bytes(b"pexels-video-content" * 1000)

    def fake_fetch_pexels_video(*args, **kwargs):
        return str(pexels_video_path)

    monkeypatch.setattr("src.steps.step4_assets.fetch_pexels_video", fake_fetch_pexels_video)

    text = "No pixelle workflow segment"
    content_key = Segment.compute_content_key(text)
    seg = Segment(
        segment_key=Segment.compute_segment_key(content_key, 1),
        content_key=content_key,
        index=1,
        start=0.0,
        end=4.0,
        duration=4.0,
        text=text,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=4.0),
        visual_plan=VisualPlan(type="broll", keywords=["technology"]),
        plan_hash="nopixellehash1234",
    )

    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="test_key",
        enable_pexels_video=True,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="ai_preferred",
    )

    assert resolved.asset_refs[0].kind == "pexels_video"
    assert resolved.asset_refs[0].fallback_reason_code is None


def test_ai_preferred_auto_mode_comparison(monkeypatch, tmp_path: Path):
    """ai_preferred and auto modes have different route ordering when Pixelle workflow is set."""
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    
    route_attempts_ai_preferred: list[str] = []
    route_attempts_auto: list[str] = []

    class TrackingPixelleAdapter:
        def __init__(self, route_list):
            self.route_list = route_list
            
        def invoke(self, request):
            self.route_list.append("pixelle")
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
                },
            )()

    def create_fake_pexels_video(route_list):
        def fake_fetch(*args, **kwargs):
            route_list.append("pexels_video")
            video_path = tmp_path / "assets" / "pexels_cache" / "videos" / "test.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(b"pexels-video" * 1000)
            return str(video_path)
        return fake_fetch

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: TrackingPixelleAdapter(route_attempts_ai_preferred))
    monkeypatch.setattr("src.steps.step4_assets.fetch_pexels_video", create_fake_pexels_video(route_attempts_ai_preferred))

    seg_ai_preferred = _make_segment_with_keywords()
    resolve_asset_for_segment(
        segment=seg_ai_preferred,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="test_key",
        enable_pexels_video=True,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="ai_preferred",
    )

    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: TrackingPixelleAdapter(route_attempts_auto))
    monkeypatch.setattr("src.steps.step4_assets.fetch_pexels_video", create_fake_pexels_video(route_attempts_auto))

    seg_auto = _make_segment_with_keywords()
    seg_auto.segment_key = f"{seg_auto.segment_key}-auto"
    resolve_asset_for_segment(
        segment=seg_auto,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="test_key",
        enable_pexels_video=True,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="auto",
    )

    assert route_attempts_ai_preferred[0] == "pixelle"
    assert route_attempts_auto[0] == "pexels_video"


def test_minimax_legacy_fallback_happy_ai_only_chain(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    routed_capabilities: list[str] = []
    pexels_called: list[str] = []

    class MinimaxFailAdapter:
        def invoke(self, request):
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(
                        category=ErrorCategory.PROVIDER,
                        message="minimax provider failed",
                    ),
                },
            )()

    class LegacySuccessAdapter:
        def invoke(self, request):
            output_path = Path(request.output_dir) / f"legacy_{request.segment_key}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"legacy-video")
            return type(
                "Resp",
                (),
                {
                    "success": True,
                    "output_path": str(output_path),
                    "error": None,
                },
            )()

    def fake_get_adapter(name: str, **kwargs):
        routed_capabilities.append(name)
        if name == "minimax_video":
            return MinimaxFailAdapter()
        if name == "digital_human":
            return LegacySuccessAdapter()
        raise AssertionError(f"unexpected capability: {name}")

    def fake_fetch_pexels_video(*args, **kwargs):
        pexels_called.append("pexels_video")
        return None

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", fake_get_adapter)
    monkeypatch.setattr("src.steps.step4_assets.fetch_pexels_video", fake_fetch_pexels_video)

    seg = _make_segment_with_keywords()
    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="test_key",
        enable_pexels_video=True,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="ai_only",
    )

    assert routed_capabilities == ["minimax_video", "digital_human"]
    assert pexels_called == []
    assert resolved.asset_refs[0].kind == "pixelle_video"
    assert resolved.asset_refs[0].fallback_reason_code == "PIXELLE_INVOCATION_FAILED"
    assert resolved.asset_refs[0].fallback_error_category == "PROVIDER"
    assert resolved.asset_refs[0].fallback_diagnostic is not None
    attempts = resolved.asset_refs[0].fallback_diagnostic["provider_attempts"]
    assert attempts[0]["provider_stage"] == "minimax_primary"
    assert attempts[0]["routed_capability"] == "minimax_video"
    assert attempts[1]["provider_stage"] == "legacy_secondary"
    assert attempts[1]["routed_capability"] == "digital_human"
    assert attempts[1]["success"] is True


def test_minimax_legacy_fallback_error_ai_only_chain_exhausted(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    routed_capabilities: list[str] = []

    class MinimaxFailAdapter:
        def invoke(self, request):
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(
                        category=ErrorCategory.PROVIDER,
                        message="minimax unavailable",
                    ),
                },
            )()

    class LegacyFailAdapter:
        def invoke(self, request):
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(
                        category=ErrorCategory.RESOURCE,
                        message="legacy quota exhausted",
                        details={"reason_code": "LEGACY_QUOTA_EXHAUSTED"},
                    ),
                },
            )()

    def fake_get_adapter(name: str, **kwargs):
        routed_capabilities.append(name)
        if name == "minimax_video":
            return MinimaxFailAdapter()
        if name == "digital_human":
            return LegacyFailAdapter()
        raise AssertionError(f"unexpected capability: {name}")

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", fake_get_adapter)

    seg = _make_segment_with_keywords()
    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="ai_only",
    )

    assert routed_capabilities == ["minimax_video", "digital_human"]
    assert resolved.asset_refs[0].kind == "ai_only_exhausted"
    assert resolved.asset_refs[0].fallback_reason_code == "AI_ONLY_ROUTES_EXHAUSTED"
    assert resolved.asset_refs[0].fallback_error_category == "POLICY"
    assert resolved.asset_refs[0].fallback_diagnostic is not None

    original = resolved.asset_refs[0].fallback_diagnostic["original_failure"]
    assert original["reason_code"] == "LEGACY_QUOTA_EXHAUSTED"
    assert original["category"] == "RESOURCE"
    assert original["provider_chain_exhausted"] is True
    attempts = original["provider_attempts"]
    assert attempts[0]["provider_stage"] == "minimax_primary"
    assert attempts[0]["routed_capability"] == "minimax_video"
    assert attempts[1]["provider_stage"] == "legacy_secondary"
    assert attempts[1]["routed_capability"] == "digital_human"


def test_step4_supported_duration_combo_normalizes_to_10s(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PIXELLE_BACKEND_MODE", "direct")
    monkeypatch.setenv("PIXELLE_MINIMAX_MODEL", "MiniMax-Hailuo-02")

    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    captured = {"target_duration": None, "model": None}

    class CaptureAdapter:
        def invoke(self, request):
            captured["target_duration"] = request.target_duration
            captured["model"] = request.model
            output_path = Path(request.output_dir) / f"minimax_{request.segment_key}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"minimax-video")
            return type(
                "Resp",
                (),
                {
                    "success": True,
                    "output_path": str(output_path),
                    "error": None,
                },
            )()

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: CaptureAdapter())

    seg = _make_segment_with_keywords()
    seg.duration = 12.0
    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
        resolution=(768, 1366),
        material_mode="auto",
    )

    assert resolved.asset_refs[0].kind == "pixelle_video"
    assert captured["model"] == "MiniMax-Hailuo-02"
    assert captured["target_duration"] == pytest.approx(10.0)


def test_step4_unsupported_duration_combo_blocked_before_provider_invocation(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PIXELLE_BACKEND_MODE", "direct")
    monkeypatch.setenv("PIXELLE_MINIMAX_MODEL", "MiniMax-Hailuo-2.3")

    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    invoke_calls = {"count": 0}

    class ShouldNotInvokeAdapter:
        def invoke(self, request):
            invoke_calls["count"] += 1
            raise AssertionError("adapter.invoke should not be called for unsupported duration combo")

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: ShouldNotInvokeAdapter())

    seg = _make_segment_with_keywords()
    seg.duration = 10.0

    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
        resolution=(1080, 1920),
        material_mode="auto",
    )

    assert invoke_calls["count"] == 0
    assert resolved.asset_refs[0].kind == "template"
    assert resolved.asset_refs[0].fallback_reason_code == "PIXELLE_MINIMAX_UNSUPPORTED_DURATION_COMBO"
    assert resolved.asset_refs[0].fallback_diagnostic is not None
    assert resolved.asset_refs[0].fallback_diagnostic["reason_code"] == "PIXELLE_MINIMAX_UNSUPPORTED_DURATION_COMBO"
    assert resolved.asset_refs[0].fallback_diagnostic["category"] == "VALIDATION"


def test_semantic_score_ranking_prefers_richer_semantic_signals():
    low_text = "Simple status update"
    low_content_key = Segment.compute_content_key(low_text)
    low_segment = Segment(
        segment_key=Segment.compute_segment_key(low_content_key, 1),
        content_key=low_content_key,
        index=1,
        start=0.0,
        end=4.0,
        duration=4.0,
        text=low_text,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=4.0),
        visual_plan=VisualPlan(type="broll", keywords=["news"], prompt="clean visual"),
        plan_hash="semantic-low",
    )

    high_text = "AI医疗诊断数据增长42%，机器人自动驾驶系统持续优化与教育普及"
    high_content_key = Segment.compute_content_key(high_text)
    high_segment = Segment(
        segment_key=Segment.compute_segment_key(high_content_key, 2),
        content_key=high_content_key,
        index=2,
        start=4.0,
        end=8.0,
        duration=4.0,
        text=high_text,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=4.0, trim_end=8.0),
        visual_plan=VisualPlan(
            type="broll",
            keywords=["ai", "医疗", "数据", "图表", "机器人"],
            prompt="Artificial intelligence medical diagnosis dashboard and autonomous robotics",
        ),
        plan_hash="semantic-high",
    )

    low_score = compute_segment_semantic_priority_score(low_segment)
    high_score = compute_segment_semantic_priority_score(high_segment)

    assert high_score["score"] > low_score["score"]
    assert high_score["score_components"]["visual_keywords"] > low_score["score_components"]["visual_keywords"]
    assert high_score["score_components"]["expanded_keywords"] >= low_score["score_components"]["expanded_keywords"]
    assert isinstance(high_score["inputs"]["expanded_keywords"], list)
    assert set(high_score["score_components"].keys()) == {
        "visual_keywords",
        "expanded_keywords",
        "prompt_terms",
        "text_richness",
    }


def test_semantic_score_tie_break_stable_index_order_on_equal_scores():
    text = "Deterministic semantic tie break"
    content_key = Segment.compute_content_key(text)

    seg_1 = Segment(
        segment_key=Segment.compute_segment_key(content_key, 1),
        content_key=content_key,
        index=3,
        start=0.0,
        end=4.0,
        duration=4.0,
        text=text,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=4.0),
        visual_plan=VisualPlan(type="broll", keywords=["technology"], prompt="digital technology"),
        plan_hash="semantic-tie-a",
    )
    seg_2 = Segment(
        segment_key=Segment.compute_segment_key(content_key, 2),
        content_key=content_key,
        index=7,
        start=4.0,
        end=8.0,
        duration=4.0,
        text=text,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=4.0, trim_end=8.0),
        visual_plan=VisualPlan(type="broll", keywords=["technology"], prompt="digital technology"),
        plan_hash="semantic-tie-b",
    )

    score_1 = compute_segment_semantic_priority_score(seg_1)
    score_2 = compute_segment_semantic_priority_score(seg_2)

    assert score_1["score"] == score_2["score"]
    assert score_1["tie_break"]["stable_index_key"] < score_2["tie_break"]["stable_index_key"]

    ranked = sorted(
        [score_2, score_1],
        key=lambda item: item["tie_break"]["rank_key"],
    )
    assert ranked[0]["tie_break"]["index"] == 3
    assert ranked[1]["tie_break"]["index"] == 7


def _build_top6_test_manifest(count: int = 8) -> Manifest:
    segments: list[Segment] = []
    for idx in range(1, count + 1):
        text = f"Top6 planner segment {idx} for deterministic allocation"
        content_key = Segment.compute_content_key(text)
        keywords = [
            "ai",
            "医疗",
            "数据",
            "图表",
            "机器人",
            "自动驾驶",
            "教育",
            "金融",
        ][:idx]
        prompt = " ".join(keywords)
        seg = Segment(
            segment_key=Segment.compute_segment_key(content_key, idx),
            content_key=content_key,
            index=idx,
            start=float((idx - 1) * 4),
            end=float(idx * 4),
            duration=4.0,
            text=text,
            audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=4.0),
            visual_plan=VisualPlan(type="broll", keywords=keywords, prompt=prompt),
            plan_hash=f"top6-{idx}",
        )
        segments.append(seg)
    return Manifest(project_id="top6-test", segments=segments)


def _build_ai_cap_test_manifest(count: int = 8) -> Manifest:
    segments: list[Segment] = []
    for idx in range(1, count + 1):
        text = f"AI cap routing segment {idx}"
        content_key = Segment.compute_content_key(text)
        keywords = [
            "ai",
            "医疗",
            "数据",
            "图表",
            "机器人",
            "自动驾驶",
            "教育",
            "金融",
        ][:idx]
        seg = Segment(
            segment_key=Segment.compute_segment_key(content_key, idx),
            content_key=content_key,
            index=idx,
            start=float((idx - 1) * 4),
            end=float(idx * 4),
            duration=4.0,
            text=text,
            audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=4.0),
            visual_plan=VisualPlan(
                type="pixelle_digital_human",
                pixelle_workflow="digital_human",
                keywords=keywords,
                prompt=" ".join(keywords),
            ),
            plan_hash=f"ai-cap-{idx}",
        )
        segments.append(seg)
    return Manifest(project_id="ai-cap-test", segments=segments, material_mode="ai_only")


def test_top6_allocation_exactly_six(monkeypatch, tmp_path: Path):
    manifest = _build_top6_test_manifest(8)
    observed: dict[str, dict] = {}

    def fake_resolve_asset_for_segment(segment, **kwargs):
        observed[segment.segment_key] = {
            "ai_selected": getattr(segment, "step4_ai_selected", None),
            "ai_allocation_map": getattr(segment, "step4_ai_allocation_map", None),
        }
        segment.asset_refs = [AssetRef(kind="template", path="", asset_hash="top6hash")]
        return segment

    monkeypatch.setattr("src.steps.step4_assets.resolve_asset_for_segment", fake_resolve_asset_for_segment)

    updated_manifest = run_step4(
        manifest=manifest,
        output_manifest=str(tmp_path / "build" / "manifest_step4.json"),
        project_root=str(tmp_path),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
    )

    allocation_map = getattr(updated_manifest, "step4_ai_allocation_map")
    selected_keys = {k for k, v in allocation_map.items() if v}
    assert len(selected_keys) == 6

    ranked = sorted(
        manifest.segments,
        key=lambda seg: compute_segment_semantic_priority_score(seg)["tie_break"]["rank_key"],
    )
    expected_selected = {seg.segment_key for seg in ranked[:6]}
    assert selected_keys == expected_selected
    assert len(selected_keys) <= 6

    assert set(observed.keys()) == {seg.segment_key for seg in manifest.segments}
    assert all(isinstance(v["ai_selected"], bool) for v in observed.values())
    assert all(v["ai_allocation_map"] == allocation_map for v in observed.values())


def test_top6_respects_target_subset(monkeypatch, tmp_path: Path):
    manifest = _build_top6_test_manifest(8)
    target_subset = [seg.segment_key for seg in manifest.segments[:7]]

    def fake_resolve_asset_for_segment(segment, **kwargs):
        segment.asset_refs = [AssetRef(kind="template", path="", asset_hash="top6hash")]
        return segment

    monkeypatch.setattr("src.steps.step4_assets.resolve_asset_for_segment", fake_resolve_asset_for_segment)

    updated_manifest = run_step4(
        manifest=manifest,
        output_manifest=str(tmp_path / "build" / "manifest_step4_subset.json"),
        project_root=str(tmp_path),
        target_segment_keys=target_subset,
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
    )

    allocation_map = getattr(updated_manifest, "step4_ai_allocation_map")
    selected_keys = {k for k, v in allocation_map.items() if v}
    assert len(selected_keys) <= 6
    assert all(k in set(target_subset) for k in selected_keys)

    expected_map = build_top6_ai_allocation_map(
        manifest.segments,
        target_segment_keys=target_subset,
        max_ai_segments=6,
    )
    expected_selected = {k for k, v in expected_map.items() if v}
    assert selected_keys == expected_selected

    outside_subset = {seg.segment_key for seg in manifest.segments} - set(target_subset)
    assert all(allocation_map[key] is False for key in outside_subset)


def test_ai_only_cap_enforced_six(monkeypatch, tmp_path: Path):
    manifest = _build_ai_cap_test_manifest(8)
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    adapter_calls: list[str] = []

    class FakeAdapter:
        def invoke(self, request):
            adapter_calls.append(request.segment_key)
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
                },
            )()

    def fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template")
        return str(path)

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FakeAdapter())
    monkeypatch.setattr("src.steps.step4_assets.generate_template_asset", fake_template)

    updated_manifest = run_step4(
        manifest=manifest,
        output_manifest=str(tmp_path / "build" / "manifest_step4_ai_cap.json"),
        project_root=str(tmp_path),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
    )

    allocation_map = getattr(updated_manifest, "step4_ai_allocation_map")
    selected_keys = {k for k, v in allocation_map.items() if v}
    non_selected_keys = {k for k, v in allocation_map.items() if not v}

    assert len(selected_keys) == 6
    assert len(adapter_calls) == 6
    assert set(adapter_calls) == selected_keys

    refs_by_key = {seg.segment_key: seg.asset_refs[0] for seg in updated_manifest.segments}
    assert all(refs_by_key[key].kind == "pixelle_video" for key in selected_keys)
    assert all(refs_by_key[key].kind in {"template", "pexels_video", "pexels_photo", "pdf_chart", "cached"} for key in non_selected_keys)
    assert all(refs_by_key[key].kind != "pixelle_video" for key in non_selected_keys)


def test_cap_blocks_cached_ai_bypass(monkeypatch, tmp_path: Path):
    from src.steps import step4_assets

    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    text = "cached ai bypass should be blocked when not selected"
    content_key = Segment.compute_content_key(text)
    seg = Segment(
        segment_key=Segment.compute_segment_key(content_key, 1),
        content_key=content_key,
        index=1,
        start=0.0,
        end=4.0,
        duration=4.0,
        text=text,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=4.0),
        visual_plan=VisualPlan(type="pixelle_digital_human", pixelle_workflow="digital_human"),
        plan_hash="cap-cache-block",
    )

    effective_cache_hash = step4_assets._compute_effective_cache_hash(seg.plan_hash or "", "digital_human")
    cached_path = Path(step4_assets._asset_cache_path(str(generated_dir), content_key, effective_cache_hash, "mp4"))
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.write_bytes(b"cached-ai-video" * 200)

    meta_path = Path(step4_assets._asset_cache_meta_path(str(cached_path)))
    meta_path.write_text(
        '{"kind": "pixelle_video", "material_mode": "ai_only", "is_ai_generated": true}',
        encoding="utf-8",
    )

    setattr(seg, "step4_ai_selected", False)
    setattr(seg, "step4_ai_allocation_map", {seg.segment_key: False})

    adapter_calls: list[str] = []

    def fail_if_adapter_called(name: str, **kwargs):
        adapter_calls.append(name)
        raise AssertionError("AI provider should not be invoked for non-selected segment")

    def fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template")
        return str(path)

    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", fail_if_adapter_called)
    monkeypatch.setattr("src.steps.step4_assets.generate_template_asset", fake_template)

    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="ai_only",
    )

    assert adapter_calls == []
    assert resolved.asset_refs[0].kind == "template"
    assert resolved.asset_refs[0].path != str(cached_path)
    assert resolved.asset_refs[0].path.endswith(".png")


# ─────────────────────────────────────────────
# Task 17: Failure-Path Integration Tests
# ─────────────────────────────────────────────
# Task 17: Failure-Path Integration Tests
# ─────────────────────────────────────────────


def test_step4_ai_only_preconditions_ok(monkeypatch, tmp_path: Path):
    """ai_only preconditions satisfied: route proceeds through AI provider chain."""
    monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    routed_capabilities: list[str] = []

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
                },
            )()

    def fake_get_adapter(name: str, **kwargs):
        routed_capabilities.append(name)
        return FakeAdapter()

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", fake_get_adapter)

    seg = _make_segment_with_keywords()
    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="ai_only",
    )

    assert routed_capabilities == ["minimax_video"]
    assert resolved.asset_refs[0].kind == "pixelle_video"
    assert resolved.asset_refs[0].fallback_reason_code is None


def test_step4_ai_only_missing_workflow(monkeypatch, tmp_path: Path):
    """ai_only missing workflow fails fast with deterministic precondition diagnostic."""
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    adapter_calls: list[str] = []

    def fake_get_adapter(name: str, **kwargs):
        adapter_calls.append(name)
        raise AssertionError("Provider adapter should not be requested when workflow is missing")

    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", fake_get_adapter)
    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)

    text = "ai_only missing workflow precondition"
    content_key = Segment.compute_content_key(text)
    seg = Segment(
        segment_key=Segment.compute_segment_key(content_key, 1),
        content_key=content_key,
        index=1,
        start=0.0,
        end=4.0,
        duration=4.0,
        text=text,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=4.0),
        visual_plan=VisualPlan(type="broll", keywords=["ai", "video"]),
        plan_hash="missingworkflowhash1234",
    )

    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="ai_only",
    )

    assert adapter_calls == []
    assert resolved.asset_refs[0].kind == "ai_only_exhausted"
    assert resolved.asset_refs[0].fallback_reason_code == "AI_ONLY_MISSING_WORKFLOW"
    assert resolved.asset_refs[0].fallback_error_category == "CONFIG"
    assert resolved.asset_refs[0].fallback_diagnostic is not None
    assert resolved.asset_refs[0].fallback_diagnostic["reason_code"] == "AI_ONLY_MISSING_WORKFLOW"
    assert resolved.asset_refs[0].fallback_diagnostic["category"] == "CONFIG"
    assert resolved.asset_refs[0].fallback_diagnostic["retryable"] is False
    assert "precondition_failure" in resolved.asset_refs[0].fallback_diagnostic


def test_step4_provider_timeout_fallback_has_full_diagnostic(monkeypatch, tmp_path: Path):
    """Provider timeout returns fallback with retryable=True and TIMEOUT category."""
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    class TimeoutAdapter:
        def invoke(self, request):
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(
                        category=ErrorCategory.TIMEOUT,
                        message="Provider timed out after 300s",
                        details={"timeout_seconds": 300.0, "operation": "wait_for_completion"},
                    ),
                },
            )()

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: TimeoutAdapter())

    def fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template")
        return str(path)

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
    assert ref.fallback_reason_code == "PIXELLE_INVOCATION_FAILED"
    assert ref.fallback_error_category == "TIMEOUT"
    assert ref.fallback_diagnostic is not None
    assert ref.fallback_diagnostic["category"] == "TIMEOUT"
    assert ref.fallback_diagnostic["retryable"] is True
    assert ref.fallback_diagnostic["reason_code"] == "PIXELLE_INVOCATION_FAILED"
    assert ref.fallback_diagnostic["guidance"]
    # fallback_hint is None for TIMEOUT per contract (line 124-128 in contracts.py)


def test_step4_validation_failure_fallback_has_full_diagnostic(monkeypatch, tmp_path: Path):
    """Validation failure returns fallback with retryable=False and VALIDATION category."""
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    class ValidationFailingAdapter:
        def invoke(self, request):
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(
                        category=ErrorCategory.VALIDATION,
                        message="avatar_id is required",
                        details={"field": "avatar_id"},
                    ),
                },
            )()

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: ValidationFailingAdapter())

    def fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template")
        return str(path)

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
    assert ref.fallback_reason_code == "PIXELLE_INVOCATION_FAILED"
    assert ref.fallback_error_category == "VALIDATION"
    assert ref.fallback_diagnostic is not None
    assert ref.fallback_diagnostic["category"] == "VALIDATION"
    assert ref.fallback_diagnostic["retryable"] is False
    assert ref.fallback_diagnostic["guidance"]


def test_step4_corrupted_artifact_fallback_has_reason_code(monkeypatch, tmp_path: Path):
    """Corrupted artifact failure propagates specific reason_code in fallback diagnostic."""
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    class CorruptedArtifactAdapter:
        def invoke(self, request):
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(
                        category=ErrorCategory.EXECUTION,
                        message="Artifact signature is invalid for format .mp4",
                        details={"reason_code": "PIXELLE_ARTIFACT_CORRUPTED", "file_path": "/tmp/bad.mp4"},
                    ),
                },
            )()

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: CorruptedArtifactAdapter())

    def fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template")
        return str(path)

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
    assert ref.fallback_reason_code == "PIXELLE_ARTIFACT_CORRUPTED"
    assert ref.fallback_error_category == "EXECUTION"
    assert ref.fallback_diagnostic is not None
    assert ref.fallback_diagnostic["reason_code"] == "PIXELLE_ARTIFACT_CORRUPTED"
    assert ref.fallback_diagnostic["category"] == "EXECUTION"
    assert ref.fallback_diagnostic["retryable"] is True


def test_step4_uncategorized_provider_error_normalized_to_provider_category(monkeypatch, tmp_path: Path):
    """Unknown provider errors normalize to PROVIDER category with retryable=True."""
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    class UnknownErrorAdapter:
        def invoke(self, request):
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(
                        category=ErrorCategory.PROVIDER,
                        message="Unknown provider error: internal server error 500",
                        details={"status_code": 500},
                    ),
                },
            )()

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: UnknownErrorAdapter())

    def fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template")
        return str(path)

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
    assert ref.fallback_reason_code == "PIXELLE_INVOCATION_FAILED"
    assert ref.fallback_error_category == "PROVIDER"
    assert ref.fallback_diagnostic is not None
    assert ref.fallback_diagnostic["category"] == "PROVIDER"
    assert ref.fallback_diagnostic["retryable"] is True
    assert ref.fallback_diagnostic["guidance"]


def test_step4_throttling_rate_limit_fallback_diagnostic(monkeypatch, tmp_path: Path):
    """Rate limiting returns fallback with RESOURCE category and PIXELLE_RATE_LIMITED reason."""
    from src.steps.pixelle_reliability_controls import (
        PixelleReliabilityControls,
        ReliabilityConfig,
        TokenBucketRateLimiter,
    )

    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    class FakeAdapter:
        def invoke(self, request):
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

    class DepletedClock:
        def __init__(self):
            self.now = 0.0
        def __call__(self):
            return self.now

    clock = DepletedClock()
    controls = PixelleReliabilityControls(
        ReliabilityConfig(
            rate_limit_per_second=1.0,
            rate_limit_burst=1,
            rate_limit_wait_seconds=0.0,
            circuit_window_size=100,
            circuit_min_requests=50,
            circuit_error_rate_threshold=0.6,
            circuit_open_seconds=30.0,
            circuit_half_open_max_calls=1,
        ),
        rate_limiter=TokenBucketRateLimiter(rate_per_second=1.0, burst=1, clock=clock),
    )
    monkeypatch.setattr("src.steps.step4_assets._pixelle_reliability_controls", controls)

    # First call succeeds and consumes the token
    first_seg = _make_segment()
    first_resolved = resolve_asset_for_segment(
        segment=first_seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
    )
    assert first_resolved.asset_refs[0].kind == "pixelle_video"

    # Second call is rate-limited
    second_seg = _make_segment()
    second_seg.segment_key = f"{second_seg.segment_key}-throttled"
    resolved = resolve_asset_for_segment(
        segment=second_seg,
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
    assert ref.fallback_reason_code == "PIXELLE_RATE_LIMITED"
    assert ref.fallback_error_category == "RESOURCE"
    assert ref.fallback_diagnostic is not None
    assert ref.fallback_diagnostic["reason_code"] == "PIXELLE_RATE_LIMITED"
    assert ref.fallback_diagnostic["category"] == "RESOURCE"
    assert ref.fallback_diagnostic["retryable"] is True


def test_step4_circuit_open_fallback_diagnostic(monkeypatch, tmp_path: Path):
    """Circuit breaker open returns fallback with PROVIDER category and PIXELLE_CIRCUIT_OPEN reason."""
    from src.steps.pixelle_reliability_controls import (
        ErrorRateCircuitBreaker,
        PixelleReliabilityControls,
        ReliabilityConfig,
    )

    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    adapter_calls = {"count": 0}

    class FailingAdapter:
        def invoke(self, request):
            adapter_calls["count"] += 1
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(category=ErrorCategory.PROVIDER, message="provider down"),
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

    # Pre-open the circuit by recording failures
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
    assert adapter_calls["count"] == 0  # Adapter should not be invoked
    assert ref.kind == "template"
    assert ref.fallback_reason_code == "PIXELLE_CIRCUIT_OPEN"
    assert ref.fallback_error_category == "PROVIDER"
    assert ref.fallback_diagnostic is not None
    assert ref.fallback_diagnostic["reason_code"] == "PIXELLE_CIRCUIT_OPEN"
    assert ref.fallback_diagnostic["category"] == "PROVIDER"
    assert ref.fallback_diagnostic["retryable"] is True
    assert ref.fallback_diagnostic["guidance"]


def test_step4_pixelle_failure_uses_artifact_reason_code_for_diagnostic(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    class FailingAdapter:
        def invoke(self, request):
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(
                        category=ErrorCategory.EXECUTION,
                        message="Artifact duration invalid",
                        details={"reason_code": "PIXELLE_ARTIFACT_INVALID_DURATION"},
                    ),
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
    assert resolved.asset_refs[0].fallback_reason_code == "PIXELLE_ARTIFACT_INVALID_DURATION"
    assert resolved.asset_refs[0].fallback_error_category == "EXECUTION"
    assert resolved.asset_refs[0].fallback_diagnostic is not None
    assert resolved.asset_refs[0].fallback_diagnostic["reason_code"] == "PIXELLE_ARTIFACT_INVALID_DURATION"


def test_step4_rejects_non_test_mvp_placeholder_and_falls_back(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    class PlaceholderAdapter:
        def invoke(self, request):
            output_path = Path(request.output_dir) / f"pixelle_{request.segment_key}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"mvp-placeholder-video")
            return type(
                "Resp",
                (),
                {
                    "success": True,
                    "output_path": str(output_path),
                    "error": None,
                    "metadata": {"capability": "digital_human", "mvp_placeholder": True},
                },
            )()

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: PlaceholderAdapter())

    def fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template")
        return str(path)

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
    assert resolved.asset_refs[0].fallback_reason_code == "PIXELLE_INVOCATION_FAILED"
    assert resolved.asset_refs[0].fallback_error_category == "EXECUTION"
    assert resolved.asset_refs[0].fallback_diagnostic is not None


def test_digital_human_adapter_provider_execution_success(tmp_path: Path):
    from pixelle_snapshot.adapters import DigitalHumanAdapter, DigitalHumanRequest
    from pixelle_snapshot.adapters.contracts import (
        ProviderSubmitResult,
        ProviderPollResult,
        ProviderFetchResult,
        ProviderJobStatus,
    )
    
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)
    artifact_path = output_dir / "test_job_123.mp4"
    artifact_path.write_bytes(b"mock video content")
    
    class MockProviderClient:
        def submit(self, capability, request, idempotency_key=None):
            return ProviderSubmitResult(
                job_id="test_job_123",
                status=ProviderJobStatus.SUBMITTED,
                metadata={"request_id": "req-abc"},
            )
        
        def wait_for_completion(self, job_id, timeout_seconds=None, cancel_on_timeout=True):
            return ProviderPollResult(
                job_id=job_id,
                status=ProviderJobStatus.SUCCEEDED,
                metadata={"duration": 2.5, "resolution": "1080x1920", "run_seconds": 1.2},
            )
        
        def fetch(self, job_id, output_dir):
            return ProviderFetchResult(
                job_id=job_id,
                output_path=str(artifact_path),
                metadata={"artifact_bytes": 18},
            )
    
    request = DigitalHumanRequest(
        segment_key="provider_test#1",
        segment_text="Provider execution test",
        segment_duration=2.5,
        project_root=str(tmp_path),
        output_dir=str(output_dir),
        avatar_id="test_avatar",
        voice_id="test_voice",
    )
    
    adapter = DigitalHumanAdapter(provider_client=MockProviderClient())
    response = adapter.invoke(request)
    
    assert response.success is True
    assert response.output_path == str(artifact_path)
    assert response.video_duration == 2.5
    assert response.video_resolution == "1080x1920"
    assert response.avatar_used == "test_avatar"
    assert response.voice_used == "test_voice"
    assert response.metadata.get("provider_job_id") == "test_job_123"
    assert response.metadata.get("capability") == "digital_human"


def test_digital_human_adapter_provider_job_failed(tmp_path: Path):
    from pixelle_snapshot.adapters import DigitalHumanAdapter, DigitalHumanRequest
    from pixelle_snapshot.adapters.contracts import (
        ProviderSubmitResult,
        ProviderPollResult,
        ProviderJobStatus,
        ErrorCategory,
    )
    
    class MockProviderClient:
        def submit(self, capability, request, idempotency_key=None):
            return ProviderSubmitResult(
                job_id="failed_job_456",
                status=ProviderJobStatus.SUBMITTED,
            )
        
        def wait_for_completion(self, job_id, timeout_seconds=None, cancel_on_timeout=True):
            return ProviderPollResult(
                job_id=job_id,
                status=ProviderJobStatus.FAILED,
                metadata={"error": "GPU memory exhausted"},
            )
    
    request = DigitalHumanRequest(
        segment_key="failure_test#1",
        segment_text="Provider failure test",
        segment_duration=2.0,
        project_root=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        avatar_id="test_avatar",
        voice_id="test_voice",
    )
    
    adapter = DigitalHumanAdapter(provider_client=MockProviderClient())
    response = adapter.invoke(request)
    
    assert response.success is False
    assert response.error is not None
    assert response.error.category == ErrorCategory.PROVIDER
    assert "failed" in response.error.message.lower()
    assert response.error.details.get("job_id") == "failed_job_456"
    assert response.metadata.get("provider_job_id") == "failed_job_456"


def test_digital_human_adapter_output_path_validation(tmp_path: Path):
    from pixelle_snapshot.adapters import DigitalHumanAdapter, DigitalHumanRequest
    from pixelle_snapshot.adapters.contracts import (
        ProviderSubmitResult,
        ProviderPollResult,
        ProviderFetchResult,
        ProviderJobStatus,
        ErrorCategory,
    )
    
    class MockProviderClient:
        def submit(self, capability, request, idempotency_key=None):
            return ProviderSubmitResult(job_id="missing_output_job", status=ProviderJobStatus.SUBMITTED)
        
        def wait_for_completion(self, job_id, timeout_seconds=None, cancel_on_timeout=True):
            return ProviderPollResult(job_id=job_id, status=ProviderJobStatus.SUCCEEDED)
        
        def fetch(self, job_id, output_dir):
            return ProviderFetchResult(
                job_id=job_id,
                output_path="/nonexistent/path/video.mp4",
            )
    
    request = DigitalHumanRequest(
        segment_key="validation_test#1",
        segment_text="Output path validation test",
        segment_duration=2.0,
        project_root=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        avatar_id="test_avatar",
        voice_id="test_voice",
    )
    
    adapter = DigitalHumanAdapter(provider_client=MockProviderClient())
    response = adapter.invoke(request)
    
    assert response.success is False
    assert response.error is not None
    assert response.error.category == ErrorCategory.EXECUTION
    assert "output file not found" in response.error.message.lower()


def test_digital_human_adapter_idempotency_key_deterministic(tmp_path: Path):
    from pixelle_snapshot.adapters import DigitalHumanAdapter, DigitalHumanRequest
    
    request = DigitalHumanRequest(
        segment_key="idempotency_test#1",
        segment_text="Idempotency test",
        segment_duration=2.0,
        project_root=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        avatar_id="avatar_a",
        voice_id="voice_b",
    )
    
    adapter = DigitalHumanAdapter()
    key1 = adapter._compute_idempotency_key(request)
    key2 = adapter._compute_idempotency_key(request)
    
    assert key1 == key2
    assert len(key1) == 32


def test_adapter_timeout_returns_timeout_category(tmp_path: Path):
    import time
    from pixelle_snapshot.adapters.base import BaseAdapter
    from pixelle_snapshot.adapters.contracts import (
        AdapterRequest,
        AdapterResponse,
        ErrorCategory,
    )
    from dataclasses import dataclass, field
    from typing import Dict, Any, Type

    @dataclass
    class SlowRequest(AdapterRequest):
        delay_seconds: float = 0.5

    @dataclass
    class SlowResponse(AdapterResponse):
        pass

    class SlowAdapter(BaseAdapter[SlowRequest, SlowResponse]):
        @property
        def capability_name(self) -> str:
            return "slow_test"

        @property
        def request_type(self) -> Type[SlowRequest]:
            return SlowRequest

        @property
        def response_type(self) -> Type[SlowResponse]:
            return SlowResponse

        def _execute(self, request: SlowRequest) -> SlowResponse:
            time.sleep(request.delay_seconds)
            return SlowResponse(success=True, segment_key=request.segment_key)

    adapter = SlowAdapter()
    request = SlowRequest(
        segment_key="timeout_test#1",
        segment_text="Timeout test",
        segment_duration=2.0,
        project_root=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        timeout_seconds=0.1,
        delay_seconds=1.0,
    )

    response = adapter.invoke(request)

    assert response.success is False
    assert response.error is not None
    assert response.error.category == ErrorCategory.TIMEOUT
    assert "timeout" in response.error.message.lower()
    assert response.error.details.get("timeout_seconds") == 0.1


def test_step4_timeout_falls_back_with_correct_category(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    class SlowAdapter:
        def invoke(self, request):
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(
                        category=ErrorCategory.TIMEOUT,
                        message="Operation timed out",
                    ),
                },
            )()

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: SlowAdapter())

    def fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template")
        return str(path)

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
    assert resolved.asset_refs[0].fallback_reason_code == "PIXELLE_INVOCATION_FAILED"
    assert resolved.asset_refs[0].fallback_error_category == "TIMEOUT"
    assert resolved.asset_refs[0].fallback_diagnostic is not None
    assert resolved.asset_refs[0].fallback_diagnostic["category"] == "TIMEOUT"
    assert resolved.asset_refs[0].fallback_diagnostic["retryable"] is True


def test_concurrency_limiter_respects_bound(monkeypatch, tmp_path: Path):
    import threading
    import time
    from src.steps import step4_assets

    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    original_limit = step4_assets._PIXELLE_CONCURRENCY_LIMIT
    monkeypatch.setattr(step4_assets, "_PIXELLE_CONCURRENCY_LIMIT", 2)
    monkeypatch.setattr(step4_assets, "_pixelle_semaphore", threading.Semaphore(2))

    concurrent_calls = []
    max_concurrent = [0]
    lock = threading.Lock()

    class SlowAdapter:
        def invoke(self, request):
            with lock:
                concurrent_calls.append(threading.current_thread().name)
                current = len(concurrent_calls)
                if current > max_concurrent[0]:
                    max_concurrent[0] = current

            time.sleep(0.05)

            with lock:
                concurrent_calls.remove(threading.current_thread().name)

            output_path = Path(request.output_dir) / f"pixelle_{request.segment_key}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"test-video")
            return type(
                "Resp",
                (),
                {
                    "success": True,
                    "output_path": str(output_path),
                    "error": None,
                },
            )()

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: SlowAdapter())

    segments = []
    for i in range(5):
        text = f"Concurrency test segment {i}"
        content_key = Segment.compute_content_key(text)
        seg = Segment(
            segment_key=Segment.compute_segment_key(content_key, i + 1),
            content_key=content_key,
            index=i + 1,
            start=float(i),
            end=float(i + 1),
            duration=1.0,
            text=text,
            audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=1.0),
            visual_plan=VisualPlan(type="pixelle_digital_human", pixelle_workflow="digital_human"),
            plan_hash=f"concurrent-hash-{i}",
        )
        segments.append(seg)

    threads = []
    for seg in segments:
        t = threading.Thread(
            target=resolve_asset_for_segment,
            kwargs={
                "segment": seg,
                "project_root": str(project_root),
                "generated_dir": str(generated_dir),
                "library_dir": str(tmp_path / "assets" / "library"),
                "pexels_api_key": "",
                "enable_pexels_video": False,
                "enable_pexels_photo": False,
                "enable_ai_image": False,
            },
        )
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max_concurrent[0] <= 2, f"Expected max 2 concurrent calls, got {max_concurrent[0]}"


def test_timeout_error_contract():
    from pixelle_snapshot.adapters.contracts import TimeoutError, ErrorCategory

    error = TimeoutError(
        message="Operation timed out after 30s",
        timeout_seconds=30.0,
        operation="test_op",
    )

    assert error.category == ErrorCategory.TIMEOUT
    assert error.details["timeout_seconds"] == 30.0
    assert error.details["operation"] == "test_op"
    assert "30s" in str(error)


def test_failure_diagnostic_contract():
    from pixelle_snapshot.adapters.contracts import (
        ErrorCategory,
        FailureDiagnostic,
        normalize_error_category,
        CATEGORY_GUIDANCE,
        REASON_CODE_GUIDANCE,
    )

    diag = FailureDiagnostic.from_error(
        ErrorCategory.PROVIDER,
        "PIXELLE_INVOCATION_FAILED",
    )
    assert diag.category == "PROVIDER"
    assert diag.reason_code == "PIXELLE_INVOCATION_FAILED"
    assert diag.retryable is True
    assert diag.guidance == REASON_CODE_GUIDANCE["PIXELLE_INVOCATION_FAILED"]
    assert diag.fallback_hint is not None

    as_dict = diag.to_dict()
    assert as_dict["category"] == "PROVIDER"
    assert as_dict["retryable"] is True


def test_normalize_error_category():
    from pixelle_snapshot.adapters.contracts import (
        ErrorCategory,
        normalize_error_category,
    )

    assert normalize_error_category(ErrorCategory.VALIDATION) == "VALIDATION"
    assert normalize_error_category("TIMEOUT") == "TIMEOUT"
    assert normalize_error_category(None) == "EXECUTION"


def test_telemetry_success_path(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

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

    ref = resolved.asset_refs[0]
    assert ref.kind == "pixelle_video"
    assert ref.fallback_reason_code is None
    assert ref.fallback_error_category is None
    assert ref.fallback_diagnostic is None


def test_telemetry_error_path_has_diagnostic(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    class FailingAdapter:
        def invoke(self, request):
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(
                        category=ErrorCategory.EXECUTION,
                        message="runtime execution failed",
                    ),
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
    assert ref.fallback_reason_code == "PIXELLE_INVOCATION_FAILED"
    assert ref.fallback_error_category == "EXECUTION"
    assert ref.fallback_diagnostic is not None
    assert ref.fallback_diagnostic["category"] == "EXECUTION"
    assert ref.fallback_diagnostic["retryable"] is True
    assert ref.fallback_diagnostic["guidance"]
    assert ref.fallback_diagnostic["fallback_hint"]


def test_telemetry_capability_unavailable_diagnostic(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: False)

    def fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template")
        return str(path)

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
    assert ref.fallback_reason_code == "PIXELLE_CAPABILITY_UNAVAILABLE"
    assert ref.fallback_error_category == "UNSUPPORTED"
    assert ref.fallback_diagnostic is not None
    assert ref.fallback_diagnostic["category"] == "UNSUPPORTED"
    assert ref.fallback_diagnostic["retryable"] is False
    assert "capability" in ref.fallback_diagnostic["guidance"].lower()


def test_digital_human_adapter_creates_output_file(tmp_path: Path):
    from pixelle_snapshot.adapters import DigitalHumanAdapter, DigitalHumanRequest
    from pixelle_snapshot.test_doubles import enable_test_mode, disable_test_mode
    
    enable_test_mode()
    try:
        request = DigitalHumanRequest(
            segment_key="test_seg_001#1",
            segment_text="Hello from digital human",
            segment_duration=2.0,
            project_root=str(tmp_path),
            output_dir=str(tmp_path / "output"),
            avatar_id="default_avatar",
            voice_id="default_voice",
        )
        
        adapter = DigitalHumanAdapter()
        response = adapter.invoke(request)
        
        assert response.success is True
        assert response.output_path is not None
        assert Path(response.output_path).exists()
        assert Path(response.output_path).stat().st_size > 0
        assert response.metadata.get("test_mode") is True
        assert response.output_path.endswith(".test.mp4")
    finally:
        disable_test_mode()


# ─────────────────────────────────────────────
# Task 2: Mode Policy Helper and ai_only Guard Tests
# ─────────────────────────────────────────────


def _make_pdf_chart_segment() -> Segment:
    """Create a segment with pdf_chart visual plan type."""
    text = "PDF chart mode policy segment"
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
        visual_plan=VisualPlan(
            type="pdf_chart",
            use_pdf_assets=[{"image": "charts/chart1.png"}],
        ),
        plan_hash="pdfcharthash1234",
    )


def test_mode_policy_is_route_allowed_returns_true_for_auto_pdf_chart():
    """Mode policy helper allows pdf_chart route when material_mode is 'auto'."""
    from src.steps.step4_assets import is_route_allowed_by_mode_policy

    assert is_route_allowed_by_mode_policy("pdf_chart", "auto") is True


def test_mode_policy_is_route_allowed_returns_true_for_ai_preferred_pdf_chart():
    """Mode policy helper allows pdf_chart route when material_mode is 'ai_preferred'."""
    from src.steps.step4_assets import is_route_allowed_by_mode_policy

    assert is_route_allowed_by_mode_policy("pdf_chart", "ai_preferred") is True


def test_mode_policy_is_route_allowed_returns_false_for_ai_only_pdf_chart():
    """Mode policy helper blocks pdf_chart route when material_mode is 'ai_only'."""
    from src.steps.step4_assets import is_route_allowed_by_mode_policy

    assert is_route_allowed_by_mode_policy("pdf_chart", "ai_only") is False


def test_mode_policy_is_route_allowed_returns_true_for_ai_only_ai_image():
    """Mode policy helper allows ai_image route when material_mode is 'ai_only'."""
    from src.steps.step4_assets import is_route_allowed_by_mode_policy

    assert is_route_allowed_by_mode_policy("ai_image", "ai_only") is True


def test_mode_policy_is_route_allowed_returns_true_for_ai_only_pixelle_video():
    """Mode policy helper allows pixelle_video route when material_mode is 'ai_only'."""
    from src.steps.step4_assets import is_route_allowed_by_mode_policy

    assert is_route_allowed_by_mode_policy("pixelle_video", "ai_only") is True


def test_mode_policy_is_route_allowed_returns_false_for_ai_only_template():
    """Mode policy helper blocks template route when material_mode is 'ai_only' (strict mode)."""
    from src.steps.step4_assets import is_route_allowed_by_mode_policy

    assert is_route_allowed_by_mode_policy("template", "ai_only") is False



def test_mode_policy_docstring_does_not_claim_template_allowed_in_ai_only():
    """Docstring must match implementation: template is not allowed in ai_only."""
    import inspect
    from src.steps.step4_assets import is_route_allowed_by_mode_policy

    doc = inspect.getdoc(is_route_allowed_by_mode_policy) or ""
    assert "ai_image, pixelle_video, template" not in doc


def test_step4_ai_only_blocks_pdf_chart_route(monkeypatch, tmp_path: Path):
    """When material_mode is 'ai_only', pdf_chart route is blocked and fails fast on missing workflow."""
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    charts_dir = tmp_path / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    (charts_dir / "chart1.png").write_bytes(b"fake-pdf-chart-image")

    template_called = []
    def fake_template(output_path: str, width: int, height: int, text: str):
        template_called.append("template")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template")
        return str(path)

    monkeypatch.setattr("src.steps.step4_assets.generate_template_asset", fake_template)

    seg = _make_pdf_chart_segment()
    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="ai_only",
    )

    assert resolved.asset_refs[0].kind != "pdf_chart"
    assert resolved.asset_refs[0].kind == "ai_only_exhausted"
    assert resolved.asset_refs[0].fallback_reason_code == "AI_ONLY_MISSING_WORKFLOW"
    assert resolved.asset_refs[0].fallback_error_category == "CONFIG"
    assert template_called == [], "Template should not be called in ai_only mode"


def test_step4_auto_mode_allows_pdf_chart_route(monkeypatch, tmp_path: Path):
    """When material_mode is 'auto' (default), pdf_chart route is allowed."""
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    charts_dir = tmp_path / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    (charts_dir / "chart1.png").write_bytes(b"fake-pdf-chart-image")

    seg = _make_pdf_chart_segment()
    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="auto",
    )

    assert resolved.asset_refs[0].kind == "pdf_chart"
    assert resolved.asset_refs[0].fallback_reason_code is None


def test_step4_ai_preferred_mode_allows_pdf_chart_as_fallback(monkeypatch, tmp_path: Path):
    """When material_mode is 'ai_preferred', pdf_chart route is allowed as fallback."""
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    charts_dir = tmp_path / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    (charts_dir / "chart1.png").write_bytes(b"fake-pdf-chart-image")

    seg = _make_pdf_chart_segment()
    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="ai_preferred",
    )

    assert resolved.asset_refs[0].kind == "pdf_chart"
    assert resolved.asset_refs[0].fallback_reason_code is None


def test_step4_material_mode_defaults_to_auto():
    """resolve_asset_for_segment defaults material_mode to 'auto' for backward compatibility."""
    import inspect
    from src.steps.step4_assets import resolve_asset_for_segment

    sig = inspect.signature(resolve_asset_for_segment)
    params = sig.parameters

    assert "material_mode" in params
    assert params["material_mode"].default == "auto"


# ─────────────────────────────────────────────
# Task 6: Explicit Auto Route Branch Regression Tests
# ─────────────────────────────────────────────


def test_auto_mode_route_priority_constant_exists():
    """AUTO_MODE_ROUTE_PRIORITY documents the canonical baseline route order."""
    from src.steps.step4_assets import AUTO_MODE_ROUTE_PRIORITY

    assert AUTO_MODE_ROUTE_PRIORITY is not None
    assert isinstance(AUTO_MODE_ROUTE_PRIORITY, tuple)
    expected_order = (
        "cached",
        "pdf_chart",
        "pexels_video",
        "pexels_photo",
        "pixelle_video",
        "ai_image",
        "template",
    )
    assert AUTO_MODE_ROUTE_PRIORITY == expected_order


def test_auto_mode_allows_all_routes():
    """Auto mode allows all routes in the canonical priority order."""
    from src.steps.step4_assets import AUTO_MODE_ROUTE_PRIORITY, is_route_allowed_by_mode_policy

    for route in AUTO_MODE_ROUTE_PRIORITY:
        assert is_route_allowed_by_mode_policy(route, "auto") is True


def test_auto_mode_allows_non_ai_routes():
    """Auto mode explicitly allows non-AI routes (pdf_chart, pexels_video, pexels_photo)."""
    from src.steps.step4_assets import NON_AI_ROUTES, is_route_allowed_by_mode_policy

    for route in NON_AI_ROUTES:
        assert is_route_allowed_by_mode_policy(route, "auto") is True


def test_auto_mode_allows_ai_routes():
    """Auto mode explicitly allows AI routes (ai_image, pixelle_video, template)."""
    from src.steps.step4_assets import AI_GENERATED_ROUTES, is_route_allowed_by_mode_policy

    for route in AI_GENERATED_ROUTES:
        assert is_route_allowed_by_mode_policy(route, "auto") is True


def test_step4_auto_mode_pexels_video_not_blocked(monkeypatch, tmp_path: Path):
    """Auto mode does not block pexels_video route (regression baseline)."""
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    pexels_cache = tmp_path / "assets" / "pexels_cache" / "videos"
    pexels_cache.mkdir(parents=True, exist_ok=True)

    pexels_video_path = pexels_cache / "test_video.mp4"
    pexels_video_path.write_bytes(b"pexels-video-content" * 1000)

    def fake_fetch_pexels_video(*args, **kwargs):
        return str(pexels_video_path)

    monkeypatch.setattr("src.steps.step4_assets.fetch_pexels_video", fake_fetch_pexels_video)

    text = "Auto mode pexels video test"
    content_key = Segment.compute_content_key(text)
    seg = Segment(
        segment_key=Segment.compute_segment_key(content_key, 1),
        content_key=content_key,
        index=1,
        start=0.0,
        end=4.0,
        duration=4.0,
        text=text,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=4.0),
        visual_plan=VisualPlan(type="broll", keywords=["technology"]),
        plan_hash="automodepexelshash",
    )

    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="test_key",
        enable_pexels_video=True,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="auto",
    )

    assert resolved.asset_refs[0].kind == "pexels_video"
    assert resolved.asset_refs[0].fallback_reason_code is None


def test_step4_auto_mode_pexels_photo_not_blocked(monkeypatch, tmp_path: Path):
    """Auto mode does not block pexels_photo route (regression baseline)."""
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    pexels_photo_path = tmp_path / "assets" / "pexels_cache" / "photos" / "test_photo.jpg"
    pexels_photo_path.parent.mkdir(parents=True, exist_ok=True)
    pexels_photo_path.write_bytes(b"pexels-photo-content" * 100)

    def fake_fetch_pexels_photo(*args, **kwargs):
        return str(pexels_photo_path)

    monkeypatch.setattr("src.steps.step4_assets.fetch_pexels_photo", fake_fetch_pexels_photo)

    text = "Auto mode pexels photo test"
    content_key = Segment.compute_content_key(text)
    seg = Segment(
        segment_key=Segment.compute_segment_key(content_key, 1),
        content_key=content_key,
        index=1,
        start=0.0,
        end=4.0,
        duration=4.0,
        text=text,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=4.0),
        visual_plan=VisualPlan(type="broll", keywords=["nature"]),
        plan_hash="automodephotohash",
    )

    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="test_key",
        enable_pexels_video=False,
        enable_pexels_photo=True,
        enable_ai_image=False,
        material_mode="auto",
    )

    assert resolved.asset_refs[0].kind == "pexels_photo"
    assert resolved.asset_refs[0].fallback_reason_code is None


def test_step4_auto_mode_route_order_pdf_before_pexels(monkeypatch, tmp_path: Path):
    """Auto mode preserves canonical order: pdf_chart takes priority over pexels."""
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    charts_dir = tmp_path / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    (charts_dir / "chart1.png").write_bytes(b"pdf-chart-image-data")

    pexels_called = {"count": 0}

    def fake_fetch_pexels_video(*args, **kwargs):
        pexels_called["count"] += 1
        return str(tmp_path / "pexels.mp4")

    monkeypatch.setattr("src.steps.step4_assets.fetch_pexels_video", fake_fetch_pexels_video)

    seg = _make_pdf_chart_segment()
    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="test_key",
        enable_pexels_video=True,
        enable_pexels_photo=True,
        enable_ai_image=False,
        material_mode="auto",
    )

    assert resolved.asset_refs[0].kind == "pdf_chart"
    assert pexels_called["count"] == 0


def test_step4_auto_mode_fallback_chain_to_template(monkeypatch, tmp_path: Path):
    """Auto mode falls back to template when no other routes succeed."""
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    def fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template-fallback")
        return str(path)

    monkeypatch.setattr("src.steps.step4_assets.generate_template_asset", fake_template)

    text = "Auto mode template fallback test"
    content_key = Segment.compute_content_key(text)
    seg = Segment(
        segment_key=Segment.compute_segment_key(content_key, 1),
        content_key=content_key,
        index=1,
        start=0.0,
        end=4.0,
        duration=4.0,
        text=text,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=4.0),
        visual_plan=VisualPlan(type="broll", keywords=["fallback"]),
        plan_hash="automodetemplatehash",
    )

    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="auto",
    )

    assert resolved.asset_refs[0].kind == "template"


def test_auto_mode_equivalent_to_omitted_material_mode(monkeypatch, tmp_path: Path):
    """Auto mode behavior is identical to omitting material_mode (backward compat)."""
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    charts_dir = tmp_path / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    (charts_dir / "chart1.png").write_bytes(b"pdf-chart-image")

    seg_explicit = _make_pdf_chart_segment()
    seg_default = _make_pdf_chart_segment()

    resolved_explicit = resolve_asset_for_segment(
        segment=seg_explicit,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="auto",
    )

    resolved_default = resolve_asset_for_segment(
        segment=seg_default,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
    )

    assert resolved_explicit.asset_refs[0].kind == resolved_default.asset_refs[0].kind
    assert resolved_explicit.asset_refs[0].fallback_reason_code == resolved_default.asset_refs[0].fallback_reason_code


def test_digital_human_adapter_validation_error_missing_avatar(tmp_path: Path):
    from pixelle_snapshot.adapters import DigitalHumanAdapter, DigitalHumanRequest
    
    request = DigitalHumanRequest(
        segment_key="test_seg_002#1",
        segment_text="Missing avatar test",
        segment_duration=2.0,
        project_root=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        voice_id="default_voice",
    )
    
    adapter = DigitalHumanAdapter()
    response = adapter.invoke(request)
    
    assert response.success is False
    assert response.error is not None
    assert response.error.category == ErrorCategory.VALIDATION
    assert "avatar" in response.error.message.lower()
    assert response.error.details.get("field") == "avatar_id"


def test_digital_human_adapter_validation_error_missing_voice(tmp_path: Path):
    from pixelle_snapshot.adapters import DigitalHumanAdapter, DigitalHumanRequest
    
    request = DigitalHumanRequest(
        segment_key="test_seg_003#1",
        segment_text="Missing voice test",
        segment_duration=2.0,
        project_root=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        avatar_id="default_avatar",
    )
    
    adapter = DigitalHumanAdapter()
    response = adapter.invoke(request)
    
    assert response.success is False
    assert response.error is not None
    assert response.error.category == ErrorCategory.VALIDATION
    assert "voice" in response.error.message.lower()
    assert response.error.details.get("field") == "voice_id"


def test_step4_digital_human_validation_error_falls_back(monkeypatch, tmp_path: Path):
    from src.steps.step4_assets import _build_pixelle_request
    
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    def failing_build_request(capability, segment, project_root, output_dir):
        raise ValueError("Missing required input for digital_human")

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("src.steps.step4_assets._build_pixelle_request", failing_build_request)

    def fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template")
        return str(path)

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
    assert resolved.asset_refs[0].fallback_reason_code == "PIXELLE_REQUEST_BUILD_FAILED"
    assert resolved.asset_refs[0].fallback_error_category == "VALIDATION"


def _make_i2v_segment() -> Segment:
    text = "I2V routing segment"
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
        visual_plan=VisualPlan(type="pixelle_i2v", pixelle_workflow="i2v"),
        plan_hash="i2vhash1234",
    )


def test_i2v_adapter_creates_output_file(tmp_path: Path):
    from pixelle_snapshot.adapters import I2VAdapter, I2VRequest
    
    request = I2VRequest(
        segment_key="test_seg_i2v_001#1",
        segment_text="Image to video test",
        segment_duration=3.0,
        project_root=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        input_image_path=str(tmp_path / "input" / "test_image.png"),
    )
    
    adapter = I2VAdapter()
    response = adapter.invoke(request)
    
    assert response.success is True
    assert response.output_path is not None
    assert Path(response.output_path).exists()
    assert Path(response.output_path).stat().st_size > 0
    assert response.metadata.get("mvp_placeholder") is True
    assert response.video_duration == 3.0
    assert response.motion_applied in ["auto", "kenburns"]


def test_i2v_adapter_validation_error_missing_image(tmp_path: Path):
    from pixelle_snapshot.adapters import I2VAdapter, I2VRequest
    
    request = I2VRequest(
        segment_key="test_seg_i2v_002#1",
        segment_text="Missing image test",
        segment_duration=2.0,
        project_root=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        input_image_path="",
    )
    
    adapter = I2VAdapter()
    response = adapter.invoke(request)
    
    assert response.success is False
    assert response.error is not None
    assert response.error.category == ErrorCategory.VALIDATION
    assert "input_image_path" in response.error.message
    assert response.error.details.get("field") == "input_image_path"


def test_i2v_adapter_validation_error_unsupported_format(tmp_path: Path):
    from pixelle_snapshot.adapters import I2VAdapter, I2VRequest
    
    request = I2VRequest(
        segment_key="test_seg_i2v_003#1",
        segment_text="Unsupported format test",
        segment_duration=2.0,
        project_root=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        input_image_path=str(tmp_path / "input" / "test_image.psd"),
    )
    
    adapter = I2VAdapter()
    response = adapter.invoke(request)
    
    assert response.success is False
    assert response.error is not None
    assert response.error.category == ErrorCategory.VALIDATION
    assert ".psd" in response.error.message.lower()
    assert response.error.details.get("field") == "input_image_path"
    assert response.error.details.get("unsupported_extension") == ".psd"


def test_step4_vendor_routing_happy(monkeypatch, tmp_path: Path):
    """Happy path: digital_human capability routes to Pixelle, produces pixelle_video asset."""
    monkeypatch.setenv("PIXELLE_BACKEND_MODE", "direct")
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    routed_capabilities: list[str] = []
    seen_request_types: list[str] = []

    class FakeAdapter:
        def invoke(self, request):
            seen_request_types.append(type(request).__name__)
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
                },
            )()

    def fake_get_adapter(name: str, **kwargs):
        routed_capabilities.append(name)
        return FakeAdapter()

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", fake_get_adapter)

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

    assert routed_capabilities == ["minimax_video"]
    assert seen_request_types == ["MinimaxVideoRequest"]
    assert resolved.asset_refs[0].kind == "pixelle_video"
    assert resolved.asset_refs[0].fallback_reason_code is None
    assert resolved.asset_refs[0].fallback_error_category is None
    assert resolved.visual_plan is not None
    assert resolved.visual_plan.asset_path is not None
    assert resolved.visual_plan.asset_path.endswith(".mp4")


def test_step4_vendor_routing_legacy_provider_mode_happy(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    routed_capabilities: list[str] = []
    seen_request_types: list[str] = []

    class FakeAdapter:
        def invoke(self, request):
            seen_request_types.append(type(request).__name__)
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
                },
            )()

    def fake_get_adapter(name: str, **kwargs):
        routed_capabilities.append(name)
        return FakeAdapter()

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", fake_get_adapter)

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

    assert routed_capabilities == ["digital_human"]
    assert seen_request_types == ["DigitalHumanRequest"]
    assert resolved.asset_refs[0].kind == "pixelle_video"
    assert resolved.asset_refs[0].fallback_reason_code is None
    assert resolved.asset_refs[0].fallback_error_category is None


def test_step4_vendor_routing_unsupported_continuity(monkeypatch, tmp_path: Path):
    """Unsupported continuity scenario: capability unavailable falls back with explicit reason."""
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: False)

    def fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template")
        return str(path)

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
    assert ref.fallback_reason_code == "PIXELLE_CAPABILITY_UNAVAILABLE"
    assert ref.fallback_error_category == "UNSUPPORTED"
    assert ref.fallback_diagnostic is not None
    assert ref.fallback_diagnostic["category"] == "UNSUPPORTED"
    assert ref.fallback_diagnostic["retryable"] is False
    assert "capability" in ref.fallback_diagnostic["guidance"].lower()


def test_step4_i2v_route_happy_path(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    routed_capabilities: list[str] = []

    class FakeI2VAdapter:
        def invoke(self, request):
            output_path = Path(request.output_dir) / f"i2v_{request.segment_key}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"i2v-video")
            return type(
                "Resp",
                (),
                {
                    "success": True,
                    "output_path": str(output_path),
                    "error": None,
                },
            )()

    def fake_get_adapter(name: str, **kwargs):
        routed_capabilities.append(name)
        return FakeI2VAdapter()

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", fake_get_adapter)

    seg = _make_i2v_segment()
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

    assert routed_capabilities == ["i2v"]
    assert resolved.asset_refs[0].kind == "pixelle_video"
    assert resolved.asset_refs[0].fallback_reason_code is None
    assert resolved.asset_refs[0].fallback_error_category is None
    assert resolved.visual_plan is not None
    assert resolved.visual_plan.asset_path is not None
    assert resolved.visual_plan.asset_path.endswith(".mp4")


def test_step4_i2v_validation_error_falls_back(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    class FailingAdapter:
        def invoke(self, request):
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(
                        category=ErrorCategory.VALIDATION,
                        message="input_image_path is required",
                    ),
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

    seg = _make_i2v_segment()
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
    assert resolved.asset_refs[0].fallback_reason_code == "PIXELLE_INVOCATION_FAILED"
    assert resolved.asset_refs[0].fallback_error_category == "VALIDATION"
    assert resolved.visual_plan is not None
    assert resolved.visual_plan.asset_path is not None
    assert resolved.visual_plan.asset_path.endswith(".png")


def _make_action_transfer_segment() -> Segment:
    text = "Action transfer routing segment"
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
        visual_plan=VisualPlan(type="pixelle_action_transfer", pixelle_workflow="action_transfer"),
        plan_hash="actiontransferhash1234",
    )


def test_action_transfer_adapter_creates_output_file(tmp_path: Path):
    from pixelle_snapshot.adapters import ActionTransferAdapter, ActionTransferRequest
    
    request = ActionTransferRequest(
        segment_key="test_seg_at_001#1",
        segment_text="Action transfer test",
        segment_duration=3.0,
        project_root=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        reference_video_path=str(tmp_path / "input" / "reference.mp4"),
        target_image_path=str(tmp_path / "input" / "target.png"),
    )
    
    adapter = ActionTransferAdapter()
    response = adapter.invoke(request)
    
    assert response.success is True
    assert response.output_path is not None
    assert Path(response.output_path).exists()
    assert Path(response.output_path).stat().st_size > 0
    assert response.metadata.get("mvp_placeholder") is True
    assert response.video_duration == 3.0
    assert response.transfer_mode_used == "motion"
    assert response.target_type == "image"


def test_action_transfer_adapter_validation_error_missing_reference(tmp_path: Path):
    from pixelle_snapshot.adapters import ActionTransferAdapter, ActionTransferRequest
    
    request = ActionTransferRequest(
        segment_key="test_seg_at_002#1",
        segment_text="Missing reference video test",
        segment_duration=2.0,
        project_root=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        reference_video_path="",
        target_image_path=str(tmp_path / "input" / "target.png"),
    )
    
    adapter = ActionTransferAdapter()
    response = adapter.invoke(request)
    
    assert response.success is False
    assert response.error is not None
    assert response.error.category == ErrorCategory.VALIDATION
    assert "reference_video_path" in response.error.message
    assert response.error.details.get("field") == "reference_video_path"


def test_action_transfer_adapter_validation_error_unsupported_format(tmp_path: Path):
    from pixelle_snapshot.adapters import ActionTransferAdapter, ActionTransferRequest
    
    request = ActionTransferRequest(
        segment_key="test_seg_at_003#1",
        segment_text="Unsupported format test",
        segment_duration=2.0,
        project_root=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        reference_video_path=str(tmp_path / "input" / "reference.wmv"),
        target_image_path=str(tmp_path / "input" / "target.png"),
    )
    
    adapter = ActionTransferAdapter()
    response = adapter.invoke(request)
    
    assert response.success is False
    assert response.error is not None
    assert response.error.category == ErrorCategory.VALIDATION
    assert ".wmv" in response.error.message.lower()
    assert response.error.details.get("field") == "reference_video_path"
    assert response.error.details.get("unsupported_extension") == ".wmv"


def test_step4_action_transfer_route_happy_path(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    routed_capabilities: list[str] = []

    class FakeActionTransferAdapter:
        def invoke(self, request):
            output_path = Path(request.output_dir) / f"action_transfer_{request.segment_key}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"action-transfer-video")
            return type(
                "Resp",
                (),
                {
                    "success": True,
                    "output_path": str(output_path),
                    "error": None,
                },
            )()

    def fake_get_adapter(name: str, **kwargs):
        routed_capabilities.append(name)
        return FakeActionTransferAdapter()

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", fake_get_adapter)

    seg = _make_action_transfer_segment()
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

    assert routed_capabilities == ["action_transfer"]
    assert resolved.asset_refs[0].kind == "pixelle_video"
    assert resolved.asset_refs[0].fallback_reason_code is None
    assert resolved.asset_refs[0].fallback_error_category is None
    assert resolved.visual_plan is not None
    assert resolved.visual_plan.asset_path is not None
    assert resolved.visual_plan.asset_path.endswith(".mp4")


def test_step4_action_transfer_validation_error_falls_back(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    class FailingAdapter:
        def invoke(self, request):
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(
                        category=ErrorCategory.VALIDATION,
                        message="reference_video_path is required",
                    ),
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

    seg = _make_action_transfer_segment()
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
    assert resolved.asset_refs[0].fallback_reason_code == "PIXELLE_INVOCATION_FAILED"
    assert resolved.asset_refs[0].fallback_error_category == "VALIDATION"
    assert resolved.visual_plan is not None
    assert resolved.visual_plan.asset_path is not None
    assert resolved.visual_plan.asset_path.endswith(".png")


# =============================================================================
# TASK 15: Auto-Mode Regression Suite for Non-AI Behavior Stability
# =============================================================================
# These tests prove that 'auto' mode preserves legacy non-AI selection behavior
# and that ai_only policy rules do not leak into auto mode.
# =============================================================================


@pytest.mark.regression
def test_auto_mode_regression_pdf_chart_branch_succeeds(monkeypatch, tmp_path: Path):
    """Regression: auto mode must successfully route pdf_chart without policy block."""
    monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    # Create charts directory with pdf_chart image
    charts_dir = tmp_path / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    (charts_dir / "chart1.png").write_bytes(b"fake-pdf-chart-image")

    seg = _make_pdf_chart_segment()
    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="auto",
    )

    assert resolved.asset_refs[0].kind == "pdf_chart"
    assert resolved.asset_refs[0].fallback_reason_code is None
    assert "MODE_POLICY_BLOCK" not in str(resolved.asset_refs[0].fallback_reason_code)


@pytest.mark.regression
def test_auto_mode_regression_pexels_video_branch_succeeds(monkeypatch, tmp_path: Path):
    """Regression: auto mode must successfully route pexels_video without policy block."""
    monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    pexels_cache = tmp_path / "assets" / "pexels_cache" / "videos"
    pexels_cache.mkdir(parents=True, exist_ok=True)
    pexels_video_path = pexels_cache / "fallback_video.mp4"
    pexels_video_path.write_bytes(b"pexels-video-content" * 1000)

    def fake_pexels_video(**kwargs):
        return str(pexels_video_path)

    monkeypatch.setattr("src.steps.step4_assets.fetch_pexels_video", fake_pexels_video)

    seg = _make_segment_with_keywords()
    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="test-api-key",
        enable_pexels_video=True,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="auto",
    )

    assert resolved.asset_refs[0].kind == "pexels_video"
    assert resolved.asset_refs[0].fallback_reason_code is None
    assert "MODE_POLICY_BLOCK" not in str(resolved.asset_refs[0].fallback_reason_code)


@pytest.mark.regression
def test_auto_mode_regression_pexels_photo_branch_succeeds(monkeypatch, tmp_path: Path):
    """Regression: auto mode must successfully route pexels_photo without policy block."""
    monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    pexels_cache = tmp_path / "assets" / "pexels_cache" / "photos"
    pexels_cache.mkdir(parents=True, exist_ok=True)
    pexels_photo_path = pexels_cache / "fallback_photo.jpg"
    pexels_photo_path.write_bytes(b"pexels-photo-content" * 100)

    def fake_pexels_photo(**kwargs):
        return str(pexels_photo_path)

    monkeypatch.setattr("src.steps.step4_assets.fetch_pexels_photo", fake_pexels_photo)

    seg = _make_segment_with_keywords()
    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="test-api-key",
        enable_pexels_video=False,
        enable_pexels_photo=True,
        enable_ai_image=False,
        material_mode="auto",
    )

    assert resolved.asset_refs[0].kind == "pexels_photo"
    assert resolved.asset_refs[0].fallback_reason_code is None
    assert "MODE_POLICY_BLOCK" not in str(resolved.asset_refs[0].fallback_reason_code)


@pytest.mark.regression
def test_auto_mode_regression_template_fallback_succeeds(monkeypatch, tmp_path: Path):
    """Regression: auto mode must successfully fall back to template without policy block."""
    monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    def fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template-content")
        return str(path)

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
        material_mode="auto",
    )

    assert resolved.asset_refs[0].kind == "template"
    assert "MODE_POLICY_BLOCK" not in str(resolved.asset_refs[0].fallback_reason_code)


@pytest.mark.regression
def test_mode_boundary_isolation_auto_allows_what_ai_only_blocks(monkeypatch, tmp_path: Path):
    """Regression: auto mode must allow routes that ai_only would block."""
    from src.steps.step4_assets import is_route_allowed_by_mode_policy

    # ai_only blocks non-AI routes
    assert is_route_allowed_by_mode_policy("pexels_video", "ai_only") is False
    assert is_route_allowed_by_mode_policy("pexels_photo", "ai_only") is False
    assert is_route_allowed_by_mode_policy("pdf_chart", "ai_only") is False
    assert is_route_allowed_by_mode_policy("template", "ai_only") is False

    # auto allows ALL routes (no policy blocking)
    assert is_route_allowed_by_mode_policy("pexels_video", "auto") is True
    assert is_route_allowed_by_mode_policy("pexels_photo", "auto") is True
    assert is_route_allowed_by_mode_policy("pdf_chart", "auto") is True
    assert is_route_allowed_by_mode_policy("template", "auto") is True
    assert is_route_allowed_by_mode_policy("ai_image", "auto") is True
    assert is_route_allowed_by_mode_policy("pixelle_video", "auto") is True


@pytest.mark.regression
def test_mode_boundary_isolation_auto_no_policy_block_diagnostic(monkeypatch, tmp_path: Path):
    """Regression: auto mode must never produce MODE_POLICY_BLOCK diagnostic."""
    monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    def fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template-content")
        return str(path)

    monkeypatch.setattr("src.steps.step4_assets.generate_template_asset", fake_template)

    # Disable all non-template routes to force fallback chain
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
        material_mode="auto",
    )

    # Check that no MODE_POLICY_BLOCK appears in any diagnostic
    for ref in resolved.asset_refs:
        assert ref.fallback_reason_code != "MODE_POLICY_BLOCK", \
            f"auto mode must not produce MODE_POLICY_BLOCK, got: {ref.fallback_reason_code}"


@pytest.mark.regression
def test_mode_boundary_isolation_auto_pexels_no_exhausted(monkeypatch, tmp_path: Path):
    """Regression: auto mode pexels failure should report PEXELS_EXHAUSTED not MODE_POLICY_BLOCK."""
    monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    def fake_pexels_video_fail(**kwargs):
        return None  # Simulate no results

    def fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template-content")
        return str(path)

    monkeypatch.setattr("src.steps.step4_assets.fetch_pexels_video", fake_pexels_video_fail)
    monkeypatch.setattr("src.steps.step4_assets.generate_template_asset", fake_template)

    seg = _make_segment_with_keywords()
    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="test-api-key",
        enable_pexels_video=True,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="auto",
    )

    # Should fall back to template with PEXELS_EXHAUSTED, not MODE_POLICY_BLOCK
    assert resolved.asset_refs[0].kind == "template"
    assert resolved.asset_refs[0].fallback_reason_code != "MODE_POLICY_BLOCK"


@pytest.mark.regression
def test_mode_boundary_isolation_policy_helper_behavior():
    """Regression: is_route_allowed_by_mode_policy must have correct boundary behavior."""
    from src.steps.step4_assets import is_route_allowed_by_mode_policy

    # Define expected behaviors for each mode
    auto_expected = {
        "pdf_chart": True,
        "pexels_video": True,
        "pexels_photo": True,
        "template": True,
        "ai_image": True,
        "pixelle_video": True,
    }

    ai_only_expected = {
        "pdf_chart": False,
        "pexels_video": False,
        "pexels_photo": False,
        "template": False,
        "ai_image": True,
        "pixelle_video": True,
    }

    # Verify auto mode allows all
    for route, expected in auto_expected.items():
        actual = is_route_allowed_by_mode_policy(route, "auto")
        assert actual == expected, f"auto mode: {route} expected {expected}, got {actual}"

    # Verify ai_only mode blocks non-AI
    for route, expected in ai_only_expected.items():
        actual = is_route_allowed_by_mode_policy(route, "ai_only")
        assert actual == expected, f"ai_only mode: {route} expected {expected}, got {actual}"


@pytest.mark.regression
def test_auto_mode_regression_pixelle_failure_falls_to_pexels(monkeypatch, tmp_path: Path):
    """Regression: auto mode pixelle failure should fall back to pexels, not block."""
    monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    class FailingAdapter:
        def invoke(self, request):
            raise AdapterError(category=ErrorCategory.EXECUTION, message="Pixelle failed")

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FailingAdapter())

    pexels_cache = tmp_path / "assets" / "pexels_cache" / "videos"
    pexels_cache.mkdir(parents=True, exist_ok=True)
    pexels_video_path = pexels_cache / "fallback_video.mp4"
    pexels_video_path.write_bytes(b"pexels-video-content" * 1000)

    def fake_pexels_video(**kwargs):
        return str(pexels_video_path)

    monkeypatch.setattr("src.steps.step4_assets.fetch_pexels_video", fake_pexels_video)

    seg = _make_segment_with_keywords()
    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="test-api-key",
        enable_pexels_video=True,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="auto",
    )

    # Should fall back to pexels_video after pixelle failure
    assert resolved.asset_refs[0].kind == "pexels_video"
    assert resolved.asset_refs[0].fallback_reason_code != "MODE_POLICY_BLOCK"


@pytest.mark.unit
def test_cap_telemetry_summary(monkeypatch, tmp_path: Path):
    """T14: Verify step4_cap_telemetry dict is exposed on manifest with correct keys."""
    manifest = _build_top6_test_manifest(8)

    # Track which routes are assigned to each segment
    route_assignments: dict[str, str] = {}

    def fake_resolve_asset_for_segment(segment, **kwargs):
        # Simulate: first 6 segments get AI route, last 2 get template fallback
        ai_selected = getattr(segment, "step4_ai_selected", False)
        if ai_selected:
            kind = "ai_image"
        else:
            kind = "template"
        route_assignments[segment.segment_key] = kind
        segment.asset_refs = [AssetRef(kind=kind, path=f"/tmp/{kind}.mp4", asset_hash=f"{kind}-hash")]
        return segment

    monkeypatch.setattr("src.steps.step4_assets.resolve_asset_for_segment", fake_resolve_asset_for_segment)

    updated_manifest = run_step4(
        manifest=manifest,
        output_manifest=str(tmp_path / "build" / "manifest_step4.json"),
        project_root=str(tmp_path),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=True,
    )

    # Verify cap_telemetry is exposed
    cap_telemetry = getattr(updated_manifest, "step4_cap_telemetry", None)
    assert cap_telemetry is not None, "step4_cap_telemetry should be set on manifest"

    # Verify required keys exist
    required_keys = {
        "total_segments",
        "ai_selected_count",
        "ai_skipped_over_cap_count",
        "ai_routed_count",
        "non_ai_replacement_count",
        "processed",
        "skipped",
        "source_counts",
    }
    assert set(cap_telemetry.keys()) == required_keys, f"Missing keys: {required_keys - set(cap_telemetry.keys())}"

    # Verify total_segments matches manifest
    assert cap_telemetry["total_segments"] == 8

    # Verify ai_selected_count matches allocation map (top 6 segments)
    assert cap_telemetry["ai_selected_count"] == 6

    # Verify ai_skipped_over_cap_count = total - ai_selected
    assert cap_telemetry["ai_skipped_over_cap_count"] == 2

    # Verify processed count
    assert cap_telemetry["processed"] == 8
    assert cap_telemetry["skipped"] == 0

    # Verify source_counts is a dict
    assert isinstance(cap_telemetry["source_counts"], dict)


@pytest.mark.unit
def test_cap_telemetry_consistency(monkeypatch, tmp_path: Path):
    """T14: Verify cap telemetry counters are internally consistent."""
    manifest = _build_ai_cap_test_manifest(8)

    def fake_resolve_asset_for_segment(segment, **kwargs):
        # AI-selected segments get pixelle_video (AI route)
        # Non-selected segments get template (non-AI fallback)
        ai_selected = getattr(segment, "step4_ai_selected", False)
        if ai_selected:
            kind = "pixelle_video"
        else:
            kind = "template"
        segment.asset_refs = [AssetRef(kind=kind, path=f"/tmp/{kind}.mp4", asset_hash=f"{kind}-hash")]
        return segment

    monkeypatch.setattr("src.steps.step4_assets.resolve_asset_for_segment", fake_resolve_asset_for_segment)

    updated_manifest = run_step4(
        manifest=manifest,
        output_manifest=str(tmp_path / "build" / "manifest_step4.json"),
        project_root=str(tmp_path),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=True,
    )

    cap_telemetry = getattr(updated_manifest, "step4_cap_telemetry", None)
    assert cap_telemetry is not None

    # CONSISTENCY CHECK 1: ai_selected_count + ai_skipped_over_cap_count == total_segments
    assert (
        cap_telemetry["ai_selected_count"] + cap_telemetry["ai_skipped_over_cap_count"]
        == cap_telemetry["total_segments"]
    ), "ai_selected_count + ai_skipped_over_cap_count must equal total_segments"

    # CONSISTENCY CHECK 2: ai_routed_count + non_ai_replacement_count == ai_selected_count
    # (only AI-selected segments can be routed or replaced)
    assert (
        cap_telemetry["ai_routed_count"] + cap_telemetry["non_ai_replacement_count"]
        == cap_telemetry["ai_selected_count"]
    ), "ai_routed_count + non_ai_replacement_count must equal ai_selected_count"

    # CONSISTENCY CHECK 3: processed + skipped == total_segments
    assert (
        cap_telemetry["processed"] + cap_telemetry["skipped"] == cap_telemetry["total_segments"]
    ), "processed + skipped must equal total_segments"

    # CONSISTENCY CHECK 4: source_counts values should sum to processed
    source_total = sum(cap_telemetry["source_counts"].values())
    assert source_total == cap_telemetry["processed"], "source_counts sum must equal processed"

    # Verify specific values for 8-segment manifest with cap=6
    assert cap_telemetry["total_segments"] == 8
    assert cap_telemetry["ai_selected_count"] == 6  # top 6 selected
    assert cap_telemetry["ai_skipped_over_cap_count"] == 2  # 2 over cap
    assert cap_telemetry["ai_routed_count"] == 6  # all 6 selected got AI route
    assert cap_telemetry["non_ai_replacement_count"] == 0  # no fallbacks for AI-selected


# ─────────────────────────────────────────────
# Task 13: ai_only cap-fallback diagnostic tests
# ─────────────────────────────────────────────


def test_ai_only_cap_fallback_diagnostic(monkeypatch, tmp_path: Path):
    """Cap-driven fallback in ai_only mode produces AI_ONLY_CAP_POLICY_FALLBACK, not ROUTES_EXHAUSTED."""
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    text = "cap policy diagnostic segment test"
    content_key = Segment.compute_content_key(text)
    seg = Segment(
        segment_key=Segment.compute_segment_key(content_key, 1),
        content_key=content_key,
        index=1,
        start=0.0,
        end=4.0,
        duration=4.0,
        text=text,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=4.0),
        visual_plan=VisualPlan(type="pixelle_digital_human", pixelle_workflow="digital_human"),
        plan_hash="cap-policy-test",
    )

    setattr(seg, "step4_ai_selected", False)
    setattr(seg, "step4_ai_allocation_map", {seg.segment_key: False})

    def fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template")
        return str(path)

    monkeypatch.setattr("src.steps.step4_assets.generate_template_asset", fake_template)

    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="ai_only",
    )

    assert resolved.asset_refs[0].kind == "template"
    assert resolved.asset_refs[0].fallback_reason_code == "AI_ONLY_CAP_POLICY_FALLBACK"
    assert resolved.asset_refs[0].fallback_error_category == "POLICY"

    diagnostic = resolved.asset_refs[0].fallback_diagnostic
    assert diagnostic is not None
    assert diagnostic.get("reason_code") == "AI_ONLY_CAP_POLICY_FALLBACK"


def test_ai_only_provider_exhaustion_still_strict(monkeypatch, tmp_path: Path):
    """Genuine AI route exhaustion in ai_only mode produces AI_ONLY_ROUTES_EXHAUSTED (strict failure)."""
    monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    text = "provider exhaustion strict test segment"
    content_key = Segment.compute_content_key(text)
    seg = Segment(
        segment_key=Segment.compute_segment_key(content_key, 1),
        content_key=content_key,
        index=1,
        start=0.0,
        end=4.0,
        duration=4.0,
        text=text,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=4.0),
        visual_plan=VisualPlan(type="pixelle_digital_human", pixelle_workflow="digital_human"),
        plan_hash="provider-exhaust-test",
    )

    setattr(seg, "step4_ai_selected", True)
    setattr(seg, "step4_ai_allocation_map", {seg.segment_key: True})

    class FailingAdapter:
        def invoke(self, request):
            raise AdapterError(category=ErrorCategory.EXECUTION, message="All providers failed")

    template_called: list[str] = []

    def fail_if_template_called(output_path: str, width: int, height: int, text: str):
        template_called.append("template")
        raise AssertionError("Template should not be called for ai_only strict failure")

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FailingAdapter())
    monkeypatch.setattr("src.steps.step4_assets.generate_template_asset", fail_if_template_called)

    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="ai_only",
    )

    assert resolved.asset_refs[0].kind == "ai_only_exhausted"
    assert resolved.asset_refs[0].fallback_reason_code == "AI_ONLY_ROUTES_EXHAUSTED"
    assert resolved.asset_refs[0].fallback_error_category == "POLICY"

    assert template_called == []

    diagnostic = resolved.asset_refs[0].fallback_diagnostic
    assert diagnostic is not None
    assert diagnostic.get("reason_code") == "AI_ONLY_ROUTES_EXHAUSTED"


# ════════════════════════════════════════════════════════════════════════════
# Task 15: Step4 Routing Contract Tests for Cap Behavior
# ════════════════════════════════════════════════════════════════════════════


def test_ai_cap_never_exceeds_six(monkeypatch, tmp_path: Path):
    """INVARIANT: AI allocation cap never exceeds 6, regardless of manifest size.

    This is the hard cap guarantee — even with 100 segments, only 6 get AI routes.
    Tests the allocation map planner + run_step4 integration.
    """
    # Test with various segment counts: 3, 6, 8, 12, 20
    for segment_count in [3, 6, 8, 12, 20]:
        # Use _build_ai_cap_test_manifest which includes pixelle_workflow
        manifest = _build_ai_cap_test_manifest(segment_count)
        generated_dir = tmp_path / f"gen_{segment_count}"
        generated_dir.mkdir(parents=True, exist_ok=True)

        adapter_calls: list[str] = []

        class FakeAdapter:
            def invoke(self, request):
                adapter_calls.append(request.segment_key)
                output_path = Path(request.output_dir) / f"pixelle_{request.segment_key}.mp4"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"ai-video")
                return type(
                    "Resp",
                    (),
                    {
                        "success": True,
                        "output_path": str(output_path),
                        "error": None,
                    },
                )()

        def fake_template(output_path: str, width: int, height: int, text: str):
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"template")
            return str(path)

        monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
        monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FakeAdapter())
        monkeypatch.setattr("src.steps.step4_assets.generate_template_asset", fake_template)

        updated_manifest = run_step4(
            manifest=manifest,
            output_manifest=str(tmp_path / f"manifest_{segment_count}.json"),
            project_root=str(tmp_path),
            pexels_api_key="",
            enable_pexels_video=False,
            enable_pexels_photo=False,
            enable_ai_image=False,
        )

        allocation_map = getattr(updated_manifest, "step4_ai_allocation_map")
        selected_count = sum(1 for v in allocation_map.values() if v)

        # HARD INVARIANT: AI cap is <= 6
        assert selected_count <= 6, f"Cap invariant violated for {segment_count} segments: got {selected_count}"

        # Adapter should only be called for selected segments
        assert len(adapter_calls) <= 6, f"Adapter calls exceeded cap for {segment_count} segments"

        # Verify allocation matches adapter calls
        selected_keys = {k for k, v in allocation_map.items() if v}
        assert set(adapter_calls) == selected_keys

        # Reset for next iteration
        adapter_calls.clear()


def test_cap_exact_six_segments(monkeypatch, tmp_path: Path):
    """BOUNDARY: With exactly 6 segments, all 6 get AI routes."""
    manifest = _build_ai_cap_test_manifest(6)
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    adapter_calls: list[str] = []

    class FakeAdapter:
        def invoke(self, request):
            adapter_calls.append(request.segment_key)
            output_path = Path(request.output_dir) / f"pixelle_{request.segment_key}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"ai-video")
            return type(
                "Resp",
                (),
                {
                    "success": True,
                    "output_path": str(output_path),
                    "error": None,
                },
            )()

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FakeAdapter())

    updated_manifest = run_step4(
        manifest=manifest,
        output_manifest=str(tmp_path / "manifest_exact6.json"),
        project_root=str(tmp_path),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
    )

    allocation_map = getattr(updated_manifest, "step4_ai_allocation_map")
    selected_keys = {k for k, v in allocation_map.items() if v}

    # EXACT BOUNDARY: All 6 segments selected
    assert len(selected_keys) == 6
    assert len(adapter_calls) == 6

    # Verify all segments got AI route
    refs_by_key = {seg.segment_key: seg.asset_refs[0] for seg in updated_manifest.segments}
    assert all(refs_by_key[key].kind == "pixelle_video" for key in selected_keys)


def test_cap_over_six_downgrades_deterministically(monkeypatch, tmp_path: Path):
    """OVER-CAP: With >6 segments, bottom-ranked segments get non-AI routes deterministically."""
    manifest = _build_ai_cap_test_manifest(10)
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    adapter_calls: list[str] = []

    class FakeAdapter:
        def invoke(self, request):
            adapter_calls.append(request.segment_key)
            output_path = Path(request.output_dir) / f"pixelle_{request.segment_key}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"ai-video")
            return type(
                "Resp",
                (),
                {
                    "success": True,
                    "output_path": str(output_path),
                    "error": None,
                },
            )()

    def fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template")
        return str(path)

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FakeAdapter())
    monkeypatch.setattr("src.steps.step4_assets.generate_template_asset", fake_template)

    updated_manifest = run_step4(
        manifest=manifest,
        output_manifest=str(tmp_path / "manifest_over6.json"),
        project_root=str(tmp_path),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
    )

    allocation_map = getattr(updated_manifest, "step4_ai_allocation_map")
    selected_keys = {k for k, v in allocation_map.items() if v}
    non_selected_keys = {k for k, v in allocation_map.items() if not v}

    # OVER-CAP: Exactly 6 selected, 4 downgraded
    assert len(selected_keys) == 6
    assert len(non_selected_keys) == 4

    # Verify AI calls match selection
    assert len(adapter_calls) == 6
    assert set(adapter_calls) == selected_keys

    # DETERMINISM: Compute expected selection via score ranking
    ranked = sorted(
        manifest.segments,
        key=lambda seg: compute_segment_semantic_priority_score(seg)["tie_break"]["rank_key"],
    )
    expected_selected = {seg.segment_key for seg in ranked[:6]}
    assert selected_keys == expected_selected

    # Verify non-selected got non-AI fallback
    refs_by_key = {seg.segment_key: seg.asset_refs[0] for seg in updated_manifest.segments}
    for key in non_selected_keys:
        assert refs_by_key[key].kind != "pixelle_video", f"Non-selected {key} got AI route"
        assert refs_by_key[key].kind in {"template", "pexels_video", "pexels_photo", "cached"}


def test_cap_tie_break_deterministic(monkeypatch, tmp_path: Path):
    """TIE-BREAK: Equal-score segments are ordered by index deterministically."""
    # Create segments with identical semantic signals (same keywords, same prompt)
    segments: list[Segment] = []
    for idx in range(1, 9):  # 8 segments with identical semantics
        text = "Identical semantic content for tie-break test"
        content_key = Segment.compute_content_key(text + str(idx))  # Unique key
        seg = Segment(
            segment_key=Segment.compute_segment_key(content_key, idx),
            content_key=content_key,
            index=idx,
            start=float((idx - 1) * 4),
            end=float(idx * 4),
            duration=4.0,
            text=text,
            audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=4.0),
            visual_plan=VisualPlan(
                type="pixelle_digital_human",
                pixelle_workflow="digital_human",
                keywords=["same", "keywords"],
                prompt="same prompt",
            ),
            plan_hash=f"tiebreak-{idx}",
        )
        segments.append(seg)

    manifest = Manifest(project_id="tiebreak-test", segments=segments, material_mode="ai_only")
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    adapter_calls: list[str] = []

    class FakeAdapter:
        def invoke(self, request):
            adapter_calls.append(request.segment_key)
            output_path = Path(request.output_dir) / f"pixelle_{request.segment_key}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"ai-video")
            return type(
                "Resp",
                (),
                {
                    "success": True,
                    "output_path": str(output_path),
                    "error": None,
                },
            )()

    def fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template")
        return str(path)

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FakeAdapter())
    monkeypatch.setattr("src.steps.step4_assets.generate_template_asset", fake_template)

    updated_manifest = run_step4(
        manifest=manifest,
        output_manifest=str(tmp_path / "manifest_tiebreak.json"),
        project_root=str(tmp_path),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
    )

    allocation_map = getattr(updated_manifest, "step4_ai_allocation_map")
    selected_keys = {k for k, v in allocation_map.items() if v}

    # TIE-BREAK: First 6 by index should be selected (lower index wins)
    expected_indices = {1, 2, 3, 4, 5, 6}
    selected_segments = [seg for seg in manifest.segments if seg.segment_key in selected_keys]
    selected_indices = {seg.index for seg in selected_segments}

    assert selected_indices == expected_indices, f"Tie-break violated: expected indices {expected_indices}, got {selected_indices}"

    # Verify determinism across runs
    allocation_map_2 = build_top6_ai_allocation_map(manifest.segments, max_ai_segments=6)
    selected_keys_2 = {k for k, v in allocation_map_2.items() if v}
    assert selected_keys == selected_keys_2, "Tie-break not deterministic across runs"


def test_target_segment_keys_limits_ai_allocation(monkeypatch, tmp_path: Path):
    """SUBSET: target_segment_keys parameter limits allocation eligibility."""
    manifest = _build_ai_cap_test_manifest(8)
    # Only process segments 1-5 (keys for indices 1-5)
    target_subset = [seg.segment_key for seg in manifest.segments[:5]]

    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    adapter_calls: list[str] = []

    class FakeAdapter:
        def invoke(self, request):
            adapter_calls.append(request.segment_key)
            output_path = Path(request.output_dir) / f"pixelle_{request.segment_key}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"ai-video")
            return type(
                "Resp",
                (),
                {
                    "success": True,
                    "output_path": str(output_path),
                    "error": None,
                },
            )()

    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FakeAdapter())

    updated_manifest = run_step4(
        manifest=manifest,
        output_manifest=str(tmp_path / "manifest_subset.json"),
        project_root=str(tmp_path),
        target_segment_keys=target_subset,
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
    )

    allocation_map = getattr(updated_manifest, "step4_ai_allocation_map")
    selected_keys = {k for k, v in allocation_map.items() if v}
    outside_keys = {seg.segment_key for seg in manifest.segments} - set(target_subset)

    # SUBSET: Selected keys must be within target_subset
    assert all(k in set(target_subset) for k in selected_keys), "Selection leaked outside target subset"

    # SUBSET: Keys outside subset must be False
    assert all(allocation_map[k] is False for k in outside_keys), "Keys outside subset were allocated"

    # Adapter should only be called for subset
    assert all(k in set(target_subset) for k in adapter_calls), "Adapter called for non-target segment"


def test_cache_interaction_respects_ai_cap(monkeypatch, tmp_path: Path):
    """CACHE: Cached AI assets do not bypass the allocation cap."""
    from src.steps import step4_assets

    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    # Create a segment that WOULD have a valid AI cache but is NOT in top-6
    text = "Cached AI segment not in top-6"
    content_key = Segment.compute_content_key(text)
    seg = Segment(
        segment_key=Segment.compute_segment_key(content_key, 99),  # High index = low priority
        content_key=content_key,
        index=99,
        start=0.0,
        end=4.0,
        duration=4.0,
        text=text,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=4.0),
        visual_plan=VisualPlan(type="pixelle_digital_human", pixelle_workflow="digital_human"),
        plan_hash="cache-cap-test",
    )

    # Write cached AI asset
    effective_cache_hash = step4_assets._compute_effective_cache_hash(seg.plan_hash or "", "digital_human")
    cached_path = Path(step4_assets._asset_cache_path(str(generated_dir), content_key, effective_cache_hash, "mp4"))
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.write_bytes(b"cached-ai-video" * 200)

    meta_path = Path(step4_assets._asset_cache_meta_path(str(cached_path)))
    meta_path.write_text(
        '{"kind": "pixelle_video", "material_mode": "ai_only", "is_ai_generated": true}',
        encoding="utf-8",
    )

    # Mark segment as NOT selected (over cap)
    setattr(seg, "step4_ai_selected", False)
    setattr(seg, "step4_ai_allocation_map", {seg.segment_key: False})

    adapter_calls: list[str] = []

    def fail_adapter(name: str, **kwargs):
        adapter_calls.append(name)
        raise AssertionError("Adapter should not be called for non-selected segment")

    def fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template")
        return str(path)

    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", fail_adapter)
    monkeypatch.setattr("src.steps.step4_assets.generate_template_asset", fake_template)

    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="ai_only",
    )

    # CACHE: Cache should be bypassed due to cap
    assert adapter_calls == [], "Adapter should not be invoked for non-selected segment"
    assert resolved.asset_refs[0].kind != "pixelle_video", "Cached AI should not bypass cap"
    assert resolved.asset_refs[0].kind != "cached", "Cached AI should be blocked"
    assert resolved.asset_refs[0].kind == "template", "Should fall back to template"


def test_ai_only_cap_fallback_diagnostic_vs_exhausted(monkeypatch, tmp_path: Path):
    """DIAGNOSTIC: Cap-driven fallback produces AI_ONLY_CAP_POLICY_FALLBACK, not ROUTES_EXHAUSTED."""
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    text = "cap policy diagnostic comparison"
    content_key = Segment.compute_content_key(text)
    seg = Segment(
        segment_key=Segment.compute_segment_key(content_key, 1),
        content_key=content_key,
        index=1,
        start=0.0,
        end=4.0,
        duration=4.0,
        text=text,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=4.0),
        visual_plan=VisualPlan(type="pixelle_digital_human", pixelle_workflow="digital_human"),
        plan_hash="cap-vs-exhausted-test",
    )

    # NOT selected (over cap)
    setattr(seg, "step4_ai_selected", False)
    setattr(seg, "step4_ai_allocation_map", {seg.segment_key: False})

    def fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template")
        return str(path)

    monkeypatch.setattr("src.steps.step4_assets.generate_template_asset", fake_template)

    resolved = resolve_asset_for_segment(
        segment=seg,
        project_root=str(project_root),
        generated_dir=str(generated_dir),
        library_dir=str(tmp_path / "assets" / "library"),
        pexels_api_key="",
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
        material_mode="ai_only",
    )

    ref = resolved.asset_refs[0]
    # DIAGNOSTIC: Must be CAP_POLICY, not ROUTES_EXHAUSTED
    assert ref.fallback_reason_code == "AI_ONLY_CAP_POLICY_FALLBACK"
    assert ref.fallback_reason_code != "AI_ONLY_ROUTES_EXHAUSTED"
    assert ref.fallback_error_category == "POLICY"

    diagnostic = ref.fallback_diagnostic
    assert diagnostic is not None
    assert diagnostic["reason_code"] == "AI_ONLY_CAP_POLICY_FALLBACK"
    assert "cap" in diagnostic.get("guidance", "").lower() or "allocation" in diagnostic.get("guidance", "").lower()
