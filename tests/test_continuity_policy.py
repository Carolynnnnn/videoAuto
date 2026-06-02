from pathlib import Path

from src.core.models import AudioRef, Segment, VisualPlan
from src.steps.continuity_policy import evaluate_continuity_policy


def _make_segment(index: int, text: str) -> Segment:
    content_key = Segment.compute_content_key(text)
    return Segment(
        segment_key=Segment.compute_segment_key(content_key, index),
        content_key=content_key,
        index=index,
        start=float(index - 1),
        end=float(index),
        duration=1.0,
        text=text,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=1.0),
        visual_plan=VisualPlan(type="pixelle_i2v", pixelle_workflow="i2v"),
        plan_hash=f"hash-{index}",
    )


def test_continuity_policy_temporal_chain(tmp_path: Path):
    project_root = tmp_path
    frame_path = tmp_path / "artifacts" / "continuity" / "frames" / "seg1_end.png"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    frame_path.write_bytes(b"frame")

    first_segment = _make_segment(1, "first")
    first_directive = evaluate_continuity_policy(
        segment=first_segment,
        previous_segment=None,
        policy_mode="frame_chain",
        continuity_seed=123,
        style_id="style-a",
        project_id="project-1",
        vendor_preference="pixelle",
        project_root=str(project_root),
        resolution=(1080, 1920),
    )
    assert first_directive.continuity_mode == "seed_lock"
    assert first_directive.fallback_reason_code == "PIXELLE_CONTINUITY_FIRST_SEGMENT"

    previous_segment = _make_segment(1, "first")
    previous_segment.prev_last_frame_path = str(frame_path)
    middle_segment = _make_segment(2, "middle")
    chained_directive = evaluate_continuity_policy(
        segment=middle_segment,
        previous_segment=previous_segment,
        policy_mode="frame_chain",
        continuity_seed=123,
        style_id="style-a",
        project_id="project-1",
        vendor_preference="pixelle",
        project_root=str(project_root),
        resolution=(1080, 1920),
    )

    assert chained_directive.continuity_mode == "temporal"
    assert chained_directive.start_frame_path == str(frame_path)
    assert chained_directive.seed == 123
    assert chained_directive.fallback_reason_code is None
    assert chained_directive.diagnostic is not None
    assert chained_directive.diagnostic["continuity_mode"] == "temporal"


def test_continuity_policy_temporal_unsupported(tmp_path: Path):
    previous_segment = _make_segment(1, "first")
    current_segment = _make_segment(2, "second")

    directive = evaluate_continuity_policy(
        segment=current_segment,
        previous_segment=previous_segment,
        policy_mode="frame_chain",
        continuity_seed=777,
        style_id="style-b",
        project_id="project-2",
        vendor_preference="minimax",
        project_root=str(tmp_path),
        resolution=(1080, 1920),
    )

    assert directive.continuity_mode == "seed_lock"
    assert directive.seed == 777
    assert directive.fallback_reason_code == "PIXELLE_CONTINUITY_TEMPORAL_UNSUPPORTED"
    assert directive.diagnostic is not None
    assert directive.diagnostic["fallback_reason_code"] == "PIXELLE_CONTINUITY_TEMPORAL_UNSUPPORTED"
    assert directive.diagnostic["fallback_diagnostic"]["category"] == "UNSUPPORTED"


def test_continuity_policy_missing_prior_artifact(tmp_path: Path):
    previous_segment = _make_segment(1, "first")
    current_segment = _make_segment(2, "second")

    directive = evaluate_continuity_policy(
        segment=current_segment,
        previous_segment=previous_segment,
        policy_mode="frame_chain",
        continuity_seed=None,
        style_id="style-c",
        project_id="project-3",
        vendor_preference="pixelle",
        project_root=str(tmp_path),
        resolution=(1080, 1920),
    )

    assert directive.continuity_mode == "seed_lock"
    assert directive.fallback_reason_code == "PIXELLE_CONTINUITY_PRIOR_ARTIFACT_MISSING"
    assert directive.seed is not None


def test_continuity_policy_style_anchor_mode():
    """Test style_anchor continuity policy."""
    segment = _make_segment(1, "test")
    
    directive = evaluate_continuity_policy(
        segment=segment,
        previous_segment=None,
        policy_mode="style_anchor",
        continuity_seed=123,
        style_id="style-x",
        project_id="project-1",
        vendor_preference="pixelle",
        project_root="/tmp",
        resolution=(1080, 1920),
    )
    
    assert directive.continuity_mode == "style_anchor"
    assert directive.reason_code == "PIXELLE_CONTINUITY_STYLE_ANCHOR"
    assert directive.seed is None


def test_continuity_policy_seed_lock_mode():
    """Test seed_lock continuity policy."""
    segment = _make_segment(1, "test")
    
    directive = evaluate_continuity_policy(
        segment=segment,
        previous_segment=None,
        policy_mode="seed_lock",
        continuity_seed=456,
        style_id="style-y",
        project_id="project-2",
        vendor_preference="pixelle",
        project_root="/tmp",
        resolution=(1080, 1920),
    )
    
    assert directive.continuity_mode == "seed_lock"
    assert directive.reason_code == "PIXELLE_CONTINUITY_SEED_LOCKED"
    assert directive.seed == 456


def test_continuity_policy_unknown_policy_fallback():
    """Test unknown policy mode falls back to seed_lock."""
    segment = _make_segment(1, "test")
    
    directive = evaluate_continuity_policy(
        segment=segment,
        previous_segment=None,
        policy_mode="invalid_mode",
        continuity_seed=789,
        style_id="style-z",
        project_id="project-3",
        vendor_preference="pixelle",
        project_root="/tmp",
        resolution=(1080, 1920),
    )
    
    assert directive.continuity_mode == "seed_lock"
    assert directive.reason_code == "PIXELLE_CONTINUITY_UNKNOWN_POLICY"
    assert directive.fallback_reason_code == "PIXELLE_CONTINUITY_UNKNOWN_POLICY"
    assert directive.seed == 789


def test_continuity_policy_vendor_fixture_not_found():
    """Test vendor with no fixture file falls back to defaults."""
    segment = _make_segment(1, "test")
    previous_segment = _make_segment(0, "prev")
    
    directive = evaluate_continuity_policy(
        segment=segment,
        previous_segment=previous_segment,
        policy_mode="frame_chain",
        continuity_seed=111,
        style_id="style-a",
        project_id="project-x",
        vendor_preference="unknown_vendor",
        project_root="/tmp",
        resolution=(1080, 1920),
    )
    
    # Should fall back to seed_lock due to no end_frame support
    assert directive.continuity_mode == "seed_lock"
    assert directive.fallback_reason_code == "PIXELLE_CONTINUITY_TEMPORAL_UNSUPPORTED"


def test_continuity_policy_vendor_fixture_load_error(tmp_path: Path):
    """Test vendor fixture with malformed JSON falls back gracefully."""
    # Create malformed fixture
    fixture_dir = tmp_path.parent.parent / "pixelle_snapshot" / "vendors" / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    bad_fixture = fixture_dir / "bad_vendor_media_contract.json"
    bad_fixture.write_text("{invalid json")
    
    segment = _make_segment(1, "test")
    previous_segment = _make_segment(0, "prev")
    
    directive = evaluate_continuity_policy(
        segment=segment,
        previous_segment=previous_segment,
        policy_mode="frame_chain",
        continuity_seed=222,
        style_id="style-b",
        project_id="project-y",
        vendor_preference="bad_vendor",
        project_root=str(tmp_path),
        resolution=(1080, 1920),
    )
    
    # Should fall back to seed_lock with default capabilities
    assert directive.continuity_mode == "seed_lock"
    assert directive.fallback_reason_code == "PIXELLE_CONTINUITY_TEMPORAL_UNSUPPORTED"
    
    # Cleanup
    bad_fixture.unlink(missing_ok=True)


def test_continuity_policy_previous_frame_from_asset_refs(tmp_path: Path):
    from src.core.models import AssetRef
    
    video_path = tmp_path / "prev_video.mp4"
    video_path.write_bytes(b"DUMMY_VIDEO")
    
    previous_segment = _make_segment(1, "first")
    previous_segment.asset_refs = [
        AssetRef(kind="pixelle_video", path=str(video_path))
    ]
    
    current_segment = _make_segment(2, "second")
    
    from unittest.mock import patch
    with patch("src.steps.continuity_policy.extract_end_frame", return_value=(str(tmp_path / "frame.png"), None)):
        directive = evaluate_continuity_policy(
            segment=current_segment,
            previous_segment=previous_segment,
            policy_mode="frame_chain",
            continuity_seed=333,
            style_id="style-c",
            project_id="project-z",
            vendor_preference="pixelle",
            project_root=str(tmp_path),
            resolution=(1080, 1920),
        )
    
    assert directive.continuity_mode == "temporal"
    assert directive.start_frame_path is not None


def test_continuity_policy_previous_segment_non_video_asset():
    from src.core.models import AssetRef
    
    previous_segment = _make_segment(1, "first")
    previous_segment.asset_refs = [
        AssetRef(kind="image", path="/tmp/image.png")
    ]
    
    current_segment = _make_segment(2, "second")
    
    directive = evaluate_continuity_policy(
        segment=current_segment,
        previous_segment=previous_segment,
        policy_mode="frame_chain",
        continuity_seed=444,
        style_id="style-d",
        project_id="project-w",
        vendor_preference="pixelle",
        project_root="/tmp",
        resolution=(1080, 1920),
    )
    
    assert directive.continuity_mode == "seed_lock"
    assert directive.fallback_reason_code == "PIXELLE_CONTINUITY_PRIOR_ARTIFACT_MISSING"
