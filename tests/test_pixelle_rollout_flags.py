import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.steps.pixelle_rollout_flags import (
    RolloutConfig,
    RolloutDecision,
    ShadowExecutionResult,
    check_rollout_eligibility,
    compute_bucket,
    get_shadow_collector,
)


# ─────────────────────────────────────────────
# Test 1: Deterministic bucketing stability
# ─────────────────────────────────────────────
def test_compute_bucket_deterministic_same_key():
    """Same segment key returns same bucket across multiple calls."""
    segment_key = "test_segment#1"
    bucket1 = compute_bucket(segment_key)
    bucket2 = compute_bucket(segment_key)
    bucket3 = compute_bucket(segment_key)
    
    assert bucket1 == bucket2 == bucket3
    assert 0 <= bucket1 < 100


def test_compute_bucket_different_keys_vary():
    """Different segment keys return different buckets (statistical test)."""
    buckets = [compute_bucket(f"segment#{i}") for i in range(100)]
    unique_buckets = set(buckets)
    
    # Should have reasonable distribution (at least 30 unique buckets out of 100)
    assert len(unique_buckets) >= 30


def test_compute_bucket_stable_across_restarts():
    """Bucket computation is stable and does not depend on runtime state."""
    segment_key = "stable_segment#42"
    expected_bucket = compute_bucket(segment_key)
    
    # Simulate multiple "restarts" by calling compute_bucket repeatedly
    for _ in range(10):
        assert compute_bucket(segment_key) == expected_bucket


# ─────────────────────────────────────────────
# Test 2: Allowlist forces eligibility
# ─────────────────────────────────────────────
def test_allowlist_forces_eligibility_even_with_zero_rollout():
    """Allowlist key is eligible even when rollout percentage is 0."""
    config = RolloutConfig(
        enable_pixelle=True,
        rollout_percentage=0,
        allowlist_keys=frozenset(["allowlisted_segment#1"]),
        shadow_mode=False,
    )
    
    decision = check_rollout_eligibility("allowlisted_segment#1", config)
    
    assert decision.eligible is True
    assert decision.reason == "allowlisted"
    assert decision.shadow_execution is False


def test_allowlist_not_eligible_when_disabled():
    """Allowlist key is not eligible when global enable_pixelle is False."""
    config = RolloutConfig(
        enable_pixelle=False,
        rollout_percentage=100,
        allowlist_keys=frozenset(["allowlisted_segment#2"]),
        shadow_mode=False,
    )
    
    decision = check_rollout_eligibility("allowlisted_segment#2", config)
    
    assert decision.eligible is False
    assert decision.reason == "disabled"


def test_allowlist_respects_shadow_mode():
    """Allowlist key in shadow mode is eligible but flagged for shadow execution."""
    config = RolloutConfig(
        enable_pixelle=True,
        rollout_percentage=100,
        allowlist_keys=frozenset(["allowlisted_shadow#3"]),
        shadow_mode=True,
    )
    
    decision = check_rollout_eligibility("allowlisted_shadow#3", config)
    
    assert decision.eligible is True
    assert decision.reason == "allowlisted"
    assert decision.shadow_execution is True


# ─────────────────────────────────────────────
# Test 3: Canary percentage deterministic in/out
# ─────────────────────────────────────────────
def test_canary_percentage_deterministic_in_bucket():
    """Segment in rollout bucket is consistently eligible."""
    # Find a segment key that falls in bucket < 50
    segment_key = None
    for i in range(1000):
        candidate = f"canary_test#{i}"
        if compute_bucket(candidate) < 50:
            segment_key = candidate
            break
    
    assert segment_key is not None
    
    config = RolloutConfig(
        enable_pixelle=True,
        rollout_percentage=50,
        allowlist_keys=frozenset(),
        shadow_mode=False,
    )
    
    # Multiple checks should return same result
    for _ in range(5):
        decision = check_rollout_eligibility(segment_key, config)
        assert decision.eligible is True
        assert decision.reason == "canary_in_bucket"
        assert decision.shadow_execution is False


def test_canary_percentage_deterministic_out_bucket():
    """Segment outside rollout bucket is consistently ineligible."""
    # Find a segment key that falls in bucket >= 50
    segment_key = None
    for i in range(1000):
        candidate = f"canary_test#{i}"
        if compute_bucket(candidate) >= 50:
            segment_key = candidate
            break
    
    assert segment_key is not None
    
    config = RolloutConfig(
        enable_pixelle=True,
        rollout_percentage=50,
        allowlist_keys=frozenset(),
        shadow_mode=False,
    )
    
    # Multiple checks should return same result
    for _ in range(5):
        decision = check_rollout_eligibility(segment_key, config)
        assert decision.eligible is False
        assert decision.reason == "canary_out_bucket"
        assert decision.shadow_execution is False


def test_canary_percentage_edge_cases():
    """Test rollout percentage edge cases (0% and 100%)."""
    segment_key = "edge_case_segment#1"
    
    # 0% rollout: no segment is eligible
    config_zero = RolloutConfig(
        enable_pixelle=True,
        rollout_percentage=0,
        allowlist_keys=frozenset(),
        shadow_mode=False,
    )
    decision_zero = check_rollout_eligibility(segment_key, config_zero)
    assert decision_zero.eligible is False
    assert decision_zero.reason == "canary_out_bucket"
    
    # 100% rollout: all segments are eligible
    config_full = RolloutConfig(
        enable_pixelle=True,
        rollout_percentage=100,
        allowlist_keys=frozenset(),
        shadow_mode=False,
    )
    decision_full = check_rollout_eligibility(segment_key, config_full)
    assert decision_full.eligible is True
    assert decision_full.reason == "canary_in_bucket"


# ─────────────────────────────────────────────
# Test 4: Shadow mode semantics
# ─────────────────────────────────────────────
def test_shadow_mode_non_eligible_segment_executes():
    """Non-eligible segment in shadow mode executes but is marked shadow_execution=True."""
    # Find a segment key that falls outside rollout bucket
    segment_key = None
    for i in range(1000):
        candidate = f"shadow_test#{i}"
        if compute_bucket(candidate) >= 50:
            segment_key = candidate
            break
    
    assert segment_key is not None
    
    config = RolloutConfig(
        enable_pixelle=True,
        rollout_percentage=50,
        allowlist_keys=frozenset(),
        shadow_mode=True,
    )
    
    decision = check_rollout_eligibility(segment_key, config)
    
    assert decision.eligible is False
    assert decision.reason == "shadow_mode"
    assert decision.shadow_execution is True


def test_shadow_mode_eligible_segment_flagged():
    """Eligible segment in shadow mode is marked shadow_execution=True."""
    # Find a segment key that falls in rollout bucket
    segment_key = None
    for i in range(1000):
        candidate = f"shadow_eligible#{i}"
        if compute_bucket(candidate) < 50:
            segment_key = candidate
            break
    
    assert segment_key is not None
    
    config = RolloutConfig(
        enable_pixelle=True,
        rollout_percentage=50,
        allowlist_keys=frozenset(),
        shadow_mode=True,
    )
    
    decision = check_rollout_eligibility(segment_key, config)
    
    assert decision.eligible is True
    assert decision.reason == "canary_in_bucket"
    assert decision.shadow_execution is True


def test_shadow_mode_does_not_override_disabled():
    """Shadow mode does not enable Pixelle when globally disabled."""
    config = RolloutConfig(
        enable_pixelle=False,
        rollout_percentage=100,
        allowlist_keys=frozenset(),
        shadow_mode=True,
    )
    
    decision = check_rollout_eligibility("any_segment#1", config)
    
    assert decision.eligible is False
    assert decision.reason == "disabled"
    assert decision.shadow_execution is False


# ─────────────────────────────────────────────
# Test 5: Shadow execution collector
# ─────────────────────────────────────────────
def test_shadow_collector_records_results():
    """Shadow execution collector records results correctly."""
    collector = get_shadow_collector()
    collector.clear()
    
    result1 = ShadowExecutionResult(
        segment_key="shadow_seg#1",
        capability="digital_human",
        executed=True,
        success=True,
        output_path="/tmp/shadow1.mp4",
    )
    
    result2 = ShadowExecutionResult(
        segment_key="shadow_seg#2",
        capability="i2v",
        executed=True,
        success=False,
        error_code="PIXELLE_PROVIDER_ERROR",
        error_category="PROVIDER",
    )
    
    collector.record(result1)
    collector.record(result2)
    
    results = collector.get_results()
    assert len(results) == 2
    assert results[0].segment_key == "shadow_seg#1"
    assert results[0].success is True
    assert results[1].segment_key == "shadow_seg#2"
    assert results[1].success is False


def test_shadow_collector_summary_statistics():
    """Shadow collector summary provides correct statistics."""
    collector = get_shadow_collector()
    collector.clear()
    
    collector.record(ShadowExecutionResult(
        segment_key="seg#1", capability="digital_human", executed=True, success=True
    ))
    collector.record(ShadowExecutionResult(
        segment_key="seg#2", capability="digital_human", executed=True, success=False
    ))
    collector.record(ShadowExecutionResult(
        segment_key="seg#3", capability="i2v", executed=True, success=True
    ))
    
    summary = collector.summary()
    
    assert summary["total"] == 3
    assert summary["executed"] == 3
    assert summary["success"] == 2
    assert summary["failed"] == 1
    assert summary["by_capability"]["digital_human"]["total"] == 2
    assert summary["by_capability"]["digital_human"]["success"] == 1
    assert summary["by_capability"]["digital_human"]["failed"] == 1
    assert summary["by_capability"]["i2v"]["total"] == 1
    assert summary["by_capability"]["i2v"]["success"] == 1


# ─────────────────────────────────────────────
# Test 6: RolloutConfig serialization
# ─────────────────────────────────────────────
def test_rollout_config_from_dict():
    """RolloutConfig.from_dict deserializes correctly."""
    config_dict = {
        "enable_pixelle": False,
        "rollout_percentage": 25,
        "allowlist_keys": ["key1", "key2"],
        "shadow_mode": True,
    }
    
    config = RolloutConfig.from_dict(config_dict)
    
    assert config.enable_pixelle is False
    assert config.rollout_percentage == 25
    assert config.allowlist_keys == frozenset(["key1", "key2"])
    assert config.shadow_mode is True


def test_rollout_config_to_dict():
    """RolloutConfig.to_dict serializes correctly."""
    config = RolloutConfig(
        enable_pixelle=True,
        rollout_percentage=75,
        allowlist_keys=frozenset(["key_a", "key_b"]),
        shadow_mode=False,
    )
    
    config_dict = config.to_dict()
    
    assert config_dict["enable_pixelle"] is True
    assert config_dict["rollout_percentage"] == 75
    assert set(config_dict["allowlist_keys"]) == {"key_a", "key_b"}
    assert config_dict["shadow_mode"] is False


def test_rollout_config_clamps_percentage():
    """RolloutConfig clamps rollout percentage to [0, 100]."""
    config_over = RolloutConfig.from_dict({"rollout_percentage": 150})
    assert config_over.rollout_percentage == 100
    
    config_under = RolloutConfig.from_dict({"rollout_percentage": -10})
    assert config_under.rollout_percentage == 0


# ─────────────────────────────────────────────
# Test 7: RolloutDecision serialization
# ─────────────────────────────────────────────
def test_rollout_decision_to_dict():
    """RolloutDecision.to_dict serializes correctly."""
    decision = RolloutDecision(
        eligible=True,
        reason="canary_in_bucket",
        bucket=42,
        shadow_execution=False,
    )
    
    decision_dict = decision.to_dict()
    
    assert decision_dict["eligible"] is True
    assert decision_dict["reason"] == "canary_in_bucket"
    assert decision_dict["bucket"] == 42
    assert decision_dict["shadow_execution"] is False


# ─────────────────────────────────────────────
# Test 8: Integration with Step4 routing
# ─────────────────────────────────────────────
def test_step4_shadow_mode_does_not_switch_final_output(monkeypatch, tmp_path: Path):
    """Non-eligible shadow execution does not change final selected output in Step4."""
    from src.core.models import AudioRef, Segment, VisualPlan
    from src.steps.step4_assets import resolve_asset_for_segment
    
    project_root = tmp_path
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    
    # Find a segment key that is NOT in rollout bucket (bucket >= 30)
    segment_key = None
    segment_text = "Shadow mode segment"
    for i in range(1000):
        content_key = Segment.compute_content_key(segment_text)
        candidate_key = Segment.compute_segment_key(content_key, i + 1)
        if compute_bucket(candidate_key) >= 30:
            segment_key = candidate_key
            break
    
    assert segment_key is not None
    
    seg = Segment(
        segment_key=segment_key,
        content_key=Segment.compute_content_key(segment_text),
        index=1,
        start=0.0,
        end=4.0,
        duration=4.0,
        text=segment_text,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=4.0),
        visual_plan=VisualPlan(type="pixelle_digital_human", pixelle_workflow="digital_human"),
        plan_hash="shadowhash1234",
    )
    
    # Shadow mode enabled but segment not in rollout bucket
    shadow_config = RolloutConfig(
        enable_pixelle=True,
        rollout_percentage=30,
        allowlist_keys=frozenset(),
        shadow_mode=True,
    )
    
    shadow_calls = {"count": 0}
    
    class FakeShadowAdapter:
        def invoke(self, request):
            shadow_calls["count"] += 1
            output_path = Path(request.output_dir) / f"shadow_{request.segment_key}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"shadow-video")
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
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name: FakeShadowAdapter())
    
    def fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template")
        return str(path)
    
    monkeypatch.setattr("src.steps.step4_assets.generate_template_asset", fake_template)
    monkeypatch.setattr("src.steps.step4_assets._pixelle_rollout_config", shadow_config)
    
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
    
    assert shadow_calls["count"] == 0
    assert resolved.asset_refs[0].kind == "template"
    assert resolved.asset_refs[0].fallback_reason_code == "PIXELLE_ROLLOUT_INELIGIBLE"
    assert resolved.visual_plan is not None
    assert resolved.visual_plan.asset_path is not None
    assert resolved.visual_plan.asset_path.endswith(".png")
