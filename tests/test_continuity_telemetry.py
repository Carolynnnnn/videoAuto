"""
Tests for Continuity Telemetry, Quality Scoring, and Diagnostics.

Test selectors:
- continuity_telemetry_happy: Happy-path telemetry extraction and quality scoring
- continuity_telemetry_strict_gate: Strict-mode validation with gate violations
"""
import pytest

from src.core.models import AudioRef, Segment, VisualPlan
from src.steps.continuity_telemetry import (
    PREVIEW_STYLE_SIMILARITY_P50_THRESHOLD,
    PREVIEW_TEMPORAL_LINK_COVERAGE_THRESHOLD,
    STRICT_FALLBACK_COVERAGE_THRESHOLD,
    STRICT_ORPHAN_ARTIFACT_COUNT_THRESHOLD,
    STRICT_STYLE_SIMILARITY_P50_THRESHOLD,
    STRICT_TEMPORAL_LINK_COVERAGE_THRESHOLD,
    ContinuityQualityError,
    ContinuityQualitySummary,
    ContinuityTelemetryRecord,
    GateViolationDetail,
    StrictGateViolation,
    StrictModeValidationResult,
    compute_quality_summary,
    extract_all_telemetry,
    format_strict_mode_failure,
    persist_segment_telemetry,
    validate_strict_mode,
)


def _make_segment(
    index: int,
    text: str,
    continuity_mode: str = "temporal",
    requested_policy: str = "frame_chain",
    fallback_reason: str | None = None,
    seed: int | None = 123,
    start_frame_path: str | None = "/tmp/frame.png",
    vendor_id: str = "pixelle",
) -> Segment:
    """Create a segment with configured continuity diagnostic."""
    content_key = Segment.compute_content_key(text)
    segment = Segment(
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
    
    # Set continuity diagnostic
    segment.continuity_diagnostic = {
        "continuity_mode": continuity_mode,
        "requested_policy": requested_policy,
        "seed": seed,
        "start_frame_path": start_frame_path,
        "vendor_id": vendor_id,
        "fallback_reason_code": fallback_reason,
    }
    
    return segment


# ─────────────────────────────────────────────
# Happy Path Tests
# ─────────────────────────────────────────────
class TestContinuityTelemetryHappy:
    """Happy-path tests for continuity telemetry extraction and quality scoring."""

    @pytest.mark.continuity_telemetry_happy
    def test_telemetry_record_from_temporal_segment(self):
        """Telemetry record extracts correct fields from temporal segment."""
        segment = _make_segment(
            index=1,
            text="test segment",
            continuity_mode="temporal",
            start_frame_path="/tmp/frames/seg1_end.png",
            seed=42,
            vendor_id="pixelle",
        )
        
        record = ContinuityTelemetryRecord.from_segment(segment)
        
        assert record.segment_key == segment.segment_key
        assert record.continuity_mode == "temporal"
        assert record.source_frame_hash is not None  # Hash computed
        assert record.seed == 42
        assert record.vendor == "pixelle"
        assert record.fallback_reason is None
        assert record.artifact_linked is True
        # Temporal mode gets high scores
        assert record.style_similarity == 0.95
        assert record.continuity_confidence == 1.0

    @pytest.mark.continuity_telemetry_happy
    def test_telemetry_record_to_dict_deterministic(self):
        """Telemetry record serialization is deterministic."""
        segment = _make_segment(index=1, text="deterministic test")
        
        record1 = ContinuityTelemetryRecord.from_segment(segment)
        record2 = ContinuityTelemetryRecord.from_segment(segment)
        
        assert record1.to_dict() == record2.to_dict()

    @pytest.mark.continuity_telemetry_happy
    def test_quality_summary_perfect_run(self):
        """Quality summary reports perfect scores for all-temporal segments."""
        segments = [
            _make_segment(index=i, text=f"segment {i}", continuity_mode="temporal")
            for i in range(1, 6)
        ]
        
        summary = compute_quality_summary(segments)
        
        assert summary.total_segments == 5
        assert summary.temporal_segments == 5
        assert summary.fallback_segments == 0
        assert summary.orphan_artifact_count == 0
        assert summary.temporal_link_coverage == 100.0
        assert summary.fallback_coverage == 100.0
        assert summary.style_similarity_p50 == 0.95
        assert len(summary.telemetry_records) == 5

    @pytest.mark.continuity_telemetry_happy
    def test_quality_summary_to_dict_exposes_required_fields(self):
        """Quality summary to_dict exposes all required summary fields."""
        segments = [_make_segment(index=1, text="test")]
        summary = compute_quality_summary(segments)
        
        d = summary.to_dict()
        
        # Required fields per task spec
        assert "temporal_link_coverage" in d
        assert "fallback_coverage" in d
        assert "style_similarity_p50" in d
        assert "orphan_artifact_count" in d

    @pytest.mark.continuity_telemetry_happy
    def test_strict_mode_validation_passes_perfect_run(self):
        """Strict mode validation passes for perfect quality run."""
        segments = [
            _make_segment(index=i, text=f"segment {i}", continuity_mode="temporal")
            for i in range(1, 6)
        ]
        summary = compute_quality_summary(segments)
        
        result = validate_strict_mode(summary)
        
        assert result.passed is True
        assert len(result.violations) == 0
        assert result.summary is not None

    @pytest.mark.continuity_telemetry_happy
    def test_fallback_coverage_counts_ineligible_frame_chain_segments(self):
        """Fallback coverage measures diagnostic coverage for ineligible frame-chain segments."""
        segments = [
            _make_segment(index=1, text="temporal", continuity_mode="temporal", fallback_reason=None),
            _make_segment(
                index=2,
                text="fallback diagnosed",
                continuity_mode="seed_lock",
                requested_policy="frame_chain",
                fallback_reason="PIXELLE_CONTINUITY_FIRST_SEGMENT",
            ),
            _make_segment(
                index=3,
                text="fallback missing",
                continuity_mode="seed_lock",
                requested_policy="frame_chain",
                fallback_reason=None,
            ),
            _make_segment(
                index=4,
                text="out of scope policy",
                continuity_mode="off",
                requested_policy="off",
                fallback_reason=None,
            ),
        ]

        summary = compute_quality_summary(segments)

        assert summary.fallback_coverage == 50.0

    @pytest.mark.continuity_telemetry_happy
    def test_persist_telemetry_appends_to_diagnostic(self):
        """Persist telemetry adds telemetry key without overwriting diagnostic."""
        segment = _make_segment(index=1, text="persist test")
        assert segment.continuity_diagnostic is not None
        original_mode = segment.continuity_diagnostic["continuity_mode"]
        
        record = ContinuityTelemetryRecord.from_segment(segment)
        persist_segment_telemetry(segment, record)
        
        # Original diagnostic preserved
        assert segment.continuity_diagnostic is not None
        assert segment.continuity_diagnostic["continuity_mode"] == original_mode
        # Telemetry added under telemetry key
        assert "telemetry" in segment.continuity_diagnostic
        assert segment.continuity_diagnostic["telemetry"]["segment_key"] == segment.segment_key

    @pytest.mark.continuity_telemetry_happy
    def test_extract_all_telemetry_includes_diagnostics(self):
        """Extract all telemetry includes continuity_diagnostic for audit."""
        segments = [_make_segment(index=i, text=f"seg {i}") for i in range(1, 3)]
        
        telemetry_list = extract_all_telemetry(segments)
        
        assert len(telemetry_list) == 2
        for entry in telemetry_list:
            assert "continuity_diagnostic" in entry
            assert entry["continuity_mode"] == "temporal"

    @pytest.mark.continuity_telemetry_happy
    def test_style_similarity_scoring_by_mode(self):
        """Style similarity scores vary by continuity mode."""
        temporal = _make_segment(index=1, text="temporal", continuity_mode="temporal")
        seed_lock = _make_segment(index=2, text="seed", continuity_mode="seed_lock")
        style_anchor = _make_segment(index=3, text="style", continuity_mode="style_anchor")
        off = _make_segment(index=4, text="off", continuity_mode="off")
        
        r_temporal = ContinuityTelemetryRecord.from_segment(temporal)
        r_seed = ContinuityTelemetryRecord.from_segment(seed_lock)
        r_style = ContinuityTelemetryRecord.from_segment(style_anchor)
        r_off = ContinuityTelemetryRecord.from_segment(off)
        
        assert r_temporal.style_similarity == 0.95
        assert r_seed.style_similarity == 0.90
        assert r_style.style_similarity == 0.85
        assert r_off.style_similarity == 0.70


# ─────────────────────────────────────────────
# Strict Gate Tests
# ─────────────────────────────────────────────
class TestContinuityTelemetryStrictGate:
    """Strict-mode gate validation tests with violations."""

    @pytest.mark.continuity_telemetry_strict_gate
    def test_strict_gate_rejects_low_temporal_coverage(self):
        """Strict mode rejects when temporal_link_coverage < 95%."""
        # 2 out of 5 temporal = 40% coverage
        segments = [
            _make_segment(index=1, text="t1", continuity_mode="temporal"),
            _make_segment(index=2, text="t2", continuity_mode="temporal"),
            _make_segment(index=3, text="s1", continuity_mode="seed_lock"),
            _make_segment(index=4, text="s2", continuity_mode="seed_lock"),
            _make_segment(index=5, text="s3", continuity_mode="seed_lock"),
        ]
        summary = compute_quality_summary(segments)
        
        result = validate_strict_mode(summary)
        
        assert result.passed is False
        assert any(v.gate == StrictGateViolation.TEMPORAL_LINK_COVERAGE_LOW for v in result.violations)
        violation = next(v for v in result.violations if v.gate == StrictGateViolation.TEMPORAL_LINK_COVERAGE_LOW)
        assert violation.actual_value == 40.0
        assert violation.threshold == STRICT_TEMPORAL_LINK_COVERAGE_THRESHOLD

    @pytest.mark.continuity_telemetry_strict_gate
    def test_strict_gate_rejects_low_style_similarity(self):
        """Strict mode rejects when style_similarity_p50 < 0.85."""
        # All off mode = 0.70 similarity (below 0.85 threshold)
        segments = [
            _make_segment(index=i, text=f"seg {i}", continuity_mode="off")
            for i in range(1, 6)
        ]
        summary = compute_quality_summary(segments)
        
        result = validate_strict_mode(summary)
        
        assert result.passed is False
        assert any(v.gate == StrictGateViolation.STYLE_SIMILARITY_LOW for v in result.violations)
        violation = next(v for v in result.violations if v.gate == StrictGateViolation.STYLE_SIMILARITY_LOW)
        assert violation.actual_value == 0.70
        assert violation.threshold == STRICT_STYLE_SIMILARITY_P50_THRESHOLD

    @pytest.mark.continuity_telemetry_strict_gate
    def test_strict_gate_rejects_fallbacks(self):
        """Strict mode rejects when ineligible frame-chain fallback diagnostics are missing."""
        segments = [
            _make_segment(index=1, text="t1", continuity_mode="temporal"),
            _make_segment(index=2, text="t2", continuity_mode="temporal"),
            _make_segment(index=3, text="fb", continuity_mode="seed_lock", fallback_reason=None),
        ]
        summary = compute_quality_summary(segments)
        
        result = validate_strict_mode(summary)
        
        assert result.passed is False
        assert any(v.gate == StrictGateViolation.FALLBACK_COVERAGE_LOW for v in result.violations)

    @pytest.mark.continuity_telemetry_strict_gate
    def test_strict_gate_rejects_orphan_artifacts(self):
        """Strict mode rejects when orphan artifacts present."""
        # Create segment with broken linkage (fallback + off mode)
        segments = [
            _make_segment(index=1, text="t1", continuity_mode="temporal"),
            _make_segment(index=2, text="orphan", continuity_mode="off", fallback_reason="PIXELLE_CONTINUITY_PRIOR_ARTIFACT_MISSING"),
        ]
        summary = compute_quality_summary(segments)
        
        result = validate_strict_mode(summary)
        
        assert result.passed is False
        assert any(v.gate == StrictGateViolation.ORPHAN_ARTIFACTS_PRESENT for v in result.violations)

    @pytest.mark.continuity_telemetry_strict_gate
    def test_strict_gate_raises_on_violation_when_requested(self):
        """Strict mode raises ContinuityQualityError when raise_on_violation=True."""
        segments = [
            _make_segment(index=i, text=f"seg {i}", continuity_mode="off")
            for i in range(1, 3)
        ]
        summary = compute_quality_summary(segments)
        
        with pytest.raises(ContinuityQualityError) as exc_info:
            validate_strict_mode(summary, raise_on_violation=True)
        
        assert "STYLE_SIMILARITY_LOW" in str(exc_info.value)
        assert exc_info.value.result.passed is False

    @pytest.mark.continuity_telemetry_strict_gate
    def test_strict_gate_multiple_violations_reported(self):
        """Strict mode reports all violations, not just first."""
        # All off mode without fallback diagnostics = multiple violations
        segments = [
            _make_segment(
                index=i,
                text=f"seg {i}",
                continuity_mode="off",
                fallback_reason=None,
            )
            for i in range(1, 6)
        ]
        summary = compute_quality_summary(segments)
        
        result = validate_strict_mode(summary)
        
        assert result.passed is False
        # Should have multiple violations
        violation_gates = {v.gate for v in result.violations}
        # At minimum: style similarity (0.70 < 0.85), fallback coverage (0% < 100%), orphan artifacts
        assert StrictGateViolation.STYLE_SIMILARITY_LOW in violation_gates
        assert StrictGateViolation.FALLBACK_COVERAGE_LOW in violation_gates

    @pytest.mark.continuity_telemetry_strict_gate
    def test_validation_result_to_dict_serializes_correctly(self):
        """Validation result serializes to dict with all fields."""
        segments = [
            _make_segment(index=1, text="t1", continuity_mode="off")
        ]
        summary = compute_quality_summary(segments)
        result = validate_strict_mode(summary)
        
        d = result.to_dict()
        
        assert "passed" in d
        assert "violations" in d
        assert "summary" in d
        for v in d["violations"]:
            assert "gate" in v
            assert "actual_value" in v
            assert "threshold" in v
            assert "message" in v

    @pytest.mark.continuity_telemetry_strict_gate
    def test_gate_violation_detail_to_dict(self):
        """GateViolationDetail serializes correctly."""
        detail = GateViolationDetail(
            gate=StrictGateViolation.TEMPORAL_LINK_COVERAGE_LOW,
            actual_value=50.0,
            threshold=95.0,
            message="Test message",
        )
        
        d = detail.to_dict()
        
        assert d["gate"] == "TEMPORAL_LINK_COVERAGE_LOW"
        assert d["actual_value"] == 50.0
        assert d["threshold"] == 95.0
        assert d["message"] == "Test message"

    @pytest.mark.continuity_telemetry_strict_gate
    def test_profile_threshold_preview_allows_borderline_case(self):
        """Preview profile allows borderline quality where release fails."""
        summary = ContinuityQualitySummary(
            total_segments=5,
            temporal_segments=4,
            frame_chain_ineligible_segments=5,
            diagnosed_frame_chain_fallback_segments=5,
            orphan_artifact_count=0,
            style_similarities=[0.80, 0.80, 0.80, 0.80, 0.80],
        )

        preview_result = validate_strict_mode(summary, profile="preview")
        release_result = validate_strict_mode(summary, profile="release")

        assert preview_result.passed is True
        assert release_result.passed is False
        assert PREVIEW_TEMPORAL_LINK_COVERAGE_THRESHOLD < STRICT_TEMPORAL_LINK_COVERAGE_THRESHOLD
        assert PREVIEW_STYLE_SIMILARITY_P50_THRESHOLD < STRICT_STYLE_SIMILARITY_P50_THRESHOLD

    @pytest.mark.continuity_telemetry_strict_gate
    def test_profile_threshold_release_failure_report_includes_context(self):
        """Failure report includes machine-assertable profile and threshold context."""
        segments = [
            _make_segment(index=i, text=f"seg {i}", continuity_mode="off")
            for i in range(1, 3)
        ]
        summary = compute_quality_summary(segments)

        result = validate_strict_mode(summary, profile="release")
        report = format_strict_mode_failure(result)

        assert result.passed is False
        assert "active_profile=release" in report
        assert "threshold_temporal_link_coverage_min=95.0000" in report
        assert "threshold_style_similarity_p50_min=0.8500" in report
        assert "threshold_fallback_coverage_min=100.0000" in report
        assert "threshold_orphan_artifact_count_max=0" in report


# ─────────────────────────────────────────────
# Edge Case Tests
# ─────────────────────────────────────────────
class TestContinuityTelemetryEdgeCases:
    """Edge case tests for telemetry and quality scoring."""

    def test_empty_segments_list(self):
        """Quality summary handles empty segment list gracefully."""
        summary = compute_quality_summary([])
        
        assert summary.total_segments == 0
        assert summary.temporal_link_coverage == 100.0
        assert summary.fallback_coverage == 100.0
        assert summary.style_similarity_p50 == 1.0

    def test_segment_with_missing_diagnostic(self):
        """Telemetry extraction handles segment with None diagnostic."""
        segment = _make_segment(index=1, text="no diag")
        segment.continuity_diagnostic = None
        
        record = ContinuityTelemetryRecord.from_segment(segment)
        
        assert record.continuity_mode == "off"
        assert record.seed is None
        assert record.fallback_reason is None

    def test_segment_with_partial_diagnostic(self):
        """Telemetry extraction handles partial diagnostic gracefully."""
        segment = _make_segment(index=1, text="partial")
        segment.continuity_diagnostic = {"continuity_mode": "temporal"}  # Missing other fields
        
        record = ContinuityTelemetryRecord.from_segment(segment)
        
        assert record.continuity_mode == "temporal"
        assert record.seed is None
        assert record.source_frame_hash is None

    def test_style_similarity_fallback_penalty(self):
        """Style similarity applies penalty for fallback."""
        with_fallback = _make_segment(
            index=1,
            text="fallback",
            continuity_mode="temporal",
            fallback_reason="SOME_REASON",
        )
        without_fallback = _make_segment(
            index=2,
            text="no fallback",
            continuity_mode="temporal",
        )
        
        r_with = ContinuityTelemetryRecord.from_segment(with_fallback)
        r_without = ContinuityTelemetryRecord.from_segment(without_fallback)
        
        # Fallback gets 0.05 penalty
        assert r_with.style_similarity == pytest.approx(0.90, rel=1e-6)  # 0.95 - 0.05
        assert r_without.style_similarity == pytest.approx(0.95, rel=1e-6)

    def test_continuity_confidence_fallback_penalty(self):
        """Continuity confidence applies penalty for fallback."""
        with_fallback = _make_segment(
            index=1,
            text="fallback",
            continuity_mode="temporal",
            fallback_reason="SOME_REASON",
        )
        
        record = ContinuityTelemetryRecord.from_segment(with_fallback)
        
        # Temporal with frame = 1.0, minus 0.10 penalty = 0.90
        assert record.continuity_confidence == pytest.approx(0.90, rel=1e-6)


class TestProfilePropagation:
    """Tests for gate profile propagation into strict validation."""

    @pytest.mark.continuity_telemetry_strict_gate
    def test_format_failure_release_header_shows_blocked(self):
        """Release profile failure report shows 'release blocked' wording."""
        segments = [
            _make_segment(index=1, text="seg 1", continuity_mode="off"),
        ]
        summary = compute_quality_summary(segments)
        result = validate_strict_mode(summary, profile="release")
        report = format_strict_mode_failure(result)

        assert "release blocked" in report
        assert "warnings only" not in report

    @pytest.mark.continuity_telemetry_strict_gate
    def test_format_failure_preview_header_shows_warnings_only(self):
        """Preview profile failure report shows 'warnings only' wording."""
        segments = [
            _make_segment(index=1, text="seg 1", continuity_mode="off"),
        ]
        summary = compute_quality_summary(segments)
        result = validate_strict_mode(summary, profile="preview")
        report = format_strict_mode_failure(result)

        assert "preview mode - warnings only" in report
        assert "release blocked" not in report

    @pytest.mark.continuity_telemetry_strict_gate
    def test_validate_strict_mode_captures_profile_in_result(self):
        """validate_strict_mode stores profile in result for downstream use."""
        summary = ContinuityQualitySummary(total_segments=1)
        
        result_release = validate_strict_mode(summary, profile="release")
        result_preview = validate_strict_mode(summary, profile="preview")
        
        assert result_release.profile == "release"
        assert result_preview.profile == "preview"

    @pytest.mark.continuity_telemetry_strict_gate
    def test_profile_propagation_preserves_release_semantics(self):
        """Profile propagation preserves existing release hard-fail behavior."""
        segments = [
            _make_segment(index=i, text=f"seg {i}", continuity_mode="off")
            for i in range(1, 6)
        ]
        summary = compute_quality_summary(segments)
        
        result = validate_strict_mode(summary, profile="release")
        
        assert result.passed is False
        assert result.profile == "release"
        assert len(result.violations) > 0
