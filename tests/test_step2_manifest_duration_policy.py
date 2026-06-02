from src.steps.step2_manifest import srt_to_manifest
from src.utils.srt_utils import split_text_deterministic, estimate_text_width_px


def _write_srt(tmp_path, blocks):
    lines = []
    for idx, start, end, text in blocks:
        lines.extend([
            str(idx),
            f"00:00:{start:06.3f}".replace('.', ',') + " --> " + f"00:00:{end:06.3f}".replace('.', ','),
            text,
            "",
        ])
    srt_path = tmp_path / "subtitle.srt"
    srt_path.write_text("\n".join(lines), encoding="utf-8")
    return srt_path


def test_reduced_segment_count(tmp_path):
    blocks = []
    start = 0.0
    for i in range(12):
        end = start + 1.0
        blocks.append((i + 1, start, end, f"w{i + 1}"))
        start = end

    srt_path = _write_srt(tmp_path, blocks)

    manifest = srt_to_manifest(
        srt_path=str(srt_path),
        audio_path="voice.wav",
        project_id="p1",
    )
    manifest_again = srt_to_manifest(
        srt_path=str(srt_path),
        audio_path="voice.wav",
        project_id="p1",
    )

    assert len(manifest.segments) < len(blocks)
    assert len(manifest.segments) <= 3
    assert [s.segment_key for s in manifest.segments] == [
        s.segment_key for s in manifest_again.segments
    ]


def test_invalid_duration_policy_rejected_max_lt_min(tmp_path):
    srt_path = _write_srt(tmp_path, [(1, 0.0, 2.0, "hello")])

    with_message = "max_duration must be >= min_duration"
    try:
        srt_to_manifest(
            srt_path=str(srt_path),
            audio_path="voice.wav",
            project_id="p1",
            duration_policy={"min_duration": 5.0, "max_duration": 2.0},
        )
        raise AssertionError("expected ValueError for invalid duration policy")
    except ValueError as exc:
        assert with_message in str(exc)


def test_sentence_boundary_preferred_over_clause_boundary():
    text = "Alpha beta. Gamma delta, epsilon zeta eta theta."
    font_size = 48
    clause_prefix = "Alpha beta. Gamma delta,"
    safe_width = int(estimate_text_width_px(clause_prefix, font_size) + 2)

    lines = split_text_deterministic(text, safe_width, font_size)

    assert len(lines) > 1
    assert lines[0].endswith(".")
    assert not lines[0].endswith(",")
    assert " ".join(lines) == text


def test_sentence_boundary_deterministic_replay():
    text = "First sentence ends here. Another sentence follows, then closes cleanly."
    font_size = 48
    safe_width = int(estimate_text_width_px("First sentence ends here. Another sentence follows,", font_size) + 2)

    first = split_text_deterministic(text, safe_width, font_size)
    second = split_text_deterministic(text, safe_width, font_size)

    assert first == second
    assert first[0].endswith(".")


# =============================================================================
# REGRESSION TESTS: Segmentation Policy (Task 16)
# =============================================================================


def test_clip_count_reduction_with_short_entries(tmp_path):
    """
    Regression: Short segments below merge_threshold should merge to reduce clip count.
    Would fail if merge logic is removed or merge_threshold not respected.
    """
    # Create 8 very short segments (0.5s each) that should merge into fewer clips
    blocks = []
    start = 0.0
    for i in range(8):
        end = start + 0.5  # Very short, below default merge_threshold (3.0s)
        blocks.append((i + 1, start, end, f"word{i + 1}"))
        start = end

    srt_path = _write_srt(tmp_path, blocks)

    manifest = srt_to_manifest(
        srt_path=str(srt_path),
        audio_path="voice.wav",
        project_id="p_clip_count",
    )

    # With 8 x 0.5s entries (4s total), should merge into fewer segments
    assert len(manifest.segments) < len(blocks), (
        f"Expected fewer segments than input blocks ({len(blocks)}), got {len(manifest.segments)}"
    )
    # Verify content is preserved (no data loss)
    combined_text = " ".join(s.text for s in manifest.segments)
    original_text = " ".join(b[3] for b in blocks)
    assert all(word in combined_text for word in original_text.split())


def test_duration_boundaries_enforced_after_segmentation(tmp_path):
    """
    Regression: After merge/split, all segments should fall within policy bounds.
    Would fail if duration policy min/max are not enforced during segmentation.
    """
    # Create a mix of very short and very long segments
    blocks = [
        (1, 0.0, 0.3, "tiny"),  # Too short
        (2, 0.3, 15.0, "This is a very long segment that should be split into multiple parts for video generation"),  # Too long
        (3, 15.0, 15.5, "small"),  # Short
    ]
    srt_path = _write_srt(tmp_path, blocks)

    policy = {
        "min_duration": 1.0,
        "max_duration": 8.0,
        "target_min_duration": 2.0,
        "target_max_duration": 6.0,
        "merge_threshold": 2.0,
        "split_threshold": 8.0,
    }

    manifest = srt_to_manifest(
        srt_path=str(srt_path),
        audio_path="voice.wav",
        project_id="p_bounds",
        duration_policy=policy,
    )

    # All segments should respect policy bounds (with tolerance for edge cases)
    for seg in manifest.segments:
        assert seg.duration >= policy["min_duration"] - 0.01, (
            f"Segment {seg.segment_key} duration {seg.duration:.3f}s below min {policy['min_duration']}s"
        )
        assert seg.duration <= policy["max_duration"] + 0.01, (
            f"Segment {seg.segment_key} duration {seg.duration:.3f}s above max {policy['max_duration']}s"
        )


def test_boundary_violation_min_duration_zero_rejected(tmp_path):
    """
    Regression: min_duration <= 0 passed as positional arg should raise ValueError.
    Would fail if validation is removed or zero/negative values allowed.
    """
    srt_path = _write_srt(tmp_path, [(1, 0.0, 2.0, "hello")])

    with_message = "min_duration must be > 0"
    try:
        srt_to_manifest(
            srt_path=str(srt_path),
            audio_path="voice.wav",
            project_id="p1",
            min_duration=0.0,
            max_duration=10.0,
        )
        raise AssertionError("expected ValueError for zero min_duration")
    except ValueError as exc:
        assert with_message in str(exc), f"Expected '{with_message}' in error, got: {exc}"


def test_boundary_violation_max_duration_negative_rejected(tmp_path):
    """
    Regression: max_duration <= 0 passed as positional arg should raise ValueError.
    Would fail if negative duration validation is removed.
    """
    srt_path = _write_srt(tmp_path, [(1, 0.0, 2.0, "hello")])

    with_message = "max_duration must be > 0"
    try:
        srt_to_manifest(
            srt_path=str(srt_path),
            audio_path="voice.wav",
            project_id="p1",
            min_duration=1.0,
            max_duration=-5.0,
        )
        raise AssertionError("expected ValueError for negative max_duration")
    except ValueError as exc:
        assert with_message in str(exc), f"Expected '{with_message}' in error, got: {exc}"


def test_boundary_violation_target_min_below_min_rejected(tmp_path):
    """
    Regression: target_min_duration < min_duration should raise ValueError.
    Would fail if target validation chain is removed.
    """
    srt_path = _write_srt(tmp_path, [(1, 0.0, 2.0, "hello")])

    with_message = "target_min_duration must be >= min_duration"
    try:
        srt_to_manifest(
            srt_path=str(srt_path),
            audio_path="voice.wav",
            project_id="p1",
            duration_policy={
                "min_duration": 3.0,
                "max_duration": 10.0,
                "target_min_duration": 2.0,  # Below min_duration
                "target_max_duration": 8.0,
            },
        )
        raise AssertionError("expected ValueError for target_min < min")
    except ValueError as exc:
        assert with_message in str(exc), f"Expected '{with_message}' in error, got: {exc}"


def test_boundary_violation_target_max_above_max_rejected(tmp_path):
    """
    Regression: target_max_duration > max_duration should raise ValueError.
    Would fail if target bounds validation is removed.
    """
    srt_path = _write_srt(tmp_path, [(1, 0.0, 2.0, "hello")])

    with_message = "target_max_duration must be <= max_duration"
    try:
        srt_to_manifest(
            srt_path=str(srt_path),
            audio_path="voice.wav",
            project_id="p1",
            duration_policy={
                "min_duration": 1.0,
                "max_duration": 8.0,
                "target_min_duration": 3.0,
                "target_max_duration": 12.0,  # Above max_duration
            },
        )
        raise AssertionError("expected ValueError for target_max > max")
    except ValueError as exc:
        assert with_message in str(exc), f"Expected '{with_message}' in error, got: {exc}"


def test_malformed_policy_merge_threshold_above_target_min_rejected(tmp_path):
    """
    Regression: merge_threshold > target_min_duration should raise ValueError.
    Would fail if merge/target chain validation is removed.
    """
    srt_path = _write_srt(tmp_path, [(1, 0.0, 2.0, "hello")])

    with_message = "merge_threshold must be <= target_min_duration"
    try:
        srt_to_manifest(
            srt_path=str(srt_path),
            audio_path="voice.wav",
            project_id="p1",
            duration_policy={
                "min_duration": 1.0,
                "max_duration": 12.0,
                "target_min_duration": 4.0,
                "target_max_duration": 10.0,
                "merge_threshold": 5.0,  # Above target_min
            },
        )
        raise AssertionError("expected ValueError for merge_threshold > target_min")
    except ValueError as exc:
        assert with_message in str(exc), f"Expected '{with_message}' in error, got: {exc}"


def test_malformed_policy_split_threshold_below_target_max_rejected(tmp_path):
    """
    Regression: split_threshold < target_max_duration should raise ValueError.
    Would fail if split/target chain validation is removed.
    """
    srt_path = _write_srt(tmp_path, [(1, 0.0, 2.0, "hello")])

    with_message = "split_threshold must be >= target_max_duration"
    try:
        srt_to_manifest(
            srt_path=str(srt_path),
            audio_path="voice.wav",
            project_id="p1",
            duration_policy={
                "min_duration": 1.0,
                "max_duration": 12.0,
                "target_min_duration": 4.0,
                "target_max_duration": 10.0,
                "merge_threshold": 3.0,
                "split_threshold": 8.0,  # Below target_max
            },
        )
        raise AssertionError("expected ValueError for split_threshold < target_max")
    except ValueError as exc:
        assert with_message in str(exc), f"Expected '{with_message}' in error, got: {exc}"


def test_malformed_policy_target_max_below_target_min_rejected(tmp_path):
    """
    Regression: target_max_duration < target_min_duration should raise ValueError.
    Would fail if target range validation is removed.
    """
    srt_path = _write_srt(tmp_path, [(1, 0.0, 2.0, "hello")])

    with_message = "target_max_duration must be >= target_min_duration"
    try:
        srt_to_manifest(
            srt_path=str(srt_path),
            audio_path="voice.wav",
            project_id="p1",
            duration_policy={
                "min_duration": 1.0,
                "max_duration": 12.0,
                "target_min_duration": 8.0,
                "target_max_duration": 5.0,  # Below target_min
            },
        )
        raise AssertionError("expected ValueError for target_max < target_min")
    except ValueError as exc:
        assert with_message in str(exc), f"Expected '{with_message}' in error, got: {exc}"


def test_deterministic_replay_manifest_segment_keys(tmp_path):
    """
    Regression: Identical SRT input should produce identical segment_keys on replay.
    Would fail if segment key generation becomes non-deterministic.
    """
    blocks = [
        (1, 0.0, 3.0, "First sentence ends here."),
        (2, 3.0, 6.0, "Second sentence follows."),
        (3, 6.0, 8.5, "Third sentence concludes."),
    ]
    srt_path = _write_srt(tmp_path, blocks)

    manifest_1 = srt_to_manifest(
        srt_path=str(srt_path),
        audio_path="voice.wav",
        project_id="p_deterministic",
    )
    manifest_2 = srt_to_manifest(
        srt_path=str(srt_path),
        audio_path="voice.wav",
        project_id="p_deterministic",
    )

    # Segment count must be identical
    assert len(manifest_1.segments) == len(manifest_2.segments), (
        f"Replay produced different segment counts: {len(manifest_1.segments)} vs {len(manifest_2.segments)}"
    )

    # Segment keys must be identical (content_key + occurrence_index)
    keys_1 = [s.segment_key for s in manifest_1.segments]
    keys_2 = [s.segment_key for s in manifest_2.segments]
    assert keys_1 == keys_2, f"Replay produced different segment keys:\n  {keys_1}\n  vs\n  {keys_2}"

    # Segment texts must be identical
    texts_1 = [s.text for s in manifest_1.segments]
    texts_2 = [s.text for s in manifest_2.segments]
    assert texts_1 == texts_2, f"Replay produced different texts:\n  {texts_1}\n  vs\n  {texts_2}"


def test_deterministic_replay_with_policy_override(tmp_path):
    """
    Regression: Same SRT + same policy should produce identical segments.
    Would fail if policy application introduces non-determinism.
    """
    blocks = []
    start = 0.0
    for i in range(10):
        end = start + 1.5
        blocks.append((i + 1, start, end, f"segment {i + 1}"))
        start = end

    srt_path = _write_srt(tmp_path, blocks)

    policy = {
        "min_duration": 2.0,
        "max_duration": 8.0,
        "target_min_duration": 4.0,
        "target_max_duration": 6.0,
        "merge_threshold": 3.0,
        "split_threshold": 8.0,
    }

    manifest_1 = srt_to_manifest(
        srt_path=str(srt_path),
        audio_path="voice.wav",
        project_id="p_policy_deterministic",
        duration_policy=policy,
    )
    manifest_2 = srt_to_manifest(
        srt_path=str(srt_path),
        audio_path="voice.wav",
        project_id="p_policy_deterministic",
        duration_policy=policy,
    )

    keys_1 = [s.segment_key for s in manifest_1.segments]
    keys_2 = [s.segment_key for s in manifest_2.segments]
    assert keys_1 == keys_2, f"Policy replay produced different segment keys:\n  {keys_1}\n  vs\n  {keys_2}"

    # Duration should be identical
    durations_1 = [round(s.duration, 3) for s in manifest_1.segments]
    durations_2 = [round(s.duration, 3) for s in manifest_2.segments]
    assert durations_1 == durations_2, f"Policy replay produced different durations:\n  {durations_1}\n  vs\n  {durations_2}"


def test_sentence_boundary_preserved_during_merge(tmp_path):
    """
    Regression: Merge should not break sentence boundaries when possible.
    Would fail if merge logic ignores text structure.
    """
    blocks = [
        (1, 0.0, 2.0, "The quick brown fox"),
        (2, 2.0, 4.0, "jumps over the lazy dog."),
        (3, 4.0, 6.0, "Another sentence here."),
    ]
    srt_path = _write_srt(tmp_path, blocks)

    manifest = srt_to_manifest(
        srt_path=str(srt_path),
        audio_path="voice.wav",
        project_id="p_sentence_merge",
        duration_policy={
            "min_duration": 1.0,
            "max_duration": 12.0,
            "merge_threshold": 3.0,
            "target_min_duration": 5.0,
            "target_max_duration": 10.0,
            "split_threshold": 12.0,
        },
    )

    assert len(manifest.segments) >= 1, "Expected at least one segment in output"
    combined_text = " ".join(s.text for s in manifest.segments)
    assert "quick brown fox" in combined_text and "lazy dog" in combined_text, (
        f"Sentence content not preserved in merged output: {combined_text}"
    )


def test_duration_budget_120s(tmp_path):
    def _fmt_ts(total_seconds: float) -> str:
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

    blocks = []
    start = 0.0
    for i in range(15):
        end = start + 10.0
        blocks.append((i + 1, start, end, f"clip-{i + 1}"))
        start = end

    lines = []
    for idx, start, end, text in blocks:
        lines.extend([
            str(idx),
            f"{_fmt_ts(start)} --> {_fmt_ts(end)}",
            text,
            "",
        ])
    srt_path = tmp_path / "subtitle.srt"
    srt_path.write_text("\n".join(lines), encoding="utf-8")

    policy = {"target_duration_minutes": 2.0}
    manifest_1 = srt_to_manifest(
        srt_path=str(srt_path),
        audio_path="voice.wav",
        project_id="p_duration_budget",
        duration_policy=policy,
    )
    manifest_2 = srt_to_manifest(
        srt_path=str(srt_path),
        audio_path="voice.wav",
        project_id="p_duration_budget",
        duration_policy=policy,
    )

    assert len(manifest_1.segments) == 12
    assert round(sum(s.duration for s in manifest_1.segments), 3) == 120.0
    assert [s.segment_key for s in manifest_1.segments] == [
        s.segment_key for s in manifest_2.segments
    ]
    assert [s.start for s in manifest_1.segments] == sorted(s.start for s in manifest_1.segments)


def test_effective_duration_metadata(tmp_path):
    """Verify budget_diagnostics metadata fields are populated and deterministic."""
    def _fmt_ts(total_seconds: float) -> str:
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

    blocks = []
    start = 0.0
    for i in range(15):
        end = start + 10.0
        blocks.append((i + 1, start, end, f"clip-{i + 1}"))
        start = end

    lines = []
    for idx, start, end, text in blocks:
        lines.extend([
            str(idx),
            f"{_fmt_ts(start)} --> {_fmt_ts(end)}",
            text,
            "",
        ])
    srt_path = tmp_path / "subtitle.srt"
    srt_path.write_text("\n".join(lines), encoding="utf-8")

    policy = {"target_duration_minutes": 2.0}
    manifest = srt_to_manifest(
        srt_path=str(srt_path),
        audio_path="voice.wav",
        project_id="p_effective_duration",
        duration_policy=policy,
    )

    diag = manifest.budget_diagnostics
    assert diag is not None, "budget_diagnostics should be populated"
    assert diag.requested_minutes == 2, f"expected requested_minutes=2, got {diag.requested_minutes}"
    assert diag.target_seconds == 120.0, f"expected target_seconds=120.0, got {diag.target_seconds}"
    assert diag.total_available_seconds == 150.0, f"expected total_available=150.0, got {diag.total_available_seconds}"
    assert diag.effective_selected_seconds == 120.0, f"expected effective_selected=120.0, got {diag.effective_selected_seconds}"
    assert diag.selected_count == 12, f"expected selected_count=12, got {diag.selected_count}"
    assert diag.dropped_count == 3, f"expected dropped_count=3, got {diag.dropped_count}"
    assert diag.budget_exhausted is True, "expected budget_exhausted=True"


def test_duration_budget_deterministic_replay(tmp_path):
    """Verify budget_diagnostics metadata is identical across repeated runs on same input."""
    def _fmt_ts(total_seconds: float) -> str:
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

    blocks = []
    start = 0.0
    for i in range(20):
        end = start + 8.0
        blocks.append((i + 1, start, end, f"segment-{i + 1}"))
        start = end

    lines = []
    for idx, start, end, text in blocks:
        lines.extend([
            str(idx),
            f"{_fmt_ts(start)} --> {_fmt_ts(end)}",
            text,
            "",
        ])
    srt_path = tmp_path / "subtitle.srt"
    srt_path.write_text("\n".join(lines), encoding="utf-8")

    policy = {"target_duration_minutes": 1.0}
    manifest_1 = srt_to_manifest(
        srt_path=str(srt_path),
        audio_path="voice.wav",
        project_id="p_deterministic_replay",
        duration_policy=policy,
    )
    manifest_2 = srt_to_manifest(
        srt_path=str(srt_path),
        audio_path="voice.wav",
        project_id="p_deterministic_replay",
        duration_policy=policy,
    )

    diag_1 = manifest_1.budget_diagnostics
    diag_2 = manifest_2.budget_diagnostics
    assert diag_1 is not None and diag_2 is not None

    assert diag_1.requested_minutes == diag_2.requested_minutes
    assert diag_1.target_seconds == diag_2.target_seconds
    assert diag_1.total_available_seconds == diag_2.total_available_seconds
    assert diag_1.effective_selected_seconds == diag_2.effective_selected_seconds
    assert diag_1.selected_count == diag_2.selected_count
    assert diag_1.dropped_count == diag_2.dropped_count
    assert diag_1.budget_exhausted == diag_2.budget_exhausted

    assert diag_1.to_dict() == diag_2.to_dict(), "Diagnostics dict should be identical across runs"


# =============================================================================
# REGRESSION TESTS: Duration Budget (Task 16)
# =============================================================================


def _fmt_ts(total_seconds: float) -> str:
    """Format seconds into HH:MM:SS,mmm SRT timestamp (shared helper for multi-minute fixtures)."""
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')


def _write_srt_multiminute(tmp_path, blocks):
    """Write SRT with proper HH:MM:SS,mmm timestamps (avoids 00:00:{seconds} overflow issues)."""
    lines = []
    for idx, start, end, text in blocks:
        lines.extend([
            str(idx),
            f"{_fmt_ts(start)} --> {_fmt_ts(end)}",
            text,
            "",
        ])
    srt_path = tmp_path / "subtitle.srt"
    srt_path.write_text("\n".join(lines), encoding="utf-8")
    return srt_path


def test_duration_budget_60s(tmp_path):
    """
    Regression: 1-minute target should cap output at 60 seconds total.
    Would fail if budget selection ignores 1-minute policy or miscalculates seconds.
    """
    # Create 10 segments x 10s each = 100s total (exceeds 60s target)
    blocks = []
    start = 0.0
    for i in range(10):
        end = start + 10.0
        blocks.append((i + 1, start, end, f"segment-{i + 1}"))
        start = end

    srt_path = _write_srt_multiminute(tmp_path, blocks)

    policy = {"target_duration_minutes": 1.0}
    manifest_1 = srt_to_manifest(
        srt_path=str(srt_path),
        audio_path="voice.wav",
        project_id="p_duration_budget_60s",
        duration_policy=policy,
    )
    manifest_2 = srt_to_manifest(
        srt_path=str(srt_path),
        audio_path="voice.wav",
        project_id="p_duration_budget_60s",
        duration_policy=policy,
    )

    # Should select 6 segments (6 x 10s = 60s) to match 1-minute budget
    assert len(manifest_1.segments) == 6, f"Expected 6 segments for 1-min budget, got {len(manifest_1.segments)}"
    total_duration = round(sum(s.duration for s in manifest_1.segments), 3)
    assert total_duration == 60.0, f"Expected 60.0s total, got {total_duration}"

    # Deterministic replay
    assert [s.segment_key for s in manifest_1.segments] == [
        s.segment_key for s in manifest_2.segments
    ]

    # Temporal ordering preserved
    assert [s.start for s in manifest_1.segments] == sorted(s.start for s in manifest_1.segments)


def test_duration_budget_180s(tmp_path):
    """
    Regression: 3-minute target should cap output at 180 seconds total.
    Would fail if budget selection ignores 3-minute policy or miscalculates seconds.
    """
    # Create 25 segments x 10s each = 250s total (exceeds 180s target)
    blocks = []
    start = 0.0
    for i in range(25):
        end = start + 10.0
        blocks.append((i + 1, start, end, f"clip-{i + 1}"))
        start = end

    srt_path = _write_srt_multiminute(tmp_path, blocks)

    policy = {"target_duration_minutes": 3.0}
    manifest_1 = srt_to_manifest(
        srt_path=str(srt_path),
        audio_path="voice.wav",
        project_id="p_duration_budget_180s",
        duration_policy=policy,
    )
    manifest_2 = srt_to_manifest(
        srt_path=str(srt_path),
        audio_path="voice.wav",
        project_id="p_duration_budget_180s",
        duration_policy=policy,
    )

    # Should select 18 segments (18 x 10s = 180s) to match 3-minute budget
    assert len(manifest_1.segments) == 18, f"Expected 18 segments for 3-min budget, got {len(manifest_1.segments)}"
    total_duration = round(sum(s.duration for s in manifest_1.segments), 3)
    assert total_duration == 180.0, f"Expected 180.0s total, got {total_duration}"

    # Deterministic replay
    assert [s.segment_key for s in manifest_1.segments] == [
        s.segment_key for s in manifest_2.segments
    ]

    # Temporal ordering preserved
    assert [s.start for s in manifest_1.segments] == sorted(s.start for s in manifest_1.segments)


def test_under_duration_budget_keeps_all(tmp_path):
    """
    Regression: When source duration is less than target, all segments should be kept.
    Would fail if budget selection drops segments when source is under-budget.
    """
    # Create 5 segments x 10s each = 50s total (under 60s 1-minute target)
    blocks = []
    start = 0.0
    for i in range(5):
        end = start + 10.0
        blocks.append((i + 1, start, end, f"short-{i + 1}"))
        start = end

    srt_path = _write_srt_multiminute(tmp_path, blocks)

    policy = {"target_duration_minutes": 1.0}  # 60s target, but only 50s available
    manifest_1 = srt_to_manifest(
        srt_path=str(srt_path),
        audio_path="voice.wav",
        project_id="p_under_duration",
        duration_policy=policy,
    )
    manifest_2 = srt_to_manifest(
        srt_path=str(srt_path),
        audio_path="voice.wav",
        project_id="p_under_duration",
        duration_policy=policy,
    )

    # All 5 segments should be retained (no dropping when under-budget)
    assert len(manifest_1.segments) == 5, f"Expected all 5 segments retained, got {len(manifest_1.segments)}"
    total_duration = round(sum(s.duration for s in manifest_1.segments), 3)
    assert total_duration == 50.0, f"Expected 50.0s (all available), got {total_duration}"

    # Deterministic replay
    assert [s.segment_key for s in manifest_1.segments] == [
        s.segment_key for s in manifest_2.segments
    ]


def test_under_duration_budget_diagnostics(tmp_path):
    """
    Regression: Under-duration scenario should set budget_exhausted=False in diagnostics.
    Would fail if diagnostics incorrectly report budget exhausted when all content is used.
    """
    # Create 4 segments x 8s each = 32s total (well under 60s 1-minute target)
    blocks = []
    start = 0.0
    for i in range(4):
        end = start + 8.0
        blocks.append((i + 1, start, end, f"tiny-{i + 1}"))
        start = end

    srt_path = _write_srt_multiminute(tmp_path, blocks)

    policy = {"target_duration_minutes": 1.0}  # 60s target, but only 32s available
    manifest = srt_to_manifest(
        srt_path=str(srt_path),
        audio_path="voice.wav",
        project_id="p_under_duration_diag",
        duration_policy=policy,
    )

    diag = manifest.budget_diagnostics
    assert diag is not None, "budget_diagnostics should be populated"
    assert diag.requested_minutes == 1, f"expected requested_minutes=1, got {diag.requested_minutes}"
    assert diag.target_seconds == 60.0, f"expected target_seconds=60.0, got {diag.target_seconds}"
    assert diag.total_available_seconds == 32.0, f"expected total_available=32.0, got {diag.total_available_seconds}"
    assert diag.effective_selected_seconds == 32.0, f"expected effective_selected=32.0, got {diag.effective_selected_seconds}"
    assert diag.selected_count == 4, f"expected selected_count=4, got {diag.selected_count}"
    assert diag.dropped_count == 0, f"expected dropped_count=0 (under-duration), got {diag.dropped_count}"
    assert diag.budget_exhausted is False, "expected budget_exhausted=False when under-duration"


def test_invalid_duration_minutes_zero_rejected(tmp_path):
    """
    Regression: target_duration_minutes=0 should raise ValueError with clear message.
    Would fail if zero value passes validation.
    """
    srt_path = _write_srt(tmp_path, [(1, 0.0, 2.0, "hello")])

    try:
        srt_to_manifest(
            srt_path=str(srt_path),
            audio_path="voice.wav",
            project_id="p_invalid_zero",
            duration_policy={"target_duration_minutes": 0},
        )
        raise AssertionError("expected ValueError for target_duration_minutes=0")
    except ValueError as exc:
        # Should mention allowed values {1, 2, 3}
        error_msg = str(exc)
        assert "1" in error_msg and "2" in error_msg and "3" in error_msg, (
            f"Expected error to mention allowed values 1, 2, 3; got: {error_msg}"
        )


def test_invalid_duration_minutes_four_rejected(tmp_path):
    """
    Regression: target_duration_minutes=4 should raise ValueError with clear message.
    Would fail if value above allowed set (1, 2, 3) passes validation.
    """
    srt_path = _write_srt(tmp_path, [(1, 0.0, 2.0, "hello")])

    try:
        srt_to_manifest(
            srt_path=str(srt_path),
            audio_path="voice.wav",
            project_id="p_invalid_four",
            duration_policy={"target_duration_minutes": 4},
        )
        raise AssertionError("expected ValueError for target_duration_minutes=4")
    except ValueError as exc:
        error_msg = str(exc)
        assert "1" in error_msg and "2" in error_msg and "3" in error_msg, (
            f"Expected error to mention allowed values 1, 2, 3; got: {error_msg}"
        )


def test_invalid_duration_minutes_negative_rejected(tmp_path):
    """
    Regression: target_duration_minutes=-1 should raise ValueError with clear message.
    Would fail if negative value passes validation.
    """
    srt_path = _write_srt(tmp_path, [(1, 0.0, 2.0, "hello")])

    try:
        srt_to_manifest(
            srt_path=str(srt_path),
            audio_path="voice.wav",
            project_id="p_invalid_negative",
            duration_policy={"target_duration_minutes": -1},
        )
        raise AssertionError("expected ValueError for target_duration_minutes=-1")
    except ValueError as exc:
        error_msg = str(exc)
        assert "1" in error_msg and "2" in error_msg and "3" in error_msg, (
            f"Expected error to mention allowed values 1, 2, 3; got: {error_msg}"
        )
