from pathlib import Path

from build_incremental import incremental_build
from src.core.diff_engine import apply_diff, compute_diff, get_segments_to_rebuild
from src.core.models import AssetRef, AudioRef, Manifest, RenderRef, Segment, VisualPlan


def _make_segment(
    *,
    index: int,
    text: str,
    start: float,
    end: float,
    segments_dir: Path,
    continuity_mode: str,
    source_segment_key: str | None,
    start_frame_path: str | None,
) -> Segment:
    content_key = Segment.compute_content_key(text)
    segment_key = Segment.compute_segment_key(content_key, 1)
    render_hash = f"rh{index:02d}"
    video_path = segments_dir / f"{content_key}_{render_hash}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"segment")

    return Segment(
        segment_key=segment_key,
        content_key=content_key,
        index=index,
        start=start,
        end=end,
        duration=round(end - start, 3),
        text=text,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=start, trim_end=end),
        visual_plan=VisualPlan(type="pixelle_i2v", pixelle_workflow="i2v"),
        plan_hash=f"plan-{index}",
        asset_refs=[
            AssetRef(
                kind="pixelle_video",
                path=str(video_path),
                asset_hash=f"asset-{index}",
            )
        ],
        render_ref=RenderRef(
            segment_video_path=str(video_path),
            render_hash=render_hash,
            status="ok",
        ),
        prev_last_frame_path=start_frame_path,
        continuity_diagnostic={
            "continuity_mode": continuity_mode,
            "source_segment_key": source_segment_key,
            "start_frame_path": start_frame_path,
            "seed": 20260316,
            "vendor_id": "pixelle",
        },
    )


def _make_old_manifest(tmp_path: Path) -> Manifest:
    segments_dir = tmp_path / "render" / "segments"
    frame1 = tmp_path / "artifacts" / "continuity" / "frames" / "seg1_end.png"
    frame2 = tmp_path / "artifacts" / "continuity" / "frames" / "seg2_end.png"
    frame1.parent.mkdir(parents=True, exist_ok=True)
    frame1.write_bytes(b"f1")
    frame2.write_bytes(b"f2")

    seg1 = _make_segment(
        index=1,
        text="alpha segment",
        start=0.0,
        end=1.0,
        segments_dir=segments_dir,
        continuity_mode="seed_lock",
        source_segment_key=None,
        start_frame_path=str(frame1),
    )
    seg2 = _make_segment(
        index=2,
        text="beta segment",
        start=1.0,
        end=2.0,
        segments_dir=segments_dir,
        continuity_mode="temporal",
        source_segment_key=seg1.segment_key,
        start_frame_path=str(frame1),
    )
    seg3 = _make_segment(
        index=3,
        text="gamma segment",
        start=2.0,
        end=3.0,
        segments_dir=segments_dir,
        continuity_mode="temporal",
        source_segment_key=seg2.segment_key,
        start_frame_path=str(frame2),
    )

    return Manifest(
        project_id="e2e_continuity_fixture",
        style_id="style-fixture",
        continuity_seed=20260316,
        vendor_preference="pixelle",
        continuity_policy="frame_chain",
        segments=[seg1, seg2, seg3],
    )


def _make_new_manifest_with_timing_change(tmp_path: Path) -> Manifest:
    segments_dir = tmp_path / "render" / "segments"
    seg1 = _make_segment(
        index=1,
        text="alpha segment",
        start=0.0,
        end=1.0,
        segments_dir=segments_dir,
        continuity_mode="off",
        source_segment_key=None,
        start_frame_path=None,
    )
    seg2 = _make_segment(
        index=2,
        text="beta segment",
        start=1.5,
        end=2.5,
        segments_dir=segments_dir,
        continuity_mode="off",
        source_segment_key=None,
        start_frame_path=None,
    )
    seg3 = _make_segment(
        index=3,
        text="gamma segment",
        start=2.0,
        end=3.0,
        segments_dir=segments_dir,
        continuity_mode="off",
        source_segment_key=None,
        start_frame_path=None,
    )

    for seg in (seg1, seg2, seg3):
        seg.render_ref = RenderRef(status="pending")
        seg.asset_refs = []
        seg.plan_hash = None
        seg.visual_plan = None
        seg.prev_last_frame_path = None
        seg.continuity_diagnostic = None

    return Manifest(project_id="e2e_continuity_fixture", segments=[seg1, seg2, seg3])


def _fmt_ts(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_srt_blocks(path: Path, blocks: list[tuple[int, float, float, str]]) -> None:
    lines: list[str] = []
    for idx, start, end, text in blocks:
        lines.extend([str(idx), f"{_fmt_ts(start)} --> {_fmt_ts(end)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _setup_project_with_srt(tmp_path: Path, blocks: list[tuple[int, float, float, str]]) -> Path:
    project_root = tmp_path / "project"
    input_dir = project_root / "input"
    build_dir = project_root / "build"
    input_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "voice_full.wav").write_bytes(b"dummy")
    _write_srt_blocks(build_dir / "subtitle.srt", blocks)
    return project_root


def _make_cap_manifest(
    *,
    project_root: Path,
    segment_count: int,
    segment_duration: float,
    with_existing_assets: bool,
) -> Manifest:
    segments_dir = project_root / "render" / "segments"
    segments: list[Segment] = []
    start = 0.0
    for idx in range(1, segment_count + 1):
        seg = _make_segment(
            index=idx,
            text=f"policy segment {idx}",
            start=start,
            end=start + segment_duration,
            segments_dir=segments_dir,
            continuity_mode="off",
            source_segment_key=None,
            start_frame_path=None,
        )
        seg.visual_plan = VisualPlan(
            type="pixelle_digital_human",
            pixelle_workflow="digital_human",
            prompt=f"policy prompt {idx}",
        )
        seg.plan_hash = f"policy-plan-{idx}"
        seg.render_ref = RenderRef(status="pending")

        if with_existing_assets:
            old_asset = project_root / "assets" / "generated" / f"old_{idx}.png"
            old_asset.parent.mkdir(parents=True, exist_ok=True)
            old_asset.write_bytes(b"old")
            seg.asset_refs = [
                AssetRef(
                    kind="template",
                    path=str(old_asset),
                    asset_hash=f"old-asset-{idx}",
                )
            ]
        else:
            seg.asset_refs = []

        segments.append(seg)
        start += segment_duration

    return Manifest(project_id="e2e_policy_fixture", material_mode="ai_only", segments=segments)


def test_e2e_full_build_continuity_metadata_persistence(tmp_path: Path):
    manifest = _make_old_manifest(tmp_path)
    manifest_path = tmp_path / "build" / "manifest.json"
    manifest.save(str(manifest_path))

    loaded = Manifest.load(str(manifest_path))
    seg1, seg2, seg3 = loaded.segments

    assert loaded.continuity_policy == "frame_chain"
    assert loaded.continuity_seed == 20260316
    assert loaded.style_id == "style-fixture"

    assert seg1.continuity_diagnostic is not None
    assert seg2.continuity_diagnostic is not None
    assert seg3.continuity_diagnostic is not None
    assert seg2.continuity_diagnostic["continuity_mode"] == "temporal"
    assert seg2.continuity_diagnostic["source_segment_key"] == seg1.segment_key
    assert seg3.continuity_diagnostic["source_segment_key"] == seg2.segment_key
    assert seg3.prev_last_frame_path == seg3.continuity_diagnostic["start_frame_path"]


def test_e2e_incremental_continuity_preserves_changed_and_unchanged_segments(tmp_path: Path):
    old_manifest = _make_old_manifest(tmp_path)
    new_manifest = _make_new_manifest_with_timing_change(tmp_path)

    diff = compute_diff(old_manifest, new_manifest)
    assert diff.added == []
    assert diff.removed == []
    assert diff.changed_timing == [old_manifest.segments[1].segment_key]
    assert set(diff.unchanged) == {
        old_manifest.segments[0].segment_key,
        old_manifest.segments[2].segment_key,
    }

    applied_manifest = apply_diff(
        old_manifest=old_manifest,
        new_manifest=new_manifest,
        diff=diff,
        segments_dir=str(tmp_path / "render" / "segments"),
        assets_dir=str(tmp_path / "assets" / "generated"),
    )

    old_by_key = {seg.segment_key: seg for seg in old_manifest.segments}
    applied_by_key = {seg.segment_key: seg for seg in applied_manifest.segments}

    unchanged_keys = {old_manifest.segments[0].segment_key, old_manifest.segments[2].segment_key}
    for key in unchanged_keys:
        assert applied_by_key[key].render_ref.status == "ok"
        assert applied_by_key[key].prev_last_frame_path == old_by_key[key].prev_last_frame_path
        assert applied_by_key[key].continuity_diagnostic == old_by_key[key].continuity_diagnostic

    changed_key = old_manifest.segments[1].segment_key
    assert applied_by_key[changed_key].render_ref.status == "pending"
    assert applied_by_key[changed_key].continuity_diagnostic == old_by_key[changed_key].continuity_diagnostic

    plan_keys, asset_keys, render_keys = get_segments_to_rebuild(applied_manifest, diff)
    assert plan_keys == []
    assert asset_keys == []
    assert render_keys == [changed_key]


def test_e2e_incremental_continuity_strict_gate_rejects_release(tmp_path: Path, monkeypatch):
    project_root = tmp_path / "project"
    input_dir = project_root / "input"
    build_dir = project_root / "build"
    input_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    (input_dir / "voice_full.wav").write_bytes(b"dummy")
    (build_dir / "subtitle.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nhello\n",
        encoding="utf-8",
    )

    old_manifest = _make_old_manifest(project_root)
    old_manifest.save(str(build_dir / "manifest.json"))

    failing_manifest = _make_new_manifest_with_timing_change(project_root)
    monkeypatch.setenv("PIXELLE_TEST_MODE", "0")
    monkeypatch.delenv("PIXELLE_CONTINUITY_STRICT_MODE", raising=False)

    def _fake_step2(**kwargs):
        failing_manifest.save(kwargs["output_manifest"])
        return failing_manifest

    monkeypatch.setattr("build_incremental.run_step2", _fake_step2)
    monkeypatch.setattr("build_incremental.run_step3", lambda **kwargs: kwargs["manifest"])
    monkeypatch.setattr("build_incremental.run_step4", lambda **kwargs: kwargs["manifest"])
    monkeypatch.setattr("build_incremental.run_step5", lambda **kwargs: kwargs["manifest"])

    step6_called = False

    def _unexpected_step6(**kwargs):
        nonlocal step6_called
        step6_called = True
        return kwargs["manifest"]

    monkeypatch.setattr("build_incremental.run_step6", _unexpected_step6)

    result = incremental_build(project_root=str(project_root))

    assert result.success is False
    assert result.error is not None
    assert "Strict continuity gate failed" in result.error
    assert "TEMPORAL_LINK_COVERAGE_LOW" in result.error
    assert "style_similarity_p50=" in result.error
    assert step6_called is False


def test_e2e_incremental_continuity_strict_gate_respects_toggle_in_test_mode(tmp_path: Path, monkeypatch):
    project_root = tmp_path / "project"
    input_dir = project_root / "input"
    build_dir = project_root / "build"
    input_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    (input_dir / "voice_full.wav").write_bytes(b"dummy")
    (build_dir / "subtitle.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nhello\n",
        encoding="utf-8",
    )

    old_manifest = _make_old_manifest(project_root)
    old_manifest.save(str(build_dir / "manifest.json"))

    failing_manifest = _make_new_manifest_with_timing_change(project_root)
    monkeypatch.setenv("PIXELLE_TEST_MODE", "1")
    monkeypatch.delenv("PIXELLE_CONTINUITY_STRICT_MODE", raising=False)

    def _fake_step2(**kwargs):
        failing_manifest.save(kwargs["output_manifest"])
        return failing_manifest

    monkeypatch.setattr("build_incremental.run_step2", _fake_step2)
    monkeypatch.setattr("build_incremental.run_step3", lambda **kwargs: kwargs["manifest"])
    monkeypatch.setattr("build_incremental.run_step4", lambda **kwargs: kwargs["manifest"])
    monkeypatch.setattr("build_incremental.run_step5", lambda **kwargs: kwargs["manifest"])
    monkeypatch.setattr("build_incremental.run_step6", lambda **kwargs: kwargs["manifest"])

    result = incremental_build(project_root=str(project_root))

    assert result.success is True


def test_e2e_ai_cap_and_duration_budget_exact6_over6(tmp_path: Path, monkeypatch):
    cases = [
        ("exact6", 6, 10.0, 6, 6),
        ("over6", 8, 8.0, 6, 8),
    ]

    for case_name, segment_count, segment_duration, expected_selected, expected_output_segments in cases:
        case_root = tmp_path / case_name
        blocks: list[tuple[int, float, float, str]] = []
        start = 0.0
        for idx in range(1, segment_count + 1):
            end = start + segment_duration
            blocks.append((idx, start, end, f"cap-case-{case_name}-{idx}"))
            start = end

        project_root = _setup_project_with_srt(case_root, blocks)
        adapter_calls: list[str] = []
        captured_manifest: Manifest | None = None

        class FakeAdapter:
            def invoke(self, request):
                adapter_calls.append(request.segment_key)
                out = Path(request.output_dir) / f"pixelle_{request.segment_key}.mp4"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b"ai-video")
                return type("Resp", (), {"success": True, "output_path": str(out), "error": None})()

        def _fake_step3(**kwargs):
            manifest = kwargs["manifest"]
            for idx, seg in enumerate(manifest.segments, start=1):
                seg.visual_plan = VisualPlan(
                    type="pixelle_digital_human",
                    pixelle_workflow="digital_human",
                    prompt=f"e2e prompt {idx}",
                )
                seg.plan_hash = f"e2e-plan-{idx}"
            return manifest

        def _fake_template(output_path: str, width: int, height: int, text: str):
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"template")
            return str(path)

        monkeypatch.setattr("build_incremental.run_step3", _fake_step3)
        monkeypatch.setattr("build_incremental.run_step5", lambda **kwargs: kwargs["manifest"])

        def _capture_step6(**kwargs):
            nonlocal captured_manifest
            captured_manifest = kwargs["manifest"]
            return kwargs["manifest"]

        monkeypatch.setattr("build_incremental.run_step6", _capture_step6)
        monkeypatch.setattr("build_incremental._should_enforce_strict_continuity_gate", lambda: False)
        monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
        monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FakeAdapter())
        monkeypatch.setattr("src.steps.step4_assets.generate_template_asset", _fake_template)

        result = incremental_build(
            project_root=str(project_root),
            material_mode="ai_only",
            enable_ai_image=False,
            duration_policy={"target_duration_minutes": 1.0, "ai_clip_cap": 6},
        )

        assert result.success is True, f"e2e incremental_build failed for case={case_name}: {result.error}"
        assert captured_manifest is not None

        loaded = Manifest.load(str(project_root / "build" / "manifest.json"))
        assert len(loaded.segments) == expected_output_segments

        allocation_map = getattr(captured_manifest, "step4_ai_allocation_map", None)
        assert allocation_map is not None
        selected_keys = {k for k, v in allocation_map.items() if v}

        assert len(selected_keys) <= 6
        assert len(adapter_calls) <= 6
        assert set(adapter_calls) == selected_keys
        assert len(selected_keys) == expected_selected

        refs_by_key = {seg.segment_key: seg.asset_refs[0] for seg in loaded.segments if seg.asset_refs}
        assert all(refs_by_key[key].kind == "pixelle_video" for key in selected_keys)

        if expected_output_segments > 6:
            non_selected_keys = {seg.segment_key for seg in loaded.segments} - selected_keys
            assert len(non_selected_keys) == expected_output_segments - 6
            assert all(refs_by_key[key].kind != "pixelle_video" for key in non_selected_keys)


def test_e2e_duration_budget_under_duration_no_padding_repetition(tmp_path: Path, monkeypatch):
    blocks: list[tuple[int, float, float, str]] = []
    start = 0.0
    for idx in range(1, 6):
        end = start + 10.0
        blocks.append((idx, start, end, f"under-{idx}"))
        start = end

    project_root = _setup_project_with_srt(tmp_path, blocks)

    monkeypatch.setattr("build_incremental.run_step3", lambda **kwargs: kwargs["manifest"])
    monkeypatch.setattr("build_incremental.run_step4", lambda **kwargs: kwargs["manifest"])
    monkeypatch.setattr("build_incremental.run_step5", lambda **kwargs: kwargs["manifest"])
    monkeypatch.setattr("build_incremental.run_step6", lambda **kwargs: kwargs["manifest"])
    monkeypatch.setattr("build_incremental._should_enforce_strict_continuity_gate", lambda: False)

    result = incremental_build(
        project_root=str(project_root),
        duration_policy={"target_duration_minutes": 1.0, "ai_clip_cap": 6},
    )

    assert result.success is True

    loaded = Manifest.load(str(project_root / "build" / "manifest.json"))
    assert len(loaded.segments) == 5
    assert round(sum(seg.duration for seg in loaded.segments), 3) == 50.0

    expected_texts = [f"under-{i}" for i in range(1, 6)]
    actual_texts = [seg.text for seg in loaded.segments]
    assert actual_texts == expected_texts
    assert len({seg.segment_key for seg in loaded.segments}) == 5


def test_e2e_incremental_target_subset_cap_isolation(tmp_path: Path, monkeypatch):
    project_root = _setup_project_with_srt(tmp_path, [(1, 0.0, 1.0, "placeholder")])

    manifest = _make_cap_manifest(
        project_root=project_root,
        segment_count=10,
        segment_duration=1.0,
        with_existing_assets=True,
    )
    target_subset = [seg.segment_key for seg in manifest.segments[:8]]
    outside_keys = {seg.segment_key for seg in manifest.segments} - set(target_subset)
    outside_paths_before = {
        seg.segment_key: seg.asset_refs[0].path
        for seg in manifest.segments
        if seg.segment_key in outside_keys
    }

    adapter_calls: list[str] = []
    captured_manifest: Manifest | None = None

    class FakeAdapter:
        def invoke(self, request):
            adapter_calls.append(request.segment_key)
            out = Path(request.output_dir) / f"pixelle_{request.segment_key}.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"ai-video")
            return type("Resp", (), {"success": True, "output_path": str(out), "error": None})()

    def _fake_template(output_path: str, width: int, height: int, text: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template")
        return str(path)

    monkeypatch.setattr("build_incremental.run_step2", lambda **kwargs: manifest)
    monkeypatch.setattr(
        "build_incremental.get_segments_to_rebuild",
        lambda _manifest, _diff: ([], target_subset, []),
    )
    monkeypatch.setattr("build_incremental.run_step5", lambda **kwargs: kwargs["manifest"])

    def _capture_step6(**kwargs):
        nonlocal captured_manifest
        captured_manifest = kwargs["manifest"]
        return kwargs["manifest"]

    monkeypatch.setattr("build_incremental.run_step6", _capture_step6)
    monkeypatch.setattr("build_incremental._should_enforce_strict_continuity_gate", lambda: False)
    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FakeAdapter())
    monkeypatch.setattr("src.steps.step4_assets.generate_template_asset", _fake_template)

    result = incremental_build(
        project_root=str(project_root),
        material_mode="ai_only",
        enable_ai_image=False,
        duration_policy={"target_duration_minutes": 1.0, "ai_clip_cap": 6},
    )

    assert result.success is True
    assert captured_manifest is not None

    loaded = Manifest.load(str(project_root / "build" / "manifest.json"))
    allocation_map = getattr(captured_manifest, "step4_ai_allocation_map", None)
    assert allocation_map is not None

    selected_keys = {k for k, v in allocation_map.items() if v}
    assert len(selected_keys) == 6
    assert all(key in set(target_subset) for key in selected_keys)
    assert all(allocation_map[key] is False for key in outside_keys)
    assert all(key in set(target_subset) for key in adapter_calls)

    loaded_by_key = {seg.segment_key: seg for seg in loaded.segments}
    for key in outside_keys:
        assert loaded_by_key[key].asset_refs[0].path == outside_paths_before[key]
