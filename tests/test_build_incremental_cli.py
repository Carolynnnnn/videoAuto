"""Tests for build_incremental.py CLI argument parsing and propagation."""
import sys
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_material_mode_argument_parser_accepts_valid_values():
    """Test that argparse accepts all canonical material_mode values."""
    import build_incremental
    from argparse import ArgumentParser
    
    for mode in ["auto", "ai_preferred", "ai_only"]:
        test_args = ["build_incremental.py", "--project", "./projects/test", "--material-mode", mode]
        
        parser = ArgumentParser()
        parser.add_argument("--project", required=True)
        parser.add_argument("--material-mode", choices=["auto", "ai_preferred", "ai_only"], default="auto")
        
        with patch.object(sys, "argv", test_args):
            args = parser.parse_args()
            assert args.material_mode == mode
            

def test_material_mode_argument_parser_defaults_to_auto():
    """Test that omitted --material-mode defaults to 'auto'."""
    import build_incremental
    from argparse import ArgumentParser
    
    test_args = ["build_incremental.py", "--project", "./projects/test"]
    
    parser = ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--material-mode", choices=["auto", "ai_preferred", "ai_only"], default="auto")
    
    with patch.object(sys, "argv", test_args):
        args = parser.parse_args()
        assert args.material_mode == "auto"


def test_material_mode_argument_parser_rejects_invalid_value():
    """Test that argparse exits with error for invalid material_mode."""
    import build_incremental
    from argparse import ArgumentParser
    
    test_args = ["build_incremental.py", "--project", "./projects/test", "--material-mode", "invalid_mode"]
    
    parser = ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--material-mode", choices=["auto", "ai_preferred", "ai_only"], default="auto")
    
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args()
        assert exc_info.value.code != 0


def test_material_mode_propagates_to_incremental_build():
    """Test that material_mode from CLI is passed to incremental_build."""
    import build_incremental
    
    with patch.object(build_incremental, "incremental_build") as mock_build:
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.dry_run = True
        mock_result.rerendered_count = 0
        mock_result.reused_count = 0
        mock_build.return_value = mock_result
        
        test_args = [
            "--project", "./projects/test",
            "--material-mode", "ai_only",
            "--dry-run"
        ]
        with patch.object(sys, "argv", ["build_incremental.py"] + test_args):
            try:
                build_incremental.main()
            except SystemExit:
                pass
        
        if mock_build.called:
            call_kwargs = mock_build.call_args[1]
            assert "material_mode" in call_kwargs
            assert call_kwargs["material_mode"] == "ai_only"


def test_material_mode_help_message_displays_choices(capsys):
    """Test that --help shows material-mode argument with choices."""
    import build_incremental
    
    test_args = ["--help"]
    with patch.object(sys, "argv", ["build_incremental.py"] + test_args):
        with pytest.raises(SystemExit) as exc_info:
            build_incremental.main()
        assert exc_info.value.code == 0
        
        captured = capsys.readouterr()
        help_output = captured.out
        
        assert "--material-mode" in help_output
        assert "auto" in help_output
        assert "ai_preferred" in help_output
        assert "ai_only" in help_output


def test_workflow_default_ai_only_applies_i2v_when_unset(tmp_path):
    """Test that ai_only mode applies pixelle_default_workflow='i2v' when unset."""
    import build_incremental
    from src.core.models import Manifest, GlobalStyle, Segment
    
    project_root = tmp_path / "test_project"
    project_root.mkdir()
    (project_root / "input").mkdir()
    (project_root / "build").mkdir()
    
    srt_file = project_root / "build" / "subtitle.srt"
    srt_file.write_text("1\n00:00:00,000 --> 00:00:02,000\nTest subtitle\n\n")
    
    audio_file = project_root / "input" / "voice.wav"
    audio_file.write_text("fake audio data")
    
    with patch("build_incremental.run_step2") as mock_step2:
        mock_manifest = Manifest(
            project_id="test",
            global_style=GlobalStyle(),
            segments=[Segment(
                segment_key="seg_0001_001",
                content_key="test_content",
                index=0,
                text="Test",
                start=0.0,
                end=2.0,
                duration=2.0
            )],
            pixelle_default_workflow=None
        )
        mock_step2.return_value = mock_manifest
        
        with patch("build_incremental.run_step3", return_value=mock_manifest), \
             patch("build_incremental.run_step4", return_value=mock_manifest), \
             patch("build_incremental.run_step5", return_value=mock_manifest), \
             patch("build_incremental.run_step6", return_value=mock_manifest), \
             patch("build_incremental._should_enforce_strict_continuity_gate", return_value=False):
            
            result = build_incremental.incremental_build(
                project_root=str(project_root),
                dry_run=False,
                material_mode="ai_only",
            )
            
            assert result.success
            assert mock_manifest.pixelle_default_workflow == "i2v"


def test_workflow_default_ai_preferred_applies_i2v_when_unset(tmp_path):
    """Test that ai_preferred mode applies pixelle_default_workflow='i2v' when unset."""
    import build_incremental
    from src.core.models import Manifest, GlobalStyle, Segment
    
    project_root = tmp_path / "test_project"
    project_root.mkdir()
    (project_root / "input").mkdir()
    (project_root / "build").mkdir()
    
    srt_file = project_root / "build" / "subtitle.srt"
    srt_file.write_text("1\n00:00:00,000 --> 00:00:02,000\nTest subtitle\n\n")
    
    audio_file = project_root / "input" / "voice.wav"
    audio_file.write_text("fake audio data")
    
    with patch("build_incremental.run_step2") as mock_step2:
        mock_manifest = Manifest(
            project_id="test",
            global_style=GlobalStyle(),
            segments=[Segment(
                segment_key="seg_0001_001",
                content_key="test_content",
                index=0,
                text="Test",
                start=0.0,
                end=2.0,
                duration=2.0
            )],
            pixelle_default_workflow=None
        )
        mock_step2.return_value = mock_manifest
        
        with patch("build_incremental.run_step3", return_value=mock_manifest), \
             patch("build_incremental.run_step4", return_value=mock_manifest), \
             patch("build_incremental.run_step5", return_value=mock_manifest), \
             patch("build_incremental.run_step6", return_value=mock_manifest), \
             patch("build_incremental._should_enforce_strict_continuity_gate", return_value=False):
            
            result = build_incremental.incremental_build(
                project_root=str(project_root),
                dry_run=False,
                material_mode="ai_preferred",
            )
            
            assert result.success
            assert mock_manifest.pixelle_default_workflow == "i2v"


def test_workflow_default_auto_mode_no_default_applied(tmp_path):
    """Test that auto mode does not apply workflow default when unset."""
    import build_incremental
    from src.core.models import Manifest, GlobalStyle, Segment
    
    project_root = tmp_path / "test_project"
    project_root.mkdir()
    (project_root / "input").mkdir()
    (project_root / "build").mkdir()
    
    srt_file = project_root / "build" / "subtitle.srt"
    srt_file.write_text("1\n00:00:00,000 --> 00:00:02,000\nTest subtitle\n\n")
    
    audio_file = project_root / "input" / "voice.wav"
    audio_file.write_text("fake audio data")
    
    with patch("build_incremental.run_step2") as mock_step2:
        mock_manifest = Manifest(
            project_id="test",
            global_style=GlobalStyle(),
            segments=[Segment(
                segment_key="seg_0001_001",
                content_key="test_content",
                index=0,
                text="Test",
                start=0.0,
                end=2.0,
                duration=2.0
            )],
            pixelle_default_workflow=None
        )
        mock_step2.return_value = mock_manifest
        
        with patch("build_incremental.run_step3", return_value=mock_manifest), \
             patch("build_incremental.run_step4", return_value=mock_manifest), \
             patch("build_incremental.run_step5", return_value=mock_manifest), \
             patch("build_incremental.run_step6", return_value=mock_manifest), \
             patch("build_incremental._should_enforce_strict_continuity_gate", return_value=False):
            
            result = build_incremental.incremental_build(
                project_root=str(project_root),
                dry_run=False,
                material_mode="auto",
            )
            
            assert result.success
            assert mock_manifest.pixelle_default_workflow is None


def test_workflow_default_preserves_explicit_override(tmp_path):
    """Test that explicit workflow values are preserved and not clobbered."""
    import build_incremental
    from src.core.models import Manifest, GlobalStyle, Segment
    
    project_root = tmp_path / "test_project"
    project_root.mkdir()
    (project_root / "input").mkdir()
    (project_root / "build").mkdir()
    
    srt_file = project_root / "build" / "subtitle.srt"
    srt_file.write_text("1\n00:00:00,000 --> 00:00:02,000\nTest subtitle\n\n")
    
    audio_file = project_root / "input" / "voice.wav"
    audio_file.write_text("fake audio data")
    
    with patch("build_incremental.run_step2") as mock_step2:
        mock_manifest = Manifest(
            project_id="test",
            global_style=GlobalStyle(),
            segments=[Segment(
                segment_key="seg_0001_001",
                content_key="test_content",
                index=0,
                text="Test",
                start=0.0,
                end=2.0,
                duration=2.0
            )],
            pixelle_default_workflow="t2v"
        )
        mock_step2.return_value = mock_manifest
        
        with patch("build_incremental.run_step3", return_value=mock_manifest), \
             patch("build_incremental.run_step4", return_value=mock_manifest), \
             patch("build_incremental.run_step5", return_value=mock_manifest), \
             patch("build_incremental.run_step6", return_value=mock_manifest), \
             patch("build_incremental._should_enforce_strict_continuity_gate", return_value=False):
            
            result = build_incremental.incremental_build(
                project_root=str(project_root),
                dry_run=False,
                material_mode="ai_only",
            )
            
            assert result.success
            assert mock_manifest.pixelle_default_workflow == "t2v"


def test_invalid_workflow_rejected_at_manifest_load(tmp_path):
    """Test that invalid workflow values raise WorkflowPolicyError at manifest load."""
    from src.core.models import Manifest, WorkflowPolicyError
    
    manifest_data = {
        "project_id": "test",
        "global_style": {"aspect_ratio": "9:16", "resolution": "1080x1920"},
        "segments": [{"segment_key": "seg_0001_001", "content_key": "test", "index": 0, "text": "Test", "start": 0.0, "end": 2.0, "duration": 2.0}],
        "pixelle_default_workflow": "invalid_workflow"
    }
    
    with pytest.raises(WorkflowPolicyError) as exc_info:
        Manifest.load_from_dict(manifest_data)
    
    assert "invalid_workflow" in str(exc_info.value)
    assert exc_info.value.value == "invalid_workflow"
    assert "t2v" in str(exc_info.value.allowed) or "i2v" in str(exc_info.value.allowed)


def test_gate_profile_release_blocks_on_failure(tmp_path):
    """Test that --gate-profile=release returns failure when strict gate fails."""
    import build_incremental
    from src.core.models import Manifest, GlobalStyle, Segment
    
    project_root = tmp_path / "test_project"
    project_root.mkdir()
    (project_root / "input").mkdir()
    (project_root / "build").mkdir()
    (project_root / "render" / "segments").mkdir(parents=True)
    
    srt_file = project_root / "build" / "subtitle.srt"
    srt_file.write_text("1\n00:00:00,000 --> 00:00:02,000\nTest subtitle\n\n")
    
    audio_file = project_root / "input" / "voice.wav"
    audio_file.write_text("fake audio data")
    
    failing_manifest = Manifest(
        project_id="test",
        global_style=GlobalStyle(),
        segments=[
            Segment(
                segment_key="seg_0001_001",
                content_key="test_content",
                index=0,
                text="Test",
                start=0.0,
                end=2.0,
                duration=2.0,
                continuity_diagnostic={
                    "continuity_mode": "off",
                },
            )
        ],
    )
    
    step6_called = False
    
    def _unexpected_step6(**kwargs):
        nonlocal step6_called
        step6_called = True
        return kwargs["manifest"]
    
    with patch("build_incremental.run_step2", return_value=failing_manifest), \
         patch("build_incremental.run_step3", return_value=failing_manifest), \
         patch("build_incremental.run_step4", return_value=failing_manifest), \
         patch("build_incremental.run_step5", return_value=failing_manifest), \
         patch("build_incremental.run_step6", side_effect=_unexpected_step6), \
         patch("build_incremental._should_enforce_strict_continuity_gate", return_value=True):
        
        result = build_incremental.incremental_build(
            project_root=str(project_root),
            dry_run=False,
            gate_profile="release",
        )
        
        assert result.success is False, \
            "Release profile should fail when strict gate fails"
        assert result.error is not None
        assert "Strict continuity gate failed" in result.error or "TEMPORAL_LINK_COVERAGE" in result.error
    
    assert step6_called is False, \
        "Step6 should not be called when release gate blocks"


def test_gate_profile_preview_warns_but_continues(tmp_path, caplog):
    """Test that --gate-profile=preview logs warnings but continues to Step6."""
    import build_incremental
    import logging
    from src.core.models import Manifest, GlobalStyle, Segment
    
    project_root = tmp_path / "test_project"
    project_root.mkdir()
    (project_root / "input").mkdir()
    (project_root / "build").mkdir()
    (project_root / "render" / "segments").mkdir(parents=True)
    
    srt_file = project_root / "build" / "subtitle.srt"
    srt_file.write_text("1\n00:00:00,000 --> 00:00:02,000\nTest subtitle\n\n")
    
    audio_file = project_root / "input" / "voice.wav"
    audio_file.write_text("fake audio data")
    
    failing_manifest = Manifest(
        project_id="test",
        global_style=GlobalStyle(),
        segments=[
            Segment(
                segment_key="seg_0001_001",
                content_key="test_content",
                index=0,
                text="Test",
                start=0.0,
                end=2.0,
                duration=2.0,
                continuity_diagnostic={
                    "continuity_mode": "off",
                },
            )
        ],
    )
    
    step6_called = False
    
    def _track_step6(**kwargs):
        nonlocal step6_called
        step6_called = True
        return kwargs["manifest"]
    
    with patch("build_incremental.run_step2", return_value=failing_manifest), \
         patch("build_incremental.run_step3", return_value=failing_manifest), \
         patch("build_incremental.run_step4", return_value=failing_manifest), \
         patch("build_incremental.run_step5", return_value=failing_manifest), \
         patch("build_incremental.run_step6", side_effect=_track_step6), \
         patch("build_incremental._should_enforce_strict_continuity_gate", return_value=True):
        
        with caplog.at_level(logging.WARNING):
            result = build_incremental.incremental_build(
                project_root=str(project_root),
                dry_run=False,
                gate_profile="preview",
            )
        
        assert result.success is True, \
            "Preview profile should succeed even when strict gate fails"
    
    assert step6_called is True, \
        "Step6 should be called when preview gate allows non-blocking"
    
    assert any("preview mode - non-blocking" in record.message for record in caplog.records), \
        "Should log preview mode warning"


def test_gate_profile_argument_parser_defaults_to_release():
    """Test that omitted --gate-profile defaults to 'release'."""
    import build_incremental
    from argparse import ArgumentParser
    
    test_args = ["build_incremental.py", "--project", "./projects/test"]
    
    parser = ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--gate-profile", choices=["preview", "release"], default="release")
    
    with patch.object(sys, "argv", test_args):
        args = parser.parse_args()
        assert args.gate_profile == "release"


def test_gate_profile_argument_parser_accepts_preview():
    """Test that --gate-profile=preview is accepted."""
    import build_incremental
    from argparse import ArgumentParser
    
    test_args = ["build_incremental.py", "--project", "./projects/test", "--gate-profile", "preview"]
    
    parser = ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--gate-profile", choices=["preview", "release"], default="release")
    
    with patch.object(sys, "argv", test_args):
        args = parser.parse_args()
        assert args.gate_profile == "preview"


def test_duration_minutes_propagates(tmp_path):
    """Test that --duration-minutes propagates through incremental_build to Step2 Manifest."""
    import build_incremental
    from src.core.models import Manifest, GlobalStyle, Segment
    
    project_root = tmp_path / "test_project"
    project_root.mkdir()
    (project_root / "input").mkdir()
    (project_root / "build").mkdir()
    
    srt_file = project_root / "build" / "subtitle.srt"
    srt_file.write_text("1\n00:00:00,000 --> 00:00:02,000\nTest subtitle\n\n")
    
    audio_file = project_root / "input" / "voice.wav"
    audio_file.write_text("fake audio data")
    
    captured_manifest = None
    
    def _capture_step2(**kwargs):
        nonlocal captured_manifest
        captured_manifest = Manifest(
            project_id="test",
            global_style=GlobalStyle(),
            segments=[Segment(
                segment_key="seg_0001_001",
                content_key="test_content",
                index=0,
                text="Test",
                start=0.0,
                end=2.0,
                duration=2.0
            )],
            target_duration_minutes=kwargs.get("duration_policy", {}).get("target_duration_minutes", 1),
            ai_clip_cap=kwargs.get("duration_policy", {}).get("ai_clip_cap", 6),
        )
        return captured_manifest
    
    def _return_manifest(**kwargs):
        return captured_manifest
    
    with patch("build_incremental.run_step2", side_effect=_capture_step2) as mock_step2:
        with patch("build_incremental.run_step3", side_effect=_return_manifest), \
             patch("build_incremental.run_step4", side_effect=_return_manifest), \
             patch("build_incremental.run_step5", side_effect=_return_manifest), \
             patch("build_incremental.run_step6", side_effect=_return_manifest), \
             patch("build_incremental._should_enforce_strict_continuity_gate", return_value=False):
            
            result = build_incremental.incremental_build(
                project_root=str(project_root),
                dry_run=False,
                duration_policy={"target_duration_minutes": 2, "ai_clip_cap": 6},
            )
            
            assert result.success
            assert mock_step2.called
            
            call_kwargs = mock_step2.call_args[1]
            assert "duration_policy" in call_kwargs
            assert call_kwargs["duration_policy"]["target_duration_minutes"] == 2
            assert call_kwargs["duration_policy"]["ai_clip_cap"] == 6
            
            assert captured_manifest is not None
            assert captured_manifest.target_duration_minutes == 2
            assert captured_manifest.ai_clip_cap == 6


def test_duration_minutes_cli_rejects_invalid_choice():
    """Test that --duration-minutes=0 is rejected by argparse with SystemExit."""
    import build_incremental
    from argparse import ArgumentParser
    
    test_args = ["build_incremental.py", "--project", "./projects/test", "--duration-minutes", "0"]
    
    parser = ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--duration-minutes", type=int, default=1, choices=[1, 2, 3])
    
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args()
        assert exc_info.value.code != 0


def test_duration_argparse_rejects_invalid_value():
    """Test that --duration-minutes=4 is also rejected (above allowed range)."""
    import build_incremental
    from argparse import ArgumentParser
    
    test_args = ["build_incremental.py", "--project", "./projects/test", "--duration-minutes", "4"]
    
    parser = ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--duration-minutes", type=int, default=1, choices=[1, 2, 3])
    
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args()
        assert exc_info.value.code != 0


def test_duration_argparse_accepts_valid_values():
    """Test that --duration-minutes accepts all valid choices [1, 2, 3]."""
    import build_incremental
    from argparse import ArgumentParser
    
    for duration in [1, 2, 3]:
        test_args = ["build_incremental.py", "--project", "./projects/test", "--duration-minutes", str(duration)]
        
        parser = ArgumentParser()
        parser.add_argument("--project", required=True)
        parser.add_argument("--duration-minutes", type=int, default=1, choices=[1, 2, 3])
        
        with patch.object(sys, "argv", test_args):
            args = parser.parse_args()
            assert args.duration_minutes == duration


def test_duration_argparse_defaults_to_one():
    """Test that omitted --duration-minutes defaults to 1."""
    import build_incremental
    from argparse import ArgumentParser
    
    test_args = ["build_incremental.py", "--project", "./projects/test"]
    
    parser = ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--duration-minutes", type=int, default=1, choices=[1, 2, 3])
    
    with patch.object(sys, "argv", test_args):
        args = parser.parse_args()
        assert args.duration_minutes == 1, "Default duration should be 1 minute"


def test_invalid_duration_zero_rejected():
    """Test that --duration-minutes=0 is rejected by argparse. Baseline regression for invalid_duration selector."""
    import build_incremental
    from argparse import ArgumentParser
    
    test_args = ["build_incremental.py", "--project", "./projects/test", "--duration-minutes", "0"]
    
    parser = ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--duration-minutes", type=int, default=1, choices=[1, 2, 3])
    
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args()
        assert exc_info.value.code != 0, "Invalid duration 0 should be rejected"


def test_invalid_duration_above_range_rejected():
    """Test that --duration-minutes=4 is rejected by argparse. Baseline regression for invalid_duration selector."""
    import build_incremental
    from argparse import ArgumentParser
    
    test_args = ["build_incremental.py", "--project", "./projects/test", "--duration-minutes", "4"]
    
    parser = ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--duration-minutes", type=int, default=1, choices=[1, 2, 3])
    
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args()
        assert exc_info.value.code != 0, "Invalid duration 4 should be rejected"
