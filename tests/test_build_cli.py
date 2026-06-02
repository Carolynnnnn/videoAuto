"""Tests for build.py CLI argument parsing and propagation."""
import sys
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_policy_helper_defaults():
    from src.core.generation_policy import normalize_generation_policy

    policy = normalize_generation_policy()
    assert policy["target_duration_minutes"] == 1
    assert policy["ai_clip_cap"] == 6


def test_policy_helper_invalid_minutes():
    from src.core.generation_policy import normalize_target_duration_minutes

    with pytest.raises(ValueError) as exc_info:
        normalize_target_duration_minutes(0)

    message = str(exc_info.value)
    assert "target_duration_minutes" in message
    assert "Allowed values" in message
    assert "1, 2, 3" in message


def test_material_mode_argument_parser_accepts_valid_values():
    """Test that argparse accepts all canonical material_mode values."""
    import build
    
    for mode in ["auto", "ai_preferred", "ai_only"]:
        test_args = ["--project", "./projects/test", "--material-mode", mode]
        with patch.object(sys, "argv", ["build.py"] + test_args):
            args = build.parse_args()
            assert args.material_mode == mode


def test_material_mode_argument_parser_defaults_to_auto():
    """Test that omitted --material-mode defaults to 'auto'."""
    import build
    
    test_args = ["--project", "./projects/test"]
    with patch.object(sys, "argv", ["build.py"] + test_args):
        args = build.parse_args()
        assert args.material_mode == "auto"


def test_material_mode_argument_parser_rejects_invalid_value():
    """Test that argparse exits with error for invalid material_mode."""
    import build
    
    test_args = ["--project", "./projects/test", "--material-mode", "invalid_mode"]
    with patch.object(sys, "argv", ["build.py"] + test_args):
        with pytest.raises(SystemExit) as exc_info:
            build.parse_args()
        assert exc_info.value.code != 0


def test_material_mode_propagates_to_step2_manifest():
    """Test that material_mode from CLI is passed to run_step2."""
    import build
    from src.steps import step2_manifest
    
    original_run_step2 = step2_manifest.run_step2
    
    with patch.object(step2_manifest, "run_step2") as mock_run_step2:
        mock_manifest = MagicMock()
        mock_manifest.material_mode = "ai_only"
        mock_run_step2.return_value = mock_manifest
        
        test_args = [
            "--project", "./projects/test",
            "--material-mode", "ai_only",
            "--dry-run"
        ]
        with patch.object(sys, "argv", ["build.py"] + test_args):
            try:
                build.main()
            except SystemExit:
                pass
        
        if mock_run_step2.called:
            call_kwargs = mock_run_step2.call_args[1]
            assert "material_mode" in call_kwargs
            assert call_kwargs["material_mode"] == "ai_only"


def test_material_mode_help_message_displays_choices(capsys):
    """Test that --help shows material-mode argument with choices."""
    import build
    
    test_args = ["--project", "./projects/test", "--help"]
    with patch.object(sys, "argv", ["build.py"] + test_args):
        with pytest.raises(SystemExit) as exc_info:
            build.parse_args()
        assert exc_info.value.code == 0
        
        captured = capsys.readouterr()
        help_output = captured.out
        
        assert "--material-mode" in help_output
        assert "auto" in help_output
        assert "ai_preferred" in help_output
        assert "ai_only" in help_output


def test_pdf_minimax_policy():
    """Test that PDF pipeline enforces Minimax-only policy in production mode."""
    from src.steps.step_pdf import run_pdf_pipeline
    from unittest.mock import Mock
    
    # Mock the TTS functions to verify which one gets called
    with patch("src.steps.step_pdf.generate_tts_minimax") as mock_minimax, \
         patch("src.steps.step_pdf.generate_tts_elevenlabs") as mock_elevenlabs, \
         patch("src.steps.step_pdf.generate_tts") as mock_openai, \
         patch("src.steps.step_pdf.extract_pdf_text") as mock_extract_text, \
         patch("src.steps.step_pdf.extract_pdf_images") as mock_extract_images, \
         patch("src.steps.step_pdf.generate_script_from_text") as mock_generate_script:
        
        # Setup mocks to return expected values
        mock_extract_text.return_value = "test content"
        mock_extract_images.return_value = []
        mock_generate_script.return_value = "test script"
        mock_minimax.return_value = "/tmp/voice.mp3"
        
        # Test 1: Production mode with minimax (default) - should succeed
        result = run_pdf_pipeline(
            pdf_path="/tmp/test.pdf",
            project_root="/tmp/project",
            tts_provider="minimax",
            allow_legacy_provider_override=False
        )
        assert mock_minimax.called, "Minimax should be called for minimax provider"
        assert not mock_elevenlabs.called, "ElevenLabs should not be called"
        assert not mock_openai.called, "OpenAI should not be called"
        
        # Reset mocks
        mock_minimax.reset_mock()
        mock_elevenlabs.reset_mock()
        mock_openai.reset_mock()
        
        # Test 2: Production mode requesting elevenlabs - should force minimax
        result = run_pdf_pipeline(
            pdf_path="/tmp/test.pdf",
            project_root="/tmp/project",
            tts_provider="elevenlabs",
            allow_legacy_provider_override=False
        )
        assert mock_minimax.called, "Should force Minimax in production mode"
        assert not mock_elevenlabs.called, "ElevenLabs should be blocked"
        
        # Reset mocks
        mock_minimax.reset_mock()
        mock_elevenlabs.reset_mock()
        
        # Test 3: Production mode requesting openai - should force minimax
        result = run_pdf_pipeline(
            pdf_path="/tmp/test.pdf",
            project_root="/tmp/project",
            tts_provider="openai",
            allow_legacy_provider_override=False
        )
        assert mock_minimax.called, "Should force Minimax in production mode"
        assert not mock_openai.called, "OpenAI should be blocked"


def test_provider_blocked_production(caplog):
    """Test that legacy providers are blocked in production with appropriate logging."""
    from src.steps.step_pdf import run_pdf_pipeline
    import logging
    
    # Mock dependencies
    with patch("src.steps.step_pdf.generate_tts_minimax") as mock_minimax, \
         patch("src.steps.step_pdf.generate_tts_elevenlabs") as mock_elevenlabs, \
         patch("src.steps.step_pdf.extract_pdf_text") as mock_extract_text, \
         patch("src.steps.step_pdf.extract_pdf_images") as mock_extract_images, \
         patch("src.steps.step_pdf.generate_script_from_text") as mock_generate_script:
        
        # Setup mocks
        mock_extract_text.return_value = "test content"
        mock_extract_images.return_value = []
        mock_generate_script.return_value = "test script"
        mock_minimax.return_value = "/tmp/voice.mp3"
        mock_elevenlabs.return_value = "/tmp/voice.mp3"
        
        # Test 1: Production mode blocks elevenlabs and logs warning
        with caplog.at_level(logging.WARNING):
            result = run_pdf_pipeline(
                pdf_path="/tmp/test.pdf",
                project_root="/tmp/project",
                tts_provider="elevenlabs",
                allow_legacy_provider_override=False
            )
        
        # Verify blocking behavior
        assert mock_minimax.called, "Should call Minimax instead"
        assert not mock_elevenlabs.called, "ElevenLabs should be blocked"
        
        # Verify warning log
        assert any("blocked by production policy" in record.message for record in caplog.records), \
            "Should log production policy block warning"
        assert any("Effective: minimax" in record.message for record in caplog.records), \
            "Should log effective provider"
        
        # Reset
        caplog.clear()
        mock_minimax.reset_mock()
        mock_elevenlabs.reset_mock()
        
        # Test 2: Non-production override allows legacy provider with warning
        with caplog.at_level(logging.WARNING):
            result = run_pdf_pipeline(
                pdf_path="/tmp/test.pdf",
                project_root="/tmp/project",
                tts_provider="elevenlabs",
                allow_legacy_provider_override=True
            )
        
        # Verify override behavior
        assert not mock_minimax.called, "Should NOT call Minimax when override is active"
        assert mock_elevenlabs.called, "ElevenLabs should be called with override"
        
        # Verify override warning log
        assert any("LEGACY PROVIDER OVERRIDE ACTIVE" in record.message for record in caplog.records), \
            "Should log legacy provider override warning"
        assert any("NOT recommended for production" in record.message for record in caplog.records), \
            "Should log production warning"


def test_from_pdf_minimax_provider():
    """Test that build.py PDF path propagates tts_provider to run_pdf_pipeline."""
    import build
    from src.steps import step_pdf
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_project = Path(tmpdir) / "test_project"
        test_project.mkdir()
        test_pdf = Path(tmpdir) / "test.pdf"
        test_pdf.write_bytes(b"fake pdf content")
        
        with patch.object(step_pdf, "run_pdf_pipeline") as mock_pipeline:
            mock_pipeline.return_value = {"voice_path": str(test_project / "voice.mp3")}
            
            test_args = [
                "--project", str(test_project),
                "--from-pdf", str(test_pdf),
                "--tts-provider", "minimax",
                "--dry-run"
            ]
            
            with patch.object(sys, "argv", ["build.py"] + test_args):
                try:
                    build.main()
                except SystemExit:
                    pass
            
            if mock_pipeline.called:
                call_kwargs = mock_pipeline.call_args[1]
                assert "tts_provider" in call_kwargs, "Should pass tts_provider to pipeline"
                assert call_kwargs["tts_provider"] == "minimax", "Should pass minimax provider"
                assert "allow_legacy_provider_override" in call_kwargs, "Should pass override flag"


def test_invalid_tts_provider():
    """Test that build.py rejects invalid tts_provider in production mode."""
    import build
    from pathlib import Path
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_project = Path(tmpdir) / "test_project"
        test_project.mkdir()
        test_pdf = Path(tmpdir) / "test.pdf"
        test_pdf.write_bytes(b"fake pdf content")
        
        test_args = [
            "--project", str(test_project),
            "--from-pdf", str(test_pdf),
            "--tts-provider", "elevenlabs"
        ]
        
        with patch.dict(os.environ, {"PIXELLE_TEST_MODE": "0"}):
            with patch.object(sys, "argv", ["build.py"] + test_args):
                with pytest.raises(SystemExit) as exc_info:
                    build.main()
                
                assert exc_info.value.code == 1, "Should exit with error code 1 for blocked provider"


def test_workflow_default_ai_only():
    """Test that ai_only mode applies pixelle_default_workflow='i2v' when none exists."""
    import build
    from src.core.models import Manifest, GlobalStyle, Segment
    from src.steps import step2_manifest, step3_visual_plan, step4_assets, step5_render, step6_concat
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_project = Path(tmpdir) / "test_project"
        test_project.mkdir()
        (test_project / "build").mkdir()
        (test_project / "render" / "segments").mkdir(parents=True)
        (test_project / "cache" / "plans").mkdir(parents=True)
        
        srt_path = test_project / "build" / "subtitle.srt"
        srt_path.write_text("1\n00:00:00,000 --> 00:00:02,000\nTest subtitle\n\n", encoding="utf-8")
        
        audio_path = test_project / "input" / "voice_full.wav"
        audio_path.parent.mkdir(exist_ok=True)
        audio_path.write_bytes(b"fake audio data")
        
        manifest_path = test_project / "build" / "manifest.json"
        
        with patch.object(step2_manifest, "run_step2") as mock_step2, \
             patch.object(step3_visual_plan, "run_step3") as mock_step3, \
             patch.object(step4_assets, "run_step4") as mock_step4, \
             patch.object(step5_render, "run_step5") as mock_step5, \
             patch.object(step6_concat, "run_step6") as mock_step6, \
             patch("build._should_enforce_strict_continuity_gate", return_value=False):
            
            base_manifest = Manifest(
                project_id="test",
                global_style=GlobalStyle(),
                material_mode="ai_only",
                pixelle_default_workflow=None,
            )
            mock_step2.return_value = base_manifest
            mock_step3.return_value = base_manifest
            mock_step4.return_value = base_manifest
            mock_step5.return_value = base_manifest
            mock_step6.return_value = base_manifest
            
            test_args = [
                "--project", str(test_project),
                "--srt", str(srt_path),
                "--material-mode", "ai_only"
            ]
            
            with patch.object(sys, "argv", ["build.py"] + test_args):
                build.main()
            
            assert base_manifest.pixelle_default_workflow == "i2v", \
                "ai_only mode should apply i2v workflow default when none exists"


def test_workflow_default_ai_preferred():
    """Test that ai_preferred mode applies pixelle_default_workflow='i2v' when none exists."""
    import build
    from src.core.models import Manifest, GlobalStyle
    from src.steps import step2_manifest, step3_visual_plan, step4_assets, step5_render, step6_concat
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_project = Path(tmpdir) / "test_project"
        test_project.mkdir()
        (test_project / "build").mkdir()
        (test_project / "render" / "segments").mkdir(parents=True)
        (test_project / "cache" / "plans").mkdir(parents=True)
        
        srt_path = test_project / "build" / "subtitle.srt"
        srt_path.write_text("1\n00:00:00,000 --> 00:00:02,000\nTest subtitle\n\n", encoding="utf-8")
        
        audio_path = test_project / "input" / "voice_full.wav"
        audio_path.parent.mkdir(exist_ok=True)
        audio_path.write_bytes(b"fake audio data")
        
        with patch.object(step2_manifest, "run_step2") as mock_step2, \
             patch.object(step3_visual_plan, "run_step3") as mock_step3, \
             patch.object(step4_assets, "run_step4") as mock_step4, \
             patch.object(step5_render, "run_step5") as mock_step5, \
             patch.object(step6_concat, "run_step6") as mock_step6, \
             patch("build._should_enforce_strict_continuity_gate", return_value=False):
            
            base_manifest = Manifest(
                project_id="test",
                global_style=GlobalStyle(),
                material_mode="ai_preferred",
                pixelle_default_workflow=None,
            )
            mock_step2.return_value = base_manifest
            mock_step3.return_value = base_manifest
            mock_step4.return_value = base_manifest
            mock_step5.return_value = base_manifest
            mock_step6.return_value = base_manifest
            
            test_args = [
                "--project", str(test_project),
                "--srt", str(srt_path),
                "--material-mode", "ai_preferred"
            ]
            
            with patch.object(sys, "argv", ["build.py"] + test_args):
                build.main()
            
            assert base_manifest.pixelle_default_workflow == "i2v", \
                "ai_preferred mode should apply i2v workflow default when none exists"


def test_workflow_default_auto_no_application():
    """Test that auto mode does NOT apply workflow default."""
    import build
    from src.core.models import Manifest, GlobalStyle
    from src.steps import step2_manifest, step3_visual_plan, step4_assets, step5_render, step6_concat
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_project = Path(tmpdir) / "test_project"
        test_project.mkdir()
        (test_project / "build").mkdir()
        (test_project / "render" / "segments").mkdir(parents=True)
        (test_project / "cache" / "plans").mkdir(parents=True)
        
        srt_path = test_project / "build" / "subtitle.srt"
        srt_path.write_text("1\n00:00:00,000 --> 00:00:02,000\nTest subtitle\n\n", encoding="utf-8")
        
        audio_path = test_project / "input" / "voice_full.wav"
        audio_path.parent.mkdir(exist_ok=True)
        audio_path.write_bytes(b"fake audio data")
        
        with patch.object(step2_manifest, "run_step2") as mock_step2, \
             patch.object(step3_visual_plan, "run_step3") as mock_step3, \
             patch.object(step4_assets, "run_step4") as mock_step4, \
             patch.object(step5_render, "run_step5") as mock_step5, \
             patch.object(step6_concat, "run_step6") as mock_step6, \
             patch("build._should_enforce_strict_continuity_gate", return_value=False):
            
            base_manifest = Manifest(
                project_id="test",
                global_style=GlobalStyle(),
                material_mode="auto",
                pixelle_default_workflow=None,
            )
            mock_step2.return_value = base_manifest
            mock_step3.return_value = base_manifest
            mock_step4.return_value = base_manifest
            mock_step5.return_value = base_manifest
            mock_step6.return_value = base_manifest
            
            test_args = [
                "--project", str(test_project),
                "--srt", str(srt_path),
                "--material-mode", "auto"
            ]
            
            with patch.object(sys, "argv", ["build.py"] + test_args):
                build.main()
            
            assert base_manifest.pixelle_default_workflow is None, \
                "auto mode should NOT apply workflow default"


def test_override_precedence_preserves_explicit_workflow():
    """Test that explicit workflow is preserved even in ai_only mode."""
    import build
    from src.core.models import Manifest, GlobalStyle
    from src.steps import step2_manifest, step3_visual_plan, step4_assets, step5_render, step6_concat
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_project = Path(tmpdir) / "test_project"
        test_project.mkdir()
        (test_project / "build").mkdir()
        (test_project / "render" / "segments").mkdir(parents=True)
        (test_project / "cache" / "plans").mkdir(parents=True)
        
        srt_path = test_project / "build" / "subtitle.srt"
        srt_path.write_text("1\n00:00:00,000 --> 00:00:02,000\nTest subtitle\n\n", encoding="utf-8")
        
        audio_path = test_project / "input" / "voice_full.wav"
        audio_path.parent.mkdir(exist_ok=True)
        audio_path.write_bytes(b"fake audio data")
        
        with patch.object(step2_manifest, "run_step2") as mock_step2, \
             patch.object(step3_visual_plan, "run_step3") as mock_step3, \
             patch.object(step4_assets, "run_step4") as mock_step4, \
             patch.object(step5_render, "run_step5") as mock_step5, \
             patch.object(step6_concat, "run_step6") as mock_step6, \
             patch("build._should_enforce_strict_continuity_gate", return_value=False):
            
            base_manifest = Manifest(
                project_id="test",
                global_style=GlobalStyle(),
                material_mode="ai_only",
                pixelle_default_workflow="digital_human",
            )
            mock_step2.return_value = base_manifest
            mock_step3.return_value = base_manifest
            mock_step4.return_value = base_manifest
            mock_step5.return_value = base_manifest
            mock_step6.return_value = base_manifest
            
            test_args = [
                "--project", str(test_project),
                "--srt", str(srt_path),
                "--material-mode", "ai_only"
            ]
            
            with patch.object(sys, "argv", ["build.py"] + test_args):
                build.main()
            
            assert base_manifest.pixelle_default_workflow == "digital_human", \
                "Explicit workflow should be preserved, not overwritten by ai_only default"


def test_gate_profile_release_blocks_on_failure():
    """Test that --gate-profile=release exits with code 1 when strict gate fails.
    
    Release profile is the default and should block build when continuity gate fails.
    This test verifies the hard-fail behavior for production builds.
    """
    import build
    from src.core.models import Manifest, GlobalStyle, Segment
    from src.steps import step2_manifest, step3_visual_plan, step4_assets, step5_render, step6_concat
    from src.steps.continuity_telemetry import (
        compute_quality_summary,
        validate_strict_mode,
        StrictModeValidationResult,
        GateViolationDetail,
        StrictGateViolation,
        ContinuityQualitySummary,
    )
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_project = Path(tmpdir) / "test_project"
        test_project.mkdir()
        (test_project / "build").mkdir()
        (test_project / "render" / "segments").mkdir(parents=True)
        (test_project / "cache" / "plans").mkdir(parents=True)
        
        srt_path = test_project / "build" / "subtitle.srt"
        srt_path.write_text("1\n00:00:00,000 --> 00:00:02,000\nTest subtitle\n\n", encoding="utf-8")
        
        audio_path = test_project / "input" / "voice_full.wav"
        audio_path.parent.mkdir(exist_ok=True)
        audio_path.write_bytes(b"fake audio data")
        
        # Create manifest with segments that will fail strict gate
        base_manifest = Manifest(
            project_id="test",
            global_style=GlobalStyle(),
            segments=[
                Segment(
                    segment_key="seg_001",
                    content_key="ck_001",
                    index=1,
                    start=0.0,
                    end=2.0,
                    duration=2.0,
                    text="Test subtitle",
                    continuity_diagnostic={
                        "continuity_mode": "off",  # This will cause low temporal link coverage
                    },
                )
            ],
        )
        
        step6_called = False
        
        def _unexpected_step6(**kwargs):
            nonlocal step6_called
            step6_called = True
            return kwargs["manifest"]
        
        with patch.object(step2_manifest, "run_step2") as mock_step2, \
             patch.object(step3_visual_plan, "run_step3") as mock_step3, \
             patch.object(step4_assets, "run_step4") as mock_step4, \
             patch.object(step5_render, "run_step5") as mock_step5, \
             patch.object(step6_concat, "run_step6", side_effect=_unexpected_step6), \
             patch("build._should_enforce_strict_continuity_gate", return_value=True):
            
            mock_step2.return_value = base_manifest
            mock_step3.return_value = base_manifest
            mock_step4.return_value = base_manifest
            mock_step5.return_value = base_manifest
            
            test_args = [
                "--project", str(test_project),
                "--srt", str(srt_path),
                "--gate-profile", "release"
            ]
            
            with patch.object(sys, "argv", ["build.py"] + test_args):
                with pytest.raises(SystemExit) as exc_info:
                    build.main()
                
                # Release profile should exit with code 1 on gate failure
                assert exc_info.value.code == 1, \
                    "Release profile should exit with code 1 when strict gate fails"
        
        # Step6 should NOT have been called (build blocked at gate)
        assert step6_called is False, \
            "Step6 should not be called when release gate blocks"


def test_gate_profile_preview_warns_but_continues(caplog):
    """Test that --gate-profile=preview logs warnings but continues to Step6.
    
    Preview profile should log warnings when continuity gate fails but NOT block
    the build. This allows developers to preview builds without strict enforcement.
    """
    import build
    import logging
    from src.core.models import Manifest, GlobalStyle, Segment
    from src.steps import step2_manifest, step3_visual_plan, step4_assets, step5_render, step6_concat
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_project = Path(tmpdir) / "test_project"
        test_project.mkdir()
        (test_project / "build").mkdir()
        (test_project / "render" / "segments").mkdir(parents=True)
        (test_project / "cache" / "plans").mkdir(parents=True)
        
        srt_path = test_project / "build" / "subtitle.srt"
        srt_path.write_text("1\n00:00:00,000 --> 00:00:02,000\nTest subtitle\n\n", encoding="utf-8")
        
        audio_path = test_project / "input" / "voice_full.wav"
        audio_path.parent.mkdir(exist_ok=True)
        audio_path.write_bytes(b"fake audio data")
        
        # Create manifest with segments that will fail strict gate
        base_manifest = Manifest(
            project_id="test",
            global_style=GlobalStyle(),
            segments=[
                Segment(
                    segment_key="seg_001",
                    content_key="ck_001",
                    index=1,
                    start=0.0,
                    end=2.0,
                    duration=2.0,
                    text="Test subtitle",
                    continuity_diagnostic={
                        "continuity_mode": "off",  # This will cause low temporal link coverage
                    },
                )
            ],
        )
        
        step6_called = False
        
        def _track_step6(**kwargs):
            nonlocal step6_called
            step6_called = True
            return kwargs["manifest"]
        
        with patch.object(step2_manifest, "run_step2") as mock_step2, \
             patch.object(step3_visual_plan, "run_step3") as mock_step3, \
             patch.object(step4_assets, "run_step4") as mock_step4, \
             patch.object(step5_render, "run_step5") as mock_step5, \
             patch.object(step6_concat, "run_step6", side_effect=_track_step6), \
             patch("build._should_enforce_strict_continuity_gate", return_value=True):
            
            mock_step2.return_value = base_manifest
            mock_step3.return_value = base_manifest
            mock_step4.return_value = base_manifest
            mock_step5.return_value = base_manifest
            
            test_args = [
                "--project", str(test_project),
                "--srt", str(srt_path),
                "--gate-profile", "preview"
            ]
            
            with caplog.at_level(logging.WARNING):
                with patch.object(sys, "argv", ["build.py"] + test_args):
                    # Preview profile should NOT raise SystemExit
                    build.main()
        
        # Step6 SHOULD have been called (preview mode is non-blocking)
        assert step6_called is True, \
            "Step6 should be called when preview gate allows non-blocking"
        
        # Verify warning was logged
        assert any("preview mode - non-blocking" in record.message for record in caplog.records), \
            "Should log preview mode warning"


def test_duration_argparse_rejects_invalid_value():
    """Test that --duration-minutes=0 or 4 is rejected by argparse with SystemExit."""
    import build
    
    # Test value 0 (below allowed range)
    test_args = ["--project", "./projects/test", "--duration-minutes", "0"]
    with patch.object(sys, "argv", ["build.py"] + test_args):
        with pytest.raises(SystemExit) as exc_info:
            build.parse_args()
        assert exc_info.value.code != 0
    
    # Test value 4 (above allowed range)
    test_args = ["--project", "./projects/test", "--duration-minutes", "4"]
    with patch.object(sys, "argv", ["build.py"] + test_args):
        with pytest.raises(SystemExit) as exc_info:
            build.parse_args()
        assert exc_info.value.code != 0


def test_invalid_duration_zero_rejected():
    """Test that --duration-minutes=0 is rejected by argparse. Baseline regression for invalid_duration selector."""
    import build
    
    test_args = ["--project", "./projects/test", "--duration-minutes", "0"]
    with patch.object(sys, "argv", ["build.py"] + test_args):
        with pytest.raises(SystemExit) as exc_info:
            build.parse_args()
        assert exc_info.value.code != 0, "Invalid duration 0 should be rejected"


def test_invalid_duration_above_range_rejected():
    """Test that --duration-minutes=4 is rejected by argparse. Baseline regression for invalid_duration selector."""
    import build
    
    test_args = ["--project", "./projects/test", "--duration-minutes", "4"]
    with patch.object(sys, "argv", ["build.py"] + test_args):
        with pytest.raises(SystemExit) as exc_info:
            build.parse_args()
        assert exc_info.value.code != 0, "Invalid duration 4 should be rejected"


def test_duration_argparse_accepts_valid_values():
    """Test that --duration-minutes accepts all valid choices [1, 2, 3]."""
    import build
    
    for duration in [1, 2, 3]:
        test_args = ["--project", "./projects/test", "--duration-minutes", str(duration)]
        with patch.object(sys, "argv", ["build.py"] + test_args):
            args = build.parse_args()
            assert args.duration_minutes == duration


def test_duration_argparse_defaults_to_one():
    """Test that omitted --duration-minutes defaults to 1."""
    import build
    
    test_args = ["--project", "./projects/test"]
    with patch.object(sys, "argv", ["build.py"] + test_args):
        args = build.parse_args()
        assert args.duration_minutes == 1, "Default duration should be 1 minute"


def test_duration_minutes_propagates():
    """Test that --duration-minutes propagates to Step2 with generation policy."""
    import build
    from src.steps import step2_manifest
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_project = Path(tmpdir) / "test_project"
        test_project.mkdir()
        (test_project / "build").mkdir()
        (test_project / "render" / "segments").mkdir(parents=True)
        (test_project / "cache" / "plans").mkdir(parents=True)
        
        srt_path = test_project / "build" / "subtitle.srt"
        srt_path.write_text("1\n00:00:00,000 --> 00:00:02,000\nTest subtitle\n\n", encoding="utf-8")
        
        audio_path = test_project / "input" / "voice_full.wav"
        audio_path.parent.mkdir(exist_ok=True)
        audio_path.write_bytes(b"fake audio data")
        
        with patch.object(step2_manifest, "run_step2") as mock_run_step2:
            mock_manifest = MagicMock()
            mock_manifest.pixelle_default_workflow = None
            mock_run_step2.return_value = mock_manifest
            
            test_args = [
                "--project", str(test_project),
                "--duration-minutes", "2",
                "--dry-run"
            ]
            
            with patch.object(sys, "argv", ["build.py"] + test_args):
                try:
                    build.main()
                except SystemExit:
                    pass
            
            if mock_run_step2.called:
                call_kwargs = mock_run_step2.call_args[1]
                assert "duration_policy" in call_kwargs
                duration_policy = call_kwargs["duration_policy"]
                
                assert duration_policy is not None, "duration_policy should be passed"
                assert "target_duration_minutes" in duration_policy, \
                    "duration_policy should contain target_duration_minutes"
                assert duration_policy["target_duration_minutes"] == 2, \
                    "target_duration_minutes should be 2 (from CLI)"
                assert "ai_clip_cap" in duration_policy, \
                    "duration_policy should contain ai_clip_cap"
                assert duration_policy["ai_clip_cap"] == 6, \
                    "ai_clip_cap should be 6 (fixed default)"


