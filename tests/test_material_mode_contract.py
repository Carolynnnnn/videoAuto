"""Tests for material_mode contract validation in Manifest and GlobalStyle."""
import pytest
from typing import Any, cast
from src.core.models import (
    Manifest, GlobalStyle, MaterialModeError, MATERIAL_MODES,
    DurationCapPolicyError, TARGET_DURATION_MINUTES_ALLOWED,
    TARGET_DURATION_MINUTES_DEFAULT, AI_CLIP_CAP_DEFAULT,
)


def test_material_mode_valid_values():
    """Test that all valid material_mode values are accepted."""
    for mode in ["auto", "ai_preferred", "ai_only"]:
        manifest = Manifest(
            project_id="test-project",
            material_mode=cast(Any, mode),
        )
        assert manifest.material_mode == mode


def test_material_mode_defaults_to_auto_when_absent():
    """Test that missing material_mode defaults to 'auto'."""
    manifest = Manifest(project_id="test-project")
    assert manifest.material_mode == "auto"


def test_material_mode_invalid_value_raises_error():
    """Test that invalid material_mode raises MaterialModeError with clear message."""
    with pytest.raises(MaterialModeError) as exc_info:
        Manifest(
            project_id="test-project",
            material_mode=cast(Any, "invalid_mode"),
        )
    
    error = exc_info.value
    assert error.value == "invalid_mode"
    assert error.allowed == MATERIAL_MODES
    assert "invalid_mode" in str(error)
    assert "auto" in str(error)
    assert "ai_preferred" in str(error)
    assert "ai_only" in str(error)


def test_material_mode_load_from_dict_with_valid_mode():
    """Test loading manifest from dict with valid material_mode."""
    data = {
        "project_id": "test-project",
        "material_mode": "ai_preferred",
        "segments": [],
    }
    manifest = Manifest.load_from_dict(data)
    assert manifest.material_mode == "ai_preferred"


def test_material_mode_load_from_dict_defaults_to_auto():
    """Test loading manifest from dict without material_mode defaults to 'auto'."""
    data = {
        "project_id": "test-project",
        "segments": [],
    }
    manifest = Manifest.load_from_dict(data)
    assert manifest.material_mode == "auto"


def test_material_mode_load_from_dict_invalid_raises_error():
    """Test loading manifest from dict with invalid material_mode raises error."""
    data = {
        "project_id": "test-project",
        "material_mode": "bad_mode",
        "segments": [],
    }
    with pytest.raises(MaterialModeError) as exc_info:
        Manifest.load_from_dict(data)
    
    error = exc_info.value
    assert error.value == "bad_mode"


def test_material_mode_save_and_load_roundtrip(tmp_path):
    """Test that material_mode is preserved through save/load cycle."""
    manifest_path = tmp_path / "manifest.json"
    
    # Create and save manifest with ai_only mode
    original = Manifest(
        project_id="test-project",
        material_mode="ai_only",
    )
    original.save(str(manifest_path))
    
    # Load and verify
    loaded = Manifest.load(str(manifest_path))
    assert loaded.material_mode == "ai_only"


def test_material_mode_save_and_load_default_mode(tmp_path):
    """Test that default 'auto' mode is preserved through save/load cycle."""
    manifest_path = tmp_path / "manifest.json"
    
    # Create and save manifest without specifying mode (should default to 'auto')
    original = Manifest(project_id="test-project")
    original.save(str(manifest_path))
    
    # Load and verify
    loaded = Manifest.load(str(manifest_path))
    assert loaded.material_mode == "auto"


def test_material_mode_to_dict_includes_field():
    """Test that material_mode is included in to_dict() output."""
    manifest = Manifest(
        project_id="test-project",
        material_mode="ai_preferred",
    )
    data = manifest.to_dict()
    assert data["material_mode"] == "ai_preferred"


def test_material_mode_contract_constants():
    """Test that MATERIAL_MODES constant contains exactly the expected values."""
    assert MATERIAL_MODES == {"auto", "ai_preferred", "ai_only"}


def test_material_mode_error_attributes():
    """Test MaterialModeError exception attributes."""
    error = MaterialModeError("wrong_mode", {"auto", "ai_preferred", "ai_only"})
    assert error.value == "wrong_mode"
    assert error.allowed == {"auto", "ai_preferred", "ai_only"}
    assert isinstance(error, ValueError)


# ─────────────────────────────────────────────
# Duration/Cap Policy Tests
# ─────────────────────────────────────────────

def test_duration_cap_defaults():
    """Test that default values are applied when fields are absent."""
    manifest = Manifest(project_id="test-project")
    assert manifest.target_duration_minutes == TARGET_DURATION_MINUTES_DEFAULT
    assert manifest.ai_clip_cap == AI_CLIP_CAP_DEFAULT


def test_duration_cap_valid_values():
    """Test that all valid target_duration_minutes values are accepted."""
    for duration in [1, 2, 3]:
        manifest = Manifest(
            project_id="test-project",
            target_duration_minutes=duration,
        )
        assert manifest.target_duration_minutes == duration


def test_duration_cap_roundtrip(tmp_path):
    """Test that duration/cap fields are preserved through save/load cycle."""
    manifest_path = tmp_path / "manifest.json"
    
    original = Manifest(
        project_id="test-project",
        target_duration_minutes=3,
        ai_clip_cap=12,
    )
    original.save(str(manifest_path))
    
    loaded = Manifest.load(str(manifest_path))
    assert loaded.target_duration_minutes == 3
    assert loaded.ai_clip_cap == 12


def test_duration_cap_backward_compatibility():
    """Test that old manifests without duration/cap fields load with defaults."""
    data = {
        "project_id": "old-manifest-project",
        "segments": [],
    }
    manifest = Manifest.load_from_dict(data)
    assert manifest.target_duration_minutes == TARGET_DURATION_MINUTES_DEFAULT
    assert manifest.ai_clip_cap == AI_CLIP_CAP_DEFAULT


def test_invalid_duration_minutes():
    """Test that invalid target_duration_minutes raises DurationCapPolicyError."""
    with pytest.raises(DurationCapPolicyError) as exc_info:
        Manifest(
            project_id="test-project",
            target_duration_minutes=cast(Any, 5),
        )
    
    error = exc_info.value
    assert error.field == "target_duration_minutes"
    assert error.value == 5
    assert error.allowed == TARGET_DURATION_MINUTES_ALLOWED
    assert "target_duration_minutes" in str(error)
    assert "5" in str(error)


def test_invalid_duration_minutes_load_from_dict():
    """Test loading manifest with invalid target_duration_minutes raises error."""
    data = {
        "project_id": "test-project",
        "target_duration_minutes": 0,
        "segments": [],
    }
    with pytest.raises(DurationCapPolicyError) as exc_info:
        Manifest.load_from_dict(data)
    
    error = exc_info.value
    assert error.field == "target_duration_minutes"
    assert error.value == 0


def test_invalid_ai_clip_cap_zero():
    """Test that ai_clip_cap=0 raises DurationCapPolicyError."""
    with pytest.raises(DurationCapPolicyError) as exc_info:
        Manifest(
            project_id="test-project",
            ai_clip_cap=cast(Any, 0),
        )
    
    error = exc_info.value
    assert error.field == "ai_clip_cap"
    assert error.value == 0


def test_invalid_ai_clip_cap_negative():
    """Test that negative ai_clip_cap raises DurationCapPolicyError."""
    with pytest.raises(DurationCapPolicyError) as exc_info:
        Manifest(
            project_id="test-project",
            ai_clip_cap=cast(Any, -1),
        )
    
    error = exc_info.value
    assert error.field == "ai_clip_cap"


def test_duration_cap_to_dict_includes_fields():
    """Test that duration/cap fields are included in to_dict() output."""
    manifest = Manifest(
        project_id="test-project",
        target_duration_minutes=2,
        ai_clip_cap=10,
    )
    data = manifest.to_dict()
    assert data["target_duration_minutes"] == 2
    assert data["ai_clip_cap"] == 10


def test_duration_cap_policy_error_attributes():
    """Test DurationCapPolicyError exception attributes."""
    error = DurationCapPolicyError("target_duration_minutes", 99, {1, 2, 3})
    assert error.field == "target_duration_minutes"
    assert error.value == 99
    assert error.allowed == {1, 2, 3}
    assert isinstance(error, ValueError)


def test_duration_cap_constants():
    """Test that duration/cap constants have expected values."""
    assert TARGET_DURATION_MINUTES_ALLOWED == {1, 2, 3}
    assert TARGET_DURATION_MINUTES_DEFAULT == 1
    assert AI_CLIP_CAP_DEFAULT == 6


def test_invalid_duration_minutes_negative():
    """Test that negative target_duration_minutes raises DurationCapPolicyError."""
    with pytest.raises(DurationCapPolicyError) as exc_info:
        Manifest(
            project_id="test-project",
            target_duration_minutes=cast(Any, -1),
        )
    
    error = exc_info.value
    assert error.field == "target_duration_minutes"
    assert error.value == -1
    assert error.allowed == TARGET_DURATION_MINUTES_ALLOWED


def test_invalid_duration_minutes_boundary_four():
    """Test that target_duration_minutes=4 (just above allowed) raises DurationCapPolicyError."""
    with pytest.raises(DurationCapPolicyError) as exc_info:
        Manifest(
            project_id="test-project",
            target_duration_minutes=cast(Any, 4),
        )
    
    error = exc_info.value
    assert error.field == "target_duration_minutes"
    assert error.value == 4


def test_invalid_duration_minutes_large_value():
    """Test that target_duration_minutes=100 raises DurationCapPolicyError."""
    with pytest.raises(DurationCapPolicyError) as exc_info:
        Manifest(
            project_id="test-project",
            target_duration_minutes=cast(Any, 100),
        )
    
    error = exc_info.value
    assert error.field == "target_duration_minutes"
    assert error.value == 100


def test_invalid_duration_minutes_negative_load_from_dict():
    """Test loading manifest with negative target_duration_minutes raises error."""
    data = {
        "project_id": "test-project",
        "target_duration_minutes": -5,
        "segments": [],
    }
    with pytest.raises(DurationCapPolicyError) as exc_info:
        Manifest.load_from_dict(data)
    
    error = exc_info.value
    assert error.field == "target_duration_minutes"
    assert error.value == -5
