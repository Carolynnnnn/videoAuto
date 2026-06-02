"""
Tests for workbench state initialization and persistence.

Covers:
- T1 Scenario: initialize workbench directories for a fixture project
- T1 Scenario: reopen an existing project with saved workbench state
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
pytest = __import__("pytest")
from src.workbench.state import (
    WorkbenchPaths,
    WorkbenchSession,
    init_workbench,
    load_session,
    save_session,
)
from src.workbench.apply import ManifestDriftConflictError, apply_pixelle_selections
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from src.gui.app import VideoAutomationApp

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app

@pytest.fixture
def fixture_project(tmp_path):
    """Create a minimal fixture project without workbench directory."""
    project_root = tmp_path / "test_project"
    project_root.mkdir()
    
    # Create standard project directories
    (project_root / "input").mkdir()
    (project_root / "build").mkdir()
    
    # Create a minimal manifest.json (formal state)
    manifest_path = project_root / "build" / "manifest.json"
    manifest_path.write_text(json.dumps({
        "project_id": "test_project",
        "build_id": "fixture-build-1",
        "segments": [],
        "global_style": {}
    }, indent=2))
    
    return project_root


def test_init_workbench_creates_layout(fixture_project):
    """
    T1 Scenario: initialize workbench directories for a fixture project
    
    Steps:
      1. Run the workbench initialization entrypoint against the fixture project
      2. Assert project/workbench/session.json and library directories now exist
      3. Assert build/manifest.json file contents are unchanged
    Expected Result: workbench layout is created and manifest is untouched
    """
    # Capture original manifest before initialization
    manifest_path = fixture_project / "build" / "manifest.json"
    original_manifest = manifest_path.read_text()
    
    # Initialize workbench
    paths = init_workbench(fixture_project)
    
    # Assert workbench paths exist
    assert paths.workbench_dir.exists()
    assert paths.workbench_dir.is_dir()
    assert paths.session_json.exists()
    assert paths.library_dir.exists()
    assert paths.library_dir.is_dir()
    assert paths.thumbnails_dir.exists()
    assert paths.cache_dir.exists()
    
    # Assert session.json was created with valid structure
    session = load_session(paths)
    assert session.created_at != ""
    assert session.last_modified != ""
    assert session.draft_selections == {}
    assert session.style_profile == {}
    
    # Assert library metadata index was created
    assert paths.library_metadata_json.exists()
    library_data = json.loads(paths.library_metadata_json.read_text())
    assert "assets" in library_data
    assert library_data["assets"] == []
    
    # Assert manifest.json is completely unchanged
    current_manifest = manifest_path.read_text()
    assert current_manifest == original_manifest


def test_reload_existing_workbench_session(fixture_project):
    """
    T1 Scenario: reopen an existing project with saved workbench state
    
    Steps:
      1. Load the workbench session from disk
      2. Assert previously saved draft values round-trip exactly
    Expected Result: persisted draft session loads successfully
    """
    # Initialize workbench first time
    paths = init_workbench(fixture_project)
    
    # Create and save a session with draft data
    session = load_session(paths)
    session.draft_selections = {
        "seg_001": {
            "source": "ai",
            "asset_id": "generated_001",
            "prompt": "abstract blue background"
        }
    }
    session.style_profile = {
        "visual_style": "modern",
        "palette": ["#0066CC", "#FFFFFF"]
    }
    session.template_preset = "modern_tech_9x16"
    save_session(paths, session)
    
    # Reload session from disk
    reloaded_session = load_session(paths)
    
    # Assert all draft values round-trip exactly
    assert reloaded_session.draft_selections == session.draft_selections
    assert reloaded_session.style_profile == session.style_profile
    assert reloaded_session.template_preset == "modern_tech_9x16"
    assert reloaded_session.created_at == session.created_at
    # last_modified should be updated during save
    assert reloaded_session.last_modified != ""


def test_init_workbench_idempotent(fixture_project):
    """Calling init_workbench multiple times is safe and preserves session."""
    # First init
    paths1 = init_workbench(fixture_project)
    session1 = load_session(paths1)
    session1.draft_selections["test"] = {"value": 123}
    save_session(paths1, session1)
    
    # Second init (should not overwrite existing session)
    paths2 = init_workbench(fixture_project)
    session2 = load_session(paths2)
    
    assert session2.draft_selections == {"test": {"value": 123}}


def test_init_workbench_invalid_path():
    """init_workbench raises ValueError for non-existent path."""
    with pytest.raises(ValueError, match="does not exist"):
        init_workbench("/nonexistent/path")


def test_load_session_missing_file(fixture_project):
    """load_session raises FileNotFoundError if session.json missing."""
    paths = WorkbenchPaths(project_root=fixture_project)
    paths.ensure_dirs()
    
    # Don't create session.json
    assert not paths.session_json.exists()
    
    with pytest.raises(FileNotFoundError):
        load_session(paths)


def test_session_serialization_roundtrip():
    """WorkbenchSession to_dict/from_dict preserves all fields."""
    original = WorkbenchSession(
        created_at="2026-03-12T10:00:00",
        last_modified="2026-03-12T11:00:00",
        draft_selections={"seg_001": {"source": "library"}},
        style_profile={"tone": "corporate"},
        template_preset="preset_1",
        recommendation_cache={"seg_001": ["rec_1", "rec_2"]},
        pixelle_default_workflow="digital_human",
        pixelle_segment_overrides={"seg_002": "i2v", "seg_003": None}
    )
    
    data = original.to_dict()
    restored = WorkbenchSession.from_dict(data)
    
    assert restored.created_at == original.created_at
    assert restored.last_modified == original.last_modified
    assert restored.draft_selections == original.draft_selections
    assert restored.style_profile == original.style_profile
    assert restored.template_preset == original.template_preset
    assert restored.recommendation_cache == original.recommendation_cache
    assert restored.pixelle_default_workflow == original.pixelle_default_workflow
    assert restored.pixelle_segment_overrides == original.pixelle_segment_overrides


def test_pixelle_workflow_backward_compatibility():
    """Loading session without Pixelle fields uses safe defaults."""
    legacy_data = {
        "created_at": "2026-03-01T10:00:00",
        "last_modified": "2026-03-01T11:00:00",
        "draft_selections": {},
        "style_profile": {}
    }
    
    session = WorkbenchSession.from_dict(legacy_data)
    
    assert session.pixelle_default_workflow is None
    assert session.pixelle_segment_overrides == {}


def test_pixelle_workflow_invalid_values_sanitized():
    """Invalid Pixelle workflow values are sanitized to None."""
    corrupt_data = {
        "created_at": "2026-03-01T10:00:00",
        "last_modified": "2026-03-01T11:00:00",
        "draft_selections": {},
        "style_profile": {},
        "pixelle_default_workflow": "invalid_workflow",
        "pixelle_segment_overrides": {
            "seg_001": "invalid_workflow",
            "seg_002": "digital_human",
            "seg_003": "i2v",
            "seg_004": None,
            "seg_005": "action_transfer"
        }
    }
    
    session = WorkbenchSession.from_dict(corrupt_data)
    
    assert session.pixelle_default_workflow is None
    assert "seg_001" not in session.pixelle_segment_overrides
    assert session.pixelle_segment_overrides["seg_002"] == "digital_human"
    assert session.pixelle_segment_overrides["seg_003"] == "i2v"
    assert session.pixelle_segment_overrides["seg_004"] is None
    assert session.pixelle_segment_overrides["seg_005"] == "action_transfer"


def test_pixelle_workflow_reload_preserves_selections(fixture_project):
    """Pixelle workflow selections persist across reload."""
    paths = init_workbench(fixture_project)
    
    session = load_session(paths)
    session.pixelle_default_workflow = "i2v"
    session.pixelle_segment_overrides = {
        "seg_001": "digital_human",
        "seg_002": None
    }
    save_session(paths, session)
    
    reloaded_session = load_session(paths)
    
    assert reloaded_session.pixelle_default_workflow == "i2v"
    assert reloaded_session.pixelle_segment_overrides == {
        "seg_001": "digital_human",
        "seg_002": None
    }


def test_pixelle_selection_draft_changes_do_not_mutate_manifest_before_apply(fixture_project):
    manifest_path = fixture_project / "build" / "manifest.json"
    original_manifest = manifest_path.read_text()

    paths = init_workbench(fixture_project)
    session = load_session(paths)
    session.pixelle_default_workflow = "i2v"
    session.pixelle_segment_overrides = {
        "seg_001": "digital_human",
        "seg_002": None,
    }
    save_session(paths, session)

    assert manifest_path.read_text() == original_manifest


def test_apply_pixelle_selections_updates_manifest_fields_only(fixture_project):
    manifest_path = fixture_project / "build" / "manifest.json"
    manifest_path.write_text(json.dumps({
        "project_id": "test_project",
        "build_id": "fixture-build-1",
        "global_style": {},
        "segments": [
            {
                "segment_key": "seg_001#1",
                "content_key": "seg_001",
                "index": 1,
                "start": 0.0,
                "end": 1.0,
                "duration": 1.0,
                "text": "first"
            },
            {
                "segment_key": "seg_002#1",
                "content_key": "seg_002",
                "index": 2,
                "start": 1.0,
                "end": 2.0,
                "duration": 1.0,
                "text": "second"
            }
        ],
        "audio_path": "audio.wav",
        "build_status": "pending"
    }, indent=2))

    paths = init_workbench(fixture_project)
    session = load_session(paths)
    session.pixelle_default_workflow = "action_transfer"
    session.pixelle_segment_overrides = {
        "seg_001#1": "digital_human",
        "seg_002#1": None,
        "seg_missing#1": "i2v",
    }
    save_session(paths, session)

    apply_result = apply_pixelle_selections(paths)
    updated_manifest = json.loads(manifest_path.read_text())

    assert apply_result.applied_default_workflow == "action_transfer"
    assert apply_result.applied_override_count == 2
    assert updated_manifest["pixelle_default_workflow"] == "action_transfer"
    assert updated_manifest["pixelle_segment_overrides"] == {
        "seg_001#1": "digital_human",
        "seg_002#1": None,
    }
    assert updated_manifest["project_id"] == "test_project"
    assert updated_manifest["build_id"] == "fixture-build-1"
    assert updated_manifest["audio_path"] == "audio.wav"
    assert updated_manifest["segments"][0]["text"] == "first"
    assert updated_manifest["segments"][1]["text"] == "second"


@pytest.mark.workbench_apply_drift_conflict
def test_apply_pixelle_selections_blocks_manifest_drift(fixture_project):
    manifest_path = fixture_project / "build" / "manifest.json"
    manifest_path.write_text(json.dumps({
        "project_id": "test_project",
        "build_id": "build-v1",
        "global_style": {},
        "segments": [
            {
                "segment_key": "seg_001#1",
                "content_key": "seg_001",
                "index": 1,
                "start": 0.0,
                "end": 1.0,
                "duration": 1.0,
                "text": "first"
            }
        ]
    }, indent=2))

    paths = init_workbench(fixture_project)
    session = load_session(paths)
    session.pixelle_default_workflow = "digital_human"
    save_session(paths, session)

    drifted_manifest = json.loads(manifest_path.read_text())
    drifted_manifest["build_id"] = "build-v2"
    manifest_path.write_text(json.dumps(drifted_manifest, indent=2))

    with pytest.raises(ManifestDriftConflictError) as exc_info:
        apply_pixelle_selections(paths)

    conflict = exc_info.value
    assert conflict.expected_build_id == "build-v1"
    assert conflict.current_build_id == "build-v2"
    assert "Reload workbench session" in conflict.recovery_action

    unchanged_manifest = json.loads(manifest_path.read_text())
    assert unchanged_manifest.get("pixelle_default_workflow") is None
    assert unchanged_manifest.get("pixelle_segment_overrides") is None

def test_workbench_ui_select_save_reload(qapp, fixture_project):
    app = VideoAutomationApp()
    
    # Note: fixture_project has no segments by default, let's add one
    import json
    manifest_path = fixture_project / "build" / "manifest.json"
    manifest_data = json.loads(manifest_path.read_text())
    manifest_data["segments"] = [{"segment_key": "seg_001", "text": "first"}]
    manifest_path.write_text(json.dumps(manifest_data))
    
    # Load session
    app._load_wb_session(str(fixture_project))
    
    # Select global default
    idx = app._wb_global_combo.findData("digital_human")
    app._wb_global_combo.setCurrentIndex(idx)
    
    # Select segment override
    seg_combo = app._wb_segment_combos["seg_001"]
    idx = seg_combo.findData("i2v")
    seg_combo.setCurrentIndex(idx)
    
    # Save session
    app._save_wb_session()
    
    # Reload session
    paths = init_workbench(fixture_project)
    session = load_session(paths)
    
    assert session.pixelle_default_workflow == "digital_human"
    assert session.pixelle_segment_overrides == {"seg_001": "i2v"}

def test_workbench_ui_unavailable_workflow_blocked(qapp, fixture_project, monkeypatch):
    # Mock is_capability_available to return False for i2v
    import pixelle_snapshot.adapters
    original_is_available = pixelle_snapshot.adapters.is_capability_available
    
    def mock_is_available(name):
        if name == "i2v":
            return False
        return original_is_available(name)
        
    monkeypatch.setattr(pixelle_snapshot.adapters, "is_capability_available", mock_is_available)
    
    import json
    manifest_path = fixture_project / "build" / "manifest.json"
    manifest_data = json.loads(manifest_path.read_text())
    manifest_data["segments"] = [{"segment_key": "seg_001", "text": "first"}]
    manifest_path.write_text(json.dumps(manifest_data))
    
    app = VideoAutomationApp()
    app._load_wb_session(str(fixture_project))
    
    # Check global combo
    idx = app._wb_global_combo.findData("i2v")
    from PyQt5.QtGui import QStandardItemModel
    model = app._wb_global_combo.model()
    if isinstance(model, QStandardItemModel):
        item = model.item(idx)
        if item is not None:
            assert not item.isEnabled()
    assert "Unavailable" in app._wb_global_combo.itemText(idx)
    
    # Check segment combo
    seg_combo = app._wb_segment_combos["seg_001"]
    idx = seg_combo.findData("i2v")
    model = seg_combo.model()
    if isinstance(model, QStandardItemModel):
        item = model.item(idx)
        if item is not None:
            assert not item.isEnabled()
    assert "Unavailable" in seg_combo.itemText(idx)


def test_manifest_backward_compat_continuity(tmp_path):
    """
    T4 Scenario: Load old manifest and save with new continuity fields
    
    Preconditions: Fixture with legacy manifest/session schema exists (no continuity fields)
    Steps:
      1. Run: pytest tests -k "manifest_backward_compat_continuity" -q
      2. Assert load succeeds without KeyError/ValueError
      3. Assert saved output includes defaults for new fields
    Expected Result: Backward-compatible migration behavior
    """
    from src.core.models import Manifest
    
    legacy_manifest_path = tmp_path / "manifest.json"
    legacy_manifest_path.write_text(json.dumps({
        "project_id": "legacy_project",
        "build_id": "legacy-build-1",
        "global_style": {"subtitle_style": "clean"},
        "segments": [
            {
                "segment_key": "abc123#1",
                "content_key": "abc123",
                "index": 1,
                "start": 0.0,
                "end": 1.0,
                "duration": 1.0,
                "text": "Hello world"
            }
        ],
        "pixelle_default_workflow": "digital_human",
        "audio_path": "voice.wav"
    }, indent=2))
    
    manifest = Manifest.load(str(legacy_manifest_path))
    
    assert manifest.project_id == "legacy_project"
    assert manifest.build_id == "legacy-build-1"
    assert manifest.pixelle_default_workflow == "digital_human"
    assert manifest.style_id is None
    assert manifest.continuity_seed is None
    assert manifest.vendor_preference is None
    assert manifest.continuity_policy is None
    assert len(manifest.segments) == 1
    assert manifest.segments[0].prev_last_frame_path is None
    
    manifest.style_id = "style-bible-001"
    manifest.continuity_seed = 42
    manifest.vendor_preference = "minimax"
    manifest.continuity_policy = "frame_chain"
    manifest.segments[0].prev_last_frame_path = "/path/to/frame.png"
    
    new_path = tmp_path / "manifest_updated.json"
    manifest.save(str(new_path))
    
    reloaded = Manifest.load(str(new_path))
    assert reloaded.style_id == "style-bible-001"
    assert reloaded.continuity_seed == 42
    assert reloaded.vendor_preference == "minimax"
    assert reloaded.continuity_policy == "frame_chain"
    assert reloaded.segments[0].prev_last_frame_path == "/path/to/frame.png"
    assert reloaded.pixelle_default_workflow == "digital_human"
    assert reloaded.audio_path == "voice.wav"


def test_continuity_policy_invalid(tmp_path):
    """
    T4 Scenario: Reject invalid continuity policy mode
    
    Preconditions: Fixture uses unknown policy enum value
    Steps:
      1. Run: pytest tests -k "continuity_policy_invalid" -q
      2. Assert explicit validation error with allowed values
    Expected Result: Invalid mode rejected predictably
    """
    from src.core.models import Manifest, ContinuityPolicyError
    
    invalid_manifest_path = tmp_path / "manifest_invalid.json"
    invalid_manifest_path.write_text(json.dumps({
        "project_id": "invalid_project",
        "build_id": "invalid-build-1",
        "global_style": {},
        "segments": [],
        "continuity_policy": "unknown_policy"
    }, indent=2))
    
    with pytest.raises(ContinuityPolicyError) as exc_info:
        Manifest.load(str(invalid_manifest_path))
    
    error = exc_info.value
    assert error.value == "unknown_policy"
    assert "frame_chain" in error.allowed
    assert "seed_lock" in error.allowed
    assert "style_anchor" in error.allowed
    assert "off" in error.allowed
    assert "Invalid continuity policy" in str(error)
    assert "Allowed values" in str(error)


def test_session_continuity_fields_backward_compat():
    """Loading session without continuity fields uses safe defaults."""
    legacy_data = {
        "created_at": "2026-03-01T10:00:00",
        "last_modified": "2026-03-01T11:00:00",
        "draft_selections": {},
        "style_profile": {},
        "pixelle_default_workflow": "i2v"
    }
    
    session = WorkbenchSession.from_dict(legacy_data)
    
    assert session.style_id is None
    assert session.continuity_seed is None
    assert session.vendor_preference is None
    assert session.continuity_policy is None
    assert session.pixelle_default_workflow == "i2v"


def test_session_continuity_fields_roundtrip():
    """WorkbenchSession with continuity fields round-trips correctly."""
    original = WorkbenchSession(
        created_at="2026-03-15T10:00:00",
        last_modified="2026-03-15T11:00:00",
        draft_selections={},
        style_profile={},
        pixelle_default_workflow="digital_human",
        pixelle_segment_overrides={},
        style_id="style-bible-002",
        continuity_seed=12345,
        vendor_preference="minimax",
        continuity_policy="seed_lock",
    )
    
    data = original.to_dict()
    restored = WorkbenchSession.from_dict(data)
    
    assert restored.style_id == "style-bible-002"
    assert restored.continuity_seed == 12345
    assert restored.vendor_preference == "minimax"
    assert restored.continuity_policy == "seed_lock"


def test_session_invalid_continuity_policy_sanitized():
    """Invalid continuity policy values are sanitized to None."""
    corrupt_data = {
        "created_at": "2026-03-01T10:00:00",
        "last_modified": "2026-03-01T11:00:00",
        "draft_selections": {},
        "style_profile": {},
        "continuity_policy": "invalid_mode"
    }
    
    session = WorkbenchSession.from_dict(corrupt_data)
    assert session.continuity_policy is None


def test_session_invalid_continuity_seed_sanitized():
    """Non-integer continuity_seed values are sanitized to None."""
    corrupt_data = {
        "created_at": "2026-03-01T10:00:00",
        "last_modified": "2026-03-01T11:00:00",
        "draft_selections": {},
        "style_profile": {},
        "continuity_seed": "not_a_number"
    }
    
    session = WorkbenchSession.from_dict(corrupt_data)
    assert session.continuity_seed is None


def test_apply_preserves_continuity_fields(fixture_project):
    """Apply transfers continuity fields from session to manifest."""
    manifest_path = fixture_project / "build" / "manifest.json"
    manifest_path.write_text(json.dumps({
        "project_id": "test_project",
        "build_id": "fixture-build-1",
        "global_style": {},
        "segments": [
            {
                "segment_key": "seg_001#1",
                "content_key": "seg_001",
                "index": 1,
                "start": 0.0,
                "end": 1.0,
                "duration": 1.0,
                "text": "first"
            }
        ]
    }, indent=2))
    
    paths = init_workbench(fixture_project)
    session = load_session(paths)
    session.pixelle_default_workflow = "i2v"
    session.style_id = "test-style"
    session.continuity_seed = 99999
    session.vendor_preference = "minimax"
    session.continuity_policy = "style_anchor"
    save_session(paths, session)
    
    result = apply_pixelle_selections(paths)
    
    assert result.applied_style_id == "test-style"
    assert result.applied_continuity_seed == 99999
    assert result.applied_vendor_preference == "minimax"
    assert result.applied_continuity_policy == "style_anchor"
    
    updated_manifest = json.loads(manifest_path.read_text())
    assert updated_manifest["style_id"] == "test-style"
    assert updated_manifest["continuity_seed"] == 99999
    assert updated_manifest["vendor_preference"] == "minimax"
    assert updated_manifest["continuity_policy"] == "style_anchor"


def test_manifest_policy_valid(tmp_path):
    """
    Validates that manifest with valid workflow/policy fields deserializes correctly.
    
    Steps:
      1. Run: pytest tests/test_workbench_state.py -k manifest_policy_valid
      2. Assert all valid pixelle_workflows accepted
      3. Assert all valid continuity_policies accepted
      4. Assert all valid material_modes accepted
    Expected Result: Valid policy values deserialize and serialize without errors
    """
    from src.core.models import Manifest, PIXELLE_WORKFLOWS, CONTINUITY_POLICIES, MATERIAL_MODES
    
    valid_manifest_path = tmp_path / "manifest_valid.json"
    valid_manifest_path.write_text(json.dumps({
        "project_id": "valid_project",
        "build_id": "valid-build-1",
        "global_style": {},
        "segments": [],
        "pixelle_default_workflow": "digital_human",
        "pixelle_segment_overrides": {
            "seg_001": "i2v",
            "seg_002": "action_transfer",
            "seg_003": None
        },
        "continuity_policy": "frame_chain",
        "material_mode": "ai_preferred"
    }, indent=2))
    
    manifest = Manifest.load(str(valid_manifest_path))
    
    assert manifest.pixelle_default_workflow == "digital_human"
    assert manifest.pixelle_segment_overrides == {
        "seg_001": "i2v",
        "seg_002": "action_transfer",
        "seg_003": None
    }
    assert manifest.continuity_policy == "frame_chain"
    assert manifest.material_mode == "ai_preferred"
    
    roundtrip_path = tmp_path / "manifest_roundtrip.json"
    manifest.save(str(roundtrip_path))
    reloaded = Manifest.load(str(roundtrip_path))
    
    assert reloaded.pixelle_default_workflow == "digital_human"
    assert reloaded.pixelle_segment_overrides == manifest.pixelle_segment_overrides
    assert reloaded.continuity_policy == "frame_chain"
    assert reloaded.material_mode == "ai_preferred"


def test_invalid_workflow_default_raises_error(tmp_path):
    """
    Validates that invalid pixelle_default_workflow raises WorkflowPolicyError.
    
    Steps:
      1. Run: pytest tests/test_workbench_state.py -k invalid_workflow
      2. Assert WorkflowPolicyError raised with field and allowed values
    Expected Result: Invalid workflow rejected with clear error message
    """
    from src.core.models import Manifest, WorkflowPolicyError, PIXELLE_WORKFLOWS
    
    invalid_manifest_path = tmp_path / "manifest_invalid_workflow.json"
    invalid_manifest_path.write_text(json.dumps({
        "project_id": "invalid_project",
        "build_id": "invalid-build-1",
        "global_style": {},
        "segments": [],
        "pixelle_default_workflow": "bogus_workflow"
    }, indent=2))
    
    with pytest.raises(WorkflowPolicyError) as exc_info:
        Manifest.load(str(invalid_manifest_path))
    
    error = exc_info.value
    assert error.value == "bogus_workflow"
    assert error.field == "pixelle_default_workflow"
    assert "digital_human" in error.allowed
    assert "i2v" in error.allowed
    assert "action_transfer" in error.allowed
    assert "Invalid workflow" in str(error)
    assert "Allowed values" in str(error)


def test_invalid_workflow_override_raises_error(tmp_path):
    """
    Validates that invalid pixelle_segment_overrides value raises WorkflowPolicyError.
    
    Steps:
      1. Run: pytest tests/test_workbench_state.py -k invalid_workflow
      2. Assert WorkflowPolicyError raised with segment key in field name
    Expected Result: Invalid override rejected with clear error including segment key
    """
    from src.core.models import Manifest, WorkflowPolicyError
    
    invalid_manifest_path = tmp_path / "manifest_invalid_override.json"
    invalid_manifest_path.write_text(json.dumps({
        "project_id": "invalid_project",
        "build_id": "invalid-build-1",
        "global_style": {},
        "segments": [],
        "pixelle_default_workflow": "digital_human",
        "pixelle_segment_overrides": {
            "seg_001": "i2v",
            "seg_002": "not_a_real_workflow",
            "seg_003": "action_transfer"
        }
    }, indent=2))
    
    with pytest.raises(WorkflowPolicyError) as exc_info:
        Manifest.load(str(invalid_manifest_path))
    
    error = exc_info.value
    assert error.value == "not_a_real_workflow"
    assert "seg_002" in error.field
    assert "pixelle_segment_overrides" in error.field
    assert "Invalid workflow" in str(error)
