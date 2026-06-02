"""
Workbench state layout and initialization contract.

Defines the filesystem structure under `project/workbench/` and provides
initialization/loading utilities for workbench draft state independent of
the formal `build/manifest.json`.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Literal, Set
import json
from datetime import datetime


CONTINUITY_POLICIES: Set[str] = {"frame_chain", "seed_lock", "style_anchor", "off"}


@dataclass
class WorkbenchPaths:
    """
    Centralized workbench path contract for project-local state.
    
    All workbench draft state lives under `project/workbench/`:
    - session.json: draft segment selections, recommendations, style profile
    - library/: reusable asset metadata and files
    - thumbnails/: generated preview thumbnails for library assets
    - cache/: workbench-specific caches (recommendation results, etc.)
    """
    project_root: Path
    
    @property
    def workbench_dir(self) -> Path:
        """Root workbench directory: project/workbench/"""
        return self.project_root / "workbench"
    
    @property
    def session_json(self) -> Path:
        """Draft workbench session state"""
        return self.workbench_dir / "session.json"

    @property
    def manifest_json(self) -> Path:
        return self.project_root / "build" / "manifest.json"
    
    @property
    def library_dir(self) -> Path:
        """Asset library storage: uploaded + AI-generated assets"""
        return self.workbench_dir / "library"
    
    @property
    def library_metadata_json(self) -> Path:
        """Library asset metadata index"""
        return self.library_dir / "index.json"
    
    @property
    def thumbnails_dir(self) -> Path:
        """Thumbnail cache for library assets"""
        return self.workbench_dir / "thumbnails"
    
    @property
    def cache_dir(self) -> Path:
        """Workbench-specific cache (recommendations, etc.)"""
        return self.workbench_dir / "cache"
    
    def ensure_dirs(self) -> None:
        """Create all necessary workbench directories."""
        dirs = [
            self.workbench_dir,
            self.library_dir,
            self.thumbnails_dir,
            self.cache_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class WorkbenchSession:
    created_at: str = ""
    last_modified: str = ""
    manifest_build_id: str = ""
    
    draft_selections: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    style_profile: Dict[str, Any] = field(default_factory=dict)
    
    template_preset: Optional[str] = None
    
    recommendation_cache: Dict[str, Any] = field(default_factory=dict)
    
    pixelle_default_workflow: Optional[str] = None
    
    pixelle_segment_overrides: Dict[str, Optional[str]] = field(default_factory=dict)
    
    style_id: Optional[str] = None
    continuity_seed: Optional[int] = None
    vendor_preference: Optional[str] = None
    continuity_policy: Optional[Literal["frame_chain", "seed_lock", "style_anchor", "off"]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkbenchSession:
        pixelle_default = data.get("pixelle_default_workflow")
        if pixelle_default and pixelle_default not in {"digital_human", "i2v", "action_transfer"}:
            pixelle_default = None
        
        pixelle_overrides = data.get("pixelle_segment_overrides", {})
        valid_overrides = {}
        for key, value in pixelle_overrides.items():
            if value is None or value in {"digital_human", "i2v", "action_transfer"}:
                valid_overrides[key] = value
        
        continuity_policy = data.get("continuity_policy")
        if continuity_policy is not None and continuity_policy not in CONTINUITY_POLICIES:
            continuity_policy = None
        
        continuity_seed = data.get("continuity_seed")
        if continuity_seed is not None and not isinstance(continuity_seed, int):
            continuity_seed = None
        
        return cls(
            created_at=data.get("created_at", ""),
            last_modified=data.get("last_modified", ""),
            manifest_build_id=data.get("manifest_build_id", ""),
            draft_selections=data.get("draft_selections", {}),
            style_profile=data.get("style_profile", {}),
            template_preset=data.get("template_preset"),
            recommendation_cache=data.get("recommendation_cache", {}),
            pixelle_default_workflow=pixelle_default,
            pixelle_segment_overrides=valid_overrides,
            style_id=data.get("style_id"),
            continuity_seed=continuity_seed,
            vendor_preference=data.get("vendor_preference"),
            continuity_policy=continuity_policy,
        )


def init_workbench(project_root: str | Path) -> WorkbenchPaths:
    """
    Initialize workbench directories for a project.
    
    Creates the workbench directory structure if it doesn't exist.
    Does NOT mutate build/manifest.json or any formal project state.
    
    Args:
        project_root: Path to project root directory
        
    Returns:
        WorkbenchPaths instance with all paths configured
        
    Raises:
        ValueError: If project_root does not exist or is not a directory
    """
    project_path = Path(project_root)
    if not project_path.exists():
        raise ValueError(f"Project root does not exist: {project_root}")
    if not project_path.is_dir():
        raise ValueError(f"Project root is not a directory: {project_root}")
    
    paths = WorkbenchPaths(project_root=project_path)
    paths.ensure_dirs()
    
    # Initialize empty session if it doesn't exist
    if not paths.session_json.exists():
        session = WorkbenchSession(
            created_at=datetime.now().isoformat(),
            last_modified=datetime.now().isoformat(),
            manifest_build_id=_read_manifest_build_id(paths.manifest_json),
        )
        save_session(paths, session)
    
    # Initialize empty library metadata if it doesn't exist
    if not paths.library_metadata_json.exists():
        paths.library_metadata_json.write_text(json.dumps({"assets": []}, indent=2))
    
    return paths


def load_session(paths: WorkbenchPaths) -> WorkbenchSession:
    """
    Load workbench session from session.json.
    
    Args:
        paths: WorkbenchPaths instance
        
    Returns:
        WorkbenchSession instance
        
    Raises:
        FileNotFoundError: If session.json does not exist
        json.JSONDecodeError: If session.json is malformed
    """
    if not paths.session_json.exists():
        raise FileNotFoundError(f"Session file not found: {paths.session_json}")
    
    data = json.loads(paths.session_json.read_text())
    return WorkbenchSession.from_dict(data)


def save_session(paths: WorkbenchPaths, session: WorkbenchSession) -> None:
    """
    Save workbench session to session.json.
    
    Updates last_modified timestamp automatically.
    Does NOT modify build/manifest.json.
    
    Args:
        paths: WorkbenchPaths instance
        session: WorkbenchSession to save
    """
    session.last_modified = datetime.now().isoformat()
    paths.session_json.write_text(json.dumps(session.to_dict(), indent=2))


def _read_manifest_build_id(manifest_path: Path) -> str:
    if not manifest_path.exists():
        return ""

    try:
        manifest_data = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return ""

    build_id = manifest_data.get("build_id")
    return build_id if isinstance(build_id, str) else ""
