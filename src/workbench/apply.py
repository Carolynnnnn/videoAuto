from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.core.models import Manifest
from src.workbench.state import WorkbenchPaths, load_session, save_session


class ManifestDriftConflictError(RuntimeError):
    def __init__(self, *, expected_build_id: str, current_build_id: str):
        self.expected_build_id = expected_build_id
        self.current_build_id = current_build_id
        self.recovery_action = "Reload workbench session from latest manifest and re-apply selections."
        super().__init__(
            "Apply blocked: manifest changed since this draft session was created. "
            "Reload the workbench session to sync with the latest manifest before applying again."
        )


@dataclass
class ApplyResult:
    manifest_path: str
    applied_default_workflow: str | None
    applied_override_count: int
    applied_style_id: Optional[str] = None
    applied_continuity_seed: Optional[int] = None
    applied_vendor_preference: Optional[str] = None
    applied_continuity_policy: Optional[str] = None


def apply_pixelle_selections(paths: WorkbenchPaths) -> ApplyResult:
    session = load_session(paths)
    manifest_path = paths.manifest_json
    manifest = Manifest.load(str(manifest_path))

    if session.manifest_build_id and session.manifest_build_id != manifest.build_id:
        raise ManifestDriftConflictError(
            expected_build_id=session.manifest_build_id,
            current_build_id=manifest.build_id,
        )

    valid_segment_keys = {segment.segment_key for segment in manifest.segments}
    applied_overrides = {
        key: value
        for key, value in session.pixelle_segment_overrides.items()
        if key in valid_segment_keys
    }

    manifest.pixelle_default_workflow = session.pixelle_default_workflow
    manifest.pixelle_segment_overrides = applied_overrides
    manifest.style_id = session.style_id
    manifest.continuity_seed = session.continuity_seed
    manifest.vendor_preference = session.vendor_preference
    manifest.continuity_policy = session.continuity_policy
    manifest.save(str(manifest_path))

    session.manifest_build_id = manifest.build_id
    save_session(paths, session)

    return ApplyResult(
        manifest_path=str(manifest_path),
        applied_default_workflow=manifest.pixelle_default_workflow,
        applied_override_count=len(applied_overrides),
        applied_style_id=manifest.style_id,
        applied_continuity_seed=manifest.continuity_seed,
        applied_vendor_preference=manifest.vendor_preference,
        applied_continuity_policy=manifest.continuity_policy,
    )
