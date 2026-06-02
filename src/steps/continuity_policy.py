from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pixelle_snapshot.adapters.contracts import ErrorCategory, FailureDiagnostic
from pixelle_snapshot.vendors import capability_matrix_from_contract, load_vendor_contract_file

from src.core.frame_artifacts import extract_end_frame
from src.core.models import Segment


@dataclass(frozen=True)
class ContinuityDirective:
    continuity_mode: str
    requested_policy: str
    reason_code: str
    fallback_reason_code: Optional[str] = None
    seed: Optional[int] = None
    start_frame_path: Optional[str] = None
    source_segment_key: Optional[str] = None
    diagnostic: Optional[Dict[str, Any]] = None

    def request_metadata(self) -> Dict[str, Any]:
        return {
            "continuity_mode": self.continuity_mode,
            "requested_policy": self.requested_policy,
            "reason_code": self.reason_code,
            "fallback_reason_code": self.fallback_reason_code,
            "seed": self.seed,
            "start_frame_path": self.start_frame_path,
            "source_segment_key": self.source_segment_key,
        }


def evaluate_continuity_policy(
    segment: Segment,
    previous_segment: Optional[Segment],
    policy_mode: Optional[str],
    continuity_seed: Optional[int],
    style_id: Optional[str],
    project_id: str,
    vendor_preference: Optional[str],
    project_root: str,
    resolution: Tuple[int, int],
) -> ContinuityDirective:
    requested_policy = policy_mode or "off"
    vendor_id = (vendor_preference or "pixelle").strip().lower() or "pixelle"
    supports_end_frame, supports_seed = _resolve_vendor_capabilities(vendor_id)

    if requested_policy == "off":
        return _make_directive(
            continuity_mode="off",
            requested_policy=requested_policy,
            reason_code="PIXELLE_CONTINUITY_DISABLED",
            vendor_id=vendor_id,
            supports_end_frame=supports_end_frame,
            supports_seed=supports_seed,
        )

    if requested_policy == "style_anchor":
        return _make_directive(
            continuity_mode="style_anchor",
            requested_policy=requested_policy,
            reason_code="PIXELLE_CONTINUITY_STYLE_ANCHOR",
            vendor_id=vendor_id,
            supports_end_frame=supports_end_frame,
            supports_seed=supports_seed,
        )

    stable_seed = _resolve_stable_seed(
        continuity_seed=continuity_seed,
        project_id=project_id,
        style_id=style_id,
        vendor_id=vendor_id,
    )

    if requested_policy == "seed_lock":
        return _make_directive(
            continuity_mode="seed_lock",
            requested_policy=requested_policy,
            reason_code="PIXELLE_CONTINUITY_SEED_LOCKED",
            seed=stable_seed,
            vendor_id=vendor_id,
            supports_end_frame=supports_end_frame,
            supports_seed=supports_seed,
        )

    if requested_policy != "frame_chain":
        return _make_directive(
            continuity_mode="seed_lock",
            requested_policy=requested_policy,
            reason_code="PIXELLE_CONTINUITY_UNKNOWN_POLICY",
            fallback_reason_code="PIXELLE_CONTINUITY_UNKNOWN_POLICY",
            seed=stable_seed,
            vendor_id=vendor_id,
            supports_end_frame=supports_end_frame,
            supports_seed=supports_seed,
            category=ErrorCategory.VALIDATION,
        )

    if previous_segment is None:
        return _make_directive(
            continuity_mode="seed_lock",
            requested_policy=requested_policy,
            reason_code="PIXELLE_CONTINUITY_FIRST_SEGMENT",
            fallback_reason_code="PIXELLE_CONTINUITY_FIRST_SEGMENT",
            seed=stable_seed,
            vendor_id=vendor_id,
            supports_end_frame=supports_end_frame,
            supports_seed=supports_seed,
            category=ErrorCategory.VALIDATION,
        )

    if not supports_end_frame:
        return _make_directive(
            continuity_mode="seed_lock",
            requested_policy=requested_policy,
            reason_code="PIXELLE_CONTINUITY_TEMPORAL_UNSUPPORTED",
            fallback_reason_code="PIXELLE_CONTINUITY_TEMPORAL_UNSUPPORTED",
            seed=stable_seed,
            vendor_id=vendor_id,
            supports_end_frame=supports_end_frame,
            supports_seed=supports_seed,
            category=ErrorCategory.UNSUPPORTED,
        )

    previous_frame_path = _resolve_previous_frame(
        previous_segment=previous_segment,
        project_root=project_root,
        resolution=resolution,
    )
    if not previous_frame_path:
        return _make_directive(
            continuity_mode="seed_lock",
            requested_policy=requested_policy,
            reason_code="PIXELLE_CONTINUITY_PRIOR_ARTIFACT_MISSING",
            fallback_reason_code="PIXELLE_CONTINUITY_PRIOR_ARTIFACT_MISSING",
            seed=stable_seed,
            vendor_id=vendor_id,
            supports_end_frame=supports_end_frame,
            supports_seed=supports_seed,
            category=ErrorCategory.EXECUTION,
        )

    return _make_directive(
        continuity_mode="temporal",
        requested_policy=requested_policy,
        reason_code="PIXELLE_CONTINUITY_TEMPORAL_CHAINED",
        seed=stable_seed,
        start_frame_path=previous_frame_path,
        source_segment_key=previous_segment.segment_key,
        vendor_id=vendor_id,
        supports_end_frame=supports_end_frame,
        supports_seed=supports_seed,
    )


def _resolve_stable_seed(
    continuity_seed: Optional[int],
    project_id: str,
    style_id: Optional[str],
    vendor_id: str,
) -> int:
    if isinstance(continuity_seed, int):
        return continuity_seed
    raw = f"{project_id}|{style_id or ''}|{vendor_id}|continuity-seed"
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8], 16)


def _resolve_vendor_capabilities(vendor_id: str) -> Tuple[bool, bool]:
    if vendor_id in {"", "pixelle", "test"}:
        return True, True

    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "pixelle_snapshot"
        / "vendors"
        / "fixtures"
        / f"{vendor_id}_media_contract.json"
    )
    if not fixture_path.exists():
        return False, True

    try:
        contract = load_vendor_contract_file(fixture_path)
        matrix = capability_matrix_from_contract(contract)
        return matrix.supports_end_frame, matrix.supports_seed
    except Exception:
        return False, True


def _resolve_previous_frame(
    previous_segment: Segment,
    project_root: str,
    resolution: Tuple[int, int],
) -> Optional[str]:
    if previous_segment.prev_last_frame_path and Path(previous_segment.prev_last_frame_path).exists():
        return previous_segment.prev_last_frame_path

    if not previous_segment.asset_refs:
        return None

    previous_asset = previous_segment.asset_refs[0]
    if previous_asset.kind not in {"pixelle_video", "pexels_video", "cached"}:
        return None
    if not previous_asset.path:
        return None
    if not Path(previous_asset.path).exists():
        return None

    artifact_dir = str(Path(project_root) / "artifacts" / "continuity" / "frames")
    frame_path, frame_error = extract_end_frame(
        segment_key=previous_segment.segment_key,
        video_path=previous_asset.path,
        video_duration=previous_segment.duration,
        resolution=f"{resolution[0]}x{resolution[1]}",
        artifact_dir=artifact_dir,
    )
    if frame_error:
        return None
    return frame_path


def _make_directive(
    continuity_mode: str,
    requested_policy: str,
    reason_code: str,
    vendor_id: str,
    supports_end_frame: bool,
    supports_seed: bool,
    fallback_reason_code: Optional[str] = None,
    seed: Optional[int] = None,
    start_frame_path: Optional[str] = None,
    source_segment_key: Optional[str] = None,
    category: Optional[ErrorCategory] = None,
) -> ContinuityDirective:
    diagnostic: Dict[str, Any] = {
        "continuity_mode": continuity_mode,
        "requested_policy": requested_policy,
        "reason_code": reason_code,
        "fallback_reason_code": fallback_reason_code,
        "vendor_id": vendor_id,
        "supports_end_frame": supports_end_frame,
        "supports_seed": supports_seed,
        "seed": seed,
        "start_frame_path": start_frame_path,
        "source_segment_key": source_segment_key,
    }

    if fallback_reason_code and category is not None:
        diagnostic["fallback_diagnostic"] = FailureDiagnostic.from_error(
            category=category,
            reason_code=fallback_reason_code,
        ).to_dict()

    return ContinuityDirective(
        continuity_mode=continuity_mode,
        requested_policy=requested_policy,
        reason_code=reason_code,
        fallback_reason_code=fallback_reason_code,
        seed=seed,
        start_frame_path=start_frame_path,
        source_segment_key=source_segment_key,
        diagnostic=diagnostic,
    )
