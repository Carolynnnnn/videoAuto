"""
Continuity Telemetry, Quality Scoring, and Diagnostics Module

This module provides:
1. ContinuityTelemetryRecord - Per-segment telemetry capture (mode, source_frame_hash, seed, vendor, fallback_reason)
2. QualityScorer - Automated scoring hooks (style_similarity, continuity_confidence, artifact_linkage)
3. ContinuityQualitySummary - Aggregated metrics for post-run audits
4. StrictModeValidator - Hard-gate validation for strict continuity enforcement

Design Principles:
- Telemetry records are per-segment and persisted via segment.continuity_diagnostic
- Quality scoring is deterministic for fixed fixtures
- Strict-mode rejects on any hard-gate violation
- Diagnostics support post-run audits without modifying happy-path fallback fields
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.core.models import Segment


# ─────────────────────────────────────────────
# Strict Mode Gate Thresholds
# ─────────────────────────────────────────────
STRICT_TEMPORAL_LINK_COVERAGE_THRESHOLD = 95.0  # percentage
STRICT_STYLE_SIMILARITY_P50_THRESHOLD = 0.85
STRICT_FALLBACK_COVERAGE_THRESHOLD = 100.0
STRICT_ORPHAN_ARTIFACT_COUNT_THRESHOLD = 0  # must be zero

PREVIEW_TEMPORAL_LINK_COVERAGE_THRESHOLD = 80.0  # percentage
PREVIEW_STYLE_SIMILARITY_P50_THRESHOLD = 0.80
PREVIEW_FALLBACK_COVERAGE_THRESHOLD = 100.0
PREVIEW_ORPHAN_ARTIFACT_COUNT_THRESHOLD = 0  # must be zero


@dataclass(frozen=True)
class StrictGateThresholds:
    temporal_link_coverage_min: float
    style_similarity_p50_min: float
    fallback_coverage_min: float
    orphan_artifact_count_max: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "temporal_link_coverage_min": self.temporal_link_coverage_min,
            "style_similarity_p50_min": self.style_similarity_p50_min,
            "fallback_coverage_min": self.fallback_coverage_min,
            "orphan_artifact_count_max": float(self.orphan_artifact_count_max),
        }


STRICT_GATE_THRESHOLDS_BY_PROFILE: Dict[str, StrictGateThresholds] = {
    "preview": StrictGateThresholds(
        temporal_link_coverage_min=PREVIEW_TEMPORAL_LINK_COVERAGE_THRESHOLD,
        style_similarity_p50_min=PREVIEW_STYLE_SIMILARITY_P50_THRESHOLD,
        fallback_coverage_min=PREVIEW_FALLBACK_COVERAGE_THRESHOLD,
        orphan_artifact_count_max=PREVIEW_ORPHAN_ARTIFACT_COUNT_THRESHOLD,
    ),
    "release": StrictGateThresholds(
        temporal_link_coverage_min=STRICT_TEMPORAL_LINK_COVERAGE_THRESHOLD,
        style_similarity_p50_min=STRICT_STYLE_SIMILARITY_P50_THRESHOLD,
        fallback_coverage_min=STRICT_FALLBACK_COVERAGE_THRESHOLD,
        orphan_artifact_count_max=STRICT_ORPHAN_ARTIFACT_COUNT_THRESHOLD,
    ),
}


def get_strict_gate_thresholds(profile: str = "release") -> StrictGateThresholds:
    normalized_profile = (profile or "release").strip().lower()
    if normalized_profile not in STRICT_GATE_THRESHOLDS_BY_PROFILE:
        supported_profiles = ", ".join(sorted(STRICT_GATE_THRESHOLDS_BY_PROFILE.keys()))
        raise ValueError(
            f"Unsupported strict gate profile '{profile}'. Supported profiles: {supported_profiles}"
        )
    return STRICT_GATE_THRESHOLDS_BY_PROFILE[normalized_profile]


class StrictGateViolation(str, Enum):
    """Types of strict-mode gate violations."""
    TEMPORAL_LINK_COVERAGE_LOW = "TEMPORAL_LINK_COVERAGE_LOW"
    STYLE_SIMILARITY_LOW = "STYLE_SIMILARITY_LOW"
    FALLBACK_COVERAGE_LOW = "FALLBACK_COVERAGE_LOW"
    ORPHAN_ARTIFACTS_PRESENT = "ORPHAN_ARTIFACTS_PRESENT"


@dataclass
class GateViolationDetail:
    """Detail of a single gate violation."""
    gate: StrictGateViolation
    actual_value: float
    threshold: float
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate": self.gate.value,
            "actual_value": self.actual_value,
            "threshold": self.threshold,
            "message": self.message,
        }


# ─────────────────────────────────────────────
# Per-Segment Telemetry Record
# ─────────────────────────────────────────────
@dataclass
class ContinuityTelemetryRecord:
    """
    Per-segment telemetry capture for continuity tracking.
    
    Fields:
    - segment_key: Segment identifier
    - continuity_mode: Selected mode (temporal, seed_lock, style_anchor, off)
    - source_frame_hash: Hash of source frame (if temporal mode)
    - seed: Continuity seed value
    - vendor: Vendor used for generation
    - fallback_reason: Reason code if fallback occurred
    - style_similarity: Computed style similarity score [0.0, 1.0]
    - continuity_confidence: Confidence in temporal linkage [0.0, 1.0]
    - artifact_linked: Whether artifact chain is intact
    """
    segment_key: str
    continuity_mode: str
    source_frame_hash: Optional[str] = None
    seed: Optional[int] = None
    vendor: Optional[str] = None
    fallback_reason: Optional[str] = None
    
    # Quality scoring fields
    style_similarity: float = 1.0  # Default: perfect similarity
    continuity_confidence: float = 1.0  # Default: perfect confidence
    artifact_linked: bool = True  # Default: properly linked

    @classmethod
    def from_segment(cls, segment: Segment) -> "ContinuityTelemetryRecord":
        """
        Extract telemetry from segment's continuity_diagnostic.
        
        Gracefully handles missing or partial diagnostics.
        """
        diag = segment.continuity_diagnostic or {}
        
        # Extract source frame hash if available
        source_frame_path = diag.get("start_frame_path")
        source_frame_hash = None
        if source_frame_path:
            source_frame_hash = _compute_frame_hash(source_frame_path)
        
        # Determine artifact linkage status
        has_fallback = diag.get("fallback_reason_code") is not None
        artifact_linked = not has_fallback and diag.get("continuity_mode") in ("temporal", "seed_lock", "style_anchor")
        
        return cls(
            segment_key=segment.segment_key,
            continuity_mode=diag.get("continuity_mode", "off"),
            source_frame_hash=source_frame_hash,
            seed=diag.get("seed"),
            vendor=diag.get("vendor_id"),
            fallback_reason=diag.get("fallback_reason_code"),
            style_similarity=_compute_style_similarity(segment, diag),
            continuity_confidence=_compute_continuity_confidence(diag),
            artifact_linked=artifact_linked,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_key": self.segment_key,
            "continuity_mode": self.continuity_mode,
            "source_frame_hash": self.source_frame_hash,
            "seed": self.seed,
            "vendor": self.vendor,
            "fallback_reason": self.fallback_reason,
            "style_similarity": self.style_similarity,
            "continuity_confidence": self.continuity_confidence,
            "artifact_linked": self.artifact_linked,
        }


def _compute_frame_hash(frame_path: str) -> str:
    """Compute deterministic hash for frame path (not content for performance)."""
    return hashlib.sha256(frame_path.encode("utf-8")).hexdigest()[:16]


def _compute_style_similarity(segment: Segment, diagnostic: Dict[str, Any]) -> float:
    """
    Compute style similarity score for segment.
    
    Scoring rules (deterministic for fixed fixtures):
    - temporal mode: 0.95 (high consistency from frame reference)
    - seed_lock mode: 0.90 (good consistency from seed)
    - style_anchor mode: 0.85 (moderate consistency from style only)
    - off mode: 0.70 (baseline consistency)
    - with fallback: penalty of -0.05
    """
    mode = diagnostic.get("continuity_mode", "off")
    base_scores = {
        "temporal": 0.95,
        "seed_lock": 0.90,
        "style_anchor": 0.85,
        "off": 0.70,
    }
    score = base_scores.get(mode, 0.70)
    
    # Apply fallback penalty
    if diagnostic.get("fallback_reason_code"):
        score -= 0.05
    
    return max(0.0, min(1.0, score))


def _compute_continuity_confidence(diagnostic: Dict[str, Any]) -> float:
    """
    Compute continuity confidence score.
    
    Scoring rules:
    - temporal with source frame: 1.0
    - temporal without frame: 0.8
    - seed_lock: 0.85
    - style_anchor: 0.75
    - off: 0.50
    - with fallback: penalty of -0.10
    """
    mode = diagnostic.get("continuity_mode", "off")
    has_frame = diagnostic.get("start_frame_path") is not None
    
    if mode == "temporal":
        score = 1.0 if has_frame else 0.8
    elif mode == "seed_lock":
        score = 0.85
    elif mode == "style_anchor":
        score = 0.75
    else:
        score = 0.50
    
    # Apply fallback penalty
    if diagnostic.get("fallback_reason_code"):
        score -= 0.10
    
    return max(0.0, min(1.0, score))


# ─────────────────────────────────────────────
# Quality Summary (Aggregated Metrics)
# ─────────────────────────────────────────────
@dataclass
class ContinuityQualitySummary:
    """
    Aggregated quality metrics for post-run audits.
    
    Required summary fields:
    - temporal_link_coverage: Percentage of segments with temporal linkage
    - fallback_coverage: Percentage of ineligible frame-chain segments with explicit fallback diagnostics
    - style_similarity_p50: Median style similarity score
    - orphan_artifact_count: Count of segments with broken artifact links
    
    Additional fields:
    - total_segments: Total segment count
    - temporal_segments: Count with temporal mode
    - fallback_segments: Count with fallback reason
    - style_similarities: List of all style scores (for percentile calculation)
    """
    total_segments: int = 0
    temporal_segments: int = 0
    fallback_segments: int = 0
    frame_chain_ineligible_segments: int = 0
    diagnosed_frame_chain_fallback_segments: int = 0
    orphan_artifact_count: int = 0
    
    style_similarities: List[float] = field(default_factory=list)
    continuity_confidences: List[float] = field(default_factory=list)
    
    telemetry_records: List[ContinuityTelemetryRecord] = field(default_factory=list)

    @property
    def temporal_link_coverage(self) -> float:
        """Percentage of segments with temporal linkage (0-100)."""
        if self.total_segments == 0:
            return 100.0
        return (self.temporal_segments / self.total_segments) * 100.0

    @property
    def fallback_coverage(self) -> float:
        """
        Percentage of ineligible frame-chain segments with fallback diagnostics.

        Ineligible frame-chain segments are those with requested_policy="frame_chain"
        where continuity_mode is not "temporal" (for example first-segment fallback,
        unsupported temporal chaining, or missing prior artifacts).

        100% means every ineligible frame-chain segment has an explicit
        fallback_reason_code diagnostic.
        """
        if self.frame_chain_ineligible_segments == 0:
            return 100.0
        return (
            self.diagnosed_frame_chain_fallback_segments
            / self.frame_chain_ineligible_segments
        ) * 100.0

    @property
    def style_similarity_p50(self) -> float:
        """Median style similarity score (50th percentile)."""
        if not self.style_similarities:
            return 1.0
        sorted_scores = sorted(self.style_similarities)
        mid = len(sorted_scores) // 2
        if len(sorted_scores) % 2 == 0:
            return (sorted_scores[mid - 1] + sorted_scores[mid]) / 2
        return sorted_scores[mid]

    @property
    def continuity_confidence_p50(self) -> float:
        """Median continuity confidence score (50th percentile)."""
        if not self.continuity_confidences:
            return 1.0
        sorted_scores = sorted(self.continuity_confidences)
        mid = len(sorted_scores) // 2
        if len(sorted_scores) % 2 == 0:
            return (sorted_scores[mid - 1] + sorted_scores[mid]) / 2
        return sorted_scores[mid]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temporal_link_coverage": round(self.temporal_link_coverage, 2),
            "fallback_coverage": round(self.fallback_coverage, 2),
            "style_similarity_p50": round(self.style_similarity_p50, 4),
            "orphan_artifact_count": self.orphan_artifact_count,
            "total_segments": self.total_segments,
            "temporal_segments": self.temporal_segments,
            "fallback_segments": self.fallback_segments,
            "continuity_confidence_p50": round(self.continuity_confidence_p50, 4),
        }


# ─────────────────────────────────────────────
# Quality Scorer
# ─────────────────────────────────────────────
def compute_quality_summary(segments: List[Segment]) -> ContinuityQualitySummary:
    """
    Compute aggregated quality summary from segment list.
    
    Processes each segment's continuity_diagnostic to build:
    - Telemetry records for each segment
    - Aggregated quality metrics
    - Frame-chain fallback diagnostic coverage for strict gate G2
    """
    summary = ContinuityQualitySummary()
    summary.total_segments = len(segments)
    
    for segment in segments:
        record = ContinuityTelemetryRecord.from_segment(segment)
        summary.telemetry_records.append(record)
        
        # Count temporal segments
        if record.continuity_mode == "temporal":
            summary.temporal_segments += 1
        
        # Count fallback segments
        if record.fallback_reason is not None:
            summary.fallback_segments += 1

        diagnostic = segment.continuity_diagnostic or {}
        if diagnostic.get("requested_policy") == "frame_chain" and record.continuity_mode != "temporal":
            summary.frame_chain_ineligible_segments += 1
            if record.fallback_reason is not None:
                summary.diagnosed_frame_chain_fallback_segments += 1
        
        # Count orphan artifacts
        if not record.artifact_linked:
            summary.orphan_artifact_count += 1
        
        # Collect scores for percentile calculation
        summary.style_similarities.append(record.style_similarity)
        summary.continuity_confidences.append(record.continuity_confidence)
    
    return summary


# ─────────────────────────────────────────────
# Strict Mode Validation
# ─────────────────────────────────────────────
@dataclass
class StrictModeValidationResult:
    """Result of strict-mode gate validation."""
    passed: bool
    profile: str = "release"
    thresholds: StrictGateThresholds = field(default_factory=get_strict_gate_thresholds)
    violations: List[GateViolationDetail] = field(default_factory=list)
    summary: Optional[ContinuityQualitySummary] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "profile": self.profile,
            "thresholds": self.thresholds.to_dict(),
            "violations": [v.to_dict() for v in self.violations],
            "summary": self.summary.to_dict() if self.summary else None,
        }


class ContinuityQualityError(Exception):
    """Raised when strict-mode validation fails."""
    def __init__(self, result: StrictModeValidationResult):
        self.result = result
        violations_str = ", ".join(v.gate.value for v in result.violations)
        super().__init__(f"Strict continuity validation failed: {violations_str}")


def is_strict_continuity_mode_enabled() -> bool:
    raw = os.environ.get("PIXELLE_CONTINUITY_STRICT_MODE", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def format_strict_mode_failure(result: StrictModeValidationResult) -> str:
    summary = result.summary or ContinuityQualitySummary()
    thresholds = result.thresholds
    profile = result.profile
    if profile == "release":
        header = "Strict continuity gate failed (release blocked)."
    else:
        header = f"Strict continuity gate failed ({profile} mode - warnings only)."
    lines = [
        header,
        (
            "Gate profile context: "
            f"active_profile={result.profile}, "
            f"threshold_temporal_link_coverage_min={thresholds.temporal_link_coverage_min:.4f}, "
            f"threshold_style_similarity_p50_min={thresholds.style_similarity_p50_min:.4f}, "
            f"threshold_fallback_coverage_min={thresholds.fallback_coverage_min:.4f}, "
            f"threshold_orphan_artifact_count_max={thresholds.orphan_artifact_count_max}"
        ),
        (
            "Summary metrics: "
            f"temporal_link_coverage={summary.temporal_link_coverage:.2f}%, "
            f"fallback_coverage={summary.fallback_coverage:.2f}%, "
            f"style_similarity_p50={summary.style_similarity_p50:.4f}, "
            f"orphan_artifact_count={summary.orphan_artifact_count}"
        ),
        "Violations:",
    ]
    for violation in result.violations:
        lines.append(
            "- "
            f"{violation.gate.value}: actual={violation.actual_value:.4f}, "
            f"threshold={violation.threshold:.4f}, detail={violation.message}"
        )
    return "\n".join(lines)


def validate_strict_mode(
    summary: ContinuityQualitySummary,
    profile: str = "release",
    raise_on_violation: bool = False,
) -> StrictModeValidationResult:
    """
    Validate quality summary against strict-mode thresholds.
    
    Hard-gate rules (all must pass):
    - temporal_link_coverage >= 95%
    - style_similarity_p50 >= 0.85
    - fallback_coverage == 100% (all ineligible frame-chain pairs diagnosed)
    - orphan_artifact_count == 0
    
    Args:
        summary: Quality summary to validate
        raise_on_violation: If True, raise ContinuityQualityError on failure
        
    Returns:
        StrictModeValidationResult with pass/fail status and violation details
    """
    violations: List[GateViolationDetail] = []
    normalized_profile = (profile or "release").strip().lower()
    thresholds = get_strict_gate_thresholds(normalized_profile)
    
    # Gate 1: Temporal link coverage
    if summary.temporal_link_coverage < thresholds.temporal_link_coverage_min:
        violations.append(GateViolationDetail(
            gate=StrictGateViolation.TEMPORAL_LINK_COVERAGE_LOW,
            actual_value=summary.temporal_link_coverage,
            threshold=thresholds.temporal_link_coverage_min,
            message=(
                f"Temporal link coverage {summary.temporal_link_coverage:.1f}% "
                f"< {thresholds.temporal_link_coverage_min}%"
            ),
        ))
    
    # Gate 2: Style similarity P50
    if summary.style_similarity_p50 < thresholds.style_similarity_p50_min:
        violations.append(GateViolationDetail(
            gate=StrictGateViolation.STYLE_SIMILARITY_LOW,
            actual_value=summary.style_similarity_p50,
            threshold=thresholds.style_similarity_p50_min,
            message=(
                f"Style similarity P50 {summary.style_similarity_p50:.2f} "
                f"< {thresholds.style_similarity_p50_min}"
            ),
        ))
    
    if summary.fallback_coverage < thresholds.fallback_coverage_min:
        violations.append(GateViolationDetail(
            gate=StrictGateViolation.FALLBACK_COVERAGE_LOW,
            actual_value=summary.fallback_coverage,
            threshold=thresholds.fallback_coverage_min,
            message=(
                "Fallback diagnostic coverage "
                f"{summary.fallback_coverage:.1f}% < {thresholds.fallback_coverage_min}% "
                "(missing fallback diagnostics for ineligible frame-chain segments)"
            ),
        ))
    
    # Gate 4: Orphan artifact count (must be 0)
    if summary.orphan_artifact_count > thresholds.orphan_artifact_count_max:
        violations.append(GateViolationDetail(
            gate=StrictGateViolation.ORPHAN_ARTIFACTS_PRESENT,
            actual_value=float(summary.orphan_artifact_count),
            threshold=float(thresholds.orphan_artifact_count_max),
            message=(
                f"Orphan artifacts: {summary.orphan_artifact_count} "
                f"> {thresholds.orphan_artifact_count_max}"
            ),
        ))
    
    result = StrictModeValidationResult(
        passed=len(violations) == 0,
        profile=normalized_profile,
        thresholds=thresholds,
        violations=violations,
        summary=summary,
    )
    
    if raise_on_violation and not result.passed:
        raise ContinuityQualityError(result)
    
    return result


# ─────────────────────────────────────────────
# Diagnostic Persistence
# ─────────────────────────────────────────────
def persist_segment_telemetry(
    segment: Segment,
    telemetry: ContinuityTelemetryRecord,
) -> None:
    """
    Persist telemetry record to segment's continuity_diagnostic.
    
    Merges telemetry fields into existing diagnostic without overwriting
    core continuity policy fields.
    """
    if segment.continuity_diagnostic is None:
        segment.continuity_diagnostic = {}
    
    # Add telemetry fields under 'telemetry' key to avoid collision
    segment.continuity_diagnostic["telemetry"] = telemetry.to_dict()


def extract_all_telemetry(segments: List[Segment]) -> List[Dict[str, Any]]:
    """
    Extract telemetry from all segments for post-run audit export.
    
    Returns list of telemetry dictionaries including segment diagnostics.
    """
    result = []
    for segment in segments:
        record = ContinuityTelemetryRecord.from_segment(segment)
        entry = record.to_dict()
        entry["continuity_diagnostic"] = segment.continuity_diagnostic
        result.append(entry)
    return result
