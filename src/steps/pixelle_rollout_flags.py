"""
Pixelle rollout flags: canary, allowlist, and shadow mode controls.

Feature Flag System:
- enable_pixelle: Global kill switch for Pixelle provider
- rollout_percentage: Canary rollout (0-100%) with deterministic bucketing
- allowlist_keys: Explicit segment keys that always route to Pixelle
- shadow_mode: Execute Pixelle calls but don't use result in final output

Deterministic Bucketing:
- Same segment_key always gets same bucket assignment
- Uses SHA256 hash modulo 100 for consistent routing
- Allows safe percentage ramps without traffic flip-flopping
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Set

from src.utils.logger import get_logger

logger = get_logger("pixelle_rollout")


@dataclass(frozen=True)
class RolloutConfig:
    """Configuration for Pixelle rollout flags."""
    
    # Global enable flag (kill switch)
    enable_pixelle: bool = True
    
    # Canary rollout percentage (0-100)
    rollout_percentage: int = 100
    
    # Explicit allowlist - segment keys that always route to Pixelle
    allowlist_keys: FrozenSet[str] = field(default_factory=frozenset)
    
    # Shadow mode: execute but don't use results
    shadow_mode: bool = False
    
    @classmethod
    def from_env(cls) -> "RolloutConfig":
        """Load rollout config from environment variables."""
        enable = os.environ.get("PIXELLE_ENABLED", "1").lower() in ("1", "true", "yes")
        
        rollout_pct = _read_int_env("PIXELLE_ROLLOUT_PERCENTAGE", 100, minimum=0, maximum=100)
        
        allowlist_raw = os.environ.get("PIXELLE_ALLOWLIST_KEYS", "")
        allowlist = frozenset(
            k.strip() for k in allowlist_raw.split(",") if k.strip()
        )
        
        shadow = os.environ.get("PIXELLE_SHADOW_MODE", "0").lower() in ("1", "true", "yes")
        
        return cls(
            enable_pixelle=enable,
            rollout_percentage=rollout_pct,
            allowlist_keys=allowlist,
            shadow_mode=shadow,
        )
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RolloutConfig":
        """Load rollout config from dict (e.g., manifest or config file)."""
        allowlist_raw = d.get("allowlist_keys", [])
        if isinstance(allowlist_raw, str):
            allowlist = frozenset(k.strip() for k in allowlist_raw.split(",") if k.strip())
        elif isinstance(allowlist_raw, (list, tuple, set, frozenset)):
            allowlist = frozenset(str(k) for k in allowlist_raw)
        else:
            allowlist = frozenset()
        
        return cls(
            enable_pixelle=bool(d.get("enable_pixelle", True)),
            rollout_percentage=max(0, min(100, int(d.get("rollout_percentage", 100)))),
            allowlist_keys=allowlist,
            shadow_mode=bool(d.get("shadow_mode", False)),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "enable_pixelle": self.enable_pixelle,
            "rollout_percentage": self.rollout_percentage,
            "allowlist_keys": sorted(self.allowlist_keys),
            "shadow_mode": self.shadow_mode,
        }


def compute_bucket(segment_key: str) -> int:
    """
    Compute deterministic bucket (0-99) for a segment key.
    
    Uses SHA256 hash to ensure:
    - Same segment_key always returns same bucket
    - Uniform distribution across buckets
    - No flip-flopping when ramping rollout percentage
    """
    hash_bytes = hashlib.sha256(segment_key.encode("utf-8")).digest()
    # Use first 4 bytes as unsigned int, then modulo 100
    bucket_value = int.from_bytes(hash_bytes[:4], byteorder="big") % 100
    return bucket_value


@dataclass
class RolloutDecision:
    """Result of a rollout eligibility check."""
    
    eligible: bool
    reason: Literal[
        "disabled",           # enable_pixelle=False
        "allowlisted",        # segment in allowlist
        "canary_in_bucket",   # within rollout percentage
        "canary_out_bucket",  # outside rollout percentage
        "shadow_mode",        # shadow mode active
    ]
    bucket: int = 0
    shadow_execution: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "eligible": self.eligible,
            "reason": self.reason,
            "bucket": self.bucket,
            "shadow_execution": self.shadow_execution,
        }


def check_rollout_eligibility(
    segment_key: str,
    config: RolloutConfig,
) -> RolloutDecision:
    """
    Check if a segment is eligible for Pixelle routing.
    
    Priority order:
    1. Global disable → not eligible
    2. Allowlist → eligible (unless disabled)
    3. Rollout percentage → deterministic bucket check
    4. Shadow mode modifier → executes but doesn't count as eligible
    
    Args:
        segment_key: Unique segment identifier
        config: Rollout configuration
        
    Returns:
        RolloutDecision with eligibility status and reason
    """
    bucket = compute_bucket(segment_key)
    
    # 1. Global kill switch
    if not config.enable_pixelle:
        logger.debug(
            "event=pixelle_rollout_disabled segment_key=%s",
            segment_key,
        )
        return RolloutDecision(
            eligible=False,
            reason="disabled",
            bucket=bucket,
            shadow_execution=False,
        )
    
    # 2. Allowlist check (explicit opt-in)
    if segment_key in config.allowlist_keys:
        logger.debug(
            "event=pixelle_allowlisted segment_key=%s",
            segment_key,
        )
        return RolloutDecision(
            eligible=True,
            reason="allowlisted",
            bucket=bucket,
            shadow_execution=config.shadow_mode,
        )
    
    # 3. Canary rollout check (deterministic bucketing)
    in_rollout = bucket < config.rollout_percentage
    
    if in_rollout:
        logger.debug(
            "event=pixelle_canary_in_bucket segment_key=%s bucket=%d threshold=%d",
            segment_key,
            bucket,
            config.rollout_percentage,
        )
        return RolloutDecision(
            eligible=True,
            reason="canary_in_bucket",
            bucket=bucket,
            shadow_execution=config.shadow_mode,
        )
    
    # Not in rollout bucket - check if shadow mode applies
    if config.shadow_mode:
        logger.debug(
            "event=pixelle_shadow_mode_eligible segment_key=%s bucket=%d threshold=%d",
            segment_key,
            bucket,
            config.rollout_percentage,
        )
        # Shadow mode: we execute but flag that results won't be used
        return RolloutDecision(
            eligible=False,
            reason="shadow_mode",
            bucket=bucket,
            shadow_execution=True,
        )
    
    logger.debug(
        "event=pixelle_canary_out_bucket segment_key=%s bucket=%d threshold=%d",
        segment_key,
        bucket,
        config.rollout_percentage,
    )
    return RolloutDecision(
        eligible=False,
        reason="canary_out_bucket",
        bucket=bucket,
        shadow_execution=False,
    )


@dataclass
class ShadowExecutionResult:
    """Result of a shadow mode execution."""
    
    segment_key: str
    capability: str
    executed: bool
    success: bool
    output_path: Optional[str] = None
    error_code: Optional[str] = None
    error_category: Optional[str] = None
    diagnostic: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_key": self.segment_key,
            "capability": self.capability,
            "executed": self.executed,
            "success": self.success,
            "output_path": self.output_path,
            "error_code": self.error_code,
            "error_category": self.error_category,
            "diagnostic": self.diagnostic,
        }


class ShadowExecutionCollector:
    """
    Collects shadow execution results for observability.
    
    Shadow mode executes Pixelle calls but doesn't affect user-facing output.
    Results are collected for monitoring and analysis.
    """
    
    def __init__(self) -> None:
        self._results: List[ShadowExecutionResult] = []
    
    def record(self, result: ShadowExecutionResult) -> None:
        """Record a shadow execution result."""
        self._results.append(result)
        logger.info(
            "event=pixelle_shadow_execution segment_key=%s capability=%s success=%s",
            result.segment_key,
            result.capability,
            result.success,
        )
    
    def get_results(self) -> List[ShadowExecutionResult]:
        """Get all collected shadow execution results."""
        return list(self._results)
    
    def clear(self) -> None:
        """Clear collected results."""
        self._results.clear()
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of shadow executions."""
        if not self._results:
            return {
                "total": 0,
                "executed": 0,
                "success": 0,
                "failed": 0,
                "by_capability": {},
            }
        
        executed = [r for r in self._results if r.executed]
        success = [r for r in executed if r.success]
        failed = [r for r in executed if not r.success]
        
        by_capability: Dict[str, Dict[str, int]] = {}
        for r in self._results:
            if r.capability not in by_capability:
                by_capability[r.capability] = {"total": 0, "success": 0, "failed": 0}
            by_capability[r.capability]["total"] += 1
            if r.executed:
                if r.success:
                    by_capability[r.capability]["success"] += 1
                else:
                    by_capability[r.capability]["failed"] += 1
        
        return {
            "total": len(self._results),
            "executed": len(executed),
            "success": len(success),
            "failed": len(failed),
            "by_capability": by_capability,
        }


# Module-level default collector for shadow executions
_shadow_collector = ShadowExecutionCollector()


def get_shadow_collector() -> ShadowExecutionCollector:
    """Get the module-level shadow execution collector."""
    return _shadow_collector


def _read_int_env(name: str, default: int, minimum: int, maximum: int = 100) -> int:
    """Read integer from environment with bounds checking."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))
