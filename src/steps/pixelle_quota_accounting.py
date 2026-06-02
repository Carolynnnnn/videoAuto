"""
Pixelle Quota Accounting and Enforcement

This module provides:
1. QuotaConfig - Configurable quota thresholds (env/profile) 
2. QuotaAccounting - Per-request cost/usage tracking
3. QuotaEnforcement - Pre-call quota checks with explicit fallback reason codes

Design Principles:
- Quota checks fail gracefully to fallback (do not block fallback chain)
- Explicit reason codes: PIXELLE_QUOTA_EXCEEDED, PIXELLE_BUDGET_EXCEEDED
- Deterministic behavior in test mode (quota enforcement disabled)
- Cost/usage metadata flows through ProviderMetrics and logs

Environment Variables:
- PIXELLE_QUOTA_ENABLED: "1" to enable quota enforcement (default: "0")
- PIXELLE_QUOTA_MAX_REQUESTS_PER_BUILD: Max requests per build ID (default: 0 = unlimited)
- PIXELLE_QUOTA_MAX_COST_USD_PER_BUILD: Max total cost per build ID (default: 0.0 = unlimited)
- PIXELLE_QUOTA_MAX_COST_USD_PER_REQUEST: Max cost per single request (default: 0.0 = unlimited)
- PIXELLE_TEST_MODE: "1" disables quota enforcement for deterministic testing
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger("pixelle_quota")


class QuotaExceededError(Exception):
    """Raised when quota check fails - triggers fallback with explicit diagnostics."""

    def __init__(
        self,
        reason_code: str,
        category: str,
        guidance: str,
        current_value: float,
        limit_value: float,
    ):
        self.reason_code = reason_code
        self.category = category
        self.guidance = guidance
        self.current_value = current_value
        self.limit_value = limit_value
        super().__init__(
            f"{reason_code}: current={current_value}, limit={limit_value}"
        )


@dataclass(frozen=True)
class QuotaConfig:
    """
    Configurable quota thresholds loaded from environment.
    
    All limits default to 0 (unlimited) unless explicitly configured.
    Test mode disables enforcement to preserve deterministic CI behavior.
    """
    enabled: bool = False
    test_mode: bool = False
    max_requests_per_build: int = 0  # 0 = unlimited
    max_cost_usd_per_build: float = 0.0  # 0.0 = unlimited
    max_cost_usd_per_request: float = 0.0  # 0.0 = unlimited

    @classmethod
    def from_env(cls) -> "QuotaConfig":
        """Load quota configuration from environment variables."""
        test_mode = os.environ.get("PIXELLE_TEST_MODE", "0") == "1"
        return cls(
            enabled=os.environ.get("PIXELLE_QUOTA_ENABLED", "0") == "1",
            test_mode=test_mode,
            max_requests_per_build=_read_int_env(
                "PIXELLE_QUOTA_MAX_REQUESTS_PER_BUILD", 0, minimum=0
            ),
            max_cost_usd_per_build=_read_float_env(
                "PIXELLE_QUOTA_MAX_COST_USD_PER_BUILD", 0.0, minimum=0.0
            ),
            max_cost_usd_per_request=_read_float_env(
                "PIXELLE_QUOTA_MAX_COST_USD_PER_REQUEST", 0.0, minimum=0.0
            ),
        )

    @property
    def is_enforcement_active(self) -> bool:
        """Check if quota enforcement is actually active."""
        return self.enabled and not self.test_mode


@dataclass
class UsageRecord:
    """Per-request usage record for quota tracking."""
    request_id: str
    segment_key: str
    capability: str
    cost_usd: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    build_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "segment_key": self.segment_key,
            "capability": self.capability,
            "cost_usd": self.cost_usd,
            "timestamp": self.timestamp.isoformat(),
            "build_id": self.build_id,
        }


@dataclass
class QuotaSnapshot:
    """Point-in-time quota usage snapshot."""
    build_id: Optional[str]
    total_requests: int
    total_cost_usd: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Limits for context
    max_requests: int = 0
    max_cost_usd: float = 0.0
    
    @property
    def requests_remaining(self) -> Optional[int]:
        if self.max_requests <= 0:
            return None
        return max(0, self.max_requests - self.total_requests)
    
    @property
    def cost_usd_remaining(self) -> Optional[float]:
        if self.max_cost_usd <= 0.0:
            return None
        return max(0.0, self.max_cost_usd - self.total_cost_usd)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "build_id": self.build_id,
            "total_requests": self.total_requests,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "timestamp": self.timestamp.isoformat(),
        }
        if self.requests_remaining is not None:
            result["requests_remaining"] = self.requests_remaining
        if self.cost_usd_remaining is not None:
            result["cost_usd_remaining"] = round(self.cost_usd_remaining, 6)
        return result


class QuotaAccounting:
    """
    Thread-safe per-request cost/usage tracking with build-level aggregation.
    
    Records usage metadata for observability and provides quota snapshots
    for enforcement decisions.
    """

    def __init__(self, config: Optional[QuotaConfig] = None):
        self._config = config or QuotaConfig.from_env()
        self._lock = threading.Lock()
        self._build_usage: Dict[str, List[UsageRecord]] = {}
        # Global tracking for builds without ID
        self._global_usage: List[UsageRecord] = []

    @property
    def config(self) -> QuotaConfig:
        return self._config

    def record_usage(
        self,
        *,
        request_id: str,
        segment_key: str,
        capability: str,
        cost_usd: float = 0.0,
        build_id: Optional[str] = None,
    ) -> UsageRecord:
        """
        Record per-request usage metadata.
        
        Returns the created UsageRecord for logging/metrics integration.
        """
        record = UsageRecord(
            request_id=request_id,
            segment_key=segment_key,
            capability=capability,
            cost_usd=cost_usd,
            build_id=build_id,
        )
        
        with self._lock:
            if build_id:
                if build_id not in self._build_usage:
                    self._build_usage[build_id] = []
                self._build_usage[build_id].append(record)
            else:
                self._global_usage.append(record)
        
        # Log usage for observability
        logger.info(
            "event=pixelle_usage_recorded request_id=%s segment_key=%s "
            "capability=%s cost_usd=%.6f build_id=%s",
            request_id,
            segment_key,
            capability,
            cost_usd,
            build_id or "none",
        )
        
        return record

    def get_snapshot(self, build_id: Optional[str] = None) -> QuotaSnapshot:
        """
        Get current quota usage snapshot for a build (or global).
        """
        with self._lock:
            if build_id and build_id in self._build_usage:
                records = self._build_usage[build_id]
            elif build_id:
                records = []
            else:
                records = self._global_usage
            
            total_requests = len(records)
            total_cost = sum(r.cost_usd for r in records)
        
        return QuotaSnapshot(
            build_id=build_id,
            total_requests=total_requests,
            total_cost_usd=total_cost,
            max_requests=self._config.max_requests_per_build,
            max_cost_usd=self._config.max_cost_usd_per_build,
        )

    def reset_build(self, build_id: str) -> None:
        """Reset usage tracking for a specific build."""
        with self._lock:
            if build_id in self._build_usage:
                del self._build_usage[build_id]

    def reset_all(self) -> None:
        """Reset all usage tracking (for testing)."""
        with self._lock:
            self._build_usage.clear()
            self._global_usage.clear()


class QuotaEnforcement:
    """
    Pre-call quota enforcement with explicit fallback reason codes.
    
    Raises QuotaExceededError when limits are breached, which triggers
    controlled fallback with explicit diagnostics.
    
    Quota check failures do NOT block the fallback chain - they simply
    cause immediate fallback to next source with diagnostic metadata.
    """

    def __init__(
        self,
        accounting: Optional[QuotaAccounting] = None,
        config: Optional[QuotaConfig] = None,
    ):
        self._config = config or QuotaConfig.from_env()
        self._accounting = accounting or QuotaAccounting(self._config)

    @property
    def config(self) -> QuotaConfig:
        return self._config

    @property
    def accounting(self) -> QuotaAccounting:
        return self._accounting

    def check_before_request(
        self,
        *,
        segment_key: str,
        capability: str,
        build_id: Optional[str] = None,
        estimated_cost_usd: float = 0.0,
    ) -> None:
        """
        Check quota limits before making a provider request.
        
        Raises QuotaExceededError if any limit would be breached,
        triggering fallback with explicit reason code and category.
        
        Args:
            segment_key: Segment being processed
            capability: Pixelle capability (digital_human, i2v, etc.)
            build_id: Optional build ID for per-build quotas
            estimated_cost_usd: Estimated cost for this request
        """
        if not self._config.is_enforcement_active:
            return
        
        snapshot = self._accounting.get_snapshot(build_id)
        
        # Check request count limit
        if self._config.max_requests_per_build > 0:
            if snapshot.total_requests >= self._config.max_requests_per_build:
                logger.warning(
                    "event=pixelle_quota_exceeded reason=request_count "
                    "segment_key=%s capability=%s build_id=%s "
                    "current=%d limit=%d",
                    segment_key,
                    capability,
                    build_id or "none",
                    snapshot.total_requests,
                    self._config.max_requests_per_build,
                )
                raise QuotaExceededError(
                    reason_code="PIXELLE_QUOTA_EXCEEDED",
                    category="RESOURCE",
                    guidance=(
                        f"Build request quota exhausted. "
                        f"Current: {snapshot.total_requests}, "
                        f"Limit: {self._config.max_requests_per_build}. "
                        f"Consider increasing PIXELLE_QUOTA_MAX_REQUESTS_PER_BUILD."
                    ),
                    current_value=float(snapshot.total_requests),
                    limit_value=float(self._config.max_requests_per_build),
                )
        
        # Check total cost limit (build-level budget)
        if self._config.max_cost_usd_per_build > 0.0:
            projected_cost = snapshot.total_cost_usd + estimated_cost_usd
            if projected_cost >= self._config.max_cost_usd_per_build:
                logger.warning(
                    "event=pixelle_budget_exceeded reason=build_cost "
                    "segment_key=%s capability=%s build_id=%s "
                    "current=%.6f projected=%.6f limit=%.6f",
                    segment_key,
                    capability,
                    build_id or "none",
                    snapshot.total_cost_usd,
                    projected_cost,
                    self._config.max_cost_usd_per_build,
                )
                raise QuotaExceededError(
                    reason_code="PIXELLE_BUDGET_EXCEEDED",
                    category="RESOURCE",
                    guidance=(
                        f"Build cost budget exceeded. "
                        f"Current: ${snapshot.total_cost_usd:.4f}, "
                        f"Projected: ${projected_cost:.4f}, "
                        f"Limit: ${self._config.max_cost_usd_per_build:.4f}. "
                        f"Consider increasing PIXELLE_QUOTA_MAX_COST_USD_PER_BUILD."
                    ),
                    current_value=snapshot.total_cost_usd,
                    limit_value=self._config.max_cost_usd_per_build,
                )
        
        # Check per-request cost limit
        if self._config.max_cost_usd_per_request > 0.0:
            if estimated_cost_usd > self._config.max_cost_usd_per_request:
                logger.warning(
                    "event=pixelle_request_cost_exceeded reason=request_cost "
                    "segment_key=%s capability=%s build_id=%s "
                    "estimated=%.6f limit=%.6f",
                    segment_key,
                    capability,
                    build_id or "none",
                    estimated_cost_usd,
                    self._config.max_cost_usd_per_request,
                )
                raise QuotaExceededError(
                    reason_code="PIXELLE_REQUEST_COST_EXCEEDED",
                    category="RESOURCE",
                    guidance=(
                        f"Single request cost exceeds limit. "
                        f"Estimated: ${estimated_cost_usd:.4f}, "
                        f"Limit: ${self._config.max_cost_usd_per_request:.4f}. "
                        f"Consider increasing PIXELLE_QUOTA_MAX_COST_USD_PER_REQUEST."
                    ),
                    current_value=estimated_cost_usd,
                    limit_value=self._config.max_cost_usd_per_request,
                )


def create_quota_diagnostic(
    error: QuotaExceededError,
) -> Dict[str, Any]:
    """
    Create a fallback diagnostic dictionary from QuotaExceededError.
    
    Follows the same schema as other fallback diagnostics (FailureDiagnostic).
    """
    return {
        "category": error.category,
        "reason_code": error.reason_code,
        "guidance": error.guidance,
        "retryable": False,  # Quota exceeded is not retryable within same build
        "fallback_hint": "Pipeline will use next fallback source.",
        "quota_details": {
            "current_value": error.current_value,
            "limit_value": error.limit_value,
        },
    }


# ─────────────────────────────────────────────
# Module-level singleton for Step4 integration
# ─────────────────────────────────────────────
_quota_enforcement: Optional[QuotaEnforcement] = None


def get_quota_enforcement() -> QuotaEnforcement:
    """Get the module-level QuotaEnforcement singleton."""
    global _quota_enforcement
    if _quota_enforcement is None:
        _quota_enforcement = QuotaEnforcement()
    return _quota_enforcement


def set_quota_enforcement(enforcement: QuotaEnforcement) -> None:
    """Set a custom QuotaEnforcement instance (for testing)."""
    global _quota_enforcement
    _quota_enforcement = enforcement


def reset_quota_enforcement() -> None:
    """Reset the module-level QuotaEnforcement singleton."""
    global _quota_enforcement
    _quota_enforcement = None


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def _read_int_env(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def _read_float_env(name: str, default: float, minimum: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(minimum, value)
