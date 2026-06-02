#!/usr/bin/env python3
"""
Rollout Gate Checker and Threshold Policy

Computes pass/fail for rollout stages (1%, 10%, 50%, etc.) from metrics input.
Provides machine-readable output including pass/fail decision and reasons.
Thresholds are configurable via environment variables and/or config file.

Usage:
    python3 -m src.steps.rollout_gate_checker --metrics-file path/to/metrics.json
    python3 -m src.steps.rollout_gate_checker --inline-metrics '{"error_rate": 0.02, ...}'

Exit codes:
    0: Gate PASS
    1: Gate FAIL (metrics exceed thresholds)
    2: Configuration or input error
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ThresholdPolicy:
    """
    Configurable threshold policy for rollout gates.
    
    All thresholds can be overridden via environment variables with prefix PIXELLE_GATE_
    """
    # Error rate thresholds (0.0 to 1.0)
    max_error_rate: float = 0.05  # 5% max error rate
    max_timeout_rate: float = 0.10  # 10% max timeout rate
    max_fallback_rate: float = 0.20  # 20% max fallback rate
    
    # Latency thresholds (seconds)
    max_p50_latency: float = 30.0  # P50 latency must be under 30s
    max_p95_latency: float = 120.0  # P95 latency must be under 120s
    max_p99_latency: float = 300.0  # P99 latency must be under 300s
    
    # Cost thresholds (USD)
    max_cost_per_request: float = 1.0  # $1 max per request
    max_total_cost: Optional[float] = None  # None = no limit
    
    # Volume thresholds
    min_sample_size: int = 10  # Minimum requests required for statistical significance
    
    @classmethod
    def from_env(cls) -> ThresholdPolicy:
        """Load policy from environment variables with PIXELLE_GATE_ prefix."""
        def get_float(key: str, default: float) -> float:
            val = os.environ.get(f"PIXELLE_GATE_{key}")
            return float(val) if val else default
        
        def get_int(key: str, default: int) -> int:
            val = os.environ.get(f"PIXELLE_GATE_{key}")
            return int(val) if val else default
        
        def get_optional_float(key: str, default: Optional[float]) -> Optional[float]:
            val = os.environ.get(f"PIXELLE_GATE_{key}")
            if val and val.lower() != "none":
                return float(val)
            return default
        
        return cls(
            max_error_rate=get_float("MAX_ERROR_RATE", 0.05),
            max_timeout_rate=get_float("MAX_TIMEOUT_RATE", 0.10),
            max_fallback_rate=get_float("MAX_FALLBACK_RATE", 0.20),
            max_p50_latency=get_float("MAX_P50_LATENCY", 30.0),
            max_p95_latency=get_float("MAX_P95_LATENCY", 120.0),
            max_p99_latency=get_float("MAX_P99_LATENCY", 300.0),
            max_cost_per_request=get_float("MAX_COST_PER_REQUEST", 1.0),
            max_total_cost=get_optional_float("MAX_TOTAL_COST", None),
            min_sample_size=get_int("MIN_SAMPLE_SIZE", 10),
        )
    
    @classmethod
    def from_file(cls, path: str | Path) -> ThresholdPolicy:
        """Load policy from JSON config file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class RolloutMetrics:
    """
    Metrics extracted from logs/evidence for rollout gate evaluation.
    """
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    fallback_requests: int = 0
    
    # Latency percentiles (seconds)
    p50_latency: Optional[float] = None
    p95_latency: Optional[float] = None
    p99_latency: Optional[float] = None
    
    # Cost metrics (USD)
    total_cost: Optional[float] = None
    avg_cost_per_request: Optional[float] = None
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate (failed / total)."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    @property
    def timeout_rate(self) -> float:
        """Calculate timeout rate (timeout / total)."""
        if self.total_requests == 0:
            return 0.0
        return self.timeout_requests / self.total_requests
    
    @property
    def fallback_rate(self) -> float:
        """Calculate fallback rate (fallback / total)."""
        if self.total_requests == 0:
            return 0.0
        return self.fallback_requests / self.total_requests
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RolloutMetrics:
        """Parse metrics from dictionary (JSON compatible)."""
        return cls(
            total_requests=data.get("total_requests", 0),
            successful_requests=data.get("successful_requests", 0),
            failed_requests=data.get("failed_requests", 0),
            timeout_requests=data.get("timeout_requests", 0),
            fallback_requests=data.get("fallback_requests", 0),
            p50_latency=data.get("p50_latency"),
            p95_latency=data.get("p95_latency"),
            p99_latency=data.get("p99_latency"),
            total_cost=data.get("total_cost"),
            avg_cost_per_request=data.get("avg_cost_per_request"),
        )


@dataclass
class GateViolation:
    """Represents a single threshold violation."""
    metric: str
    current_value: float
    threshold_value: float
    severity: str = "BLOCKING"  # BLOCKING or WARNING
    
    def __str__(self) -> str:
        return (
            f"[{self.severity}] {self.metric}: {self.current_value:.4f} "
            f"exceeds threshold {self.threshold_value:.4f}"
        )


@dataclass
class GateResult:
    """
    Machine-readable rollout gate evaluation result.
    """
    passed: bool
    violations: List[GateViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Optional[RolloutMetrics] = None
    policy: Optional[ThresholdPolicy] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as JSON-serializable dictionary."""
        return {
            "passed": self.passed,
            "violations": [
                {
                    "metric": v.metric,
                    "current_value": v.current_value,
                    "threshold_value": v.threshold_value,
                    "severity": v.severity,
                }
                for v in self.violations
            ],
            "warnings": self.warnings,
            "metrics": {
                "total_requests": self.metrics.total_requests if self.metrics else 0,
                "error_rate": self.metrics.error_rate if self.metrics else 0.0,
                "timeout_rate": self.metrics.timeout_rate if self.metrics else 0.0,
                "fallback_rate": self.metrics.fallback_rate if self.metrics else 0.0,
            } if self.metrics else {},
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Export as formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def format_report(self) -> str:
        """Format human-readable report for console output."""
        lines = []
        lines.append("=" * 60)
        lines.append("ROLLOUT GATE EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Status: {'PASS' if self.passed else 'FAIL'}")
        lines.append("")
        
        if self.metrics:
            lines.append("Metrics Summary:")
            lines.append(f"  Total Requests:    {self.metrics.total_requests}")
            lines.append(f"  Error Rate:        {self.metrics.error_rate:.2%}")
            lines.append(f"  Timeout Rate:      {self.metrics.timeout_rate:.2%}")
            lines.append(f"  Fallback Rate:     {self.metrics.fallback_rate:.2%}")
            if self.metrics.p50_latency is not None:
                lines.append(f"  P50 Latency:       {self.metrics.p50_latency:.2f}s")
            if self.metrics.p95_latency is not None:
                lines.append(f"  P95 Latency:       {self.metrics.p95_latency:.2f}s")
            if self.metrics.avg_cost_per_request is not None:
                lines.append(f"  Avg Cost/Request:  ${self.metrics.avg_cost_per_request:.4f}")
            lines.append("")
        
        if self.violations:
            lines.append("Violations:")
            for violation in self.violations:
                lines.append(f"  {violation}")
            lines.append("")
        
        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)


def evaluate_rollout_gate(
    metrics: RolloutMetrics,
    policy: ThresholdPolicy,
) -> GateResult:
    """
    Evaluate rollout gate based on metrics and policy thresholds.
    
    Args:
        metrics: Collected metrics from rollout stage
        policy: Configured threshold policy
    
    Returns:
        GateResult with pass/fail decision and violation details
    """
    violations: List[GateViolation] = []
    warnings: List[str] = []
    
    # Check minimum sample size
    if metrics.total_requests < policy.min_sample_size:
        warnings.append(
            f"Insufficient sample size: {metrics.total_requests} < {policy.min_sample_size} "
            f"(results may not be statistically significant)"
        )
    
    # Check error rate
    if metrics.error_rate > policy.max_error_rate:
        violations.append(GateViolation(
            metric="error_rate",
            current_value=metrics.error_rate,
            threshold_value=policy.max_error_rate,
            severity="BLOCKING",
        ))
    
    # Check timeout rate
    if metrics.timeout_rate > policy.max_timeout_rate:
        violations.append(GateViolation(
            metric="timeout_rate",
            current_value=metrics.timeout_rate,
            threshold_value=policy.max_timeout_rate,
            severity="BLOCKING",
        ))
    
    # Check fallback rate
    if metrics.fallback_rate > policy.max_fallback_rate:
        violations.append(GateViolation(
            metric="fallback_rate",
            current_value=metrics.fallback_rate,
            threshold_value=policy.max_fallback_rate,
            severity="BLOCKING",
        ))
    
    # Check latency thresholds
    if metrics.p50_latency is not None and metrics.p50_latency > policy.max_p50_latency:
        violations.append(GateViolation(
            metric="p50_latency",
            current_value=metrics.p50_latency,
            threshold_value=policy.max_p50_latency,
            severity="BLOCKING",
        ))
    
    if metrics.p95_latency is not None and metrics.p95_latency > policy.max_p95_latency:
        violations.append(GateViolation(
            metric="p95_latency",
            current_value=metrics.p95_latency,
            threshold_value=policy.max_p95_latency,
            severity="BLOCKING",
        ))
    
    if metrics.p99_latency is not None and metrics.p99_latency > policy.max_p99_latency:
        violations.append(GateViolation(
            metric="p99_latency",
            current_value=metrics.p99_latency,
            threshold_value=policy.max_p99_latency,
            severity="BLOCKING",
        ))
    
    # Check cost thresholds
    if metrics.avg_cost_per_request is not None and metrics.avg_cost_per_request > policy.max_cost_per_request:
        violations.append(GateViolation(
            metric="avg_cost_per_request",
            current_value=metrics.avg_cost_per_request,
            threshold_value=policy.max_cost_per_request,
            severity="BLOCKING",
        ))
    
    if (
        policy.max_total_cost is not None
        and metrics.total_cost is not None
        and metrics.total_cost > policy.max_total_cost
    ):
        violations.append(GateViolation(
            metric="total_cost",
            current_value=metrics.total_cost,
            threshold_value=policy.max_total_cost,
            severity="BLOCKING",
        ))
    
    # Determine pass/fail
    passed = len(violations) == 0
    
    return GateResult(
        passed=passed,
        violations=violations,
        warnings=warnings,
        metrics=metrics,
        policy=policy,
    )


def main() -> int:
    """
    CLI entry point for rollout gate checker.
    
    Returns:
        0 if gate passes, 1 if gate fails, 2 on error
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate rollout gate from metrics and thresholds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        help="Path to JSON file containing metrics",
    )
    parser.add_argument(
        "--inline-metrics",
        type=str,
        help="JSON string with inline metrics",
    )
    parser.add_argument(
        "--policy-file",
        type=str,
        help="Path to JSON file with threshold policy (default: use env vars)",
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=["human", "json"],
        default="human",
        help="Output format (default: human)",
    )
    
    args = parser.parse_args()
    
    # Load metrics
    try:
        if args.metrics_file:
            with open(args.metrics_file) as f:
                metrics_data = json.load(f)
        elif args.inline_metrics:
            metrics_data = json.loads(args.inline_metrics)
        else:
            print("ERROR: Either --metrics-file or --inline-metrics is required", file=sys.stderr)
            return 2
        
        metrics = RolloutMetrics.from_dict(metrics_data)
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"ERROR: Failed to load metrics: {e}", file=sys.stderr)
        return 2
    
    # Load policy
    try:
        if args.policy_file:
            policy = ThresholdPolicy.from_file(args.policy_file)
        else:
            policy = ThresholdPolicy.from_env()
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"ERROR: Failed to load policy: {e}", file=sys.stderr)
        return 2
    
    # Evaluate gate
    result = evaluate_rollout_gate(metrics, policy)
    
    # Output result
    if args.output == "json":
        print(result.to_json())
    else:
        print(result.format_report())
    
    # Return exit code
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
