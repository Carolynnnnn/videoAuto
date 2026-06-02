import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
import pytest

from src.steps.rollout_gate_checker import (
    GateResult,
    GateViolation,
    RolloutMetrics,
    ThresholdPolicy,
    evaluate_rollout_gate,
    main,
)


class TestThresholdPolicy:
    def test_from_env_defaults(self, monkeypatch):
        # Clear all PIXELLE_GATE_ env vars
        for key in list(os.environ.keys()):
            if key.startswith("PIXELLE_GATE_"):
                monkeypatch.delenv(key, raising=False)
        
        policy = ThresholdPolicy.from_env()
        
        assert policy.max_error_rate == 0.05
        assert policy.max_timeout_rate == 0.10
        assert policy.max_fallback_rate == 0.20
        assert policy.max_p50_latency == 30.0
        assert policy.max_p95_latency == 120.0
        assert policy.max_p99_latency == 300.0
        assert policy.max_cost_per_request == 1.0
        assert policy.max_total_cost is None
        assert policy.min_sample_size == 10
    
    def test_from_env_custom_values(self, monkeypatch):
        monkeypatch.setenv("PIXELLE_GATE_MAX_ERROR_RATE", "0.02")
        monkeypatch.setenv("PIXELLE_GATE_MAX_TIMEOUT_RATE", "0.05")
        monkeypatch.setenv("PIXELLE_GATE_MAX_FALLBACK_RATE", "0.15")
        monkeypatch.setenv("PIXELLE_GATE_MAX_P50_LATENCY", "20.0")
        monkeypatch.setenv("PIXELLE_GATE_MAX_P95_LATENCY", "100.0")
        monkeypatch.setenv("PIXELLE_GATE_MAX_P99_LATENCY", "200.0")
        monkeypatch.setenv("PIXELLE_GATE_MAX_COST_PER_REQUEST", "0.50")
        monkeypatch.setenv("PIXELLE_GATE_MAX_TOTAL_COST", "10.0")
        monkeypatch.setenv("PIXELLE_GATE_MIN_SAMPLE_SIZE", "20")
        
        policy = ThresholdPolicy.from_env()
        
        assert policy.max_error_rate == 0.02
        assert policy.max_timeout_rate == 0.05
        assert policy.max_fallback_rate == 0.15
        assert policy.max_p50_latency == 20.0
        assert policy.max_p95_latency == 100.0
        assert policy.max_p99_latency == 200.0
        assert policy.max_cost_per_request == 0.50
        assert policy.max_total_cost == 10.0
        assert policy.min_sample_size == 20
    
    def test_from_file(self, tmp_path: Path):
        policy_file = tmp_path / "policy.json"
        policy_file.write_text(json.dumps({
            "max_error_rate": 0.03,
            "max_timeout_rate": 0.08,
            "max_fallback_rate": 0.25,
            "max_p50_latency": 25.0,
            "max_p95_latency": 150.0,
            "max_p99_latency": 350.0,
            "max_cost_per_request": 0.75,
            "max_total_cost": 15.0,
            "min_sample_size": 15,
        }))
        
        policy = ThresholdPolicy.from_file(policy_file)
        
        assert policy.max_error_rate == 0.03
        assert policy.max_timeout_rate == 0.08
        assert policy.max_fallback_rate == 0.25
        assert policy.max_p50_latency == 25.0
        assert policy.max_p95_latency == 150.0
        assert policy.max_p99_latency == 350.0
        assert policy.max_cost_per_request == 0.75
        assert policy.max_total_cost == 15.0
        assert policy.min_sample_size == 15


class TestRolloutMetrics:
    def test_from_dict_complete(self):
        data = {
            "total_requests": 100,
            "successful_requests": 95,
            "failed_requests": 5,
            "timeout_requests": 2,
            "fallback_requests": 10,
            "p50_latency": 12.5,
            "p95_latency": 45.2,
            "p99_latency": 89.7,
            "total_cost": 25.50,
            "avg_cost_per_request": 0.255,
        }
        
        metrics = RolloutMetrics.from_dict(data)
        
        assert metrics.total_requests == 100
        assert metrics.successful_requests == 95
        assert metrics.failed_requests == 5
        assert metrics.timeout_requests == 2
        assert metrics.fallback_requests == 10
        assert metrics.p50_latency == 12.5
        assert metrics.p95_latency == 45.2
        assert metrics.p99_latency == 89.7
        assert metrics.total_cost == 25.50
        assert metrics.avg_cost_per_request == 0.255
    
    def test_from_dict_partial(self):
        data = {
            "total_requests": 50,
            "failed_requests": 2,
        }
        
        metrics = RolloutMetrics.from_dict(data)
        
        assert metrics.total_requests == 50
        assert metrics.failed_requests == 2
        assert metrics.successful_requests == 0
        assert metrics.p50_latency is None
    
    def test_error_rate_calculation(self):
        metrics = RolloutMetrics(
            total_requests=100,
            failed_requests=5,
        )
        assert metrics.error_rate == 0.05
    
    def test_timeout_rate_calculation(self):
        metrics = RolloutMetrics(
            total_requests=100,
            timeout_requests=10,
        )
        assert metrics.timeout_rate == 0.10
    
    def test_fallback_rate_calculation(self):
        metrics = RolloutMetrics(
            total_requests=100,
            fallback_requests=20,
        )
        assert metrics.fallback_rate == 0.20
    
    def test_rate_calculation_zero_requests(self):
        metrics = RolloutMetrics(
            total_requests=0,
            failed_requests=0,
        )
        assert metrics.error_rate == 0.0
        assert metrics.timeout_rate == 0.0
        assert metrics.fallback_rate == 0.0


class TestEvaluateRolloutGate:
    def test_gate_pass_happy_path(self):
        metrics = RolloutMetrics(
            total_requests=100,
            successful_requests=98,
            failed_requests=2,
            timeout_requests=5,
            fallback_requests=10,
            p50_latency=15.0,
            p95_latency=80.0,
            p99_latency=150.0,
            avg_cost_per_request=0.25,
            total_cost=25.0,
        )
        
        policy = ThresholdPolicy()
        
        result = evaluate_rollout_gate(metrics, policy)
        
        assert result.passed is True
        assert len(result.violations) == 0
        assert result.metrics == metrics
        assert result.policy == policy
    
    def test_gate_fail_error_rate_exceeded(self):
        metrics = RolloutMetrics(
            total_requests=100,
            failed_requests=10,  # 10% error rate > 5% threshold
        )
        
        policy = ThresholdPolicy(max_error_rate=0.05)
        
        result = evaluate_rollout_gate(metrics, policy)
        
        assert result.passed is False
        assert len(result.violations) == 1
        violation = result.violations[0]
        assert violation.metric == "error_rate"
        assert violation.current_value == 0.10
        assert violation.threshold_value == 0.05
        assert violation.severity == "BLOCKING"
    
    def test_gate_fail_timeout_rate_exceeded(self):
        metrics = RolloutMetrics(
            total_requests=100,
            timeout_requests=15,  # 15% timeout rate > 10% threshold
        )
        
        policy = ThresholdPolicy(max_timeout_rate=0.10)
        
        result = evaluate_rollout_gate(metrics, policy)
        
        assert result.passed is False
        assert len(result.violations) == 1
        violation = result.violations[0]
        assert violation.metric == "timeout_rate"
        assert violation.current_value == 0.15
        assert violation.threshold_value == 0.10
    
    def test_gate_fail_fallback_rate_exceeded(self):
        metrics = RolloutMetrics(
            total_requests=100,
            fallback_requests=30,  # 30% fallback rate > 20% threshold
        )
        
        policy = ThresholdPolicy(max_fallback_rate=0.20)
        
        result = evaluate_rollout_gate(metrics, policy)
        
        assert result.passed is False
        assert len(result.violations) == 1
        violation = result.violations[0]
        assert violation.metric == "fallback_rate"
        assert violation.current_value == 0.30
        assert violation.threshold_value == 0.20
    
    def test_gate_fail_p50_latency_exceeded(self):
        metrics = RolloutMetrics(
            total_requests=50,
            p50_latency=45.0,  # 45s > 30s threshold
        )
        
        policy = ThresholdPolicy(max_p50_latency=30.0)
        
        result = evaluate_rollout_gate(metrics, policy)
        
        assert result.passed is False
        assert len(result.violations) == 1
        violation = result.violations[0]
        assert violation.metric == "p50_latency"
        assert violation.current_value == 45.0
        assert violation.threshold_value == 30.0
    
    def test_gate_fail_p95_latency_exceeded(self):
        metrics = RolloutMetrics(
            total_requests=50,
            p95_latency=150.0,  # 150s > 120s threshold
        )
        
        policy = ThresholdPolicy(max_p95_latency=120.0)
        
        result = evaluate_rollout_gate(metrics, policy)
        
        assert result.passed is False
        assert len(result.violations) == 1
        violation = result.violations[0]
        assert violation.metric == "p95_latency"
        assert violation.current_value == 150.0
        assert violation.threshold_value == 120.0
    
    def test_gate_fail_cost_per_request_exceeded(self):
        metrics = RolloutMetrics(
            total_requests=50,
            avg_cost_per_request=1.50,  # $1.50 > $1.00 threshold
        )
        
        policy = ThresholdPolicy(max_cost_per_request=1.0)
        
        result = evaluate_rollout_gate(metrics, policy)
        
        assert result.passed is False
        assert len(result.violations) == 1
        violation = result.violations[0]
        assert violation.metric == "avg_cost_per_request"
        assert violation.current_value == 1.50
        assert violation.threshold_value == 1.0
    
    def test_gate_fail_total_cost_exceeded(self):
        metrics = RolloutMetrics(
            total_requests=50,
            total_cost=150.0,  # $150 > $100 threshold
        )
        
        policy = ThresholdPolicy(max_total_cost=100.0)
        
        result = evaluate_rollout_gate(metrics, policy)
        
        assert result.passed is False
        assert len(result.violations) == 1
        violation = result.violations[0]
        assert violation.metric == "total_cost"
        assert violation.current_value == 150.0
        assert violation.threshold_value == 100.0
    
    def test_gate_multiple_violations(self):
        metrics = RolloutMetrics(
            total_requests=100,
            failed_requests=10,  # 10% error rate > 5% threshold
            timeout_requests=15,  # 15% timeout rate > 10% threshold
            p50_latency=50.0,  # 50s > 30s threshold
        )
        
        policy = ThresholdPolicy()
        
        result = evaluate_rollout_gate(metrics, policy)
        
        assert result.passed is False
        assert len(result.violations) == 3
        violation_metrics = {v.metric for v in result.violations}
        assert "error_rate" in violation_metrics
        assert "timeout_rate" in violation_metrics
        assert "p50_latency" in violation_metrics
    
    def test_gate_warning_insufficient_sample_size(self):
        metrics = RolloutMetrics(
            total_requests=5,  # < 10 min sample size
        )
        
        policy = ThresholdPolicy(min_sample_size=10)
        
        result = evaluate_rollout_gate(metrics, policy)
        
        assert result.passed is True  # Still passes, but with warning
        assert len(result.warnings) == 1
        assert "Insufficient sample size" in result.warnings[0]
        assert "5 < 10" in result.warnings[0]


class TestGateResultSerialization:
    def test_to_dict_passed(self):
        result = GateResult(
            passed=True,
            violations=[],
            warnings=["Sample warning"],
            metrics=RolloutMetrics(
                total_requests=100,
                failed_requests=2,
            ),
        )
        
        data = result.to_dict()
        
        assert data["passed"] is True
        assert data["violations"] == []
        assert data["warnings"] == ["Sample warning"]
        assert data["metrics"]["total_requests"] == 100
        assert data["metrics"]["error_rate"] == 0.02
    
    def test_to_dict_failed(self):
        result = GateResult(
            passed=False,
            violations=[
                GateViolation(
                    metric="error_rate",
                    current_value=0.10,
                    threshold_value=0.05,
                    severity="BLOCKING",
                ),
            ],
            metrics=RolloutMetrics(
                total_requests=100,
                failed_requests=10,
            ),
        )
        
        data = result.to_dict()
        
        assert data["passed"] is False
        assert len(data["violations"]) == 1
        violation = data["violations"][0]
        assert violation["metric"] == "error_rate"
        assert violation["current_value"] == 0.10
        assert violation["threshold_value"] == 0.05
        assert violation["severity"] == "BLOCKING"
    
    def test_to_json_format(self):
        result = GateResult(
            passed=True,
            violations=[],
            warnings=[],
            metrics=RolloutMetrics(total_requests=50),
        )
        
        json_str = result.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["passed"] is True
        assert parsed["violations"] == []
    
    def test_format_report_pass(self):
        result = GateResult(
            passed=True,
            violations=[],
            warnings=[],
            metrics=RolloutMetrics(
                total_requests=100,
                failed_requests=2,
                timeout_requests=5,
                fallback_requests=10,
                p50_latency=15.0,
                avg_cost_per_request=0.25,
            ),
        )
        
        report = result.format_report()
        
        assert "Status: PASS" in report
        assert "Total Requests:    100" in report
        assert "Error Rate:        2.00%" in report
        assert "P50 Latency:       15.00s" in report
        assert "Avg Cost/Request:  $0.2500" in report
    
    def test_format_report_fail(self):
        result = GateResult(
            passed=False,
            violations=[
                GateViolation(
                    metric="error_rate",
                    current_value=0.10,
                    threshold_value=0.05,
                    severity="BLOCKING",
                ),
            ],
            warnings=["Warning message"],
            metrics=RolloutMetrics(total_requests=100, failed_requests=10),
        )
        
        report = result.format_report()
        
        assert "Status: FAIL" in report
        assert "Violations:" in report
        assert "error_rate" in report
        assert "Warnings:" in report
        assert "Warning message" in report


class TestMainCLI:
    def test_main_metrics_file_pass(self, tmp_path: Path, monkeypatch):
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps({
            "total_requests": 100,
            "failed_requests": 2,
            "timeout_requests": 5,
            "fallback_requests": 10,
        }))
        
        monkeypatch.setattr(
            "sys.argv",
            ["rollout_gate_checker", "--metrics-file", str(metrics_file)],
        )
        
        exit_code = main()
        
        assert exit_code == 0  # PASS
    
    def test_main_metrics_file_fail(self, tmp_path: Path, monkeypatch):
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps({
            "total_requests": 100,
            "failed_requests": 10,  # 10% error rate > 5% threshold
        }))
        
        monkeypatch.setattr(
            "sys.argv",
            ["rollout_gate_checker", "--metrics-file", str(metrics_file)],
        )
        
        exit_code = main()
        
        assert exit_code == 1  # FAIL
    
    def test_main_inline_metrics_pass(self, monkeypatch):
        metrics_json = json.dumps({
            "total_requests": 50,
            "failed_requests": 1,
        })
        
        monkeypatch.setattr(
            "sys.argv",
            ["rollout_gate_checker", "--inline-metrics", metrics_json],
        )
        
        exit_code = main()
        
        assert exit_code == 0  # PASS
    
    def test_main_inline_metrics_fail(self, monkeypatch):
        metrics_json = json.dumps({
            "total_requests": 50,
            "failed_requests": 10,  # 20% error rate > 5% threshold
        })
        
        monkeypatch.setattr(
            "sys.argv",
            ["rollout_gate_checker", "--inline-metrics", metrics_json],
        )
        
        exit_code = main()
        
        assert exit_code == 1  # FAIL
    
    def test_main_policy_file(self, tmp_path: Path, monkeypatch):
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps({
            "total_requests": 100,
            "failed_requests": 8,  # 8% error rate
        }))
        
        policy_file = tmp_path / "policy.json"
        policy_file.write_text(json.dumps({
            "max_error_rate": 0.10,  # 10% threshold (higher than default 5%)
        }))
        
        monkeypatch.setattr(
            "sys.argv",
            [
                "rollout_gate_checker",
                "--metrics-file", str(metrics_file),
                "--policy-file", str(policy_file),
            ],
        )
        
        exit_code = main()
        
        assert exit_code == 0  # PASS with custom threshold
    
    def test_main_json_output(self, tmp_path: Path, monkeypatch, capsys):
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps({
            "total_requests": 100,
            "failed_requests": 2,
        }))
        
        monkeypatch.setattr(
            "sys.argv",
            [
                "rollout_gate_checker",
                "--metrics-file", str(metrics_file),
                "--output", "json",
            ],
        )
        
        exit_code = main()
        
        assert exit_code == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["passed"] is True
        assert "metrics" in output
    
    def test_main_missing_metrics_error(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["rollout_gate_checker"],
        )
        
        exit_code = main()
        
        assert exit_code == 2  # Configuration error
    
    def test_main_invalid_metrics_file_error(self, tmp_path: Path, monkeypatch):
        metrics_file = tmp_path / "nonexistent.json"
        
        monkeypatch.setattr(
            "sys.argv",
            ["rollout_gate_checker", "--metrics-file", str(metrics_file)],
        )
        
        exit_code = main()
        
        assert exit_code == 2  # Configuration error
