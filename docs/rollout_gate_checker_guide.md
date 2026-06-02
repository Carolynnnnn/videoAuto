# Rollout Gate Checker - Usage Guide

## Overview

The rollout gate checker is an automated tool for evaluating whether a Pixelle rollout stage (1%, 10%, 50%, etc.) meets quality and reliability thresholds based on collected metrics.

## Quick Start

```bash
# Basic usage with metrics file
python3 -m src.steps.rollout_gate_checker --metrics-file path/to/metrics.json

# With custom policy
python3 -m src.steps.rollout_gate_checker \
  --metrics-file metrics.json \
  --policy-file custom_policy.json

# JSON output for automation
python3 -m src.steps.rollout_gate_checker \
  --metrics-file metrics.json \
  --output json
```

## Exit Codes

- **0**: Gate PASS - metrics within thresholds
- **1**: Gate FAIL - one or more thresholds exceeded
- **2**: Configuration error (invalid metrics/policy file)

## Metrics Input Format

Create a JSON file with your rollout metrics:

```json
{
  "total_requests": 100,
  "successful_requests": 95,
  "failed_requests": 5,
  "timeout_requests": 2,
  "fallback_requests": 10,
  "p50_latency": 12.5,
  "p95_latency": 45.2,
  "p99_latency": 89.7,
  "total_cost": 25.50,
  "avg_cost_per_request": 0.255
}
```

**Required fields**: `total_requests`  
**Optional fields**: All others (omitted fields default to 0 or None)

## Threshold Policy Configuration

### Via Environment Variables

Set thresholds using `PIXELLE_GATE_*` prefix:

```bash
export PIXELLE_GATE_MAX_ERROR_RATE=0.05        # 5% max error rate
export PIXELLE_GATE_MAX_TIMEOUT_RATE=0.10      # 10% max timeout rate
export PIXELLE_GATE_MAX_FALLBACK_RATE=0.20     # 20% max fallback rate
export PIXELLE_GATE_MAX_P50_LATENCY=30.0       # 30s P50 latency
export PIXELLE_GATE_MAX_P95_LATENCY=120.0      # 120s P95 latency
export PIXELLE_GATE_MAX_P99_LATENCY=300.0      # 300s P99 latency
export PIXELLE_GATE_MAX_COST_PER_REQUEST=1.0   # $1.00 per request
export PIXELLE_GATE_MAX_TOTAL_COST=100.0       # $100 total (optional)
export PIXELLE_GATE_MIN_SAMPLE_SIZE=10         # 10 requests minimum
```

### Via Policy File

Create `policy.json`:

```json
{
  "max_error_rate": 0.05,
  "max_timeout_rate": 0.10,
  "max_fallback_rate": 0.20,
  "max_p50_latency": 30.0,
  "max_p95_latency": 120.0,
  "max_p99_latency": 300.0,
  "max_cost_per_request": 1.0,
  "max_total_cost": null,
  "min_sample_size": 10
}
```

Then use with `--policy-file policy.json`

**Note**: Policy file overrides environment variables if provided.

## Default Thresholds

If no configuration is provided, these production-safe defaults apply:

| Metric | Default Threshold |
|--------|-------------------|
| Error Rate | 5% |
| Timeout Rate | 10% |
| Fallback Rate | 20% |
| P50 Latency | 30 seconds |
| P95 Latency | 120 seconds |
| P99 Latency | 300 seconds |
| Cost per Request | $1.00 |
| Total Cost | None (unlimited) |
| Min Sample Size | 10 requests |

## Output Formats

### Human-Readable (default)

```
============================================================
ROLLOUT GATE EVALUATION REPORT
============================================================
Status: FAIL

Metrics Summary:
  Total Requests:    100
  Error Rate:        10.00%
  Timeout Rate:      5.00%
  Fallback Rate:     15.00%
  P50 Latency:       12.50s
  P95 Latency:       45.20s
  Avg Cost/Request:  $0.2550

Violations:
  [BLOCKING] error_rate: 0.1000 exceeds threshold 0.0500

Warnings:
  - Insufficient sample size: 5 < 10 (results may not be statistically significant)

============================================================
```

### Machine-Readable JSON

```json
{
  "passed": false,
  "violations": [
    {
      "metric": "error_rate",
      "current_value": 0.1,
      "threshold_value": 0.05,
      "severity": "BLOCKING"
    }
  ],
  "warnings": [
    "Insufficient sample size: 5 < 10 (results may not be statistically significant)"
  ],
  "metrics": {
    "total_requests": 100,
    "error_rate": 0.1,
    "timeout_rate": 0.05,
    "fallback_rate": 0.15
  }
}
```

## Integration Examples

### CI/CD Pipeline

```bash
#!/bin/bash
# Rollout gate check in CI

python3 -m src.steps.rollout_gate_checker \
  --metrics-file rollout_metrics.json \
  --output json > gate_result.json

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "✓ Gate PASSED - proceed with rollout"
  exit 0
elif [ $EXIT_CODE -eq 1 ]; then
  echo "✗ Gate FAILED - rollout blocked"
  cat gate_result.json
  exit 1
else
  echo "✗ Configuration error"
  exit 2
fi
```

### Staged Rollout Strategy

Use different thresholds for different rollout stages:

```bash
# Stage 1: 1% traffic - strict thresholds
export PIXELLE_GATE_MAX_ERROR_RATE=0.01  # 1%
export PIXELLE_GATE_MAX_TIMEOUT_RATE=0.05  # 5%
python3 -m src.steps.rollout_gate_checker --metrics-file stage1_metrics.json

# Stage 2: 10% traffic - relaxed thresholds
export PIXELLE_GATE_MAX_ERROR_RATE=0.03  # 3%
export PIXELLE_GATE_MAX_TIMEOUT_RATE=0.08  # 8%
python3 -m src.steps.rollout_gate_checker --metrics-file stage2_metrics.json

# Stage 3: 50% traffic - production thresholds
export PIXELLE_GATE_MAX_ERROR_RATE=0.05  # 5%
export PIXELLE_GATE_MAX_TIMEOUT_RATE=0.10  # 10%
python3 -m src.steps.rollout_gate_checker --metrics-file stage3_metrics.json
```

## Troubleshooting

### Gate fails with "Insufficient sample size"

**Symptom**: Warning message about sample size < min_sample_size  
**Solution**: Collect more metrics or lower `PIXELLE_GATE_MIN_SAMPLE_SIZE`  
**Note**: This is a warning only - gate can still pass

### Configuration error (exit code 2)

**Common causes**:
- Missing or invalid metrics file
- Malformed JSON in metrics or policy file
- Invalid numeric values (e.g., negative latency)

**Solution**: Validate JSON syntax and required fields

### Multiple violations

**Symptom**: Gate fails with multiple threshold breaches  
**Solution**: Address highest-impact violations first (typically error rate, then timeouts)

## Best Practices

1. **Start conservative**: Use strict thresholds for early rollout stages (1%, 10%)
2. **Monitor trends**: Track metrics over time to establish baseline
3. **Adjust gradually**: Relax thresholds as confidence increases
4. **Document decisions**: Keep audit trail of threshold changes and rationale
5. **Automate evidence**: Use JSON output for automated evidence collection

## See Also

- Example policy: `pixelle_snapshot/rollout_gate_policy.example.json`
- Plan context: `.sisyphus/plans/pixelle-production-api-first.md` (Task 18)
- Test coverage: `tests/test_rollout_gate_checker.py`
