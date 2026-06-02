import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pixelle_snapshot.adapters.contracts import (
    ErrorCategory,
    ExecutionError,
    ProviderExecutionMapping,
    ProviderFetchResult,
    ProviderJobStatus,
    ProviderPollResult,
    ProviderSubmitResult,
)


def test_provider_execution_mapping_filters_provider_payload() -> None:
    submit = ProviderSubmitResult(
        job_id="job-123",
        status=ProviderJobStatus.SUBMITTED,
        metadata={"request_id": "req-1", "raw_payload": {"secret": "x"}},
    )
    poll = ProviderPollResult(
        job_id="job-123",
        status=ProviderJobStatus.SUCCEEDED,
        metadata={"run_seconds": 3.2, "provider_trace": "opaque"},
    )
    fetch = ProviderFetchResult(
        job_id="job-123",
        output_path="/tmp/output.mp4",
        metadata={"artifact_bytes": 1024, "provider_blob": {"foo": "bar"}},
    )

    mapped = ProviderExecutionMapping.from_lifecycle(submit=submit, poll=poll, fetch=fetch)

    assert mapped.job_id == "job-123"
    assert mapped.status == ProviderJobStatus.SUCCEEDED
    assert mapped.output_path == "/tmp/output.mp4"
    assert mapped.metadata["provider_job_id"] == "job-123"
    assert mapped.metadata["provider_status"] == "SUCCEEDED"
    assert mapped.metadata["request_id"] == "req-1"
    assert mapped.metadata["run_seconds"] == 3.2
    assert mapped.metadata["artifact_bytes"] == 1024
    assert "raw_payload" not in mapped.metadata
    assert "provider_trace" not in mapped.metadata
    assert "provider_blob" not in mapped.metadata


def test_provider_execution_mapping_supports_error_without_fetch() -> None:
    submit = ProviderSubmitResult(job_id="job-456")
    poll = ProviderPollResult(job_id="job-456", status=ProviderJobStatus.FAILED)
    error = ExecutionError("provider execution failed")

    mapped = ProviderExecutionMapping.from_lifecycle(
        submit=submit,
        poll=poll,
        fetch=None,
        error=error,
    )

    assert mapped.output_path is None
    assert mapped.error is error
    assert mapped.status == ProviderJobStatus.FAILED
    assert mapped.metadata["provider_status"] == "FAILED"


def test_step4_contract_import_boundary_blocks_provider_symbols() -> None:
    step4_path = Path(__file__).parent.parent / "src" / "steps" / "step4_assets.py"
    source = step4_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    allowed_contract_symbols = {
        "ErrorCategory",
        "FailureDiagnostic",
        "normalize_error_category",
    }
    forbidden_provider_symbols = {
        "ProviderLifecycleClient",
        "ProviderSubmitResult",
        "ProviderPollResult",
        "ProviderFetchResult",
        "ProviderCancelResult",
        "ProviderExecutionMapping",
        "ProviderJobStatus",
    }

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "pixelle_snapshot.adapters.contracts":
            for alias in node.names:
                imported.add(alias.name)

    assert imported <= allowed_contract_symbols
    assert not (imported & forbidden_provider_symbols)

    for symbol in forbidden_provider_symbols:
        assert symbol not in source


def test_provider_error_category_remains_normalized() -> None:
    assert ErrorCategory.PROVIDER.value == "PROVIDER"
