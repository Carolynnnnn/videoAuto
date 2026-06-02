import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pixelle_snapshot.adapters.contracts import AdapterError, ErrorCategory
from src.core.models import AudioRef, Segment, VisualPlan
from src.steps.pixelle_retry_policy import (
    ERROR_RETRY_MATRIX,
    PixelleRetryPolicy,
    classify_provider_error,
)
from src.steps.step4_assets import _resolve_pixelle_asset


def _make_segment() -> Segment:
    text = "Retry policy segment"
    content_key = Segment.compute_content_key(text)
    return Segment(
        segment_key=Segment.compute_segment_key(content_key, 1),
        content_key=content_key,
        index=1,
        start=0.0,
        end=4.0,
        duration=4.0,
        text=text,
        audio_ref=AudioRef(type="full", path="/tmp/audio.wav", trim_start=0.0, trim_end=4.0),
        visual_plan=VisualPlan(type="pixelle_digital_human", pixelle_workflow="digital_human"),
        plan_hash="retryhash1234",
    )


def test_retry_matrix_maps_to_existing_error_categories() -> None:
    assert ERROR_RETRY_MATRIX[ErrorCategory.VALIDATION.value] is False
    assert ERROR_RETRY_MATRIX[ErrorCategory.MISSING_INPUT.value] is False
    assert ERROR_RETRY_MATRIX[ErrorCategory.UNSUPPORTED.value] is False
    assert ERROR_RETRY_MATRIX[ErrorCategory.EXECUTION.value] is True
    assert ERROR_RETRY_MATRIX[ErrorCategory.TIMEOUT.value] is True
    assert ERROR_RETRY_MATRIX[ErrorCategory.RESOURCE.value] is True
    assert ERROR_RETRY_MATRIX[ErrorCategory.PROVIDER.value] is True


def test_auth_like_provider_error_maps_to_validation_non_retryable() -> None:
    err = AdapterError(category=ErrorCategory.PROVIDER, message="Invalid API key for provider")
    classification = classify_provider_error(err)

    assert classification.category == ErrorCategory.VALIDATION.value
    assert classification.retryable is False


def test_retry_policy_backoff_is_exponential_and_bounded() -> None:
    policy = PixelleRetryPolicy(max_attempts=4, base_delay_seconds=0.5, max_delay_seconds=1.0)

    assert policy.backoff_seconds(1) == 0.5
    assert policy.backoff_seconds(2) == 1.0
    assert policy.backoff_seconds(3) == 1.0


@pytest.mark.resiliency_retry_then_success
def test_resolve_pixelle_asset_retries_retryable_failures(monkeypatch, tmp_path: Path) -> None:
    attempts = {"count": 0}

    class FlakyAdapter:
        def invoke(self, request):
            attempts["count"] += 1
            if attempts["count"] < 3:
                return type(
                    "Resp",
                    (),
                    {
                        "success": False,
                        "output_path": None,
                        "error": AdapterError(category=ErrorCategory.TIMEOUT, message="timed out"),
                    },
                )()
            output_path = Path(request.output_dir) / f"pixelle_{request.segment_key}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"pixelle-video")
            return type(
                "Resp",
                (),
                {
                    "success": True,
                    "output_path": str(output_path),
                    "error": None,
                },
            )()

    monkeypatch.setenv("PIXELLE_PROVIDER_MAX_RETRIES", "3")
    monkeypatch.setenv("PIXELLE_PROVIDER_RETRY_BASE_DELAY", "0")
    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: FlakyAdapter())

    result = _resolve_pixelle_asset(
        segment=_make_segment(),
        project_root=str(tmp_path),
        generated_dir=str(tmp_path / "generated"),
        effective_capability="digital_human",
    )

    assert attempts["count"] == 3
    assert result.kind == "pixelle_video"
    assert result.path is not None


@pytest.mark.resiliency_non_retryable
def test_resolve_pixelle_asset_stops_on_non_retryable_auth_failure(monkeypatch, tmp_path: Path) -> None:
    attempts = {"count": 0}

    class AuthFailAdapter:
        def invoke(self, request):
            attempts["count"] += 1
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(category=ErrorCategory.PROVIDER, message="Unauthorized API key"),
                },
            )()

    monkeypatch.setenv("PIXELLE_PROVIDER_MAX_RETRIES", "5")
    monkeypatch.setenv("PIXELLE_PROVIDER_RETRY_BASE_DELAY", "0")
    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: AuthFailAdapter())

    result = _resolve_pixelle_asset(
        segment=_make_segment(),
        project_root=str(tmp_path),
        generated_dir=str(tmp_path / "generated"),
        effective_capability="digital_human",
    )

    assert attempts["count"] == 1
    assert result.fallback_reason_code == "PIXELLE_INVOCATION_FAILED"
    assert result.fallback_error_category == ErrorCategory.VALIDATION.value


def test_resolve_pixelle_asset_respects_max_attempt_bound(monkeypatch, tmp_path: Path) -> None:
    attempts = {"count": 0}

    class AlwaysTimeoutAdapter:
        def invoke(self, request):
            attempts["count"] += 1
            return type(
                "Resp",
                (),
                {
                    "success": False,
                    "output_path": None,
                    "error": AdapterError(category=ErrorCategory.TIMEOUT, message="provider timeout"),
                },
            )()

    monkeypatch.setenv("PIXELLE_PROVIDER_MAX_RETRIES", "2")
    monkeypatch.setenv("PIXELLE_PROVIDER_RETRY_BASE_DELAY", "0")
    monkeypatch.setattr("pixelle_snapshot.adapters.is_capability_available", lambda name, **kwargs: True)
    monkeypatch.setattr("pixelle_snapshot.adapters.get_adapter", lambda name, **kwargs: AlwaysTimeoutAdapter())

    result = _resolve_pixelle_asset(
        segment=_make_segment(),
        project_root=str(tmp_path),
        generated_dir=str(tmp_path / "generated"),
        effective_capability="digital_human",
    )

    assert attempts["count"] == 2
    assert result.fallback_reason_code == "PIXELLE_INVOCATION_FAILED"
    assert result.fallback_error_category == ErrorCategory.TIMEOUT.value


# ============================================================================
# Tests for PixelleRetryPolicy.from_env() - covers lines 81-94
# ============================================================================


def test_from_env_uses_defaults_when_no_env_vars_set(monkeypatch) -> None:
    monkeypatch.delenv("PIXELLE_PROVIDER_MAX_RETRIES", raising=False)
    monkeypatch.delenv("PIXELLE_PROVIDER_RETRY_BASE_DELAY", raising=False)
    monkeypatch.delenv("PIXELLE_PROVIDER_RETRY_MAX_DELAY", raising=False)
    monkeypatch.delenv("PIXELLE_PROVIDER_RETRY_EXP_BASE", raising=False)

    policy = PixelleRetryPolicy.from_env()

    assert policy.max_attempts == 3
    assert policy.base_delay_seconds == 1.0
    assert policy.max_delay_seconds == 8.0
    assert policy.exponential_base == 2.0


def test_from_env_reads_all_env_vars_correctly(monkeypatch) -> None:
    monkeypatch.setenv("PIXELLE_PROVIDER_MAX_RETRIES", "5")
    monkeypatch.setenv("PIXELLE_PROVIDER_RETRY_BASE_DELAY", "0.5")
    monkeypatch.setenv("PIXELLE_PROVIDER_RETRY_MAX_DELAY", "10.0")
    monkeypatch.setenv("PIXELLE_PROVIDER_RETRY_EXP_BASE", "3.0")

    policy = PixelleRetryPolicy.from_env()

    assert policy.max_attempts == 5
    assert policy.base_delay_seconds == 0.5
    assert policy.max_delay_seconds == 10.0
    assert policy.exponential_base == 3.0


def test_from_env_clamps_max_delay_to_base_delay_if_smaller(monkeypatch) -> None:
    monkeypatch.setenv("PIXELLE_PROVIDER_RETRY_BASE_DELAY", "5.0")
    monkeypatch.setenv("PIXELLE_PROVIDER_RETRY_MAX_DELAY", "2.0")

    policy = PixelleRetryPolicy.from_env()

    assert policy.base_delay_seconds == 5.0
    assert policy.max_delay_seconds == 5.0


def test_from_env_enforces_minimum_values(monkeypatch) -> None:
    monkeypatch.setenv("PIXELLE_PROVIDER_MAX_RETRIES", "0")
    monkeypatch.setenv("PIXELLE_PROVIDER_RETRY_BASE_DELAY", "-1.0")
    monkeypatch.setenv("PIXELLE_PROVIDER_RETRY_MAX_DELAY", "-5.0")
    monkeypatch.setenv("PIXELLE_PROVIDER_RETRY_EXP_BASE", "0.5")

    policy = PixelleRetryPolicy.from_env()

    assert policy.max_attempts == 1
    assert policy.base_delay_seconds == 0.0
    assert policy.max_delay_seconds == 0.0
    assert policy.exponential_base == 1.0


def test_from_env_handles_invalid_int_gracefully(monkeypatch) -> None:
    monkeypatch.setenv("PIXELLE_PROVIDER_MAX_RETRIES", "not_a_number")

    policy = PixelleRetryPolicy.from_env()

    assert policy.max_attempts == 3


def test_from_env_handles_invalid_float_gracefully(monkeypatch) -> None:
    monkeypatch.setenv("PIXELLE_PROVIDER_RETRY_BASE_DELAY", "invalid")
    monkeypatch.setenv("PIXELLE_PROVIDER_RETRY_MAX_DELAY", "also_invalid")

    policy = PixelleRetryPolicy.from_env()

    assert policy.base_delay_seconds == 1.0
    assert policy.max_delay_seconds == 8.0


# ============================================================================
# Tests for _normalize_category() - covers lines 108-148
# ============================================================================


def test_normalize_category_returns_execution_when_error_is_none() -> None:
    classification = classify_provider_error(None)

    assert classification.category == ErrorCategory.EXECUTION.value
    assert classification.retryable is True


def test_normalize_category_adapts_adapter_error_with_provider_category_and_auth_marker() -> None:
    err = AdapterError(category=ErrorCategory.PROVIDER, message="forbidden credential error")
    classification = classify_provider_error(err)

    assert classification.category == ErrorCategory.VALIDATION.value
    assert classification.retryable is False


def test_normalize_category_preserves_non_provider_adapter_error_category() -> None:
    err = AdapterError(category=ErrorCategory.TIMEOUT, message="connection timeout")
    classification = classify_provider_error(err)

    assert classification.category == ErrorCategory.TIMEOUT.value
    assert classification.retryable is True


def test_normalize_category_maps_status_code_400_to_validation() -> None:
    class CustomError(Exception):
        details = {"status_code": 400}

    err = CustomError("bad request")
    classification = classify_provider_error(err)

    assert classification.category == ErrorCategory.VALIDATION.value
    assert classification.retryable is False


def test_normalize_category_maps_status_code_401_to_validation() -> None:
    class CustomError(Exception):
        details = {"status_code": 401}

    classification = classify_provider_error(CustomError("unauthorized"))
    assert classification.category == ErrorCategory.VALIDATION.value


def test_normalize_category_maps_status_code_403_to_validation() -> None:
    class CustomError(Exception):
        details = {"status_code": 403}

    classification = classify_provider_error(CustomError("forbidden"))
    assert classification.category == ErrorCategory.VALIDATION.value


def test_normalize_category_maps_status_code_422_to_validation() -> None:
    class CustomError(Exception):
        details = {"status_code": 422}

    classification = classify_provider_error(CustomError("unprocessable"))
    assert classification.category == ErrorCategory.VALIDATION.value


def test_normalize_category_maps_status_code_404_to_missing_input() -> None:
    class CustomError(Exception):
        details = {"status_code": 404}

    classification = classify_provider_error(CustomError("not found"))
    assert classification.category == ErrorCategory.MISSING_INPUT.value
    assert classification.retryable is False


def test_normalize_category_maps_status_code_408_to_timeout() -> None:
    class CustomError(Exception):
        details = {"status_code": 408}

    classification = classify_provider_error(CustomError("request timeout"))
    assert classification.category == ErrorCategory.TIMEOUT.value
    assert classification.retryable is True


def test_normalize_category_maps_status_code_504_to_timeout() -> None:
    class CustomError(Exception):
        details = {"status_code": 504}

    classification = classify_provider_error(CustomError("gateway timeout"))
    assert classification.category == ErrorCategory.TIMEOUT.value


def test_normalize_category_maps_status_code_429_to_resource() -> None:
    class CustomError(Exception):
        details = {"status_code": 429}

    classification = classify_provider_error(CustomError("rate limited"))
    assert classification.category == ErrorCategory.RESOURCE.value
    assert classification.retryable is True


def test_normalize_category_maps_5xx_status_to_provider() -> None:
    class CustomError(Exception):
        details = {"status_code": 500}

    classification = classify_provider_error(CustomError("internal server error"))
    assert classification.category == ErrorCategory.PROVIDER.value
    assert classification.retryable is True


def test_normalize_category_maps_503_status_to_provider() -> None:
    class CustomError(Exception):
        details = {"status_code": 503}

    classification = classify_provider_error(CustomError("service unavailable"))
    assert classification.category == ErrorCategory.PROVIDER.value


def test_normalize_category_detects_auth_marker_in_message() -> None:
    class CustomError(Exception):
        pass

    classification = classify_provider_error(CustomError("invalid token provided"))
    assert classification.category == ErrorCategory.VALIDATION.value
    assert classification.retryable is False


def test_normalize_category_detects_missing_input_marker() -> None:
    class CustomError(Exception):
        pass

    classification = classify_provider_error(CustomError("input file not found"))
    assert classification.category == ErrorCategory.MISSING_INPUT.value


def test_normalize_category_detects_unsupported_marker() -> None:
    class CustomError(Exception):
        pass

    classification = classify_provider_error(CustomError("unsupported format"))
    assert classification.category == ErrorCategory.UNSUPPORTED.value
    assert classification.retryable is False


def test_normalize_category_detects_timeout_marker() -> None:
    class CustomError(Exception):
        pass

    classification = classify_provider_error(CustomError("operation timed out"))
    assert classification.category == ErrorCategory.TIMEOUT.value


def test_normalize_category_detects_resource_marker() -> None:
    class CustomError(Exception):
        pass

    classification = classify_provider_error(CustomError("rate limit exceeded"))
    assert classification.category == ErrorCategory.RESOURCE.value


def test_normalize_category_detects_validation_marker() -> None:
    class CustomError(Exception):
        pass

    classification = classify_provider_error(CustomError("invalid request format"))
    assert classification.category == ErrorCategory.VALIDATION.value


def test_normalize_category_defaults_to_provider_when_no_markers_match() -> None:
    class CustomError(Exception):
        pass

    classification = classify_provider_error(CustomError("unknown error occurred"))
    assert classification.category == ErrorCategory.PROVIDER.value
    assert classification.retryable is True


def test_normalize_category_handles_non_dict_details_attribute() -> None:
    class CustomError(Exception):
        details = "not a dict"

    classification = classify_provider_error(CustomError("some error"))
    assert classification.category == ErrorCategory.PROVIDER.value


# ============================================================================
# Tests for backoff_seconds edge cases
# ============================================================================


def test_backoff_seconds_handles_zero_attempt_number() -> None:
    policy = PixelleRetryPolicy(base_delay_seconds=1.0, exponential_base=2.0, max_delay_seconds=10.0)

    delay = policy.backoff_seconds(0)

    assert delay == 1.0


def test_backoff_seconds_handles_negative_attempt_number() -> None:
    policy = PixelleRetryPolicy(base_delay_seconds=1.0, exponential_base=2.0, max_delay_seconds=10.0)

    delay = policy.backoff_seconds(-5)

    assert delay == 1.0
