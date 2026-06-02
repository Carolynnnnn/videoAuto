from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

from pixelle_snapshot.adapters.contracts import AdapterError, ErrorCategory, normalize_error_category


ERROR_RETRY_MATRIX: Dict[str, bool] = {
    ErrorCategory.VALIDATION.value: False,
    ErrorCategory.MISSING_INPUT.value: False,
    ErrorCategory.UNSUPPORTED.value: False,
    ErrorCategory.EXECUTION.value: True,
    ErrorCategory.TIMEOUT.value: True,
    ErrorCategory.RESOURCE.value: True,
    ErrorCategory.PROVIDER.value: True,
}


_AUTH_MARKERS = (
    "auth",
    "unauthorized",
    "forbidden",
    "credential",
    "api key",
    "token",
    "permission denied",
)

_VALIDATION_MARKERS = (
    "invalid",
    "malformed",
    "bad request",
    "unprocessable",
    "missing required",
)

_MISSING_INPUT_MARKERS = (
    "missing input",
    "input file",
    "not found",
    "no such file",
)

_UNSUPPORTED_MARKERS = (
    "unsupported",
    "not implemented",
)

_TIMEOUT_MARKERS = (
    "timeout",
    "timed out",
    "deadline exceeded",
)

_RESOURCE_MARKERS = (
    "rate limit",
    "too many requests",
    "quota",
    "out of memory",
    "insufficient",
)


@dataclass(frozen=True)
class ProviderErrorClassification:
    category: str
    retryable: bool


@dataclass(frozen=True)
class PixelleRetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 8.0
    exponential_base: float = 2.0

    @classmethod
    def from_env(cls) -> "PixelleRetryPolicy":
        max_attempts = _read_int_env("PIXELLE_PROVIDER_MAX_RETRIES", 3, minimum=1)
        base_delay_seconds = _read_float_env("PIXELLE_PROVIDER_RETRY_BASE_DELAY", 1.0, minimum=0.0)
        max_delay_seconds = _read_float_env("PIXELLE_PROVIDER_RETRY_MAX_DELAY", 8.0, minimum=0.0)
        exponential_base = _read_float_env("PIXELLE_PROVIDER_RETRY_EXP_BASE", 2.0, minimum=1.0)

        if max_delay_seconds < base_delay_seconds:
            max_delay_seconds = base_delay_seconds

        return cls(
            max_attempts=max_attempts,
            base_delay_seconds=base_delay_seconds,
            max_delay_seconds=max_delay_seconds,
            exponential_base=exponential_base,
        )

    def backoff_seconds(self, attempt_number: int) -> float:
        exponent = max(0, attempt_number - 1)
        delay = self.base_delay_seconds * (self.exponential_base ** exponent)
        return min(delay, self.max_delay_seconds)


def classify_provider_error(error: Any) -> ProviderErrorClassification:
    category = _normalize_category(error)
    return ProviderErrorClassification(category=category, retryable=ERROR_RETRY_MATRIX.get(category, True))


def _normalize_category(error: Any) -> str:
    if error is None:
        return ErrorCategory.EXECUTION.value

    if isinstance(error, AdapterError):
        category = normalize_error_category(error.category)
        if category == ErrorCategory.PROVIDER.value:
            message = f"{type(error).__name__} {error}".lower()
            if _contains_any(message, _AUTH_MARKERS):
                return ErrorCategory.VALIDATION.value
        return category

    details = getattr(error, "details", {})
    status_code = details.get("status_code") if isinstance(details, dict) else None

    if status_code in (400, 401, 403, 422):
        return ErrorCategory.VALIDATION.value
    if status_code == 404:
        return ErrorCategory.MISSING_INPUT.value
    if status_code in (408, 504):
        return ErrorCategory.TIMEOUT.value
    if status_code == 429:
        return ErrorCategory.RESOURCE.value
    if isinstance(status_code, int) and status_code >= 500:
        return ErrorCategory.PROVIDER.value

    message = f"{type(error).__name__} {error}".lower()

    if _contains_any(message, _AUTH_MARKERS):
        return ErrorCategory.VALIDATION.value
    if _contains_any(message, _MISSING_INPUT_MARKERS):
        return ErrorCategory.MISSING_INPUT.value
    if _contains_any(message, _UNSUPPORTED_MARKERS):
        return ErrorCategory.UNSUPPORTED.value
    if _contains_any(message, _TIMEOUT_MARKERS):
        return ErrorCategory.TIMEOUT.value
    if _contains_any(message, _RESOURCE_MARKERS):
        return ErrorCategory.RESOURCE.value
    if _contains_any(message, _VALIDATION_MARKERS):
        return ErrorCategory.VALIDATION.value

    return ErrorCategory.PROVIDER.value


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


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
