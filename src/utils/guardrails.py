"""
Validation and guardrail utilities for input sanitization and edge case handling.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, List, Any, Union
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger("guardrails")


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_value: Any = None
    
    @classmethod
    def success(cls, value: Any = None, warnings: Optional[List[str]] = None):
        return cls(is_valid=True, errors=[], warnings=warnings or [], sanitized_value=value)
    
    @classmethod
    def failure(cls, errors: List[str], warnings: Optional[List[str]] = None):
        return cls(is_valid=False, errors=errors, warnings=warnings or [])


def validate_file_path(
    path: Union[str, Path],
    must_exist: bool = True,
    allowed_extensions: Optional[List[str]] = None,
    max_size_bytes: Optional[int] = None,
) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    
    try:
        path = Path(path)
    except (TypeError, ValueError) as e:
        return ValidationResult.failure([f"Invalid path format: {e}"])
    
    if must_exist and not path.exists():
        errors.append(f"File does not exist: {path}")
    
    if allowed_extensions and path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
        errors.append(f"Invalid extension: {path.suffix}. Allowed: {allowed_extensions}")
    
    if path.exists():
        if max_size_bytes and path.stat().st_size > max_size_bytes:
            errors.append(f"File too large: {path.stat().st_size} > {max_size_bytes} bytes")
        
        if path.stat().st_size == 0:
            warnings.append(f"File is empty: {path}")
    
    if errors:
        return ValidationResult.failure(errors, warnings)
    return ValidationResult.success(path, warnings)


def validate_url(url: str) -> ValidationResult:
    errors: List[str] = []
    
    if not url:
        return ValidationResult.failure(["URL cannot be empty"])
    
    url_pattern = re.compile(
        r'^https?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )
    
    if not url_pattern.match(url):
        errors.append(f"Invalid URL format: {url}")
    
    if errors:
        return ValidationResult.failure(errors)
    return ValidationResult.success(url)


def validate_time_range(
    start: float,
    end: float,
    min_duration: float = 0.1,
    max_duration: float = 3600.0,
) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    
    if start < 0:
        errors.append(f"Start time cannot be negative: {start}")
    
    if end < start:
        errors.append(f"End time ({end}) must be >= start time ({start})")
    
    duration = end - start
    if duration < min_duration:
        errors.append(f"Duration too short: {duration:.3f}s < {min_duration}s")
    
    if duration > max_duration:
        warnings.append(f"Duration very long: {duration:.3f}s > {max_duration}s")
    
    if errors:
        return ValidationResult.failure(errors, warnings)
    return ValidationResult.success((start, end), warnings)


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    filename = re.sub(r'_+', '_', filename)
    filename = filename.strip('_. ')
    
    if len(filename) > max_length:
        name_part, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        max_name_len = max_length - len(ext) - 1 if ext else max_length
        filename = name_part[:max_name_len] + ('.' + ext if ext else '')
    
    if not filename:
        filename = "unnamed"
    
    return filename


def sanitize_text_for_subtitle(
    text: str,
    max_length: int = 500,
    remove_special_chars: bool = True,
) -> str:
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    
    if remove_special_chars:
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    if len(text) > max_length:
        text = text[:max_length - 3] + "..."
        logger.warning(f"Text truncated to {max_length} chars")
    
    return text


def ensure_positive_number(
    value: Any,
    default: float,
    name: str = "value",
) -> float:
    try:
        num = float(value)
        if num <= 0:
            logger.warning(f"{name} must be positive, got {num}. Using default: {default}")
            return default
        return num
    except (TypeError, ValueError):
        logger.warning(f"Invalid {name}: {value}. Using default: {default}")
        return default


def ensure_in_range(
    value: Any,
    min_val: float,
    max_val: float,
    default: float,
    name: str = "value",
) -> float:
    try:
        num = float(value)
        if num < min_val or num > max_val:
            logger.warning(f"{name} out of range [{min_val}, {max_val}], got {num}. Using default: {default}")
            return default
        return num
    except (TypeError, ValueError):
        logger.warning(f"Invalid {name}: {value}. Using default: {default}")
        return default


def safe_get(
    data: dict,
    key: str,
    default: Any = None,
    expected_type: Optional[type] = None,
) -> Any:
    value = data.get(key, default)
    
    if value is None:
        return default
    
    if expected_type is not None and not isinstance(value, expected_type):
        logger.warning(f"Key '{key}' expected {expected_type.__name__}, got {type(value).__name__}. Using default.")
        return default
    
    return value


def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


class GuardedOperation:
    def __init__(
        self,
        operation_name: str,
        fallback_value: Any = None,
        log_errors: bool = True,
        reraise: bool = False,
    ):
        self.operation_name = operation_name
        self.fallback_value = fallback_value
        self.log_errors = log_errors
        self.reraise = reraise
        self.error: Optional[Exception] = None
    
    def __enter__(self) -> "GuardedOperation":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            self.error = exc_val
            if self.log_errors:
                logger.error(f"[{self.operation_name}] Operation failed: {exc_type.__name__}: {exc_val}")
            
            if self.reraise:
                return False
            return True
        return False
    
    @property
    def succeeded(self) -> bool:
        return self.error is None
    
    def get_result_or_fallback(self, result: Any) -> Any:
        if self.error is not None:
            return self.fallback_value
        return result
