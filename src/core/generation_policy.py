from __future__ import annotations

from typing import Any, Mapping, Optional, TypedDict


TARGET_DURATION_MINUTES_ALLOWED = {1, 2, 3}
TARGET_DURATION_MINUTES_DEFAULT = 1
AI_CLIP_CAP_DEFAULT = 6


class GenerationPolicy(TypedDict):
    target_duration_minutes: int
    ai_clip_cap: int


def minutes_to_seconds(minutes: int) -> int:
    return minutes * 60


def normalize_target_duration_minutes(value: Optional[Any]) -> int:
    resolved = TARGET_DURATION_MINUTES_DEFAULT if value is None else value
    if resolved not in TARGET_DURATION_MINUTES_ALLOWED:
        allowed = ", ".join(str(v) for v in sorted(TARGET_DURATION_MINUTES_ALLOWED))
        raise ValueError(
            f"Invalid target_duration_minutes '{resolved}'. "
            f"Allowed values: {allowed}"
        )
    return resolved


def normalize_ai_clip_cap(value: Optional[Any]) -> int:
    resolved = AI_CLIP_CAP_DEFAULT if value is None else value
    if not isinstance(resolved, int) or resolved < 1:
        raise ValueError(
            f"Invalid ai_clip_cap '{resolved}'. "
            "Allowed values: positive integer >= 1"
        )
    return resolved


def normalize_generation_policy(
    target_duration_minutes: Optional[Any] = None,
    ai_clip_cap: Optional[Any] = None,
) -> GenerationPolicy:
    return {
        "target_duration_minutes": normalize_target_duration_minutes(target_duration_minutes),
        "ai_clip_cap": normalize_ai_clip_cap(ai_clip_cap),
    }


def normalize_generation_policy_from_mapping(values: Optional[Mapping[str, Any]]) -> GenerationPolicy:
    source = values or {}
    return normalize_generation_policy(
        target_duration_minutes=source.get("target_duration_minutes"),
        ai_clip_cap=source.get("ai_clip_cap"),
    )


def index_tie_break_key(index: int) -> int:
    return index


__all__ = [
    "AI_CLIP_CAP_DEFAULT",
    "GenerationPolicy",
    "TARGET_DURATION_MINUTES_ALLOWED",
    "TARGET_DURATION_MINUTES_DEFAULT",
    "index_tie_break_key",
    "minutes_to_seconds",
    "normalize_ai_clip_cap",
    "normalize_generation_policy",
    "normalize_generation_policy_from_mapping",
    "normalize_target_duration_minutes",
]
