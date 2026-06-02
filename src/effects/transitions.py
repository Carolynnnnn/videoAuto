from __future__ import annotations

SUPPORTED_TRANSITIONS = {
    "fade",
    "dissolve",
    "wipeleft",
    "wiperight",
    "slideup",
    "slidedown",
    "circleopen",
    "circleclose",
}


def _format_seconds(value: float) -> str:
    return f"{round(value, 3):g}"


def calculate_transition_offset(segment_duration: float, transition_duration: float = 0.5) -> float:
    if segment_duration <= 0:
        raise ValueError("Segment duration must be greater than 0")
    if transition_duration <= 0:
        raise ValueError("Transition duration must be greater than 0")
    if transition_duration >= segment_duration:
        raise ValueError("Transition duration must be less than segment duration")
    return round(segment_duration - transition_duration, 3)


def generate_transition(
    transition_type: str,
    segment_duration: float,
    transition_duration: float = 0.5,
) -> str:
    if transition_type not in SUPPORTED_TRANSITIONS:
        raise ValueError(f"Unsupported transition: {transition_type}")

    offset = calculate_transition_offset(segment_duration, transition_duration)
    return (
        f"xfade=transition={transition_type}:"
        f"duration={_format_seconds(transition_duration)}:"
        f"offset={_format_seconds(offset)}"
    )
