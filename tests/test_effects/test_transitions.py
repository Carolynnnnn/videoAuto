from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.effects.transitions import calculate_transition_offset, generate_transition


@pytest.mark.parametrize(
    "transition_type",
    [
        "fade",
        "dissolve",
        "wipeleft",
        "wiperight",
        "slideup",
        "slidedown",
        "circleopen",
        "circleclose",
    ],
)
def test_generate_transition_supports_all_required_types(transition_type: str) -> None:
    result = generate_transition(transition_type, segment_duration=5.0)

    assert result == f"xfade=transition={transition_type}:duration=0.5:offset=4.5"


def test_calculate_transition_offset_uses_segment_end_overlap() -> None:
    assert calculate_transition_offset(segment_duration=7.0, transition_duration=0.5) == 6.5


def test_generate_transition_supports_custom_duration() -> None:
    result = generate_transition("dissolve", segment_duration=6.0, transition_duration=1.2)

    assert result == "xfade=transition=dissolve:duration=1.2:offset=4.8"


def test_transition_duration_must_be_shorter_than_segment_duration() -> None:
    with pytest.raises(ValueError, match="less than segment duration"):
        calculate_transition_offset(segment_duration=0.5, transition_duration=0.5)


def test_generate_transition_rejects_unsupported_type() -> None:
    with pytest.raises(ValueError, match="Unsupported transition"):
        generate_transition("zoom", segment_duration=5.0)
