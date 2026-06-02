from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.effects.stickers import (
    calculate_position,
    generate_overlay_filter,
    validate_gif,
)


def _create_gif(path: Path, size: tuple[int, int]) -> None:
    image = Image.new("RGBA", size, (255, 0, 0, 180))
    image.save(path, format="GIF")


def test_validate_gif_accepts_valid_gif(tmp_path: Path) -> None:
    gif_path = tmp_path / "ok.gif"
    _create_gif(gif_path, (320, 240))

    result = validate_gif(str(gif_path))

    assert result["valid"] is True
    assert result["format"] == "GIF"
    assert result["width"] == 320
    assert result["height"] == 240


def test_validate_gif_rejects_non_gif(tmp_path: Path) -> None:
    png_path = tmp_path / "bad.png"
    Image.new("RGB", (64, 64), (0, 0, 0)).save(png_path, format="PNG")

    result = validate_gif(str(png_path))

    assert result["valid"] is False
    assert "gif" in result["error"].lower()


def test_validate_gif_rejects_large_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gif_path = tmp_path / "large.gif"
    _create_gif(gif_path, (320, 240))

    monkeypatch.setattr("src.effects.stickers.os.path.getsize", lambda _: 10 * 1024 * 1024 + 1)

    result = validate_gif(str(gif_path))

    assert result == {
        "valid": False,
        "error": "File exceeds 10MB limit",
    }


def test_validate_gif_rejects_large_resolution(tmp_path: Path) -> None:
    gif_path = tmp_path / "huge.gif"
    _create_gif(gif_path, (1921, 1080))

    result = validate_gif(str(gif_path))

    assert result["valid"] is False
    assert "resolution" in result["error"].lower()


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        ("top-left", (0, 0)),
        ("top-center", (440, 0)),
        ("top-right", (880, 0)),
        ("center-left", (0, 910)),
        ("center", (440, 910)),
        ("center-right", (880, 910)),
        ("bottom-left", (0, 1820)),
        ("bottom-center", (440, 1820)),
        ("bottom-right", (880, 1820)),
    ],
)
def test_calculate_anchor_position(anchor: str, expected: tuple[int, int]) -> None:
    assert calculate_position(anchor, 1080, 1920, 200, 100) == expected


def test_calculate_anchor_position_rejects_unknown_anchor() -> None:
    with pytest.raises(ValueError):
        calculate_position("unknown", 1080, 1920, 200, 100)


def test_build_overlay_filter_contains_enable_between_expression() -> None:
    filter_text = generate_overlay_filter(
        sticker_stream="[1:v]",
        base_stream="[0:v]",
        output_stream="[vout]",
        anchor="center",
        start_time=1.0,
        duration=2.5,
        scale=0.5,
        transparency=0.8,
        video_width=1080,
        video_height=1920,
        sticker_width=200,
        sticker_height=100,
    )

    assert "enable='between(t,1.0,3.5)'" in filter_text


def test_build_overlay_filter_contains_scale_and_transparency() -> None:
    filter_text = generate_overlay_filter(
        sticker_stream="[1:v]",
        base_stream="[0:v]",
        output_stream="[vout]",
        anchor="top-right",
        start_time=0.0,
        duration=1.0,
        scale=0.75,
        transparency=0.45,
        video_width=1080,
        video_height=1920,
        sticker_width=200,
        sticker_height=100,
    )

    assert "format=rgba" in filter_text
    assert "scale=iw*0.75:ih*0.75" in filter_text
    assert "colorchannelmixer=aa=0.45" in filter_text
    assert "overlay=930:0" in filter_text


def test_build_overlay_filter_rejects_zero_duration() -> None:
    with pytest.raises(ValueError):
        generate_overlay_filter(
            sticker_stream="[1:v]",
            base_stream="[0:v]",
            output_stream="[vout]",
            anchor="center",
            start_time=1.0,
            duration=0.0,
            scale=1.0,
            transparency=1.0,
            video_width=1080,
            video_height=1920,
            sticker_width=200,
            sticker_height=100,
        )


# Integration tests for Step5 sticker rendering


def test_step5_extract_sticker_effects_from_segment(tmp_path: Path) -> None:
    """Test extraction of sticker metadata from segment visual_plan."""
    from src.core.models import Segment, VisualPlan, OverlayItem
    from src.steps.step5_render import _extract_sticker_effects

    sticker_path = tmp_path / "sticker.gif"
    _create_gif(sticker_path, (100, 100))

    segment = Segment(
        segment_key="test#1",
        content_key="test",
        index=1,
        start=0.0,
        end=3.0,
        duration=3.0,
        text="Test segment",
        visual_plan=VisualPlan(
            overlay=[
                OverlayItem(
                    kind="sticker",
                    extra={
                        "asset_path": str(sticker_path),
                        "anchor": "top-right",
                        "scale": 0.5,
                        "transparency": 0.9,
                        "start_time": 0.5,
                        "duration": 2.0,
                    },
                )
            ]
        ),
    )

    result = _extract_sticker_effects(segment)

    assert len(result) == 1
    assert result[0]["asset_path"] == str(sticker_path)
    assert result[0]["anchor"] == "top-right"
    assert result[0]["scale"] == 0.5
    assert result[0]["transparency"] == 0.9
    assert result[0]["start_time"] == 0.5
    assert result[0]["duration"] == 2.0


def test_step5_extract_sticker_ignores_non_sticker_overlay(tmp_path: Path) -> None:
    """Test that non-sticker overlay items are ignored."""
    from src.core.models import Segment, VisualPlan, OverlayItem
    from src.steps.step5_render import _extract_sticker_effects

    segment = Segment(
        segment_key="test#1",
        content_key="test",
        index=1,
        start=0.0,
        end=3.0,
        duration=3.0,
        text="Test segment",
        visual_plan=VisualPlan(
            overlay=[
                OverlayItem(kind="subtitle_emphasis", extra={"tokens": ["test"]}),
                OverlayItem(kind="highlight", extra={"target": "center"}),
            ]
        ),
    )

    result = _extract_sticker_effects(segment)

    assert len(result) == 0


def test_step5_extract_sticker_handles_missing_asset_path(tmp_path: Path) -> None:
    """Test that stickers without asset_path are skipped with warning."""
    from src.core.models import Segment, VisualPlan, OverlayItem
    from src.steps.step5_render import _extract_sticker_effects

    segment = Segment(
        segment_key="test#1",
        content_key="test",
        index=1,
        start=0.0,
        end=3.0,
        duration=3.0,
        text="Test segment",
        visual_plan=VisualPlan(
            overlay=[
                OverlayItem(kind="sticker", extra={"anchor": "center"}),
            ]
        ),
    )

    result = _extract_sticker_effects(segment)

    assert len(result) == 0


def test_step5_build_sticker_filter_validates_gif(tmp_path: Path) -> None:
    """Test that _build_sticker_filter validates GIF and returns None for invalid files."""
    from src.steps.step5_render import _build_sticker_filter

    # Invalid file (doesn't exist)
    result = _build_sticker_filter(
        "/nonexistent/sticker.gif", 1, 0.5, 1.0, "center", 0.0, 2.0, 1080, 1920
    )

    assert result is None


def test_step5_build_sticker_filter_returns_correct_tuple(tmp_path: Path) -> None:
    """Test that _build_sticker_filter returns correct tuple for valid GIF."""
    from src.steps.step5_render import _build_sticker_filter

    sticker_path = tmp_path / "valid.gif"
    _create_gif(sticker_path, (200, 100))

    result = _build_sticker_filter(
        str(sticker_path), 1, 0.5, 0.8, "center", 0.0, 2.0, 1080, 1920
    )

    assert result is not None
    filter_str, x_pos, y_pos, start_time, end_time, sticker_index = result

    # Check filter string contains expected elements
    assert "[1:v]format=rgba" in filter_str
    assert "scale=iw*0.5:ih*0.5" in filter_str
    assert "colorchannelmixer=aa=0.8" in filter_str
    assert "[stk1]" in filter_str

    # Check position (centered, 200*0.5=100 width, 100*0.5=50 height)
    assert x_pos == int((1080 - 100) / 2)
    assert y_pos == int((1920 - 50) / 2)

    # Check timing
    assert start_time == 0.0
    assert end_time == 2.0
    assert sticker_index == 1


def test_step5_build_sticker_filter_handles_different_anchors(tmp_path: Path) -> None:
    """Test anchor positioning logic in _build_sticker_filter."""
    from src.steps.step5_render import _build_sticker_filter

    sticker_path = tmp_path / "valid.gif"
    _create_gif(sticker_path, (200, 100))

    # Test top-left
    result = _build_sticker_filter(
        str(sticker_path), 1, 1.0, 1.0, "top-left", 0.0, 1.0, 1080, 1920
    )
    assert result is not None
    _, x_pos, y_pos, _, _, _ = result
    assert x_pos == 0
    assert y_pos == 0

    # Test bottom-right
    result = _build_sticker_filter(
        str(sticker_path), 1, 1.0, 1.0, "bottom-right", 0.0, 1.0, 1080, 1920
    )
    assert result is not None
    _, x_pos, y_pos, _, _, _ = result
    assert x_pos == 1080 - 200
    assert y_pos == 1920 - 100


def test_step5_subtitle_emphasis_tokens_are_deduped_without_extra_lines() -> None:
    from src.core.models import GlobalStyle, OverlayItem, Segment, VisualPlan
    from src.steps.step5_render import _build_subtitle_filter

    text = "AI正在推动效率增长"
    segment = Segment(
        segment_key="test#1",
        content_key="test",
        index=1,
        start=0.0,
        end=4.0,
        duration=4.0,
        text=text,
        visual_plan=VisualPlan(
            type="template",
            overlay=[
                OverlayItem(
                    kind="subtitle_emphasis",
                    target="subtitle",
                    strength=0.8,
                    extra={"tokens": ["效率", "效率", "增长", text]},
                )
            ],
        ),
    )
    style = GlobalStyle(subtitle_style="clean", font_size=48)

    subtitle_filter = _build_subtitle_filter(segment, style, width=1080, height=1920)

    assert subtitle_filter.count("drawtext=text='效率'") == 1
    assert subtitle_filter.count("drawtext=text='增长'") == 1
    assert subtitle_filter.count(f"drawtext=text='{text}'") == 1
