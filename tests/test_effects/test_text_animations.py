from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.effects.text_animations import (
    generate_fade,
    generate_scale,
    generate_slide,
    generate_flashing,
    generate_text_animation,
)
from src.core.models import GlobalStyle, OverlayItem, Segment, VisualPlan
from src.steps.step3_visual_plan import _extract_emphasis_tokens
from src.steps.step5_render import _build_subtitle_filter, _estimate_text_width, _wrap_subtitle_lines


def _font_config() -> dict[str, object]:
    return {
        "font_paths": [
            "/missing/primary-font.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    }


def test_generate_fade_builds_alpha_expression_and_cjk_text() -> None:
    text_filter = generate_fade(
        text="测试文字动画",
        start=1.0,
        duration=3.0,
        color="0xFFFFFF",
        font_size=56,
        x="(w-text_w)/2",
        y="(h-text_h)/2",
        config=_font_config(),
    )

    assert "drawtext=text='测试文字动画'" in text_filter
    assert ":fontcolor_expr=" in text_filter
    assert "0xFFFFFF" in text_filter
    assert "0x00" in text_filter
    assert "0xFF" in text_filter


def test_generate_slide_contains_enter_and_exit_position_expressions() -> None:
    text_filter = generate_slide(
        text="slide demo",
        start=2.0,
        duration=4.0,
        color="0x00FF00",
        font_size=48,
        x="(w-text_w)/2",
        y="h*0.78",
        axis="x",
        distance=300,
        config=_font_config(),
    )

    assert "if(lt(t,2.5)" in text_filter
    assert "if(gt(t,5.5)" in text_filter
    assert ":y=h*0.78" in text_filter


def test_generate_scale_contains_small_to_large_to_small_fontsize_expr() -> None:
    text_filter = generate_scale(
        text="scale demo",
        start=0.5,
        duration=3.0,
        color="0xFFAA00",
        font_size=60,
        x="(w-text_w)/2",
        y="h*0.78",
        min_scale=0.6,
        max_scale=1.3,
        config=_font_config(),
    )

    assert ":fontsize='if(lt(t,2.0)" in text_filter
    assert "36.0" in text_filter
    assert "78.0" in text_filter


def test_generate_flashing_contains_alpha_oscillation_expr() -> None:
    text_filter = generate_flashing(
        text="flashing demo",
        start=1.0,
        duration=3.0,
        color="0xFF0000",
        font_size=50,
        x="(w-text_w)/2",
        y="h*0.8",
        flash_speed=3.0,
        config=_font_config(),
    )

    assert ":fontcolor_expr=" in text_filter
    assert "sin((t-1.0)*3.0*PI)" in text_filter
    assert "191*" in text_filter
    assert "+64" in text_filter


def test_generate_flashing_rejects_invalid_speed() -> None:
    with pytest.raises(ValueError):
        generate_flashing(
            text="bad speed",
            start=0.0,
            duration=2.0,
            color="0xFFFFFF",
            font_size=40,
            x="10",
            y="20",
            flash_speed=15.0,
            config=_font_config(),
        )
    with pytest.raises(ValueError):
        generate_flashing(
            text="bad speed",
            start=0.0,
            duration=2.0,
            color="0xFFFFFF",
            font_size=40,
            x="10",
            y="20",
            flash_speed=-1.0,
            config=_font_config(),
        )


def test_generate_text_animation_dispatches_presets() -> None:
    fade_filter = generate_text_animation(
        animation="fade",
        text="fade",
        start=0.0,
        duration=2.0,
        color="0xFFFFFF",
        font_size=40,
        x="10",
        y="20",
        config=_font_config(),
    )
    slide_filter = generate_text_animation(
        animation="slide",
        text="slide",
        start=0.0,
        duration=2.0,
        color="0xFFFFFF",
        font_size=40,
        x="10",
        y="20",
        config=_font_config(),
    )
    scale_filter = generate_text_animation(
        animation="scale",
        text="scale",
        start=0.0,
        duration=2.0,
        color="0xFFFFFF",
        font_size=40,
        x="10",
        y="20",
        config=_font_config(),
    )

    assert ":fontcolor_expr=" in fade_filter
    assert "if(lt(t," in slide_filter
    assert ":fontsize='if(lt(t," in scale_filter


def test_generate_text_animation_rejects_unknown_preset() -> None:
    with pytest.raises(ValueError):
        generate_text_animation(
            animation="spin",
            text="bad",
            start=0.0,
            duration=2.0,
            color="0xFFFFFF",
            font_size=40,
            x="10",
            y="20",
            config=_font_config(),
        )


def test_picks_font_from_config_cascade(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_exists(path: str) -> bool:
        return path == "/usr/share/fonts/truetype/arphic/uming.ttc"

    monkeypatch.setattr("src.effects.text_animations.Path.exists", lambda p: fake_exists(str(p)))

    text_filter = generate_fade(
        text="测试文字动画",
        start=0.0,
        duration=2.0,
        color="0x112233",
        font_size=40,
        x="10",
        y="20",
        config=_font_config(),
    )

    assert ":fontfile='/usr/share/fonts/truetype/arphic/uming.ttc'" in text_filter


def _make_segment(text: str, overlay: list[OverlayItem] | None = None) -> Segment:
    return Segment(
        segment_key="seg#1",
        content_key="content1",
        index=1,
        start=0.0,
        end=4.0,
        duration=4.0,
        text=text,
        visual_plan=VisualPlan(type="template", overlay=overlay or []),
    )


def test_wrap_long_chinese_subtitle_stays_within_safe_width() -> None:
    long_text = "人工智能正在重塑医疗诊断与教育服务效率这是一个非常长的字幕需要自动换行避免超出画面边界"
    max_width = 780
    lines = _wrap_subtitle_lines(long_text, max_width_px=max_width, font_size=48, max_lines=3)

    assert len(lines) >= 2
    for line in lines:
        assert _estimate_text_width(line.replace("…", ""), 48) <= max_width + 48


def test_extract_emphasis_tokens_accepts_multiple_schema_variants() -> None:
    plan_dict = {
        "subtitle_emphasis": ["增长", {"text": "效率"}],
        "emphasis_words": "创新, 未来",
    }

    tokens = _extract_emphasis_tokens(plan_dict)
    assert "增长" in tokens
    assert "效率" in tokens
    assert "创新" in tokens
    assert "未来" in tokens


def test_subtitle_filter_adds_emphasis_drawtext_layers() -> None:
    segment = _make_segment(
        "AI正在推动效率增长",
        overlay=[OverlayItem(kind="subtitle_emphasis", target="subtitle", strength=0.8, extra={"tokens": ["效率", "增长"]})],
    )
    style = GlobalStyle(subtitle_style="clean", font_size=48)

    subtitle_filter = _build_subtitle_filter(segment, style, width=1080, height=1920)

    assert "drawtext=text='效率'" in subtitle_filter
    assert "drawtext=text='增长'" in subtitle_filter
    assert ":fontcolor=yellow" in subtitle_filter
    assert ":borderw=2" in subtitle_filter


def test_subtitle_filter_does_not_render_literal_n_for_wrapped_lines() -> None:
    segment = _make_segment(
        "这是一个很长的中文句子用于测试换行后不会把换行符渲染成字母n并保持显示正常"
    )
    style = GlobalStyle(subtitle_style="clean", font_size=48)

    subtitle_filter = _build_subtitle_filter(segment, style, width=1080, height=1920)

    assert "drawtext=text='" in subtitle_filter
    assert r"\n" in subtitle_filter
    assert "\\\\n" not in subtitle_filter


def test_subtitle_filter_uses_safe_horizontal_margin() -> None:
    segment = _make_segment("测试字幕安全边距")
    style = GlobalStyle(subtitle_style="clean", font_size=48)

    subtitle_filter = _build_subtitle_filter(segment, style, width=1080, height=1920)

    assert r":x=max(43\\,(w-text_w)/2)" in subtitle_filter


def test_subtitle_filter_escapes_commas_in_ffmpeg_drawtext() -> None:
    segment = _make_segment("第一行,第二行")
    style = GlobalStyle(subtitle_style="clean", font_size=48)

    subtitle_filter = _build_subtitle_filter(segment, style, width=1080, height=1920)

    assert r"第一行\,第二行" in subtitle_filter


def test_subtitle_filter_ignores_overlong_emphasis_token() -> None:
    text = "AI正在推动效率增长"
    segment = _make_segment(
        text,
        overlay=[
            OverlayItem(
                kind="subtitle_emphasis",
                target="subtitle",
                strength=0.8,
                extra={"tokens": [text, "效率"]},
            )
        ],
    )
    style = GlobalStyle(subtitle_style="clean", font_size=48)

    subtitle_filter = _build_subtitle_filter(segment, style, width=1080, height=1920)

    assert "drawtext=text='效率'" in subtitle_filter
    assert subtitle_filter.count(f"drawtext=text='{text}'") == 1


@pytest.mark.parametrize(
    ("preset", "marker"),
    [
        ("fade", ":fontcolor_expr="),
        ("slide", "if(lt(t,"),
        ("flashing", "sin((t-0.0)*2.0*PI)"),
    ],
)
def test_step5_subtitle_filter_keeps_animation_presets(preset: str, marker: str) -> None:
    segment = _make_segment("这是动画字幕")
    style = GlobalStyle(subtitle_style=preset, font_size=42)

    subtitle_filter = _build_subtitle_filter(segment, style, width=1080, height=1920)

    assert marker in subtitle_filter


# ─────────────────────────────────────────────
# Step3 Sticker Overlay Generation Tests
# ─────────────────────────────────────────────
from src.steps.step3_visual_plan import (
    _should_add_sticker,
    _generate_sticker_overlay,
    _get_fallback_sticker_path,
)


def test_should_add_sticker_every_third_segment() -> None:
    """Stickers added deterministically every 3rd segment (index 2, 5, 8...)."""
    assert not _should_add_sticker(0, "hello world")
    assert not _should_add_sticker(1, "hello world")
    assert _should_add_sticker(2, "hello world")  # 3rd segment (0-indexed: 2)
    assert not _should_add_sticker(3, "hello world")
    assert not _should_add_sticker(4, "hello world")
    assert _should_add_sticker(5, "hello world")  # 6th segment


def test_should_add_sticker_requires_minimum_words() -> None:
    """Stickers require at least 2 words in segment text."""
    assert not _should_add_sticker(2, "single")  # Only 1 word
    assert _should_add_sticker(2, "two words")  # 2 words - passes
    assert _should_add_sticker(2, "three full words")  # 3 words - passes


def test_generate_sticker_overlay_returns_valid_overlay_item() -> None:
    """Generated sticker overlay has required schema fields."""
    overlay = _generate_sticker_overlay(
        segment_index=2,
        segment_text="hello world test",
        segment_duration=5.0,
    )
    
    # May be None if no sticker asset available
    if overlay is None:
        sticker_path = _get_fallback_sticker_path()
        assert sticker_path is None, "Should return overlay when sticker path exists"
        return
    
    # Verify schema
    assert overlay.kind == "sticker"
    assert overlay.target == "video"
    assert isinstance(overlay.extra, dict)
    
    # Required fields for Step5
    extra = overlay.extra
    assert "asset_path" in extra
    assert "anchor" in extra
    assert "scale" in extra
    assert "transparency" in extra
    assert "start_time" in extra
    assert "duration" in extra
    
    # Value constraints
    assert isinstance(extra["asset_path"], str)
    assert extra["anchor"] in ["center", "top-right", "bottom-left", "top-left", "bottom-right"]
    assert 0 < extra["scale"] <= 1.0
    assert 0 <= extra["transparency"] <= 1.0
    assert extra["start_time"] >= 0
    assert extra["duration"] > 0


def test_generate_sticker_overlay_timing_bounded_by_duration() -> None:
    """Sticker timing must not exceed segment duration."""
    overlay = _generate_sticker_overlay(
        segment_index=2,
        segment_text="hello world test",
        segment_duration=3.0,
    )
    
    if overlay is None:
        return  # No sticker asset available
    
    extra = overlay.extra
    # start_time + duration should not exceed segment_duration
    end_time = extra["start_time"] + extra["duration"]
    assert end_time <= 3.0, f"Sticker ends at {end_time}, exceeds segment duration 3.0"


def test_generate_sticker_overlay_returns_none_for_non_qualifying_segments() -> None:
    """Non-qualifying segments should not get sticker overlays."""
    # Not 3rd segment
    overlay1 = _generate_sticker_overlay(
        segment_index=0,
        segment_text="hello world",
        segment_duration=5.0,
    )
    assert overlay1 is None
    
    # Single word text
    overlay2 = _generate_sticker_overlay(
        segment_index=2,
        segment_text="single",
        segment_duration=5.0,
    )
    assert overlay2 is None


def test_sticker_anchor_deterministic_by_index() -> None:
    """Anchor selection is deterministic based on segment index."""
    # Same index should always produce same anchor
    overlay1 = _generate_sticker_overlay(2, "hello world", 5.0)
    overlay2 = _generate_sticker_overlay(2, "different text", 5.0)
    
    if overlay1 and overlay2:
        assert overlay1.extra["anchor"] == overlay2.extra["anchor"]


def test_should_add_sticker_supports_chinese_text() -> None:
    """Chinese/CJK text should be eligible based on character count, not word split."""
    # Chinese text with 4+ CJK characters should pass
    assert _should_add_sticker(2, "这是中文字幕")  # 6 CJK chars
    assert _should_add_sticker(2, "人工智能")  # 4 CJK chars - minimum
    
    # Less than 4 CJK chars should fail
    assert not _should_add_sticker(2, "中文")  # Only 2 CJK chars
    assert not _should_add_sticker(2, "测试")  # Only 2 CJK chars
    
    # Mixed text: CJK count takes priority if >= 4
    assert _should_add_sticker(2, "AI人工智能")  # 4 CJK chars


def test_generate_sticker_overlay_short_segment_strict_timing() -> None:
    """Very short segments must have strict timing bound: end_time <= segment_duration."""
    # Test with very short segment (0.6s)
    overlay = _generate_sticker_overlay(
        segment_index=2,
        segment_text="hello world",
        segment_duration=0.6,
    )
    
    if overlay is None:
        return  # No sticker asset available
    
    extra = overlay.extra
    end_time = extra["start_time"] + extra["duration"]
    assert end_time <= 0.6, f"Sticker ends at {end_time}, exceeds segment duration 0.6"
    
    # Test boundary: segment too short for sticker (< 0.5s)
    overlay_too_short = _generate_sticker_overlay(
        segment_index=2,
        segment_text="hello world",
        segment_duration=0.4,
    )
    assert overlay_too_short is None, "Should skip sticker for segments < 0.5s"


# ─────────────────────────────────────────────
# SRT Utils Text Splitting and Width Tests
# ─────────────────────────────────────────────
from src.utils.srt_utils import (
    split_text_deterministic,
    merge_short_segments,
    split_long_segments,
    estimate_text_width_px,
)
from src.core.models import SRTEntry


def test_split_text_deterministic_punctuation_split() -> None:
    """split_text_deterministic splits long text at Chinese punctuation."""
    # Long text that needs splitting, with punctuation
    text = "这是第一句，这是第二句。这是第三句！这是第四句？这是第五句；这是第六句："
    safe_width = 400  # Small width to force splits
    font_size = 48
    
    lines = split_text_deterministic(text, safe_width, font_size)
    
    # Should split into multiple lines
    assert len(lines) > 1
    
    # Each line should respect width constraint
    for line in lines:
        width = estimate_text_width_px(line, font_size)
        assert width <= safe_width + font_size, f"Line '{line}' exceeds safe width: {width} > {safe_width}"
    
    # Verify text preserved after rejoining
    rejoined = "".join(lines).replace(" ", "")
    original = text.replace(" ", "")
    assert rejoined == original


def test_split_text_deterministic_word_boundary() -> None:
    """split_text_deterministic splits long English text at word boundaries (spaces)."""
    text = "The quick brown fox jumps over the lazy dog and continues running through the forest"
    safe_width = 500  # Small width to force splits
    font_size = 48
    
    lines = split_text_deterministic(text, safe_width, font_size)
    
    # Should split into multiple lines
    assert len(lines) > 1
    
    # Each line should respect width constraint
    for line in lines:
        width = estimate_text_width_px(line, font_size)
        assert width <= safe_width + font_size, f"Line '{line}' exceeds safe width: {width} > {safe_width}"
    
    # Verify text preserved after rejoining
    rejoined = " ".join(lines)
    assert rejoined == text


def test_split_text_deterministic_no_punctuation() -> None:
    """split_text_deterministic handles text with no punctuation or spaces (hard split)."""
    # Long continuous text with no punctuation marks or spaces
    text = "人工智能技术正在重塑医疗诊断教育服务效率提升这是一个非常长的字幕需要自动换行避免超出画面边界"
    safe_width = 400  # Small width to force splits
    font_size = 48
    
    lines = split_text_deterministic(text, safe_width, font_size)
    
    # Should split into multiple lines even without punctuation
    assert len(lines) > 1
    
    # Each line should respect width constraint (with small tolerance for edge cases)
    for line in lines:
        width = estimate_text_width_px(line, font_size)
        # Allow one character overflow for hard splits
        assert width <= safe_width + font_size * 1.5, f"Line '{line}' exceeds safe width: {width} > {safe_width}"
    
    # Verify no text lost
    rejoined = "".join(lines)
    assert len(rejoined) == len(text)


def test_merge_short_segments_respects_safe_width() -> None:
    """merge_short_segments does not merge if resulting text exceeds safe width."""
    # Create two short segments where merging would exceed safe width
    entries = [
        SRTEntry(index=1, start=0.0, end=0.8, text="这是一个非常长的字幕文本内容"),
        SRTEntry(index=2, start=0.8, end=1.5, text="这是第二条非常长的字幕文本"),
    ]
    
    video_width = 1080
    font_size = 48
    safe_width = int(video_width * 0.92)  # 993px
    
    # First segment is short (<1.0s), but merging would make text too wide
    merged = merge_short_segments(
        entries, 
        min_duration=1.0, 
        video_width=video_width, 
        font_size=font_size
    )
    
    # Should NOT merge due to width constraint
    assert len(merged) == 2, "Should not merge when combined text exceeds safe width"
    assert merged[0].text == "这是一个非常长的字幕文本内容"
    assert merged[1].text == "这是第二条非常长的字幕文本"


def test_split_long_segments_uses_safe_width() -> None:
    """split_long_segments splits based on safe width, not just duration."""
    # Create a segment with very long text that exceeds safe width
    long_text = "人工智能正在重塑医疗诊断与教育服务效率这是一个非常长的字幕需要自动换行避免超出画面边界"
    
    entries = [
        SRTEntry(index=1, start=0.0, end=5.0, text=long_text),  # Duration OK but text too wide
    ]
    
    video_width = 1080
    font_size = 48
    safe_width = int(video_width * 0.92)  # 993px
    
    # Text width exceeds safe_width
    text_width = estimate_text_width_px(long_text, font_size)
    assert text_width > safe_width, "Test setup: text should exceed safe width"
    
    # split_long_segments should split based on width even if duration is acceptable
    result = split_long_segments(
        entries,
        max_duration=8.0,  # Duration is fine (5.0 < 8.0)
        video_width=video_width,
        font_size=font_size
    )
    
    # Should split into multiple segments due to width constraint
    assert len(result) > 1, "Should split long text even if duration is acceptable"
    
    # Each segment's text should respect safe width
    for seg in result:
        seg_width = estimate_text_width_px(seg.text, font_size)
        assert seg_width <= safe_width + font_size, f"Segment text '{seg.text}' exceeds safe width: {seg_width} > {safe_width}"


# ─────────────────────────────────────────────
# Effect Toggle Regression Tests (enable_subtitle_effects)
# ─────────────────────────────────────────────
def test_enable_subtitle_effects_false_removes_emphasis_filters() -> None:
    """When enable_subtitle_effects=False, no yellow emphasis drawtext filters are added."""
    segment = _make_segment(
        "AI正在推动效率增长",
        overlay=[
            OverlayItem(
                kind="subtitle_emphasis",
                target="subtitle",
                strength=0.8,
                extra={"tokens": ["效率", "增长"]},
            )
        ],
    )
    style = GlobalStyle(subtitle_style="clean", font_size=48, enable_subtitle_effects=False)

    subtitle_filter = _build_subtitle_filter(segment, style, width=1080, height=1920)

    # Main subtitle should still be present
    assert "drawtext=text='AI正在推动效率增长'" in subtitle_filter or "drawtext=text='AI正在推动效率增长\\nAI正在推动效率增长'" in subtitle_filter

    # Emphasis tokens should NOT be rendered in yellow
    assert "drawtext=text='效率'" not in subtitle_filter
    assert "drawtext=text='增长'" not in subtitle_filter
    assert ":fontcolor=yellow" not in subtitle_filter


def test_enable_subtitle_effects_true_preserves_emphasis_filters() -> None:
    """When enable_subtitle_effects=True (default), yellow emphasis filters are preserved."""
    segment = _make_segment(
        "AI正在推动效率增长",
        overlay=[
            OverlayItem(
                kind="subtitle_emphasis",
                target="subtitle",
                strength=0.8,
                extra={"tokens": ["效率", "增长"]},
            )
        ],
    )
    style = GlobalStyle(subtitle_style="clean", font_size=48, enable_subtitle_effects=True)

    subtitle_filter = _build_subtitle_filter(segment, style, width=1080, height=1920)

    # Main subtitle should be present
    assert "drawtext=text='AI正在推动效率增长'" in subtitle_filter or "drawtext=text='AI正在推动效率增长\\nAI正在推动效率增长'" in subtitle_filter

    # Emphasis tokens SHOULD be rendered in yellow
    assert "drawtext=text='效率'" in subtitle_filter
    assert "drawtext=text='增长'" in subtitle_filter
    assert ":fontcolor=yellow" in subtitle_filter
    assert ":borderw=2" in subtitle_filter
