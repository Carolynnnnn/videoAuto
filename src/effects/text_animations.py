from __future__ import annotations

from pathlib import Path
from typing import Any


def _escape_ffmpeg_text(text: str) -> str:
    escaped = text.replace("\\", "\\\\")
    escaped = escaped.replace("'", "\\'")
    escaped = escaped.replace(":", "\\:")
    escaped = escaped.replace("[", "\\[").replace("]", "\\]")
    escaped = escaped.replace(",", "\\,")
    return escaped


def _normalize_color(color: str) -> str:
    value = color.strip()
    if value.startswith("#"):
        value = "0x" + value[1:]
    if not value.startswith("0x"):
        raise ValueError("Color must use 0xRRGGBB format")
    hex_part = value[2:]
    if len(hex_part) != 6:
        raise ValueError("Color must use 0xRRGGBB format")
    int(hex_part, 16)
    return f"0x{hex_part.upper()}"


def _resolve_font_file(config: dict[str, Any] | None) -> str:
    if not config:
        return ""

    candidates: list[str] = []
    fonts = config.get("fonts")
    if isinstance(fonts, dict):
        for key in ("cjk_fallback", "fallback", "paths"):
            value = fonts.get(key)
            if isinstance(value, list):
                candidates.extend(str(item) for item in value)

    for key in ("cjk_font_paths", "font_paths"):
        value = config.get(key)
        if isinstance(value, list):
            candidates.extend(str(item) for item in value)

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return f":fontfile='{candidate}'"
    return ""


def _base_drawtext(
    text: str,
    font_size: int,
    x: str,
    y: str,
    color: str,
    config: dict[str, Any] | None,
) -> str:
    escaped_text = _escape_ffmpeg_text(text)
    normalized_color = _normalize_color(color)
    font_file = _resolve_font_file(config)
    return (
        f"drawtext=text='{escaped_text}'"
        f":fontsize={font_size}"
        f":fontcolor={normalized_color}FF"
        f":x={x}"
        f":y={y}"
        f"{font_file}"
    )


def generate_fade(
    text: str,
    start: float,
    duration: float,
    color: str,
    font_size: int,
    x: str,
    y: str,
    config: dict[str, Any] | None = None,
) -> str:
    normalized_color = _normalize_color(color)
    rgb = normalized_color[2:]
    end = start + duration
    fade_window = 0.5
    fade_out_start = max(start, end - fade_window)
    alpha_expr = (
        f"if(lt(t,{start + fade_window:.3f}),"
        f"lerp(0x00,0xFF,(t-{start:.3f})/{fade_window}),"
        f"if(lt(t,{fade_out_start:.3f}),0xFF,"
        f"lerp(0xFF,0x00,(t-{fade_out_start:.3f})/{fade_window})))"
    )
    fade_color_expr = f"0x{rgb}%{{eif\\:{alpha_expr}\\:x\\:2}}"
    base = _base_drawtext(text, font_size, x, y, normalized_color, config)
    return f"{base}:fontcolor_expr='{fade_color_expr}'"


def generate_slide(
    text: str,
    start: float,
    duration: float,
    color: str,
    font_size: int,
    x: str,
    y: str,
    axis: str = "x",
    distance: int = 180,
    config: dict[str, Any] | None = None,
) -> str:
    end = start + duration
    phase = 0.5
    slide_out_start = max(start, end - phase)
    base = _base_drawtext(text, font_size, x, y, color, config)

    if axis not in {"x", "y"}:
        raise ValueError("axis must be either 'x' or 'y'")

    if axis == "x":
        x_expr = (
            f"if(lt(t,{start + phase:.1f}),"
            f"{x}+{distance}*(1-(t-{start:.1f})/{phase}),"
            f"if(gt(t,{slide_out_start:.1f}),"
            f"{x}+{distance}*((t-{slide_out_start:.1f})/{phase}),{x}))"
        )
        return f"{base}:x='{x_expr}':y={y}"

    y_expr = (
        f"if(lt(t,{start + phase:.1f}),"
        f"{y}+{distance}*(1-(t-{start:.1f})/{phase}),"
        f"if(gt(t,{slide_out_start:.1f}),"
        f"{y}+{distance}*((t-{slide_out_start:.1f})/{phase}),{y}))"
    )
    return f"{base}:x={x}:y='{y_expr}'"


def generate_scale(
    text: str,
    start: float,
    duration: float,
    color: str,
    font_size: int,
    x: str,
    y: str,
    min_scale: float = 0.7,
    max_scale: float = 1.2,
    config: dict[str, Any] | None = None,
) -> str:
    if min_scale <= 0 or max_scale <= 0 or max_scale < min_scale:
        raise ValueError("Invalid scale range")

    mid = start + (duration / 2.0)
    min_size = round(font_size * min_scale, 1)
    max_size = round(font_size * max_scale, 1)
    base = _base_drawtext(text, font_size, x, y, color, config)
    size_expr = (
        f"if(lt(t,{mid:.1f}),"
        f"{min_size}+({max_size}-{min_size})*((t-{start:.1f})/{(duration / 2.0):.1f}),"
        f"{max_size}-({max_size}-{min_size})*((t-{mid:.1f})/{(duration / 2.0):.1f}))"
    )
    return f"{base}:fontsize='{size_expr}'"


def generate_flashing(
    text: str,
    start: float,
    duration: float,
    color: str,
    font_size: int,
    x: str,
    y: str,
    flash_speed: float = 2.0,
    config: dict[str, Any] | None = None,
) -> str:
    if flash_speed <= 0 or flash_speed > 10.0:
        raise ValueError("Invalid flash speed, must be between 0 and 10.0")

    normalized_color = _normalize_color(color)
    rgb = normalized_color[2:]
    base = _base_drawtext(text, font_size, x, y, normalized_color, config)
    
    # Alpha oscillates between 0x40 (64) and 0xFF (255) to preserve readability
    # sin(t * speed * PI) goes from -1 to 1
    # We map it to 0.25 to 1.0 alpha
    alpha_expr = f"191*(0.5*sin((t-{start:.1f})*{flash_speed:.1f}*PI)+0.5)+64"
    flash_color_expr = f"0x{rgb}%{{eif\\:{alpha_expr}\\:x\\:2}}"
    
    return f"{base}:fontcolor_expr='{flash_color_expr}'"


def generate_text_animation(
    animation: str,
    text: str,
    start: float,
    duration: float,
    color: str,
    font_size: int,
    x: str,
    y: str,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> str:
    if animation == "fade":
        return generate_fade(text, start, duration, color, font_size, x, y, config=config)
    if animation == "slide":
        return generate_slide(text, start, duration, color, font_size, x, y, config=config, **kwargs)
    if animation == "scale":
        return generate_scale(text, start, duration, color, font_size, x, y, config=config, **kwargs)
    if animation == "flashing":
        return generate_flashing(text, start, duration, color, font_size, x, y, config=config, **kwargs)
    raise ValueError(f"Unsupported animation preset: {animation}")
