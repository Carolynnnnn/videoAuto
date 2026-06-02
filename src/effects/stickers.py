from __future__ import annotations

import os
from typing import Any

from PIL import Image, UnidentifiedImageError

MAX_GIF_BYTES = 10 * 1024 * 1024
MAX_GIF_WIDTH = 1920
MAX_GIF_HEIGHT = 1080

ANCHORS = {
    "top-left",
    "top-center",
    "top-right",
    "center-left",
    "center",
    "center-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
}


def validate_gif(
    asset_path: str,
    max_size_bytes: int = MAX_GIF_BYTES,
    max_width: int = MAX_GIF_WIDTH,
    max_height: int = MAX_GIF_HEIGHT,
) -> dict[str, Any]:
    if not os.path.exists(asset_path):
        return {"valid": False, "error": "File does not exist"}

    size_bytes = os.path.getsize(asset_path)
    if size_bytes > max_size_bytes:
        return {"valid": False, "error": "File exceeds 10MB limit"}

    with open(asset_path, "rb") as file_obj:
        header = file_obj.read(6)
    if header not in {b"GIF87a", b"GIF89a"}:
        return {"valid": False, "error": "File must be GIF format"}

    try:
        with Image.open(asset_path) as image:
            image_format = (image.format or "").upper()
            width, height = image.size
    except (UnidentifiedImageError, OSError):
        return {"valid": False, "error": "Unable to read GIF file"}

    if image_format != "GIF":
        return {"valid": False, "error": "File must be GIF format"}

    if width > max_width or height > max_height:
        return {
            "valid": False,
            "error": f"Resolution exceeds {max_width}x{max_height} limit",
        }

    return {
        "valid": True,
        "format": image_format,
        "width": width,
        "height": height,
        "size_bytes": size_bytes,
    }


def calculate_position(
    anchor: str,
    video_width: int,
    video_height: int,
    sticker_width: int,
    sticker_height: int,
) -> tuple[int, int]:
    if anchor not in ANCHORS:
        raise ValueError(f"Unsupported anchor: {anchor}")

    x_positions = {
        "left": 0,
        "center": int((video_width - sticker_width) / 2),
        "right": video_width - sticker_width,
    }
    y_positions = {
        "top": 0,
        "center": int((video_height - sticker_height) / 2),
        "bottom": video_height - sticker_height,
    }

    vertical, horizontal = anchor.split("-") if "-" in anchor else ("center", "center")
    if anchor == "center":
        vertical = "center"
        horizontal = "center"

    return x_positions[horizontal], y_positions[vertical]


def build_sticker_overlay_filter(
    sticker_stream: str,
    base_stream: str,
    output_stream: str,
    anchor: str,
    start_time: float,
    duration: float,
    scale: float,
    transparency: float,
    video_width: int,
    video_height: int,
    sticker_width: int,
    sticker_height: int,
) -> str:
    if duration <= 0:
        raise ValueError("Duration must be greater than 0")
    if start_time < 0:
        raise ValueError("Start time must be >= 0")
    if scale <= 0 or scale > 1.0:
        raise ValueError("Scale must be within (0, 1]")
    if transparency < 0 or transparency > 1.0:
        raise ValueError("Transparency must be within [0, 1]")

    scaled_width = int(sticker_width * scale)
    scaled_height = int(sticker_height * scale)
    x_pos, y_pos = calculate_position(
        anchor=anchor,
        video_width=video_width,
        video_height=video_height,
        sticker_width=scaled_width,
        sticker_height=scaled_height,
    )

    end_time = start_time + duration
    processed_sticker_stream = "[sticker_fx]"

    sticker_filter = (
        f"{sticker_stream}format=rgba,"
        f"scale=iw*{scale}:ih*{scale},"
        f"colorchannelmixer=aa={transparency}"
        f"{processed_sticker_stream}"
    )
    overlay_filter = (
        f"{base_stream}{processed_sticker_stream}"
        f"overlay={x_pos}:{y_pos}:"
        f"enable='between(t,{start_time},{end_time})'"
        f"{output_stream}"
    )
    return f"{sticker_filter};{overlay_filter}"


def generate_overlay_filter(
    sticker_stream: str,
    base_stream: str,
    output_stream: str,
    anchor: str,
    start_time: float,
    duration: float,
    scale: float,
    transparency: float,
    video_width: int,
    video_height: int,
    sticker_width: int,
    sticker_height: int,
) -> str:
    return build_sticker_overlay_filter(
        sticker_stream=sticker_stream,
        base_stream=base_stream,
        output_stream=output_stream,
        anchor=anchor,
        start_time=start_time,
        duration=duration,
        scale=scale,
        transparency=transparency,
        video_width=video_width,
        video_height=video_height,
        sticker_width=sticker_width,
        sticker_height=sticker_height,
    )


def calculate_anchor_position(
    anchor: str,
    video_width: int,
    video_height: int,
    sticker_width: int,
    sticker_height: int,
) -> tuple[int, int]:
    return calculate_position(
        anchor=anchor,
        video_width=video_width,
        video_height=video_height,
        sticker_width=sticker_width,
        sticker_height=sticker_height,
    )


def build_sticker_overlay_filter_from_effect(
    effect: Any,
    sticker_stream: str,
    base_stream: str,
    output_stream: str,
    video_width: int,
    video_height: int,
) -> str:
    validation = validate_gif(effect.asset_path)
    if not validation.get("valid", False):
        raise ValueError(validation["error"])

    return build_sticker_overlay_filter(
        sticker_stream=sticker_stream,
        base_stream=base_stream,
        output_stream=output_stream,
        anchor=effect.position,
        start_time=effect.start_time,
        duration=effect.duration,
        scale=effect.scale,
        transparency=getattr(effect, "transparency", 1.0),
        video_width=video_width,
        video_height=video_height,
        sticker_width=validation["width"],
        sticker_height=validation["height"],
    )
