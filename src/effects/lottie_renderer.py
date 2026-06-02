from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.effects.stickers import calculate_anchor_position

try:
    from pylottie_convert.animation import Animation as PylottieAnimation
except Exception:
    PylottieAnimation = None

Animation = PylottieAnimation


def _require_animation_class() -> None:
    if Animation is None:
        raise RuntimeError("pylottie-convert is required for Lottie rendering")


def _hash_lottie(lottie_path: str, target_fps: int) -> str:
    payload = Path(lottie_path).read_bytes() + f"|{target_fps}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _read_lottie_timeline(lottie_path: str) -> tuple[float, int, int]:
    data = json.loads(Path(lottie_path).read_text(encoding="utf-8"))
    native_fps = float(data.get("fr", 30.0))
    ip = int(data.get("ip", 0))
    op = int(data.get("op", ip))
    return native_fps, ip, op


def _frame_to_image(frame: Any) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGBA")
    if isinstance(frame, np.ndarray):
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        return Image.fromarray(frame, "RGBA")
    raise TypeError(f"Unsupported frame type: {type(frame)!r}")


def _cache_is_valid(metadata_path: Path, frames_dir: Path) -> bool:
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    frame_count = int(metadata.get("frame_count", 0))
    if frame_count <= 0:
        return False
    first = frames_dir / "frame_0001.png"
    last = frames_dir / f"frame_{frame_count:04d}.png"
    return first.exists() and last.exists()


def convert_lottie_to_frames(
    lottie_path: str,
    cache_root: str = "assets/lottie_cache",
    target_fps: int = 30,
) -> dict[str, Any]:
    _require_animation_class()
    animation_cls = Animation
    if animation_cls is None:
        raise RuntimeError("pylottie-convert is required for Lottie rendering")
    assert animation_cls is not None
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")
    if not Path(lottie_path).exists():
        raise FileNotFoundError(lottie_path)

    cache_key = _hash_lottie(lottie_path, target_fps)
    frames_dir = Path(cache_root) / cache_key
    frames_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = frames_dir / "metadata.json"
    frame_pattern = str(frames_dir / "frame_%04d.png")

    if _cache_is_valid(metadata_path, frames_dir):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["used_cache"] = True
        return metadata

    anim = animation_cls()
    anim.load_file(lottie_path)

    native_fps, ip, op = _read_lottie_timeline(lottie_path)
    if native_fps <= 0:
        native_fps = float(getattr(anim, "fps", 30.0) or 30.0)

    native_frame_count = max(1, op - ip + 1)
    duration_seconds = native_frame_count / native_fps
    output_frame_count = max(1, int(round(duration_seconds * target_fps)))

    for frame_idx in range(output_frame_count):
        source_time = frame_idx / float(target_fps)
        source_absolute = ip + int(round(source_time * native_fps))
        source_absolute = min(max(source_absolute, ip), op)
        source_zero_based = source_absolute - ip
        rgba_frame = anim.render(source_zero_based)
        image = _frame_to_image(rgba_frame)
        image.save(frames_dir / f"frame_{frame_idx + 1:04d}.png", format="PNG")

    width = int(getattr(anim, "width", 0) or 0)
    height = int(getattr(anim, "height", 0) or 0)
    metadata = {
        "cache_key": cache_key,
        "lottie_path": lottie_path,
        "frames_dir": str(frames_dir),
        "frame_pattern": frame_pattern,
        "frame_rate": int(target_fps),
        "native_fps": float(native_fps),
        "native_frame_count": int(native_frame_count),
        "frame_count": int(output_frame_count),
        "duration_seconds": float(duration_seconds),
        "width": width,
        "height": height,
        "used_cache": False,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def generate_lottie_overlay(
    effect: Any,
    sequence_info: dict[str, Any],
    video_width: int,
    video_height: int,
    image_input_index: int,
    base_stream: str,
    output_stream: str,
    video_fps: int | None = None,
) -> dict[str, str]:
    frame_rate = int(sequence_info["frame_rate"])
    frame_pattern = sequence_info["frame_pattern"]
    sticker_width = int(sequence_info.get("width", 0))
    sticker_height = int(sequence_info.get("height", 0))
    scale = float(getattr(effect, "scale", 1.0))
    transparency = float(getattr(effect, "transparency", 1.0))
    start_time = float(getattr(effect, "start_time", 0.0))
    duration = float(getattr(effect, "duration", 0.0))
    position = str(getattr(effect, "position", "center"))

    if duration <= 0:
        raise ValueError("Duration must be greater than 0")
    if scale <= 0:
        raise ValueError("Scale must be greater than 0")

    scaled_width = int(sticker_width * scale)
    scaled_height = int(sticker_height * scale)
    x_pos, y_pos = calculate_anchor_position(
        anchor=position,
        video_width=video_width,
        video_height=video_height,
        sticker_width=scaled_width,
        sticker_height=scaled_height,
    )

    end_time = start_time + duration
    image_stream = f"[{image_input_index}:v]"
    processed_stream = "[lottie_fx]"
    sync_fps = int(video_fps) if video_fps else frame_rate

    filter_text = (
        f"{image_stream}format=rgba,"
        f"fps={sync_fps},"
        f"scale=iw*{scale}:ih*{scale},"
        f"colorchannelmixer=aa={transparency}"
        f"{processed_stream};"
        f"{base_stream}{processed_stream}"
        f"overlay={x_pos}:{y_pos}:enable='between(t,{start_time},{end_time})'"
        f"{output_stream}"
    )

    input_args = f"-framerate {frame_rate} -i {shlex.quote(frame_pattern)}"
    return {"input_args": input_args, "filter": filter_text}
