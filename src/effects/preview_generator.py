"""Static frame preview generator for video effects.

Generates single-frame static previews optimized for <1s generation time.
Supports overlay preview for stickers, text animations, and other effects.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from src.effects.stickers import calculate_position, ANCHORS


# Cache directory for preview results
PREVIEW_CACHE_DIR = ".cache/previews"


def _get_cache_dir() -> Path:
    """Get or create the preview cache directory."""
    cache_dir = Path(PREVIEW_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _compute_preview_hash(
    video_path: str,
    timestamp: float,
    effects: list[dict[str, Any]] | None = None,
) -> str:
    """Compute a hash for caching based on input parameters."""
    hash_input = f"{video_path}:{timestamp:.3f}"
    if effects:
        for effect in effects:
            hash_input += f":{sorted(effect.items())}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:16]


def _run_ffmpeg(cmd: str, timeout: int = 5) -> tuple[int, str, str]:
    """Run FFmpeg command with timeout."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout
    )
    return result.returncode, result.stdout, result.stderr


def extract_frame(
    video_path: str,
    output_path: str,
    timestamp: float,
    timeout: int = 5,
) -> bool:
    """
    Extract a single frame from video at the specified timestamp.
    
    Uses FFmpeg for fast frame extraction: ffmpeg -ss TIME -i INPUT -frames:v 1 OUTPUT
    
    Args:
        video_path: Path to the video file
        output_path: Path for the output PNG file
        timestamp: Time in seconds to extract frame from
        timeout: Maximum seconds to wait for extraction
        
    Returns:
        True if extraction succeeded, False otherwise
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(video_path):
        return False
    
    # Use -ss before -i for fast seeking (input seeking)
    cmd = (
        f'ffmpeg -y -ss {timestamp:.3f} -i "{video_path}" '
        f'-frames:v 1 -q:v 2 "{output_path}" -loglevel error'
    )
    
    try:
        rc, _, _ = _run_ffmpeg(cmd, timeout=timeout)
        return rc == 0 and Path(output_path).exists()
    except subprocess.TimeoutExpired:
        return False


def extract_middle_frame(
    video_path: str,
    output_path: str,
    duration: float | None = None,
    timeout: int = 5,
) -> bool:
    """
    Extract the middle frame of a video segment.
    
    Args:
        video_path: Path to the video file
        output_path: Path for the output PNG file
        duration: Total duration of video (if known, avoids probe)
        timeout: Maximum seconds to wait
        
    Returns:
        True if extraction succeeded, False otherwise
    """
    if duration is None:
        # Probe video duration using ffprobe
        probe_cmd = (
            f'ffprobe -v error -show_entries format=duration '
            f'-of default=noprint_wrappers=1:nokey=1 "{video_path}"'
        )
        try:
            result = subprocess.run(
                probe_cmd, shell=True, capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                duration = float(result.stdout.strip())
            else:
                duration = 0.0
        except (subprocess.TimeoutExpired, ValueError):
            duration = 0.0
    
    middle_timestamp = duration / 2.0
    return extract_frame(video_path, output_path, middle_timestamp, timeout)


def generate_sticker_preview(
    background_path: str,
    sticker_path: str,
    output_path: str,
    anchor: str,
    scale: float = 1.0,
    video_width: int = 1080,
    video_height: int = 1920,
) -> bool:
    """
    Generate a preview with sticker overlay positioned correctly.
    
    Args:
        background_path: Path to background image/frame
        sticker_path: Path to sticker image (GIF/PNG)
        output_path: Output path for preview
        anchor: Position anchor (e.g., "center", "top-right")
        scale: Sticker scale factor (0-1]
        video_width: Target video width
        video_height: Target video height
        
    Returns:
        True if preview generated successfully
    """
    if not os.path.exists(background_path) or not os.path.exists(sticker_path):
        return False
    
    if anchor not in ANCHORS:
        return False
    
    try:
        # Load images
        background = Image.open(background_path).convert("RGBA")
        background = background.resize((video_width, video_height), Image.Resampling.LANCZOS)
        
        sticker = Image.open(sticker_path).convert("RGBA")
        
        # Scale sticker
        if scale != 1.0:
            new_width = int(sticker.width * scale)
            new_height = int(sticker.height * scale)
            sticker = sticker.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Calculate position
        x_pos, y_pos = calculate_position(
            anchor=anchor,
            video_width=video_width,
            video_height=video_height,
            sticker_width=sticker.width,
            sticker_height=sticker.height,
        )
        
        # Composite
        background.paste(sticker, (x_pos, y_pos), sticker)
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        background.save(output_path, "PNG")
        return True
        
    except Exception:
        return False


def generate_text_preview(
    background_path: str,
    output_path: str,
    text: str,
    x: int | str,
    y: int | str,
    font_size: int = 48,
    color: str = "white",
    video_width: int = 1080,
    video_height: int = 1920,
) -> bool:
    """
    Generate a preview with text overlay at the specified position.
    
    Args:
        background_path: Path to background image/frame
        output_path: Output path for preview
        text: Text content to render
        x: X position (int) or expression like "(w-text_w)/2"
        y: Y position (int) or expression like "h*0.78"
        font_size: Font size in pixels
        color: Text color name
        video_width: Target video width
        video_height: Target video height
        
    Returns:
        True if preview generated successfully
    """
    if not os.path.exists(background_path):
        return False
    
    try:
        # Load background
        background = Image.open(background_path).convert("RGBA")
        background = background.resize((video_width, video_height), Image.Resampling.LANCZOS)
        
        draw = ImageDraw.Draw(background)
        
        # Try to load a font, fallback to default
        font = None
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]
        for fp in font_paths:
            if Path(fp).exists():
                try:
                    font = ImageFont.truetype(fp, font_size)
                    break
                except Exception:
                    continue
        
        if font is None:
            font = ImageFont.load_default()
        
        # Calculate text size for centering if needed
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Handle string expressions for position
        if isinstance(x, str):
            if "w" in x or "text_w" in x:
                # Parse common centering expression
                x = (video_width - text_width) // 2
            else:
                x = int(float(x))
        
        if isinstance(y, str):
            if "h" in y:
                # Parse relative position like "h*0.78"
                if "*" in y:
                    factor = float(y.split("*")[1])
                    y = int(video_height * factor)
                else:
                    y = int(video_height * 0.78)
            else:
                y = int(float(y))
        
        # Draw text
        draw.text((x, y), text, fill=color, font=font)
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        background.save(output_path, "PNG")
        return True
        
    except Exception:
        return False


def generate_effect_preview(
    video_path: str,
    output_path: str,
    effects: list[dict[str, Any]] | None = None,
    timestamp: float | None = None,
    video_width: int = 1080,
    video_height: int = 1920,
    use_cache: bool = True,
) -> dict[str, Any]:
    """
    Generate a comprehensive preview with all effects overlaid.
    
    Main entry point for preview generation. Extracts middle frame
    and overlays all effect markers (stickers, text positions).
    
    Args:
        video_path: Path to video or image asset
        output_path: Output path for preview image
        effects: List of effect configurations, each containing:
            - type: "sticker" | "text"
            - For sticker: asset_path, anchor, scale
            - For text: text, x, y, font_size, color
        timestamp: Specific timestamp to extract (None = middle frame)
        video_width: Target video width
        video_height: Target video height
        use_cache: Whether to use caching
        
    Returns:
        dict with keys:
            - success: bool
            - preview_path: str (output path if successful)
            - cached: bool (whether result was from cache)
            - error: str | None
    """
    if not os.path.exists(video_path):
        return {"success": False, "error": "Video path does not exist", "cached": False}
    
    # Check cache
    preview_hash = _compute_preview_hash(video_path, timestamp or 0.0, effects)
    cache_dir = _get_cache_dir()
    cached_path = cache_dir / f"{preview_hash}.png"
    
    if use_cache and cached_path.exists():
        # Copy to output if different path requested
        if str(cached_path) != output_path:
            import shutil
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(cached_path, output_path)
        return {
            "success": True,
            "preview_path": output_path,
            "cached": True,
            "error": None,
        }
    
    # Determine if input is video or image
    ext = Path(video_path).suffix.lower()
    is_video = ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    
    # Create temp frame path
    temp_frame = cache_dir / f"temp_{preview_hash}.png"
    
    try:
        if is_video:
            # Extract frame from video
            if timestamp is not None:
                success = extract_frame(video_path, str(temp_frame), timestamp)
            else:
                success = extract_middle_frame(video_path, str(temp_frame))
            
            if not success:
                return {"success": False, "error": "Failed to extract frame", "cached": False}
            
            base_image_path = str(temp_frame)
        else:
            # Use image directly
            base_image_path = video_path
        
        # Load and resize base image
        base_image = Image.open(base_image_path).convert("RGBA")
        base_image = base_image.resize((video_width, video_height), Image.Resampling.LANCZOS)
        
        # Apply effects
        if effects:
            for effect in effects:
                effect_type = effect.get("type", "")
                
                if effect_type == "sticker":
                    sticker_path = effect.get("asset_path")
                    if sticker_path and os.path.exists(sticker_path):
                        try:
                            sticker = Image.open(sticker_path).convert("RGBA")
                            scale = effect.get("scale", 1.0)
                            if scale != 1.0:
                                new_w = int(sticker.width * scale)
                                new_h = int(sticker.height * scale)
                                sticker = sticker.resize((new_w, new_h), Image.Resampling.LANCZOS)
                            
                            anchor = effect.get("anchor", "center")
                            if anchor in ANCHORS:
                                x_pos, y_pos = calculate_position(
                                    anchor, video_width, video_height,
                                    sticker.width, sticker.height
                                )
                                base_image.paste(sticker, (x_pos, y_pos), sticker)
                        except Exception:
                            pass  # Skip invalid stickers
                
                elif effect_type == "text":
                    text = effect.get("text", "")
                    if text:
                        draw = ImageDraw.Draw(base_image)
                        font_size = effect.get("font_size", 48)
                        color = effect.get("color", "white")
                        
                        # Load font
                        font = None
                        for fp in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
                            if Path(fp).exists():
                                try:
                                    font = ImageFont.truetype(fp, font_size)
                                    break
                                except Exception:
                                    continue
                        if font is None:
                            font = ImageFont.load_default()
                        
                        # Calculate position
                        bbox = draw.textbbox((0, 0), text, font=font)
                        text_w = bbox[2] - bbox[0]
                        text_h = bbox[3] - bbox[1]
                        
                        x = effect.get("x", (video_width - text_w) // 2)
                        y = effect.get("y", int(video_height * 0.78))
                        
                        if isinstance(x, str):
                            x = (video_width - text_w) // 2
                        if isinstance(y, str):
                            y = int(video_height * 0.78)
                        
                        draw.text((x, y), text, fill=color, font=font)
        
        # Save to cache and output
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        base_image.save(output_path, "PNG")
        
        if use_cache:
            base_image.save(cached_path, "PNG")
        
        # Clean up temp file
        if temp_frame.exists():
            temp_frame.unlink()
        
        return {
            "success": True,
            "preview_path": output_path,
            "cached": False,
            "error": None,
        }
        
    except Exception as e:
        return {"success": False, "error": str(e), "cached": False}


def clear_preview_cache() -> int:
    """Clear all cached preview images. Returns number of files deleted."""
    cache_dir = _get_cache_dir()
    count = 0
    for file in cache_dir.glob("*.png"):
        try:
            file.unlink()
            count += 1
        except Exception:
            pass
    return count


def get_cached_preview(
    video_path: str,
    timestamp: float = 0.0,
    effects: list[dict[str, Any]] | None = None,
) -> str | None:
    """
    Check if a preview is already cached and return its path.
    
    Args:
        video_path: Path to video file
        timestamp: Timestamp used for preview
        effects: Effect configurations
        
    Returns:
        Path to cached preview if exists, None otherwise
    """
    preview_hash = _compute_preview_hash(video_path, timestamp, effects)
    cached_path = _get_cache_dir() / f"{preview_hash}.png"
    
    if cached_path.exists():
        return str(cached_path)
    return None
