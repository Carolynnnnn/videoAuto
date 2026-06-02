"""
Step 5：渲染每段 Segment 视频（v2）

关键变更：
  - 输出路径：render/segments/{content_key}_{render_hash}.mp4
  - render_hash = hash(plan_hash + start/end + subtitle_style + motion + asset_hashes)
  - render_hash 相同 → 直接复用旧 seg mp4（不重渲）
  - 支持只渲染指定 segment_keys（增量更新时只处理 need_rerender）
  - RenderRef 字段对齐 v2 模型（segment_video_path / render_hash / status: ok）
"""
from __future__ import annotations
import os
import subprocess
import hashlib
import shutil
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from src.core.models import Manifest, Segment, RenderRef, GlobalStyle
from src.utils.logger import get_logger
from src.effects.text_animations import generate_text_animation
from src.effects.stickers import validate_gif

logger = get_logger("step5_render")


# ─────────────────────────────────────────────
# Overflow Safety Constants
# ─────────────────────────────────────────────
# Maximum subtitle width as ratio of video width (92% of video width)
MAX_SUBTITLE_WIDTH_RATIO = 0.92

# Minimum horizontal margin as percentage of video width (4%)
MIN_MARGIN_PCT = 0.04

# Maximum single token character width before forced truncation
MAX_UNSPLITTABLE_TOKEN_WIDTH_RATIO = 0.88

# ─────────────────────────────────────────────
# FFmpeg 工具函数
# ─────────────────────────────────────────────
def _run_ffmpeg(cmd: str, timeout: int = 120) -> Tuple[int, str, str]:
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout
    )
    return result.returncode, result.stdout, result.stderr


def _escape_ffmpeg_text(text: str) -> str:
    text = text.replace("\\", "\\\\")
    text = text.replace("\n", r"\n")
    text = text.replace("'", "\\'")
    text = text.replace(":", "\\:")
    text = text.replace("[", "\\[").replace("]", "\\]")
    text = text.replace(",", r"\,")
    return text


_CJK_FONT_PATHS = [
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]


def _resolve_cjk_font_path() -> str:
    for fp in _CJK_FONT_PATHS:
        if Path(fp).exists():
            return fp
    return ""


def _char_unit_width(char: str) -> float:
    if char.isspace():
        return 0.35
    if ord(char) < 128:
        return 0.56
    return 1.0


def _estimate_text_width(text: str, font_size: int) -> float:
    return sum(_char_unit_width(ch) for ch in text) * font_size


def _get_safe_subtitle_width(video_width: int, font_size: int) -> int:
    """
    Calculate safe subtitle width using width-based formula.
    
    Returns maximum width in pixels for subtitle text to prevent overflow.
    Formula: max_width = video_width * MAX_SUBTITLE_WIDTH_RATIO
    """
    safe_width = int(video_width * MAX_SUBTITLE_WIDTH_RATIO)
    return safe_width


def _normalize_animation_color(color: str) -> str:
    value = color.strip()
    if value.startswith("0x"):
        return value
    if value.startswith("#"):
        return f"0x{value[1:]}"

    named = {
        "white": "0xFFFFFF",
        "black": "0x000000",
        "yellow": "0xFFFF00",
        "red": "0xFF0000",
        "green": "0x00FF00",
        "blue": "0x0000FF",
    }
    return named.get(value.lower(), "0xFFFFFF")


def _wrap_subtitle_lines(text: str, max_width_px: int, font_size: int, max_lines: int = 3, video_width: int = 0) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return [""]

    # Check for unsplittable tokens that exceed safe bounds
    # For CJK text, we don't treat the whole string as unsplittable if it has no spaces
    has_cjk = any('\u4e00' <= ch <= '\u9fff' for ch in cleaned)
    is_single_token = " " not in cleaned
    
    unsplittable_threshold = int(video_width * MAX_UNSPLITTABLE_TOKEN_WIDTH_RATIO) if video_width > 0 else max_width_px
    if is_single_token and not has_cjk and _estimate_text_width(cleaned, font_size) > unsplittable_threshold:
        # Token too long, truncate with ellipsis and log warning
        truncated = ""
        for ch in cleaned:
            if _estimate_text_width(truncated + ch + "...", font_size) <= unsplittable_threshold:
                truncated += ch
            else:
                break
        truncated += "..."
        logger.warning(
            f"Unsplittable token exceeds safe width. Original: '{cleaned[:50]}...', "
            f"Truncated: '{truncated}'"
        )
        return [truncated]

    lines: List[str] = []
    pending = cleaned
    break_chars = set(" ,，。！？；：、")

    while pending and len(lines) < max_lines:
        current = ""
        width_px = 0.0
        last_break_index = -1

        for idx, ch in enumerate(pending):
            next_width = width_px + (_char_unit_width(ch) * font_size)
            if next_width > max_width_px and current:
                split_index = last_break_index + 1 if last_break_index >= 0 else idx
                line = pending[:split_index].strip()
                if not line:
                    line = pending[:idx]
                    split_index = idx
                lines.append(line)
                pending = pending[split_index:].strip()
                break
            current += ch
            width_px = next_width
            if ch in break_chars:
                last_break_index = idx
        else:
            lines.append(pending)
            pending = ""

    if pending:
        lines[-1] = f"{lines[-1]}…"

    return lines


def _extract_subtitle_emphasis_tokens(segment: Segment) -> List[str]:
    if not segment.visual_plan:
        return []

    tokens: List[str] = []
    for item in segment.visual_plan.overlay:
        if not isinstance(item.extra, dict) or item.kind != "subtitle_emphasis":
            continue
        raw_tokens = item.extra.get("tokens")
        if isinstance(raw_tokens, list):
            for token in raw_tokens:
                if isinstance(token, str):
                    value = token.strip()
                    if value:
                        tokens.append(value)

    deduped: List[str] = []
    seen = set()
    base_text = re.sub(r"\s+", "", segment.text)
    for token in tokens:
        compact = re.sub(r"\s+", "", token)
        if not compact:
            continue
        if len(compact) > 8:
            continue
        if compact == base_text:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(token)
        if len(deduped) >= 6:
            break
    return deduped


def _build_emphasis_filters(
    lines: List[str],
    emphasis_tokens: List[str],
    style: GlobalStyle,
    width: int,
    top_y: int,
    line_height: int,
    font_file: str,
) -> List[str]:
    filters: List[str] = []
    if not emphasis_tokens:
        return filters

    horizontal_margin = max(24, int(width * 0.04))

    for line_index, line in enumerate(lines):
        line_width = _estimate_text_width(line, style.font_size)
        line_x = max(float(horizontal_margin), (width - line_width) / 2.0)
        y_value = top_y + line_index * line_height
        lowered_line = line.lower()

        for token in emphasis_tokens:
            search = token.lower()
            if not search:
                continue

            start_at = 0
            while True:
                hit = lowered_line.find(search, start_at)
                if hit < 0:
                    break

                prefix = line[:hit]
                token_text = line[hit:hit + len(token)]
                token_width = _estimate_text_width(token_text, style.font_size)
                x_value = line_x + _estimate_text_width(prefix, style.font_size)
                max_x = max(float(horizontal_margin), width - horizontal_margin - token_width)
                x_value = min(max(float(horizontal_margin), x_value), max_x)
                escaped_token = _escape_ffmpeg_text(token_text)
                filters.append(
                    (
                        f"drawtext=text='{escaped_token}'"
                        f":fontsize={style.font_size}"
                        f":fontcolor=yellow"
                        f":x={x_value:.1f}"
                        f":y={y_value}"
                        f":borderw=2"
                        f":bordercolor=black@0.9"
                        f":shadowx=1"
                        f":shadowy=1"
                        f":shadowcolor=black@0.7"
                        f"{font_file}"
                    )
                )

                start_at = hit + len(token)

    return filters


# ─────────────────────────────────────────────
# Sticker overlay helpers
# ─────────────────────────────────────────────
def _extract_sticker_effects(segment: Segment) -> List[Dict[str, Any]]:
    """Extract sticker overlay metadata from segment visual_plan."""
    """Extract sticker overlay metadata from segment visual_plan."""
    from typing import Dict, Any
    if not segment.visual_plan:
        return []

    stickers: List[Dict[str, Any]] = []
    for item in segment.visual_plan.overlay:
        if item.kind != "sticker":
            continue
        if not isinstance(item.extra, dict):
            continue

        asset_path = item.extra.get("asset_path") or item.extra.get("path")
        if not asset_path:
            logger.debug(f"  Sticker overlay missing asset_path: {item.extra}")
            continue

        stickers.append({
            "asset_path": asset_path,
            "anchor": item.extra.get("anchor", "center"),
            "scale": item.extra.get("scale", 0.3),
            "transparency": item.extra.get("transparency", 1.0),
            "start_time": item.extra.get("start_time", 0.0),
            "duration": item.extra.get("duration", segment.duration),
        })

    return stickers


def _build_sticker_filter(sticker_path: str, sticker_index: int, scale: float, transparency: float, anchor: str, start_time: float, duration: float, width: int, height: int) -> Optional[Tuple[str, int, int, float, float, int]]:
    """
    Validate sticker and build FFmpeg filter component.
    Returns (filter_string, x_pos, y_pos, start_time, end_time, sticker_index) or None if invalid.
    """
    """
    Validate sticker and build FFmpeg filter component.
    Returns (filter_string, sticker_width, sticker_height) or None if invalid.
    """
    validation = validate_gif(sticker_path)
    if not validation.get("valid", False):
        logger.warning(f"  Sticker validation failed: {validation.get('error', 'unknown')} - path={sticker_path}")
        return None

    sticker_width = validation["width"]
    sticker_height = validation["height"]
    scaled_width = int(sticker_width * scale)
    scaled_height = int(sticker_height * scale)

    # Calculate position based on anchor
    x_map = {"left": 0, "center": int((width - scaled_width) / 2), "right": width - scaled_width}
    y_map = {"top": 0, "center": int((height - scaled_height) / 2), "bottom": height - scaled_height}

    if anchor == "center":
        vertical, horizontal = "center", "center"
    elif "-" in anchor:
        parts = anchor.split("-")
        vertical, horizontal = parts[0], parts[1] if len(parts) > 1 else "center"
    else:
        vertical, horizontal = "center", "center"

    x_pos = x_map.get(horizontal, x_map["center"])
    y_pos = y_map.get(vertical, y_map["center"])
    end_time = start_time + duration

    # Build filter: scale + transparency applied to sticker stream
    filter_str = (
        f"[{sticker_index}:v]format=rgba,"
        f"scale=iw*{scale}:ih*{scale},"
        f"colorchannelmixer=aa={transparency}[stk{sticker_index}]"
    )

    return (filter_str, x_pos, y_pos, start_time, end_time, sticker_index)


# ─────────────────────────────────────────────
# 镜头运动滤镜
# ─────────────────────────────────────────────
def _build_motion_filter(
    preset: str,
    speed: float,
    duration: float,
    width: int,
    height: int,
) -> str:
    fps = 30
    total_frames = max(1, int(duration * fps))
    zoom_speed = speed * 0.0015

    if preset == "soft_kenburns":
        return (
            f"zoompan=z='min(zoom+{zoom_speed:.4f},1.15)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={total_frames}:s={width}x{height}:fps={fps}"
        )
    elif preset == "push_in":
        return (
            f"zoompan=z='min(zoom+{zoom_speed*1.5:.4f},1.3)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={total_frames}:s={width}x{height}:fps={fps}"
        )
    elif preset == "zoom_out":
        return (
            f"zoompan=z='if(lte(zoom,1.0),1.3,max(zoom-{zoom_speed*1.5:.4f},1.0))'"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={total_frames}:s={width}x{height}:fps={fps}"
        )
    elif preset == "pan_left":
        return (
            f"zoompan=z='1.1':x='iw/2-(iw/zoom/2)+{int(width*0.05)}*on/{total_frames}'"
            f":y='ih/2-(ih/zoom/2)':d={total_frames}:s={width}x{height}:fps={fps}"
        )
    elif preset == "pan_right":
        return (
            f"zoompan=z='1.1':x='iw/2-(iw/zoom/2)-{int(width*0.05)}*on/{total_frames}'"
            f":y='ih/2-(ih/zoom/2)':d={total_frames}:s={width}x{height}:fps={fps}"
        )
    else:  # static
        return f"scale={width}:{height},setsar=1"


# ─────────────────────────────────────────────
# 字幕滤镜
# ─────────────────────────────────────────────
def _build_subtitle_filter(segment: Segment, style: GlobalStyle, width: int, height: int) -> str:
    text = segment.text
    horizontal_margin = max(24, int(width * MIN_MARGIN_PCT))
    safe_width = _get_safe_subtitle_width(width, style.font_size)
    lines = _wrap_subtitle_lines(
        text,
        max_width_px=safe_width,
        font_size=style.font_size,
        max_lines=3,
        video_width=width,
    )
    wrapped_text = "\n".join(lines)
    escaped = _escape_ffmpeg_text(wrapped_text)
    line_height = max(1, int(style.font_size * 1.3))
    block_height = line_height * max(1, len(lines))
    bottom_margin = max(24, int(height * 0.08))
    top_y = max(int(height * 0.55), height - block_height - bottom_margin)
    box = 1 if style.subtitle_bg else 0
    box_color = style.subtitle_bg_color if style.subtitle_bg else "black@0"
    font_path = _resolve_cjk_font_path()
    font_file = f":fontfile='{font_path}'" if font_path else ""
    emphasis_tokens = _extract_subtitle_emphasis_tokens(segment)

    # Skip emphasis filters if subtitle effects are disabled
    if not getattr(style, 'enable_subtitle_effects', True):
        emphasis_tokens = []

    # 检查是否使用动画预设
    if style.subtitle_style in ["fade", "slide", "scale", "flashing"]:
        safe_center_x = f"max({horizontal_margin}\\\\,(w-text_w)/2)"
        # 动画预设
        base_filter = generate_text_animation(
            animation=style.subtitle_style,
            text=wrapped_text,
            start=0.0,
            duration=10.0, # 默认一个足够长的持续时间，或者可以从segment获取
            color=_normalize_animation_color(style.font_color),
            font_size=style.font_size,
            x=safe_center_x,
            y=str(top_y),
            config={"font_paths": [font_path] if font_path else []},
        )
        emphasis_filters = _build_emphasis_filters(
            lines=lines,
            emphasis_tokens=emphasis_tokens,
            style=style,
            width=width,
            top_y=top_y,
            line_height=line_height,
            font_file=font_file,
        )
        return ",".join([base_filter] + emphasis_filters)
    else:
        # 静态字幕
        base_filter = (
            f"drawtext=text='{escaped}'"
            f":fontsize={style.font_size}"
            f":fontcolor={style.font_color}"
            f":x=max({horizontal_margin}\\\\,(w-text_w)/2)"
            f":y={top_y}"
            f":box={box}"
            f":boxcolor={box_color}"
            f":boxborderw=12"
            f":line_spacing=8"
            f"{font_file}"
        )
        emphasis_filters = _build_emphasis_filters(
            lines=lines,
            emphasis_tokens=emphasis_tokens,
            style=style,
            width=width,
            top_y=top_y,
            line_height=line_height,
            font_file=font_file,
        )
        return ",".join([base_filter] + emphasis_filters)


# ─────────────────────────────────────────────
# 渲染缓存检查
# ─────────────────────────────────────────────
def _check_render_cache(
    segment: Segment,
    global_style: GlobalStyle,
    segments_dir: str,
) -> Optional[str]:
    """
    检查渲染缓存。
    缓存路径：render/segments/{content_key}_{render_hash}.mp4
    """
    render_hash = segment.compute_render_hash(global_style.render_related_fields())
    cache_path = os.path.join(segments_dir, f"{segment.content_key}_{render_hash}.mp4")
    if os.path.exists(cache_path):
        return cache_path
    return None


# ─────────────────────────────────────────────
# 渲染单段视频
# ─────────────────────────────────────────────
def render_segment(
    segment: Segment,
    output_path: str,
    style: GlobalStyle,
    max_retries: int = 3,
) -> bool:
    """渲染单个 Segment 为 mp4 视频。"""
    width = style.resolution_w
    height = style.resolution_h
    fps = style.fps
    duration = segment.duration

    # 获取素材路径
    asset_path = None
    if segment.visual_plan and segment.visual_plan.asset_path:
        asset_path = segment.visual_plan.asset_path
    elif segment.asset_refs:
        asset_path = segment.asset_refs[0].path if segment.asset_refs else None

    if not asset_path or not Path(asset_path).exists():
        logger.warning(f"  [seg {segment.index}] 素材不存在，生成临时背景")
        from src.steps.step4_assets import generate_template_asset
        tmp_bg = str(Path(output_path).parent / f"tmp_bg_{segment.content_key}.png")
        asset_path = generate_template_asset(tmp_bg, width, height)

    audio_ref = segment.audio_ref
    audio_path = audio_ref.path
    trim_start = audio_ref.trim_start
    trim_end = audio_ref.trim_end
    audio_duration = trim_end - trim_start if trim_end > trim_start else duration

    motion_preset = "static"
    motion_speed = 1.0
    if segment.visual_plan:
        motion_preset = segment.visual_plan.motion.preset
        motion_speed = segment.visual_plan.motion.speed

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries):
        try:
            asset_ext = Path(asset_path).suffix.lower()
            is_video_asset = asset_ext in (".mp4", ".mov", ".avi", ".mkv")

            # Extract stickers first (need to inject inputs before building video_input)
            subtitle_filter = _build_subtitle_filter(segment, style, width, height)
            sticker_effects = _extract_sticker_effects(segment)

            # Build filter graph with optional sticker overlays
            sticker_inputs_part = ""
            sticker_filters = []
            overlay_chain_parts = []

            # Validate and prepare sticker filters
            for idx, sticker in enumerate(sticker_effects, start=1):
                result = _build_sticker_filter(
                    sticker["asset_path"],
                    idx,
                    sticker["scale"],
                    sticker["transparency"],
                    sticker["anchor"],
                    sticker["start_time"],
                    sticker["duration"],
                    width,
                    height,
                )
                if result is None:
                    continue

                filter_str, x_pos, y_pos, start_time, end_time, sticker_index = result
                sticker_inputs_part += f' -i "{sticker["asset_path"]}" '
                sticker_filters.append(filter_str)
                overlay_chain_parts.append({
                    "stream": f"[stk{sticker_index}]",
                    "x": x_pos,
                    "y": y_pos,
                    "start": start_time,
                    "end": end_time,
                })

            # Now build video_input with sticker inputs injected
            if is_video_asset:
                video_input = (
                    f'-i "{asset_path}" '
                    f'{sticker_inputs_part}'
                    f'-ss {trim_start:.3f} -i "{audio_path}" '
                )
                video_filter = (
                    f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
                    f"crop={width}:{height},setsar=1,fps={fps}[v]"
                )
            else:
                motion_filter = _build_motion_filter(
                    motion_preset, motion_speed, duration, width, height
                )
                video_input = (
                    f'-loop 1 -t {duration:.3f} -i "{asset_path}" '
                    f'{sticker_inputs_part}'
                    f'-ss {trim_start:.3f} -t {audio_duration:.3f} -i "{audio_path}" '
                )
                video_filter = f"[0:v]{motion_filter},setsar=1,fps={fps}[v]"

            # Construct full filter_complex
            if sticker_filters:
                # With stickers: video -> [v] -> subtitle -> [vsub] -> overlay stickers -> [vout]
                full_filter = f"{video_filter};[v]{subtitle_filter}[vsub]"
                full_filter += ";" + ";".join(sticker_filters)

                current_stream = "[vsub]"
                for i, overlay_info in enumerate(overlay_chain_parts):
                    out_stream = "[vout]" if i == len(overlay_chain_parts) - 1 else f"[vtmp{i}]"
                    overlay_expr = (
                        f"{current_stream}{overlay_info['stream']}"
                        f"overlay={overlay_info['x']}:{overlay_info['y']}:"
                        f"enable='between(t,{overlay_info['start']},{overlay_info['end']})'{out_stream}"
                    )
                    full_filter += f";{overlay_expr}"
                    current_stream = out_stream
            else:
                # No stickers: video -> [v] -> subtitle -> [vout]
                full_filter = f"{video_filter};[v]{subtitle_filter}[vout]"

            # Calculate audio stream index (0=asset, 1..N=stickers, N+1=audio)
            audio_stream_index = len(sticker_filters) + 1

            if Path(audio_path).exists():
                cmd = (
                    f'ffmpeg -y '
                    f'{video_input}'
                    f'-filter_complex "{full_filter}" '
                    f'-map "[vout]" -map {audio_stream_index}:a '
                    f'-t {duration:.3f} '
                    f'-c:v libx264 -preset fast -crf 23 '
                    f'-c:a aac -b:a 128k '
                    f'-pix_fmt yuv420p '
                    f'"{output_path}" -loglevel warning'
                )
            else:
                cmd = (
                    f'ffmpeg -y '
                    f'-loop 1 -t {duration:.3f} -i "{asset_path}" '
                    f'-filter_complex "{full_filter}" '
                    f'-map "[vout]" '
                    f'-t {duration:.3f} '
                    f'-c:v libx264 -preset fast -crf 23 '
                    f'-pix_fmt yuv420p '
                    f'"{output_path}" -loglevel warning'
                )

            logger.debug(f"  FFmpeg: {cmd[:120]}...")
            rc, stdout, stderr = _run_ffmpeg(cmd, timeout=180)

            if rc == 0 and Path(output_path).exists():
                logger.info(f"  [seg {segment.index}] 渲染成功: {os.path.basename(output_path)}")
                return True
            else:
                logger.warning(f"  [seg {segment.index}] 渲染失败 (attempt {attempt+1}): {stderr[-200:]}")

        except subprocess.TimeoutExpired:
            logger.warning(f"  [seg {segment.index}] 渲染超时 (attempt {attempt+1})")
        except Exception as e:
            logger.warning(f"  [seg {segment.index}] 渲染异常 (attempt {attempt+1}): {e}")

    logger.error(f"  [seg {segment.index}] 所有重试失败，生成占位片段")
    _render_placeholder(segment, output_path, style)
    return False


def _render_placeholder(segment: Segment, output_path: str, style: GlobalStyle) -> None:
    """生成占位片段（黑色背景+字幕）"""
    width = style.resolution_w
    height = style.resolution_h
    duration = segment.duration
    escaped_text = _escape_ffmpeg_text(segment.text)

    cmd = (
        f'ffmpeg -y '
        f'-f lavfi -i color=c=black:size={width}x{height}:rate={style.fps} '
        f'-t {duration:.3f} '
        f'-vf "drawtext=text=\'{escaped_text}\':fontsize=40:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2" '
        f'-c:v libx264 -preset fast -crf 28 -pix_fmt yuv420p '
        f'"{output_path}" -loglevel error'
    )
    _run_ffmpeg(cmd, timeout=60)


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────
def run_step5(
    manifest: Manifest,
    output_manifest: str,
    segments_dir: str,
    target_segment_keys: Optional[List[str]] = None,
    max_retries: int = 3,
    force_rerender: bool = False,
) -> Manifest:
    """
    执行 Step 5：渲染所有 Segment 视频（v2）

    缓存策略：
      - 输出路径：render/segments/{content_key}_{render_hash}.mp4
      - render_hash 相同 → 直接复用旧 seg mp4

    :param manifest: 输入 Manifest
    :param output_manifest: 更新后 manifest.json 路径
    :param segments_dir: 输出 segments 目录
    :param target_segment_keys: 只渲染这些 segment_key（None=全部）
    :param max_retries: 最大重试次数
    :param force_rerender: 强制重渲（忽略缓存）
    :return: 更新后的 Manifest
    """
    logger.info("=" * 50)
    logger.info("Step 5: 渲染 Segment 视频 (v2)")

    style = manifest.global_style
    target_keys = set(target_segment_keys) if target_segment_keys is not None else None
    if target_keys is not None:
        manifest_keys = {seg.segment_key for seg in manifest.segments}
        unknown_keys = sorted(target_keys - manifest_keys)
        if unknown_keys:
            joined = ", ".join(unknown_keys)
            raise ValueError(f"Invalid target_segment_keys for Step5: {joined}")

    Path(segments_dir).mkdir(parents=True, exist_ok=True)

    success_count = 0
    skip_count = 0
    fail_count = 0

    for seg in manifest.segments:
        if target_keys is not None and seg.segment_key not in target_keys:
            skip_count += 1
            continue

        # 计算 render_hash
        render_hash = seg.compute_render_hash(style.render_related_fields())

        # 缓存路径：{content_key}_{render_hash}.mp4
        output_path = os.path.join(segments_dir, f"{seg.content_key}_{render_hash}.mp4")

        # 检查渲染缓存
        if not force_rerender and os.path.exists(output_path):
            logger.info(f"  [seg {seg.index}] 渲染缓存命中: {os.path.basename(output_path)}")
            seg.render_ref = RenderRef(
                segment_video_path=output_path,
                render_hash=render_hash,
                status="ok",
            )
            skip_count += 1
            continue

        logger.info(f"  渲染 [{seg.index}/{len(manifest.segments)}]: {seg.text[:30]}...")

        success = render_segment(seg, output_path, style, max_retries)

        if success:
            seg.render_ref = RenderRef(
                segment_video_path=output_path,
                render_hash=render_hash,
                status="ok",
                error=None,
            )
            success_count += 1
        else:
            seg.render_ref = RenderRef(
                segment_video_path=output_path,
                render_hash=render_hash,
                status="failed",
                error="渲染失败，已使用占位片段",
            )
            fail_count += 1

    logger.info(f"Step 5 完成: 成功 {success_count}, 跳过 {skip_count}, 失败 {fail_count}")

    os.makedirs(os.path.dirname(output_manifest), exist_ok=True)
    manifest.save(output_manifest)
    logger.info(f"Manifest 已更新: {output_manifest}")

    return manifest
