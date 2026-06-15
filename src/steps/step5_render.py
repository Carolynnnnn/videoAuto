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
# Render Engine Version
# 更改此值可使全部渲染缓存失效，强制重渲
# ─────────────────────────────────────────────
RENDER_ENGINE_VERSION = "v9"   # v9: 段落不烘焙字幕，step6 叠加 SRT（天然口播同步）

# ─────────────────────────────────────────────
# Overflow Safety Constants
# ─────────────────────────────────────────────
# Maximum subtitle width as ratio of video width (75% of video width)
# 保守值：中英混排时 FFmpeg 实际渲染宽度比 Python 估算偏大，留出足够余量
MAX_SUBTITLE_WIDTH_RATIO = 0.75

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


def _get_video_duration(path: str) -> float:
    """用 ffprobe 获取视频文件时长（秒）。失败时返回 0.0。"""
    cmd = (
        f'ffprobe -v error -select_streams v:0 '
        f'-show_entries stream=duration '
        f'-of default=noprint_wrappers=1:nokey=1 "{path}"'
    )
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        val = result.stdout.strip()
        if val and val != "N/A":
            return float(val)
        # 有些容器不在 stream 层报 duration，尝试 format 层
        cmd2 = (
            f'ffprobe -v error -show_entries format=duration '
            f'-of default=noprint_wrappers=1:nokey=1 "{path}"'
        )
        result2 = subprocess.run(cmd2, shell=True, capture_output=True, text=True, timeout=10)
        val2 = result2.stdout.strip()
        return float(val2) if val2 and val2 != "N/A" else 0.0
    except Exception:
        return 0.0


def _escape_ffmpeg_text(text: str) -> str:
    text = text.replace("\\", "\\\\")
    text = text.replace("\n", r"\n")
    text = text.replace("'", "\\'")
    text = text.replace(":", "\\:")
    text = text.replace("[", "\\[").replace("]", "\\]")
    text = text.replace(",", r"\,")
    text = text.replace("%", "％")   # FFmpeg 8 drawtext 中 % 无法可靠转义，用全角 ％ (U+FF05) 替代，视觉一致
    return text


_CJK_FONT_PATHS = [
    # macOS — 无空格路径优先（避免 FFmpeg filter 字符串解析失败）
    "/System/Library/Fonts/Supplemental/Songti.ttc",     # 宋体（无空格，首选）
    "/Library/Fonts/Arial Unicode.ttf",                  # Arial Unicode（无空格）
    # macOS — 路径含空格（在 _resolve_cjk_font_path 中会自动转义）
    "/System/Library/Fonts/STHeiti Medium.ttc",          # 黑体-简
    "/System/Library/Fonts/STHeiti Light.ttc",           # 黑体-繁
    "/System/Library/Fonts/Hiragino Sans GB.ttc",        # 冬青黑体
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    # Linux 字体路径（服务器/Docker 环境）
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]


def _resolve_cjk_font_path() -> str:
    """返回可用 CJK 字体路径。路径中的空格会被转义为 \\ ，防止 FFmpeg filter 解析失败。"""
    for fp in _CJK_FONT_PATHS:
        if Path(fp).exists():
            return fp.replace(" ", "\\ ")  # FFmpeg filter 路径空格转义
    return ""


def _char_unit_width(char: str) -> float:
    if char.isspace():
        return 0.35
    if ord(char) < 128:
        # STHeiti 等 CJK 字体中 Latin/数字实际渲染接近 0.75 倍字号
        # 保守偏高估算：宁可换行早一点，也不让文字溢出画面右边
        return 0.75
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

    # 行首禁止出现的标点（中文排版禁则）
    _NO_START_PUNCT = set("，。！？；：、…,.")

    def _fix_line_start(lines: List[str], pending: str) -> str:
        """把 pending 开头的禁则标点移到上一行末尾，返回修正后的 pending。"""
        while pending and pending[0] in _NO_START_PUNCT and lines:
            lines[-1] = lines[-1] + pending[0]
            pending = pending[1:].lstrip(" ")
        return pending

    lines: List[str] = []
    pending = cleaned
    # 可在此处断行的字符：空格、CJK 标点（ASCII 标点不单独断，与前词保持一致）
    break_chars = set(" ，。！？；：、")

    while pending and len(lines) < max_lines:
        width_px = 0.0
        last_break_index = -1   # 最近一个合法断点位置（包含该字符）

        for idx, ch in enumerate(pending):
            ch_w = _char_unit_width(ch) * font_size
            next_width = width_px + ch_w

            if next_width > max_width_px and idx > 0:
                # ── 确定断点 ──────────────────────────────────────────
                if last_break_index >= 0:
                    # 在上一个合法断点后截断
                    split_index = last_break_index + 1
                else:
                    # 无合法断点：先尝试回退到当前 ASCII 词首
                    word_start = idx
                    while word_start > 0 and ord(pending[word_start - 1]) < 128 \
                            and not pending[word_start - 1].isspace():
                        word_start -= 1

                    if word_start > 0:
                        # 词首有内容（词前有其他字符）→ 在词首截断，保持单词完整
                        split_index = word_start
                    else:
                        # 整行第一个词就超长：向后扫描找到词尾的第一个空格
                        # 让这个超长词独占一行（允许视觉溢出），不截断单词
                        word_end = idx
                        while word_end < len(pending) and not pending[word_end].isspace():
                            word_end += 1
                        if word_end < len(pending):
                            # 找到了词尾空格，在空格后截断
                            split_index = word_end + 1
                        else:
                            # 整个 pending 都没有空格（单一超长词），不得不截断
                            split_index = idx

                line = pending[:split_index].strip()
                if not line:          # 极端情况：split_index=0，强制取第一个字符
                    line = pending[0]
                    split_index = 1
                lines.append(line)
                pending = pending[split_index:].lstrip(" ")
                # ── 行首禁则：标点不允许出现在下一行开头 ─────────────
                pending = _fix_line_start(lines, pending)
                break

            width_px = next_width
            if ch in break_chars:
                last_break_index = idx
        else:
            # 当前 pending 完全放得下，收入最后一行
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
    """
    运动滤镜：先将图片 prescale 到精确输出分辨率（cover+居中裁剪），
    再对已对齐帧应用 zoompan。

    关键设计：
      - prescale 确保 zoompan 输入尺寸 == 输出尺寸（1080×1920）
        → z=1.0 永远安全，彻底消除"输入<输出"导致的 zoompan 抖动
      - zoompan 使用纯参数式 z='1.0+on*speed'（无反馈变量），无起始帧跳变
      - 不使用 crop 的 t 变量（浮点边界溢出会导致末尾跳帧抖动）
    """
    fps = 30
    total_frames = max(1, int(duration * fps))
    zoom_speed = speed * 0.0015

    # prescale：将任意输入图片缩放为精确 width×height（cover 模式，中心裁剪）
    prescale = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"setsar=1"
    )

    if preset == "soft_kenburns":
        max_zoom = 1.08
        return (
            f"{prescale},"
            f"zoompan=z='min(1.0+on*{zoom_speed:.6f}\\,{max_zoom})'"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={total_frames}:s={width}x{height}:fps={fps}"
            f",setsar=1"
        )
    elif preset == "push_in":
        max_zoom = 1.25
        zs = zoom_speed * 1.5
        return (
            f"{prescale},"
            f"zoompan=z='min(1.0+on*{zs:.6f}\\,{max_zoom})'"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={total_frames}:s={width}x{height}:fps={fps}"
            f",setsar=1"
        )
    elif preset == "zoom_out":
        start_zoom = 1.25
        zs = zoom_speed * 1.5
        return (
            f"{prescale},"
            f"zoompan=z='max(1.0\\,{start_zoom}-on*{zs:.6f})'"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={total_frames}:s={width}x{height}:fps={fps}"
            f",setsar=1"
        )
    elif preset == "pan_left":
        pan_px = int(width * 0.04)
        return (
            f"{prescale},"
            f"zoompan=z='1.0':x='iw/2-(iw/zoom/2)+{pan_px}*on/{total_frames}'"
            f":y='ih/2-(ih/zoom/2)':d={total_frames}:s={width}x{height}:fps={fps}"
            f",setsar=1"
        )
    elif preset == "pan_right":
        pan_px = int(width * 0.04)
        return (
            f"{prescale},"
            f"zoompan=z='1.0':x='iw/2-(iw/zoom/2)-{pan_px}*on/{total_frames}'"
            f":y='ih/2-(ih/zoom/2)':d={total_frames}:s={width}x{height}:fps={fps}"
            f",setsar=1"
        )
    else:  # static
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1"
        )


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
    # Emphasis token 位置基于 Python 字符宽度估算，中英混排时与 FFmpeg 实际渲染偏差过大
    # 导致黄色高亮文字叠在主字幕上方造成遮挡，暂时完全禁用
    emphasis_tokens: List[str] = []

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
            f":borderw=2"
            f":bordercolor=black@0.9"
            f":shadowx=2:shadowy=2:shadowcolor=black@0.8"
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
    video_start_offset: float = 0.0,
    video_pts_factor: float = 1.0,
    effective_duration: Optional[float] = None,
    skip_subtitle: bool = False,
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
    # audio_duration：实际口播时长（用于音频输入的 -t 限制）
    audio_duration = trim_end - trim_start if trim_end > trim_start else duration
    # seg_out_duration：本段视频+字幕的输出时长
    #   = effective_duration（由调用方传入，含本段口播后的静音间隙）
    #   若未传入则回退到 audio_duration
    # 关键：seg_out_duration 必须覆盖到下一段开始，保证视频+原始音频时间轴完全对齐
    seg_out_duration = effective_duration if (effective_duration and effective_duration > 0) else audio_duration

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
            subtitle_filter = "" if skip_subtitle else _build_subtitle_filter(segment, style, width, height)
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
                # 不用 -ss/-t 输入裁剪，改为滤镜链内精确裁剪：
                #   setpts={pts_factor}*PTS  → 拉伸/压缩视频时间轴，使单次播放恰好覆盖所有使用该素材的口播段
                #   trim=start=…:duration=… → 取出本段对应的时间窗口
                #   setpts=PTS-STARTPTS      → 重置时间戳从 0 开始
                video_input = (
                    f'-i "{asset_path}" '
                    f'{sticker_inputs_part}'
                    f'-ss {trim_start:.3f} -t {audio_duration:.3f} -i "{audio_path}" '
                )
                video_filter = (
                    f"[0:v]setpts={video_pts_factor:.6f}*PTS,"
                    f"trim=start={video_start_offset:.3f}:duration={seg_out_duration:.3f},"
                    f"setpts=PTS-STARTPTS,"
                    f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                    f"crop={width}:{height},setsar=1,fps={fps}[v]"
                )
            else:
                motion_filter = _build_motion_filter(
                    motion_preset, motion_speed, seg_out_duration, width, height
                )
                video_input = (
                    # -framerate 30 强制输入流为 30fps，确保 crop 滤镜中 t 变量以
                    # 1/30s 为步进，避免默认 25fps 导致的重复帧 judder（画面抖动）
                    f'-loop 1 -t {seg_out_duration:.3f} -framerate {fps} -i "{asset_path}" '
                    f'{sticker_inputs_part}'
                    f'-ss {trim_start:.3f} -t {audio_duration:.3f} -i "{audio_path}" '
                )
                # motion_filter 末尾已含 setsar=1,fps={fps}，此处不重复追加
                video_filter = f"[0:v]{motion_filter}[v]"

            # Construct full filter_complex
            # sub_out: stream label after optional subtitle baking
            if subtitle_filter:
                sub_part = f";[v]{subtitle_filter}[vsub]"
                sub_out = "[vsub]"
            else:
                sub_part = ""
                sub_out = "[v]"

            if sticker_filters:
                full_filter = f"{video_filter}{sub_part}"
                full_filter += ";" + ";".join(sticker_filters)
                current_stream = sub_out
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
                full_filter = f"{video_filter}{sub_part};{sub_out}copy[vout]"

            # Calculate audio stream index (0=asset, 1..N=stickers, N+1=audio)
            audio_stream_index = len(sticker_filters) + 1

            if Path(audio_path).exists():
                cmd = (
                    f'ffmpeg -y '
                    f'{video_input}'
                    f'-filter_complex "{full_filter}" '
                    f'-map "[vout]" -map {audio_stream_index}:a '
                    f'-t {seg_out_duration:.3f} '
                    f'-c:v libx264 -preset fast -crf 23 '
                    f'-c:a aac -b:a 128k '
                    f'-pix_fmt yuv420p '
                    f'"{output_path}" -loglevel warning'
                )
            else:
                cmd = (
                    f'ffmpeg -y '
                    f'-loop 1 -t {seg_out_duration:.3f} -i "{asset_path}" '
                    f'-filter_complex "{full_filter}" '
                    f'-map "[vout]" '
                    f'-t {seg_out_duration:.3f} '
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
    skip_subtitle: bool = True,
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

    # ── 预计算每段的有效输出时长（含段后静音间隙，对齐原始音频时间轴）────────────
    # effective_duration[i] = segments[i+1].trim_start - segments[i].trim_start
    # 最后一段 = trim_end - trim_start
    # 这样所有段视频时长之和 = 原始音频总时长，保证音画无漂移
    _all_segs = manifest.segments
    _effective_dur: Dict[str, float] = {}   # segment_key → effective_duration
    _fps = manifest.global_style.fps or 30
    for _i, _s in enumerate(_all_segs):
        _ar = _s.audio_ref
        if _i + 1 < len(_all_segs):
            _next_ar = _all_segs[_i + 1].audio_ref
            _ed = _next_ar.trim_start - _ar.trim_start
        else:
            _ed = _ar.trim_end - _ar.trim_start
        # 确保不低于实际口播时长（防止负间隙异常）
        _min_ed = _ar.trim_end - _ar.trim_start if _ar.trim_end > _ar.trim_start else _s.duration
        _ed = max(_ed, _min_ed)
        # 取整到帧边界（round 到最近帧），消除帧率取整造成的跨段累积漂移
        _ed_frames = round(_ed * _fps)
        _effective_dur[_s.segment_key] = _ed_frames / _fps

    # ── 预计算每个视频素材需要覆盖的总时长（用于计算播放速度因子）────────────
    _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
    _video_total_needed: Dict[str, float] = {}
    for _s in manifest.segments:
        _a = (_s.visual_plan.asset_path if _s.visual_plan else None) or \
             (_s.asset_refs[0].path if _s.asset_refs else None)
        if _a and Path(_a).suffix.lower() in _VIDEO_EXTS:
            # 使用 effective_duration（含间隙），与视频实际输出时长一致
            _video_total_needed[_a] = _video_total_needed.get(_a, 0.0) + _effective_dur.get(_s.segment_key, _s.duration)

    # 缓存 ffprobe 结果（避免同一文件多次调用）
    _video_file_dur_cache: Dict[str, float] = {}

    def _cached_video_dur(path: str) -> float:
        if path not in _video_file_dur_cache:
            _video_file_dur_cache[path] = _get_video_duration(path)
        return _video_file_dur_cache[path]

    _prev_asset_path: Optional[str] = None  # 用于检测连续相同图片
    _video_playback_pos: Dict[str, float] = {}  # 每个视频素材的累计播放偏移

    for seg in manifest.segments:
        if target_keys is not None and seg.segment_key not in target_keys:
            skip_count += 1
            continue

        # 同图连续段抑制：若当前段与上一段使用相同图片，强制 static motion
        # 避免每段 Ken Burns 从 z=1.0 重置时产生的可见"跳回"抖动
        _cur_asset = (
            (seg.visual_plan.asset_path if seg.visual_plan else None)
            or (seg.asset_refs[0].path if seg.asset_refs else None)
        )
        if _cur_asset and _cur_asset == _prev_asset_path and seg.visual_plan:
            seg.visual_plan.motion.preset = "static"
        _prev_asset_path = _cur_asset

        # 计算视频素材参数（需在 render_hash 前确定，以纳入哈希）
        _asset_ext = Path(_cur_asset).suffix.lower() if _cur_asset else ""
        _is_vid = _asset_ext in _VIDEO_EXTS
        # 累计偏移：在 setpts 拉伸后的时间轴上的起始位置（单位：秒，已是拉伸后时间）
        _video_offset = _video_playback_pos.get(_cur_asset, 0.0) if (_cur_asset and _is_vid) else 0.0
        # pts 因子：让视频单次播完恰好覆盖所有使用该素材的口播段总时长
        if _is_vid and _cur_asset:
            _file_dur = _cached_video_dur(_cur_asset)
            _total_needed = _video_total_needed.get(_cur_asset, seg.duration)
            # 只在视频不够长时才减速（拉伸到恰好覆盖），视频足够长时保持原速（不加速）
            # pts_factor > 1 → 视频变慢；pts_factor = 1 → 原速；< 1 → 不允许
            _pts_factor = max(1.0, _total_needed / _file_dur) if _file_dur > 0.1 else 1.0
        else:
            _pts_factor = 1.0

        # 计算 render_hash（含渲染引擎版本 + 偏移 + pts因子，保证缓存隔离）
        _eff_dur = _effective_dur.get(seg.segment_key, seg.duration)
        _sub_tag = "nosub" if skip_subtitle else "sub"
        _offset_extra = (
            f"{RENDER_ENGINE_VERSION}|{_sub_tag}|voff={_video_offset:.3f}|pts={_pts_factor:.6f}|ed={_eff_dur:.3f}"
            if _is_vid else f"{RENDER_ENGINE_VERSION}|{_sub_tag}|ed={_eff_dur:.3f}"
        )
        render_hash = seg.compute_render_hash(style.render_related_fields(), extra=_offset_extra)

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
            # 缓存命中时仍需推进播放位置（以 effective_duration 为准）
            if _cur_asset and _is_vid:
                _video_playback_pos[_cur_asset] = _video_offset + _eff_dur
            skip_count += 1
            continue

        logger.info(f"  渲染 [{seg.index}/{len(manifest.segments)}]: {seg.text[:30]}...")

        success = render_segment(
            seg, output_path, style, max_retries,
            video_start_offset=_video_offset,
            video_pts_factor=_pts_factor,
            effective_duration=_eff_dur,
            skip_subtitle=skip_subtitle,
        )

        # 无论成功与否，推进该视频素材的播放位置（以 effective_duration 为准）
        if _cur_asset and _is_vid:
            _video_playback_pos[_cur_asset] = _video_offset + _eff_dur

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
