"""
SRT 字幕文件解析与写入工具
"""
from __future__ import annotations
import re
from typing import List, Optional
from pathlib import Path
from src.core.models import SRTEntry

try:
    from src.steps.step5_render import _get_safe_subtitle_width
except Exception:
    def _get_safe_subtitle_width(video_width: int, font_size: int) -> int:
        return int(video_width * 0.92)


def parse_srt(path: str) -> List[SRTEntry]:
    """解析 SRT 文件，返回 SRTEntry 列表"""
    content = Path(path).read_text(encoding="utf-8")
    entries: List[SRTEntry] = []

    # 按空行分割 block
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue

        time_match = re.match(
            r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})",
            lines[1].strip(),
        )
        if not time_match:
            continue

        def to_sec(h, m, s, ms):
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

        start = to_sec(*time_match.groups()[:4])
        end = to_sec(*time_match.groups()[4:])
        text = "\n".join(lines[2:]).strip()

        entries.append(SRTEntry(index=idx, start=start, end=end, text=text))

    return entries


def write_srt(entries: List[SRTEntry], path: str) -> None:
    """将 SRTEntry 列表写入 SRT 文件"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry.to_srt_block())
            f.write("\n")


def validate_srt(entries: List[SRTEntry]) -> List[str]:
    """
    检查 SRT 合法性，返回问题列表。
    检查项：重叠时间段、过长段、过短段
    """
    issues = []
    for i, e in enumerate(entries):
        if e.duration < 0.5:
            issues.append(f"[seg {e.index}] 时长过短: {e.duration:.2f}s (< 0.5s)")
        if e.duration > 10.0:
            issues.append(f"[seg {e.index}] 时长过长: {e.duration:.2f}s (> 10s)")
        if i > 0 and e.start < entries[i - 1].end - 0.05:
            issues.append(
                f"[seg {e.index}] 时间重叠: start={e.start:.3f} < prev_end={entries[i-1].end:.3f}"
            )
        if not e.text.strip():
            issues.append(f"[seg {e.index}] 空字幕文本")
    return issues


def _char_unit_width(char: str) -> float:
    if char.isspace():
        return 0.35
    if ord(char) < 128:
        return 0.56
    return 1.0


def estimate_text_width_px(text: str, font_size: int) -> float:
    return sum(_char_unit_width(ch) for ch in text) * font_size


def split_text_deterministic(text: str, safe_width_px: int, font_size: int) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return [""]

    if estimate_text_width_px(cleaned, font_size) <= safe_width_px:
        return [cleaned]

    sentence_break_chars = set("。！？.!?")
    clause_break_chars = set("，、；：,;:")
    word_break_chars = set(" \t")
    result: List[str] = []
    pending = cleaned

    while pending:
        if estimate_text_width_px(pending, font_size) <= safe_width_px:
            result.append(pending)
            break

        width_px = 0.0
        split_at = -1
        sentence_split = -1
        clause_split = -1
        word_split = -1

        for idx, ch in enumerate(pending):
            next_width = width_px + (_char_unit_width(ch) * font_size)
            if next_width > safe_width_px:
                split_at = idx
                break
            width_px = next_width
            if ch in sentence_break_chars:
                sentence_split = idx + 1
            elif ch in clause_break_chars:
                clause_split = idx + 1
            elif ch in word_break_chars:
                word_split = idx + 1

        if split_at == -1:
            result.append(pending)
            break

        candidate = sentence_split
        if candidate <= 0:
            candidate = clause_split
        if candidate <= 0:
            candidate = word_split
        if candidate <= 0:
            candidate = split_at
        if candidate <= 0:
            candidate = 1

        chunk = pending[:candidate].strip()
        if not chunk:
            chunk = pending[:split_at].strip() or pending[:1]
            candidate = max(split_at, 1)

        result.append(chunk)
        pending = pending[candidate:].strip()

    return result


def merge_short_segments(
    entries: List[SRTEntry],
    min_duration: float = 1.0,
    video_width: int = 1080,
    font_size: int = 48,
    max_duration: Optional[float] = None,
) -> List[SRTEntry]:
    """
    合并过短字幕段（< min_duration 秒）到相邻段
    """
    if not entries:
        return entries

    merged: List[SRTEntry] = []
    safe_width_px = _get_safe_subtitle_width(video_width, font_size)
    i = 0
    while i < len(entries):
        e = entries[i]
        if e.duration < min_duration and i + 1 < len(entries):
            # 合并到下一段
            next_e = entries[i + 1]
            merged_text = f"{e.text} {next_e.text}".strip()
            merged_duration = next_e.end - e.start
            width_ok = estimate_text_width_px(merged_text, font_size) <= safe_width_px
            duration_ok = max_duration is None or merged_duration <= max_duration
            if not width_ok or not duration_ok:
                merged.append(e)
                i += 1
                continue
            combined = SRTEntry(
                index=e.index,
                start=e.start,
                end=next_e.end,
                text=merged_text,
            )
            merged.append(combined)
            i += 2
        else:
            merged.append(e)
            i += 1

    # 重新编号
    for idx, entry in enumerate(merged, 1):
        entry.index = idx

    return merged


def split_long_segments(
    entries: List[SRTEntry],
    max_duration: float = 8.0,
    video_width: int = 1080,
    font_size: int = 48,
) -> List[SRTEntry]:
    """
    拆分过长字幕段（> max_duration 秒），按句号/逗号分割文本
    """
    result: List[SRTEntry] = []
    safe_width_px = _get_safe_subtitle_width(video_width, font_size)
    for e in entries:
        text_width = estimate_text_width_px(e.text, font_size)
        if e.duration <= max_duration and text_width <= safe_width_px:
            result.append(e)
            continue

        sentences = split_text_deterministic(
            text=e.text,
            safe_width_px=safe_width_px,
            font_size=font_size,
        )

        if len(sentences) <= 1 and e.duration <= max_duration:
            result.append(e)
            continue

        if len(sentences) <= 1:
            split_count = max(2, int(e.duration / max_duration) + 1)
            text_value = e.text.strip() or "[split]"
            sentences = [text_value] * split_count

        # 按文字数量比例分配时间
        total_chars = max(1, sum(max(1, len(s)) for s in sentences))
        cur_start = e.start
        for i, sent in enumerate(sentences):
            ratio = max(1, len(sent)) / total_chars
            dur = e.duration * ratio
            sub = SRTEntry(
                index=e.index,
                start=round(cur_start, 3),
                end=round(e.end if i == len(sentences) - 1 else (cur_start + dur), 3),
                text=sent,
            )
            result.append(sub)
            cur_start += dur

    # 重新编号
    for idx, entry in enumerate(result, 1):
        entry.index = idx

    return result
