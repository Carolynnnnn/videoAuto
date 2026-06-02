"""
Step 1：音频 → 字幕时间轴（SRT 生成）

支持两种策略：
  Strategy A: Whisper ASR 转写（带时间戳）→ 映射到脚本
  Strategy B: 纯 Whisper 转写（不依赖脚本）

优先使用 openai-whisper 本地模型；若不可用则调用 OpenAI Whisper API。
"""
from __future__ import annotations
import os
import re
import json
import difflib
from pathlib import Path
from typing import List, Optional
from openai import OpenAI

from src.core.models import SRTEntry
from src.utils.srt_utils import write_srt, validate_srt, merge_short_segments, split_long_segments
from src.utils.logger import get_logger

logger = get_logger("step1_align")


# ─────────────────────────────────────────────
# 策略 A：OpenAI Whisper API 转写（带时间戳）
# ─────────────────────────────────────────────
def transcribe_with_whisper_api(audio_path: str) -> List[dict]:
    """
    调用 OpenAI Whisper API 转写音频，返回带时间戳的 segments 列表。
    每个 segment: {"start": float, "end": float, "text": str}
    """
    client = OpenAI()
    logger.info(f"调用 Whisper API 转写: {audio_path}")

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    segments = []
    response_segments = response.segments
    if response_segments is None:
        response_segments = []

    for seg in response_segments:
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        })

    logger.info(f"Whisper 转写完成，共 {len(segments)} 段")
    return segments


# ─────────────────────────────────────────────
# 策略 B：本地 Whisper（若已安装）
# ─────────────────────────────────────────────
def transcribe_with_whisper_local(audio_path: str, model_size: str = "base") -> List[dict]:
    """
    使用本地 openai-whisper 转写，返回带时间戳的 segments 列表。
    """
    try:
        import whisper  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError("本地 whisper 未安装，请运行: pip install openai-whisper")

    logger.info(f"本地 Whisper ({model_size}) 转写: {audio_path}")
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language="zh", word_timestamps=False)

    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        })

    logger.info(f"本地 Whisper 转写完成，共 {len(segments)} 段")
    return segments


# ─────────────────────────────────────────────
# 脚本文本 → 句子列表
# ─────────────────────────────────────────────
def parse_script_sentences(script_path: str) -> List[str]:
    """
    从 script.md 解析句子列表（去掉 Markdown 标记）
    """
    text = Path(script_path).read_text(encoding="utf-8")
    # 去掉 Markdown 标题、代码块、注释
    text = re.sub(r"^#+\s.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"\*\*|__|\*|_|~~|`", "", text)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)

    # 按句子分割
    sentences = re.split(r"[。！？\n]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
    return sentences


# ─────────────────────────────────────────────
# 将转写 segments 对齐到脚本句子
# ─────────────────────────────────────────────
def align_to_script(
    whisper_segments: List[dict],
    script_sentences: List[str],
) -> List[SRTEntry]:
    """
    将 Whisper 转写结果对齐到脚本句子。
    使用 SequenceMatcher 做相似度匹配，保留时间戳。
    """
    logger.info(f"对齐脚本句子 ({len(script_sentences)} 句) 与转写结果 ({len(whisper_segments)} 段)")

    # 将 whisper segments 文本拼接为一个大字符串，用于匹配
    whisper_full = " ".join(s["text"] for s in whisper_segments)

    entries: List[SRTEntry] = []
    # 构建字符位置 → 时间戳的映射
    char_to_time = []
    pos = 0
    for seg in whisper_segments:
        text = seg["text"] + " "
        dur = seg["end"] - seg["start"]
        for i, ch in enumerate(text):
            t = seg["start"] + dur * (i / len(text))
            char_to_time.append((pos + i, t))
        pos += len(text)

    def get_time_at_char(char_pos: int) -> float:
        if not char_to_time:
            return 0.0
        for cp, t in char_to_time:
            if cp >= char_pos:
                return t
        return char_to_time[-1][1]

    # 逐句在 whisper_full 中查找最佳匹配位置
    search_start = 0
    for idx, sent in enumerate(script_sentences):
        # 在 whisper_full[search_start:] 中找最相似子串
        window = min(len(sent) * 3, len(whisper_full) - search_start)
        best_ratio = 0.0
        best_pos = search_start
        best_len = len(sent)

        for offset in range(0, max(1, window - len(sent) + 1), max(1, len(sent) // 4)):
            candidate = whisper_full[search_start + offset: search_start + offset + len(sent) + 10]
            ratio = difflib.SequenceMatcher(None, sent, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = search_start + offset
                best_len = len(sent)

        start_time = get_time_at_char(best_pos)
        end_time = get_time_at_char(best_pos + best_len)

        if end_time <= start_time:
            end_time = start_time + max(1.5, len(sent) * 0.15)

        entries.append(SRTEntry(
            index=idx + 1,
            start=round(start_time, 3),
            end=round(end_time, 3),
            text=sent,
        ))
        search_start = best_pos + best_len // 2

    return entries


# ─────────────────────────────────────────────
# 直接使用 Whisper 结果生成 SRT（不对齐脚本）
# ─────────────────────────────────────────────
def whisper_segments_to_srt(whisper_segments: List[dict]) -> List[SRTEntry]:
    """将 Whisper 转写结果直接转为 SRTEntry 列表"""
    entries = []
    for idx, seg in enumerate(whisper_segments, 1):
        entries.append(SRTEntry(
            index=idx,
            start=round(seg["start"], 3),
            end=round(seg["end"], 3),
            text=seg["text"].strip(),
        ))
    return entries


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────
def run_step1(
    audio_path: str,
    output_srt: str,
    script_path: Optional[str] = None,
    use_local_whisper: bool = False,
    whisper_model: str = "base",
    min_segment_duration: float = 1.5,
    max_segment_duration: float = 8.0,
    video_width: int = 1080,
    subtitle_font_size: int = 48,
) -> List[SRTEntry]:
    """
    执行 Step 1：音频 → SRT 字幕

    :param audio_path: 音频文件路径（wav/mp3）
    :param output_srt: 输出 SRT 文件路径
    :param script_path: 脚本文件路径（可选，用于对齐）
    :param use_local_whisper: 是否使用本地 Whisper
    :param whisper_model: 本地 Whisper 模型大小
    :param min_segment_duration: 最小段时长（秒）
    :param max_segment_duration: 最大段时长（秒）
    :return: SRTEntry 列表
    """
    logger.info("=" * 50)
    logger.info("Step 1: 音频 → 字幕时间轴 (SRT)")
    logger.info(f"  音频: {audio_path}")
    logger.info(f"  脚本: {script_path or '(无，直接使用转写结果)'}")
    logger.info("=" * 50)

    # 1. 转写
    if use_local_whisper:
        whisper_segs = transcribe_with_whisper_local(audio_path, whisper_model)
    else:
        whisper_segs = transcribe_with_whisper_api(audio_path)

    # 2. 生成 SRTEntry
    if script_path and Path(script_path).exists():
        script_sentences = parse_script_sentences(script_path)
        if script_sentences:
            entries = align_to_script(whisper_segs, script_sentences)
        else:
            logger.warning("脚本为空，直接使用转写结果")
            entries = whisper_segments_to_srt(whisper_segs)
    else:
        entries = whisper_segments_to_srt(whisper_segs)

    # 3. 合并过短段 & 拆分过长段
    entries = merge_short_segments(
        entries,
        min_duration=min_segment_duration,
        video_width=video_width,
        font_size=subtitle_font_size,
        max_duration=max_segment_duration,
    )
    entries = split_long_segments(
        entries,
        max_duration=max_segment_duration,
        video_width=video_width,
        font_size=subtitle_font_size,
    )

    # 4. 验证
    issues = validate_srt(entries)
    if issues:
        logger.warning(f"SRT 验证发现 {len(issues)} 个问题:")
        for issue in issues:
            logger.warning(f"  {issue}")

    # 5. 写入文件
    write_srt(entries, output_srt)
    logger.info(f"SRT 已写入: {output_srt}  (共 {len(entries)} 段)")

    return entries
