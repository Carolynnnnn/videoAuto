"""
Step 1b: Whisper 字级时间戳对齐 SRT

在 Step 1 生成 SRT 之后、Step 2 构建 Manifest 之前运行。
使用 OpenAI Whisper API 的 word-level timestamps，将每条 SRT 条目的
起止时间精确定位到首字开口 / 尾字收口，消除 silencedetect 误判带来的截断问题。

缓存策略：以音频文件 SHA256 为 key，结果缓存到 {project_root}/cache/whisper_words/。
同一口播重复构建时不重复计费。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import List, Optional

from src.utils.logger import get_logger

logger = get_logger("step1b_word_align")


# ── 工具函数 ──────────────────────────────────────────────────────────────

def _audio_hash(audio_path: str) -> str:
    h = hashlib.sha256()
    with open(audio_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _srt_ts(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _parse_srt(path: str) -> List[dict]:
    entries = []
    blocks = re.split(r"\n\n+", Path(path).read_text(encoding="utf-8").strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        m = re.match(
            r"(\d+):(\d+):(\d+),(\d+)\s+-->\s+(\d+):(\d+):(\d+),(\d+)", lines[1]
        )
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0]*3600 + g[1]*60 + g[2] + g[3]/1000
        end   = g[4]*3600 + g[5]*60 + g[6] + g[7]/1000
        entries.append({
            "idx": int(lines[0].strip()),
            "start": start,
            "end": end,
            "text": " ".join(lines[2:]).strip(),
        })
    return entries


def _write_srt(entries: List[dict], path: str) -> None:
    lines = []
    for e in entries:
        lines += [str(e["idx"]), f"{_srt_ts(e['start'])} --> {_srt_ts(e['end'])}", e["text"], ""]
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ── Whisper API 调用 ──────────────────────────────────────────────────────

def _fetch_word_timestamps(audio_path: str, api_key: str) -> List[dict]:
    """调用 OpenAI whisper-1，返回字级时间戳列表 [{word, start, end}, ...]"""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="zh",
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )
    words = []
    for w in (result.words or []):
        word = w.word.strip()
        if word:  # 跳过空符号（如 % 对应的空词）
            words.append({"word": word, "start": w.start, "end": w.end})
    return words


# ── SRT 对齐核心 ──────────────────────────────────────────────────────────

def _strip_punct(s: str) -> str:
    return re.sub(r"[^\w]", "", s, flags=re.UNICODE).replace("_", "")


def _match_entry(words: List[dict], text: str, search_from: int) -> Optional[tuple]:
    """
    在 words[search_from:] 中找到最匹配 text 的连续词段。
    返回 (first_idx, last_idx)，未找到返回 None。
    """
    target = _strip_punct(text)
    if not target:
        return None
    max_window = min(40, len(words) - search_from)
    for window in range(1, max_window + 1):
        for i in range(search_from, len(words) - window + 1):
            chunk = "".join(_strip_punct(w["word"]) for w in words[i:i+window])
            if not chunk:
                continue
            overlap = len(set(target) & set(chunk)) / max(len(target), 1)
            if overlap >= 0.75 and (target in chunk or chunk in target):
                return i, i + window - 1
    return None


def _align(entries: List[dict], words: List[dict]) -> List[dict]:
    """
    对每条 SRT 条目：
      start = 首词 start
      end   = 下一条首词 start（无空白间隔）；最后一条 = 末词 end + 150ms

    匹配失败时保留原始时间戳并记录警告。
    """
    aligned = []
    cursor = 0

    speech_starts = []  # 每条的首词 start，供后续条目设置 end 用
    matches = []

    for entry in entries:
        match = _match_entry(words, entry["text"], search_from=cursor)
        if match is None:
            logger.warning(f"  [entry {entry['idx']}] 未匹配词序列，保留原时间: {entry['text']!r}")
            matches.append(None)
            speech_starts.append(entry["start"])
        else:
            fi, li = match
            matches.append((fi, li))
            speech_starts.append(words[fi]["start"])
            cursor = fi + 1

    for i, (entry, match) in enumerate(zip(entries, matches)):
        if match is None:
            aligned.append(dict(entry))
            continue

        fi, li = match
        new_start = words[fi]["start"]

        # end = 下一条首词 start（seamless），最后一条加 150ms buffer
        if i + 1 < len(speech_starts):
            new_end = speech_starts[i + 1]
        else:
            new_end = words[li]["end"] + 0.15

        # 防止 end <= start（极端情况）
        if new_end <= new_start:
            new_end = words[li]["end"] + 0.15

        old_start, old_end = entry["start"], entry["end"]
        logger.info(
            f"  [entry {entry['idx']}] {entry['text']!r}  "
            f"start {old_start:.3f}→{new_start:.3f} ({new_start-old_start:+.3f}s)  "
            f"end {old_end:.3f}→{new_end:.3f} ({new_end-old_end:+.3f}s)"
        )
        aligned.append({**entry, "start": new_start, "end": new_end})

    return aligned


# ── 公开入口 ──────────────────────────────────────────────────────────────

def run_step1b(
    audio_path: str,
    srt_path: str,
    project_root: str,
    api_key: Optional[str] = None,
) -> bool:
    """
    对齐 srt_path 中的字幕时间戳并原地更新该文件。
    返回 True 表示成功对齐，False 表示跳过或失败（原文件不变）。
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("Step 1b: 未配置 OPENAI_API_KEY，跳过字级对齐")
        return False

    if not Path(srt_path).exists():
        logger.warning(f"Step 1b: SRT 文件不存在: {srt_path}")
        return False

    cache_dir = Path(project_root) / "cache" / "whisper_words"
    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_hash = _audio_hash(audio_path)
    cache_file = cache_dir / f"{audio_hash}.json"

    logger.info("=" * 50)
    logger.info("Step 1b: Whisper 字级时间戳对齐")

    # 读取或获取词级时间戳
    if cache_file.exists():
        logger.info(f"  词级缓存命中: {cache_file.name}")
        words = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        logger.info(f"  调用 OpenAI whisper-1 获取字级时间戳...")
        try:
            words = _fetch_word_timestamps(audio_path, api_key)
        except Exception as e:
            logger.warning(f"  Whisper API 调用失败，跳过对齐: {e}")
            return False
        cache_file.write_text(json.dumps(words, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"  已缓存 {len(words)} 个词 → {cache_file.name}")

    # 解析并对齐 SRT
    try:
        entries = _parse_srt(srt_path)
        aligned = _align(entries, words)
        _write_srt(aligned, srt_path)
        logger.info(f"Step 1b 完成: {len(aligned)} 条字幕已对齐 → {srt_path}")
        return True
    except Exception as e:
        logger.warning(f"  对齐写入失败，保留原 SRT: {e}")
        return False
