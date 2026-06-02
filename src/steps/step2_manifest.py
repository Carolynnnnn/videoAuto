"""
Step 2：SRT → Manifest 生成（v2）

关键变更：
  - Segment 标识策略升级为 content_key + occurrence_index
  - segment_key = content_key + '#' + occurrence_index
  - 不使用 start_time 作为主键
  - 支持短字幕合并（< min_duration）和长字幕拆分（> max_duration）
"""
from __future__ import annotations
import os
import math
from typing import List, Optional, Dict, Literal, cast, Any, Tuple
from datetime import datetime

from src.core.models import Manifest, Segment, AudioRef, GlobalStyle, MATERIAL_MODES, MaterialModeError, BudgetDiagnostics
from src.core.generation_policy import (
    TARGET_DURATION_MINUTES_DEFAULT,
    AI_CLIP_CAP_DEFAULT,
    minutes_to_seconds,
)
from src.utils.srt_utils import (
    parse_srt,
    SRTEntry,
    estimate_text_width_px,
    split_text_deterministic,
    _get_safe_subtitle_width,
)
from src.utils.logger import get_logger

logger = get_logger("step2_manifest")

# 默认时长限制
DEFAULT_MIN_DURATION = 1.5   # 秒：低于此值自动合并到前/后段
DEFAULT_MAX_DURATION = 10.0  # 秒：超过此值自动拆段

DEFAULT_TARGET_MIN_DURATION = 8.0
DEFAULT_TARGET_MAX_DURATION = 10.0
DEFAULT_MERGE_THRESHOLD = 3.0


def _reassign_occurrence_index(items: List[Dict]) -> List[Dict]:
    count: Dict[str, int] = {}
    for item in items:
        ck = item["content_key"]
        count[ck] = count.get(ck, 0) + 1
        item["occurrence_index"] = count[ck]
    return items


def _build_duration_policy(
    min_duration: float,
    max_duration: float,
    duration_policy: Optional[Dict[str, float]],
) -> Dict[str, float]:
    base_min = float(min_duration)
    base_max = float(max_duration)

    if base_min <= 0:
        raise ValueError("duration policy invalid: min_duration must be > 0")
    if base_max <= 0:
        raise ValueError("duration policy invalid: max_duration must be > 0")
    if base_max < base_min:
        raise ValueError("duration policy invalid: max_duration must be >= min_duration")

    normalized: Dict[str, float] = {
        "min_duration": base_min,
        "max_duration": base_max,
        "target_min_duration": max(base_min, min(DEFAULT_TARGET_MIN_DURATION, base_max)),
        "target_max_duration": min(DEFAULT_TARGET_MAX_DURATION, base_max),
        "merge_threshold": max(base_min, min(DEFAULT_MERGE_THRESHOLD, base_max)),
        "split_threshold": base_max,
    }

    if duration_policy:
        for key in normalized.keys():
            if key in duration_policy and duration_policy[key] is not None:
                normalized[key] = float(duration_policy[key])
        if "target_duration_minutes" in duration_policy:
            normalized["target_duration_minutes"] = duration_policy["target_duration_minutes"]
        if "ai_clip_cap" in duration_policy:
            normalized["ai_clip_cap"] = duration_policy["ai_clip_cap"]

    if normalized["max_duration"] < normalized["min_duration"]:
        raise ValueError("duration policy invalid: max_duration must be >= min_duration")
    if normalized["target_max_duration"] < normalized["target_min_duration"]:
        raise ValueError("duration policy invalid: target_max_duration must be >= target_min_duration")
    if normalized["target_min_duration"] < normalized["min_duration"]:
        raise ValueError("duration policy invalid: target_min_duration must be >= min_duration")
    if normalized["target_max_duration"] > normalized["max_duration"]:
        raise ValueError("duration policy invalid: target_max_duration must be <= max_duration")
    if normalized["merge_threshold"] < normalized["min_duration"]:
        raise ValueError("duration policy invalid: merge_threshold must be >= min_duration")
    if normalized["merge_threshold"] > normalized["target_min_duration"]:
        raise ValueError("duration policy invalid: merge_threshold must be <= target_min_duration")
    if normalized["split_threshold"] < normalized["target_max_duration"]:
        raise ValueError("duration policy invalid: split_threshold must be >= target_max_duration")
    if normalized["split_threshold"] > normalized["max_duration"]:
        raise ValueError("duration policy invalid: split_threshold must be <= max_duration")

    return normalized


def _assign_occurrence_index(entries: List[SRTEntry]) -> List[Dict]:
    """
    为每个 SRT 条目分配 occurrence_index（同 content_key 的第几次出现）。
    返回 [{entry, content_key, occurrence_index}, ...]
    """
    count: Dict[str, int] = {}
    result = []
    for entry in entries:
        ck = Segment.compute_content_key(entry.text)
        count[ck] = count.get(ck, 0) + 1
        result.append({
            "entry": entry,
            "content_key": ck,
            "occurrence_index": count[ck],
        })
    return result


def _merge_short_entries(
    items: List[Dict],
    merge_threshold: float,
    safe_width_px: int,
    font_size: int,
    max_merge_duration: float,
) -> List[Dict]:
    """
    合并过短的条目（< min_duration）到相邻段。
    合并规则：优先合并到前一段；若是第一段则合并到后一段。
    """
    if not items:
        return items

    merged = []
    i = 0
    while i < len(items):
        item = items[i]
        entry = item["entry"]
        duration = entry.end - entry.start

        if duration < merge_threshold and len(items) > 1:
            if merged:
                # 合并到前一段
                prev = merged[-1]
                prev_entry = prev["entry"]
                merged_text = f"{prev_entry.text} {entry.text}".strip()
                merged_duration = entry.end - prev_entry.start
                if (
                    estimate_text_width_px(merged_text, font_size) <= safe_width_px
                    and merged_duration <= max_merge_duration
                ):
                    prev_entry.end = entry.end
                    prev_entry.text = merged_text
                    prev["content_key"] = Segment.compute_content_key(prev_entry.text)
                    logger.debug(f"  合并短段 [{entry.text[:20]}] → 前一段")
                else:
                    merged.append(item)
            elif i + 1 < len(items):
                # 合并到后一段
                next_item = items[i + 1]
                next_entry = next_item["entry"]
                merged_text = f"{entry.text} {next_entry.text}".strip()
                merged_duration = next_entry.end - entry.start
                if (
                    estimate_text_width_px(merged_text, font_size) <= safe_width_px
                    and merged_duration <= max_merge_duration
                ):
                    next_entry.start = entry.start
                    next_entry.text = merged_text
                    next_item["content_key"] = Segment.compute_content_key(next_entry.text)
                    logger.debug(f"  合并短段 [{entry.text[:20]}] → 后一段")
                else:
                    merged.append(item)
        else:
            merged.append(item)
        i += 1

    return merged


def _split_long_entries(
    items: List[Dict],
    split_threshold: float,
    target_max_duration: float,
    safe_width_px: int,
    font_size: int,
) -> List[Dict]:
    """
    拆分过长的条目（> max_duration）。
    """
    result = []
    for item in items:
        entry = item["entry"]
        duration = entry.end - entry.start
        width_px = estimate_text_width_px(entry.text, font_size)

        if duration <= split_threshold and width_px <= safe_width_px:
            result.append(item)
            continue

        sentences = split_text_deterministic(
            text=entry.text,
            safe_width_px=safe_width_px,
            font_size=font_size,
        )

        if len(sentences) <= 1:
            n = max(2, math.ceil(duration / max(target_max_duration, 0.001)))
            text_value = entry.text.strip() or "[split]"
            sentences = [text_value] * n

        if len(sentences) >= 2:
            n = len(sentences)
            seg_dur = duration / n
            for j, sent in enumerate(sentences):
                new_entry = SRTEntry(
                    index=entry.index,
                    start=round(entry.start + j * seg_dur, 3),
                    end=round(entry.end if j == n - 1 else (entry.start + (j + 1) * seg_dur), 3),
                    text=sent,
                )
                ck = Segment.compute_content_key(sent)
                result.append({"entry": new_entry, "content_key": ck, "occurrence_index": 0})
            logger.debug(f"  拆分长段 [{entry.text[:20]}...] → {n} 段")

    return _reassign_occurrence_index(result)


def _merge_subtarget_entries(
    items: List[Dict],
    target_min_duration: float,
    target_max_duration: float,
    safe_width_px: int,
    font_size: int,
) -> List[Dict]:
    if not items:
        return items

    merged: List[Dict[str, Any]] = []
    i = 0
    while i < len(items):
        item = items[i]
        entry = item["entry"]
        duration = entry.end - entry.start

        if duration >= target_min_duration:
            merged.append(item)
            i += 1
            continue

        if i + 1 < len(items):
            next_item = items[i + 1]
            next_entry = next_item["entry"]
            merged_text = f"{entry.text} {next_entry.text}".strip()
            merged_duration = next_entry.end - entry.start
            if (
                estimate_text_width_px(merged_text, font_size) <= safe_width_px
                and merged_duration <= target_max_duration
            ):
                next_entry.start = entry.start
                next_entry.text = merged_text
                next_item["content_key"] = Segment.compute_content_key(next_entry.text)
                i += 1
                continue

        if merged:
            prev = merged[-1]
            prev_entry = prev["entry"]
            merged_text = f"{prev_entry.text} {entry.text}".strip()
            merged_duration = entry.end - prev_entry.start
            if (
                estimate_text_width_px(merged_text, font_size) <= safe_width_px
                and merged_duration <= target_max_duration
            ):
                prev_entry.end = entry.end
                prev_entry.text = merged_text
                prev["content_key"] = Segment.compute_content_key(prev_entry.text)
                i += 1
                continue

        merged.append(item)
        i += 1

    return _reassign_occurrence_index(cast(List[Dict], merged))


def _apply_duration_budget(
    items: List[Dict],
    target_duration_seconds: float,
    requested_minutes: int,
) -> Tuple[List[Dict], BudgetDiagnostics]:
    total_available = sum(item["entry"].end - item["entry"].start for item in items) if items else 0.0
    
    if not items or target_duration_seconds <= 0:
        diagnostics = BudgetDiagnostics(
            requested_minutes=requested_minutes,
            target_seconds=target_duration_seconds,
            total_available_seconds=total_available,
            effective_selected_seconds=total_available,
            selected_count=len(items),
            dropped_count=0,
            budget_exhausted=False,
        )
        return items, diagnostics

    if total_available <= target_duration_seconds:
        diagnostics = BudgetDiagnostics(
            requested_minutes=requested_minutes,
            target_seconds=target_duration_seconds,
            total_available_seconds=total_available,
            effective_selected_seconds=total_available,
            selected_count=len(items),
            dropped_count=0,
            budget_exhausted=False,
        )
        return items, diagnostics

    selected: List[Dict] = []
    accumulated = 0.0
    for item in items:
        selected.append(item)
        accumulated += item["entry"].end - item["entry"].start
        if accumulated >= target_duration_seconds:
            break

    dropped_count = len(items) - len(selected)
    diagnostics = BudgetDiagnostics(
        requested_minutes=requested_minutes,
        target_seconds=target_duration_seconds,
        total_available_seconds=round(total_available, 3),
        effective_selected_seconds=round(accumulated, 3),
        selected_count=len(selected),
        dropped_count=dropped_count,
        budget_exhausted=True,
    )
    return selected, diagnostics


def srt_to_manifest(
    srt_path: str,
    audio_path: str,
    project_id: str,
    global_style: Optional[GlobalStyle] = None,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    material_mode: str = "auto",
    duration_policy: Optional[Dict[str, float]] = None,
) -> Manifest:
    """
    将 SRT 文件解析为 Manifest（v2，使用 segment_key 策略）。
    """
    if material_mode not in MATERIAL_MODES:
        raise MaterialModeError(material_mode, MATERIAL_MODES)
    
    if global_style is None:
        global_style = GlobalStyle()

    logger.info("=" * 50)
    logger.info("Step 2: SRT → Manifest 生成 (v2)")
    logger.info(f"  SRT: {srt_path}")
    logger.info(f"  音频: {audio_path}")
    logger.info("=" * 50)

    entries = parse_srt(srt_path)
    if not entries:
        raise ValueError(f"SRT 文件为空或解析失败: {srt_path}")
    logger.info(f"  解析 SRT: {len(entries)} 条")

    safe_width_px = _get_safe_subtitle_width(global_style.resolution_w, global_style.font_size)
    policy = _build_duration_policy(
        min_duration=min_duration,
        max_duration=max_duration,
        duration_policy=duration_policy,
    )
    logger.info(
        "  时长策略: "
        f"min={policy['min_duration']:.2f}s "
        f"max={policy['max_duration']:.2f}s "
        f"target={policy['target_min_duration']:.2f}-{policy['target_max_duration']:.2f}s "
        f"merge<{policy['merge_threshold']:.2f}s "
        f"split>{policy['split_threshold']:.2f}s"
    )

    items = _assign_occurrence_index(entries)
    items = _merge_short_entries(
        items,
        merge_threshold=policy["merge_threshold"],
        safe_width_px=safe_width_px,
        font_size=global_style.font_size,
        max_merge_duration=policy["target_max_duration"],
    )
    logger.info(f"  合并短段后: {len(items)} 条")
    items = _split_long_entries(
        items,
        split_threshold=policy["split_threshold"],
        target_max_duration=policy["target_max_duration"],
        safe_width_px=safe_width_px,
        font_size=global_style.font_size,
    )
    logger.info(f"  拆分长段后: {len(items)} 条")
    items = _merge_subtarget_entries(
        items,
        target_min_duration=policy["target_min_duration"],
        target_max_duration=policy["target_max_duration"],
        safe_width_px=safe_width_px,
        font_size=global_style.font_size,
    )
    logger.info(f"  压缩短窗口后: {len(items)} 条")

    target_duration_minutes = int(policy.get("target_duration_minutes", TARGET_DURATION_MINUTES_DEFAULT))
    target_duration_seconds = float(minutes_to_seconds(target_duration_minutes))
    items, budget_diagnostics = _apply_duration_budget(
        items,
        target_duration_seconds=target_duration_seconds,
        requested_minutes=target_duration_minutes,
    )
    logger.info(
        f"  应用时长预算后: {len(items)} 条 "
        f"(target={target_duration_minutes}m/{target_duration_seconds:.1f}s)"
    )
    logger.info(
        f"  预算诊断: selected={budget_diagnostics.selected_count}, "
        f"dropped={budget_diagnostics.dropped_count}, "
        f"effective={budget_diagnostics.effective_selected_seconds:.1f}s, "
        f"exhausted={budget_diagnostics.budget_exhausted}"
    )

    segments: List[Segment] = []
    for idx, item in enumerate(items, start=1):
        entry = item["entry"]
        content_key = item["content_key"]
        occurrence_index = item["occurrence_index"]
        segment_key = Segment.compute_segment_key(content_key, occurrence_index)

        seg = Segment(
            segment_key=segment_key,
            content_key=content_key,
            index=idx,
            start=entry.start,
            end=entry.end,
            duration=round(entry.end - entry.start, 3),
            text=entry.text,
            audio_ref=AudioRef(
                type="full",
                path=audio_path,
                trim_start=entry.start,
                trim_end=entry.end,
            ),
        )
        segments.append(seg)
        logger.debug(
            f"  Segment [{idx:03d}] key={segment_key[:20]} "
            f"{entry.start:.2f}→{entry.end:.2f}s '{entry.text[:30]}'"
        )

    manifest = Manifest(
        project_id=project_id,
        build_id=datetime.utcnow().isoformat() + "Z",
        global_style=global_style,
        segments=segments,
        audio_path=audio_path,
        build_status="pending",
        material_mode=cast(Literal["auto", "ai_preferred", "ai_only"], material_mode),
        target_duration_minutes=target_duration_minutes,
        ai_clip_cap=int(policy.get("ai_clip_cap", AI_CLIP_CAP_DEFAULT)),
        budget_diagnostics=budget_diagnostics,
    )

    logger.info(f"  生成 Manifest: {len(segments)} 段")
    if segments:
        total_dur = segments[-1].end - segments[0].start
        avg_dur = sum(s.duration for s in segments) / len(segments)
        logger.info(f"  总时长: {total_dur:.1f}s，平均段时长: {avg_dur:.2f}s")

    return manifest


def run_step2(
    srt_path: str,
    audio_path: str,
    project_id: str,
    output_manifest: str,
    global_style: Optional[GlobalStyle] = None,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    material_mode: str = "auto",
    duration_policy: Optional[Dict[str, float]] = None,
) -> Manifest:
    """Step 2 完整执行：SRT → Manifest，并保存到文件。"""
    manifest = srt_to_manifest(
        srt_path=srt_path,
        audio_path=audio_path,
        project_id=project_id,
        global_style=global_style,
        min_duration=min_duration,
        max_duration=max_duration,
        material_mode=material_mode,
        duration_policy=duration_policy,
    )
    os.makedirs(os.path.dirname(output_manifest), exist_ok=True)
    manifest.save(output_manifest)
    logger.info(f"  Manifest 已保存: {output_manifest}")
    return manifest
