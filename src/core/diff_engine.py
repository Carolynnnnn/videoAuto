"""
Diff 引擎 v2：字幕变动识别与分级变更判定

核心逻辑：
  1. 以 segment_key（content_key + '#' + occurrence_index）为主键做匹配
  2. 分级判定变更类型：TEXT / TIMING / STYLE
  3. 根据变更类型决定触发范围：
     - TEXT  → 重做 visual_plan + 素材 + 渲染
     - TIMING → 复用素材，只重渲
     - STYLE  → 复用素材，只重渲
  4. unchanged → 直接复用旧 seg mp4

验收标准覆盖：
  AC1: 改 1 条文案 → diff 中只出现 1 个 changed(TEXT)
  AC2: 插入 2 条 → added=2，旧段尽量 unchanged
  AC3: 只调时间轴 → changed(TIMING)，素材不重新生成
"""
from __future__ import annotations
import os
import shutil
from typing import Dict, List, Optional, Tuple

from .models import (
    Manifest, Segment, DiffResult, SegmentChange,
    ChangeType, GlobalStyle, RenderRef,
)
from src.utils.logger import get_logger

logger = get_logger("diff_engine")

# 时间变化阈值（秒）：超过此值视为 TIMING changed
TIMING_THRESHOLD = 0.2


def _build_segment_map(manifest: Manifest) -> Dict[str, Segment]:
    """将 Manifest 的 segments 转为 {segment_key: Segment} 字典"""
    return {s.segment_key: s for s in manifest.segments}


def _global_style_changed(old_gs: GlobalStyle, new_gs: GlobalStyle) -> bool:
    """判断全局样式是否发生了影响渲染的变化"""
    return old_gs.render_related_fields() != new_gs.render_related_fields()


def _text_changed(old_seg: Segment, new_seg: Segment) -> bool:
    """判断文本内容是否变化（基于 content_key，已规范化）"""
    return old_seg.content_key != new_seg.content_key


def _timing_changed(old_seg: Segment, new_seg: Segment) -> bool:
    """判断时间轴是否变化（超过阈值）"""
    return (
        abs(old_seg.start - new_seg.start) > TIMING_THRESHOLD
        or abs(old_seg.end - new_seg.end) > TIMING_THRESHOLD
    )


def compute_diff(
    old_manifest: Manifest,
    new_manifest: Manifest,
    time_threshold: float = TIMING_THRESHOLD,
) -> DiffResult:
    """
    计算新旧 Manifest 的 Diff，返回分级变更结果。

    匹配策略：
      - 以 segment_key 为主键（content_key + '#' + occurrence_index）
      - new 有 old 没有 → added
      - old 有 new 没有 → removed
      - 同 segment_key → 进入分级比较

    分级判定优先级：TEXT > TIMING > STYLE > unchanged
    """
    diff = DiffResult()
    old_map = _build_segment_map(old_manifest)
    new_map = _build_segment_map(new_manifest)

    style_changed = _global_style_changed(old_manifest.global_style, new_manifest.global_style)
    if style_changed:
        logger.warning(
            f"全局样式变化: {old_manifest.global_style.render_related_fields()} "
            f"→ {new_manifest.global_style.render_related_fields()}"
        )

    old_keys = set(old_map.keys())
    new_keys = set(new_map.keys())

    # ── added ──
    for key in sorted(new_keys - old_keys):
        diff.added.append(key)
        diff.changes.append(SegmentChange(
            segment_key=key,
            change_type=ChangeType.ADDED,
            old_segment=None,
            new_segment=new_map[key],
        ))

    # ── removed ──
    for key in sorted(old_keys - new_keys):
        diff.removed.append(key)
        diff.changes.append(SegmentChange(
            segment_key=key,
            change_type=ChangeType.REMOVED,
            old_segment=old_map[key],
            new_segment=None,
        ))

    # ── 共有 key → 分级比较 ──
    for key in sorted(old_keys & new_keys):
        old_seg = old_map[key]
        new_seg = new_map[key]

        if _text_changed(old_seg, new_seg):
            # TEXT 优先级最高（content_key 不同说明文本真的变了）
            diff.changed_text.append(key)
            diff.changes.append(SegmentChange(
                segment_key=key,
                change_type=ChangeType.TEXT,
                old_segment=old_seg,
                new_segment=new_seg,
            ))
            logger.debug(
                f"  TEXT changed [{key[:12]}]: "
                f"'{old_seg.text[:20]}' → '{new_seg.text[:20]}'"
            )
        elif _timing_changed(old_seg, new_seg):
            diff.changed_timing.append(key)
            diff.changes.append(SegmentChange(
                segment_key=key,
                change_type=ChangeType.TIMING,
                old_segment=old_seg,
                new_segment=new_seg,
            ))
            logger.debug(
                f"  TIMING changed [{key[:12]}]: "
                f"start {old_seg.start:.2f}→{new_seg.start:.2f}, "
                f"end {old_seg.end:.2f}→{new_seg.end:.2f}"
            )
        elif style_changed:
            diff.changed_style.append(key)
            diff.changes.append(SegmentChange(
                segment_key=key,
                change_type=ChangeType.STYLE,
                old_segment=old_seg,
                new_segment=new_seg,
            ))
        else:
            diff.unchanged.append(key)

    logger.info(
        f"Diff 结果: added={len(diff.added)}, removed={len(diff.removed)}, "
        f"changed_text={len(diff.changed_text)}, changed_timing={len(diff.changed_timing)}, "
        f"changed_style={len(diff.changed_style)}, unchanged={len(diff.unchanged)}"
    )
    return diff


def apply_diff(
    old_manifest: Manifest,
    new_manifest: Manifest,
    diff: DiffResult,
    segments_dir: str,
    assets_dir: str,
) -> Manifest:
    """
    将 diff 应用到 new_manifest：

    - unchanged → 从 old_manifest 复制 visual_plan、plan_hash、asset_refs、render_ref
    - changed(TIMING/STYLE) → 复制 visual_plan、plan_hash、asset_refs（复用素材），清空 render_ref
    - changed(TEXT) / added → 清空所有，等待后续步骤重建
    - removed → 不出现在 new_manifest 中

    返回已应用 diff 的 new_manifest（就地修改）。
    """
    old_map = _build_segment_map(old_manifest)

    unchanged_set = set(diff.unchanged)
    timing_style_set = set(diff.changed_timing + diff.changed_style)

    for seg in new_manifest.segments:
        key = seg.segment_key

        if key in unchanged_set:
            # 完全复用旧产物
            old_seg = old_map[key]
            seg.visual_plan = old_seg.visual_plan
            seg.plan_hash = old_seg.plan_hash
            seg.asset_refs = old_seg.asset_refs
            seg.render_ref = old_seg.render_ref
            seg.prev_last_frame_path = old_seg.prev_last_frame_path
            seg.continuity_diagnostic = old_seg.continuity_diagnostic

            # 如果 index 变了（插入/删除导致序号偏移），需要复制视频文件
            if old_seg.render_ref.segment_video_path and old_seg.render_ref.status == "ok":
                old_video = old_seg.render_ref.segment_video_path
                new_video = os.path.join(
                    segments_dir,
                    f"{seg.content_key}_{old_seg.render_ref.render_hash}.mp4"
                )
                if os.path.exists(old_video) and not os.path.exists(new_video):
                    os.makedirs(segments_dir, exist_ok=True)
                    shutil.copy2(old_video, new_video)
                    logger.debug(f"  复制片段: {os.path.basename(old_video)} → {os.path.basename(new_video)}")
                # 更新路径
                seg.render_ref = RenderRef(
                    segment_video_path=new_video if os.path.exists(new_video) else old_video,
                    render_hash=old_seg.render_ref.render_hash,
                    status="ok",
                )
            logger.debug(f"  复用产物 (UNCHANGED): [{key[:12]}]")

        elif key in timing_style_set:
            # 复用素材（plan_hash 不变），清空 render_ref 等待重渲
            old_seg = old_map[key]
            seg.visual_plan = old_seg.visual_plan
            seg.plan_hash = old_seg.plan_hash
            seg.asset_refs = old_seg.asset_refs
            seg.render_ref = RenderRef(status="pending")
            seg.prev_last_frame_path = old_seg.prev_last_frame_path
            seg.continuity_diagnostic = old_seg.continuity_diagnostic
            logger.debug(f"  复用素材 (TIMING/STYLE): [{key[:12]}]")

        # changed(TEXT) / added → 保持空，后续步骤会填充

    return new_manifest


def get_segments_to_rebuild(
    new_manifest: Manifest,
    diff: DiffResult,
) -> Tuple[List[str], List[str], List[str]]:
    """
    根据 diff 返回三个重建列表：
      - plan_rebuild_keys: 需要重新生成 visual_plan 的 segment_keys
      - asset_rebuild_keys: 需要重新生成/检索素材的 segment_keys
      - render_rebuild_keys: 需要重渲片段的 segment_keys

    触发规则：
      TEXT  → plan + assets + render
      TIMING → render only（复用 plan + assets）
      STYLE  → render only（复用 plan + assets）
      ADDED  → plan + assets + render
    """
    plan_rebuild_keys = diff.need_new_visual_plan
    asset_rebuild_keys = diff.need_new_assets
    render_rebuild_keys = diff.need_rerender
    return plan_rebuild_keys, asset_rebuild_keys, render_rebuild_keys


def check_render_cache(
    segment: Segment,
    global_style: GlobalStyle,
    segments_dir: str,
) -> Optional[str]:
    """
    检查渲染缓存：
    - 计算 render_hash
    - 如果 segments_dir 中存在对应文件，返回文件路径
    - 否则返回 None

    缓存路径：render/segments/{content_key}_{render_hash}.mp4
    """
    render_hash = segment.compute_render_hash(global_style.render_related_fields())
    cache_path = os.path.join(segments_dir, f"{segment.content_key}_{render_hash}.mp4")
    if os.path.exists(cache_path):
        return cache_path
    return None


def check_asset_cache(
    segment: Segment,
    assets_dir: str,
) -> Optional[str]:
    """
    检查素材缓存：
    - 缓存路径：assets/generated/{content_key}_{plan_hash}.{ext}
    - 如果存在，返回路径；否则返回 None
    """
    if not segment.plan_hash:
        return None
    for ext in ("png", "mp4", "jpg", "jpeg"):
        cache_path = os.path.join(
            assets_dir, f"{segment.content_key}_{segment.plan_hash}.{ext}"
        )
        if os.path.exists(cache_path):
            return cache_path
    return None


def print_diff_summary(diff: DiffResult) -> str:
    """生成可读的 diff 摘要字符串"""
    lines = [
        "=" * 50,
        "Diff 摘要（v2 分级变更）",
        "=" * 50,
        f"  新增 (ADDED):           {len(diff.added):3d} 段",
        f"  删除 (REMOVED):         {len(diff.removed):3d} 段",
        f"  文本变更 (TEXT):         {len(diff.changed_text):3d} 段  → 重做 plan+素材+渲染",
        f"  时间变更 (TIMING):       {len(diff.changed_timing):3d} 段  → 复用素材，只重渲",
        f"  样式变更 (STYLE):        {len(diff.changed_style):3d} 段  → 复用素材，只重渲",
        f"  未变 (UNCHANGED):        {len(diff.unchanged):3d} 段  → 直接复用旧产物",
        "-" * 50,
        f"  需重渲总计:              {len(diff.need_rerender):3d} 段",
        f"  复用素材总计:            {len(diff.changed_timing) + len(diff.changed_style):3d} 段",
        f"  完全复用总计:            {len(diff.unchanged):3d} 段",
        "=" * 50,
    ]
    return "\n".join(lines)
