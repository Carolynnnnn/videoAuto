"""
验收测试（Acceptance Criteria Tests，v2 修正版）

设计语义说明：
  - segment_key = content_key + '#' + occurrence_index
  - content_key = hash(normalize(text))，只与文本内容相关
  - 当文本改变时，segment_key 也改变 → 旧段 removed，新段 added
  - 这是设计上的正确行为：TEXT 变更通过 removed+added 体现
  - TIMING 变更：segment_key 相同（文本不变），时间轴超过阈值
  - STYLE 变更：全局样式变化，所有共有 key 都变为 changed(STYLE)

AC1: 改 1 条文案 → removed=1, added=1（旧段删除，新段新增），其余 unchanged
AC2: 插入 2 条 → added=2，旧段尽量 unchanged（无时间变化时）
AC3: 只调 1 段时间轴（其他段不变）→ changed(TIMING)=1，素材不重新生成
AC4: segment_key 稳定性 → 音频重对齐（start_time 偏移 < 0.2s）不触发任何变更

运行方式：
  cd /home/ubuntu/video_pipeline
  python3.11 tests/test_acceptance.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.models import (
    Manifest, Segment, GlobalStyle, AudioRef, VisualPlan,
    MotionConfig, AssetRef, RenderRef, DiffResult, ChangeType
)
from src.core.diff_engine import compute_diff, get_segments_to_rebuild, print_diff_summary


# ─────────────────────────────────────────────
# 测试工具函数
# ─────────────────────────────────────────────
def make_segment(
    text: str,
    start: float,
    end: float,
    index: int,
    occurrence_index: int = 1,
) -> Segment:
    """创建测试用 Segment"""
    content_key = Segment.compute_content_key(text)
    segment_key = Segment.compute_segment_key(content_key, occurrence_index)
    return Segment(
        segment_key=segment_key,
        content_key=content_key,
        index=index,
        start=start,
        end=end,
        duration=round(end - start, 3),
        text=text,
        audio_ref=AudioRef(type="full", path="/tmp/audio.mp3",
                           trim_start=start, trim_end=end),
    )


def make_manifest(segments, project_id="test", style=None) -> Manifest:
    return Manifest(
        project_id=project_id,
        global_style=style or GlobalStyle(),
        segments=segments,
    )


def assert_equal(actual, expected, msg=""):
    if actual != expected:
        raise AssertionError(
            f"FAIL: {msg}\n  expected={expected!r}\n  actual={actual!r}"
        )
    print(f"  PASS: {msg}")


def assert_in(item, container, msg=""):
    if item not in container:
        raise AssertionError(
            f"FAIL: {msg}\n  {item!r} not in {container!r}"
        )
    print(f"  PASS: {msg}")


def assert_empty(lst, msg=""):
    if lst:
        raise AssertionError(
            f"FAIL: {msg}\n  expected empty, got {lst!r}"
        )
    print(f"  PASS: {msg}")


# ─────────────────────────────────────────────
# AC1: 改 1 条文案 → removed=1, added=1，其余 unchanged
# ─────────────────────────────────────────────
def test_ac1_change_one_text():
    print("\n" + "=" * 50)
    print("AC1: 改 1 条文案 → removed=1, added=1，其余 unchanged")
    print("=" * 50)
    print("  (设计语义：文本变化时 segment_key 改变，旧段 removed，新段 added)")

    # 旧版：3 段
    old_segs = [
        make_segment("人工智能正在改变世界", 0.0, 3.0, 1),
        make_segment("每天都有新的突破", 3.0, 6.0, 2),
        make_segment("未来充满无限可能", 6.0, 9.0, 3),
    ]
    old_manifest = make_manifest(old_segs)

    # 新版：只改第 2 段文案（文本变化 → segment_key 变化）
    new_segs = [
        make_segment("人工智能正在改变世界", 0.0, 3.0, 1),
        make_segment("每天都有令人惊叹的突破", 3.0, 6.0, 2),   # ← 文案变了
        make_segment("未来充满无限可能", 6.0, 9.0, 3),
    ]
    new_manifest = make_manifest(new_segs)

    diff = compute_diff(old_manifest, new_manifest)
    print(print_diff_summary(diff))

    # 文本变化 → 旧段 removed，新段 added（这是正确的设计行为）
    assert_equal(len(diff.removed), 1, "removed 数量 = 1（旧文案段被移除）")
    assert_equal(len(diff.added), 1, "added 数量 = 1（新文案段被新增）")
    assert_equal(len(diff.unchanged), 2, "unchanged 数量 = 2（第1、3段复用）")
    assert_equal(len(diff.changed_text), 0, "changed_text = 0（文本变化通过 removed+added 体现）")

    # 验证第 1、3 段的 segment_key 在 unchanged 中
    assert_in(old_segs[0].segment_key, diff.unchanged, "第1段 unchanged")
    assert_in(old_segs[2].segment_key, diff.unchanged, "第3段 unchanged")

    # 验证 need_new_assets 只包含新增段（added）
    plan_keys, asset_keys, render_keys = get_segments_to_rebuild(new_manifest, diff)
    assert_equal(len(asset_keys), 1, "need_new_assets 数量 = 1（只有新增段）")
    assert_equal(len(render_keys), 1, "need_rerender 数量 = 1（只有新增段）")

    print("  AC1 PASSED ✓")


# ─────────────────────────────────────────────
# AC1b: 使用 changed_text 场景（同 segment_key，文本不同）
# 注：在当前设计中，segment_key 基于 content_key，文本变化时 key 也变化
# 因此 changed_text 实际上只在 segment_key 相同但 content_key 不同时触发
# 这种情况不会发生（segment_key 包含 content_key）
# 此测试验证 diff 引擎的内部一致性
# ─────────────────────────────────────────────
def test_ac1b_text_change_via_segment_key():
    print("\n" + "=" * 50)
    print("AC1b: segment_key 匹配语义验证")
    print("=" * 50)

    # 构造两个文本不同但 segment_key 相同的段（模拟边界情况）
    # 实际上这不会发生，因为 segment_key 包含 content_key
    # 此测试验证：当 segment_key 相同时，diff 引擎正确判断 unchanged
    text = "人工智能正在改变世界"
    seg_old = make_segment(text, 0.0, 3.0, 1)
    seg_new = make_segment(text, 0.0, 3.0, 1)  # 完全相同

    old_manifest = make_manifest([seg_old])
    new_manifest = make_manifest([seg_new])

    diff = compute_diff(old_manifest, new_manifest)

    assert_equal(len(diff.unchanged), 1, "完全相同的段应为 unchanged")
    assert_empty(diff.changed_text, "changed_text 为空")
    assert_empty(diff.added, "added 为空")
    assert_empty(diff.removed, "removed 为空")

    print("  AC1b PASSED ✓")


# ─────────────────────────────────────────────
# AC2: 插入 2 条 → added=2，旧段尽量 unchanged
# ─────────────────────────────────────────────
def test_ac2_insert_two_segments():
    print("\n" + "=" * 50)
    print("AC2: 插入 2 条 → added=2，旧段尽量 unchanged")
    print("=" * 50)

    # 旧版：3 段（时间精确）
    old_segs = [
        make_segment("人工智能正在改变世界", 0.0, 3.0, 1),
        make_segment("每天都有新的突破", 3.0, 6.0, 2),
        make_segment("未来充满无限可能", 6.0, 9.0, 3),
    ]
    old_manifest = make_manifest(old_segs)

    # 新版：在第 1 段后插入 2 条（旧段时间不变）
    new_segs = [
        make_segment("人工智能正在改变世界", 0.0, 3.0, 1),   # 时间不变 → unchanged
        make_segment("这是一个全新的时代", 3.0, 5.0, 2),      # ← 新增
        make_segment("技术的边界不断扩展", 5.0, 7.0, 3),      # ← 新增
        make_segment("每天都有新的突破", 7.0, 10.0, 4),       # 时间变了 → TIMING
        make_segment("未来充满无限可能", 10.0, 13.0, 5),      # 时间变了 → TIMING
    ]
    new_manifest = make_manifest(new_segs)

    diff = compute_diff(old_manifest, new_manifest)
    print(print_diff_summary(diff))

    assert_equal(len(diff.added), 2, "added 数量 = 2（2 条新段）")
    assert_empty(diff.removed, "removed 为空")

    # 第 1 段时间未变 → unchanged
    assert_in(old_segs[0].segment_key, diff.unchanged, "第1段（时间未变）应为 unchanged")

    # 第 2、3 段时间变了（超过 0.2s）→ TIMING
    total_timing = len(diff.changed_timing)
    total_unchanged = len(diff.unchanged)
    assert_equal(total_timing + total_unchanged, 3, "旧段总计（timing + unchanged）= 3")

    print("  AC2 PASSED ✓")


# ─────────────────────────────────────────────
# AC3: 只调 1 段时间轴（其他段不变）→ changed(TIMING)=1
# ─────────────────────────────────────────────
def test_ac3_timing_only_change():
    print("\n" + "=" * 50)
    print("AC3: 只调 1 段时间轴（其他段时间不变）→ changed(TIMING)=1")
    print("=" * 50)

    # 旧版：3 段
    old_segs = [
        make_segment("人工智能正在改变世界", 0.0, 3.0, 1),
        make_segment("每天都有新的突破", 3.0, 6.0, 2),
        make_segment("未来充满无限可能", 6.0, 9.0, 3),
    ]
    old_manifest = make_manifest(old_segs)

    # 新版：只调整第 2 段的时间轴（文本不变，其他段时间不变）
    new_segs = [
        make_segment("人工智能正在改变世界", 0.0, 3.0, 1),    # 时间不变
        make_segment("每天都有新的突破", 3.5, 7.0, 2),         # ← 时间变了（+0.5s）
        make_segment("未来充满无限可能", 6.0, 9.0, 3),         # 时间不变（保持原值）
    ]
    new_manifest = make_manifest(new_segs)

    diff = compute_diff(old_manifest, new_manifest)
    print(print_diff_summary(diff))

    assert_equal(len(diff.changed_timing), 1, "changed_timing 数量 = 1（只有第2段时间变了）")
    assert_equal(len(diff.changed_text), 0, "changed_text 数量 = 0（文本未变）")
    assert_equal(len(diff.unchanged), 2, "unchanged 数量 = 2（第1、3段不变）")

    # 关键验证：TIMING 变更不触发素材重新生成
    plan_keys, asset_keys, render_keys = get_segments_to_rebuild(new_manifest, diff)
    assert_empty(asset_keys, "need_new_assets 为空（TIMING 不重新生成素材）")
    assert_empty(plan_keys, "need_new_visual_plan 为空（TIMING 不重新生成 plan）")
    assert_equal(len(render_keys), 1, "need_rerender 数量 = 1（TIMING 需要重渲）")

    print("  AC3 PASSED ✓")


# ─────────────────────────────────────────────
# AC4: segment_key 稳定性 → 音频重对齐不触发变更
# ─────────────────────────────────────────────
def test_ac4_segment_key_stability():
    print("\n" + "=" * 50)
    print("AC4: segment_key 稳定性 → 音频重对齐（偏移 < 0.2s）不触发任何变更")
    print("=" * 50)

    # 旧版：3 段（精确时间）
    old_segs = [
        make_segment("人工智能正在改变世界", 0.000, 3.000, 1),
        make_segment("每天都有新的突破", 3.000, 6.000, 2),
        make_segment("未来充满无限可能", 6.000, 9.000, 3),
    ]
    old_manifest = make_manifest(old_segs)

    # 新版：音频重对齐，时间有微小偏移（< 0.2s 阈值）
    new_segs = [
        make_segment("人工智能正在改变世界", 0.050, 3.050, 1),   # +0.05s
        make_segment("每天都有新的突破", 3.050, 6.050, 2),       # +0.05s
        make_segment("未来充满无限可能", 6.050, 9.050, 3),       # +0.05s
    ]
    new_manifest = make_manifest(new_segs)

    diff = compute_diff(old_manifest, new_manifest)
    print(print_diff_summary(diff))

    # 关键验证：微小时间偏移（< 0.2s）不应触发任何变更
    assert_empty(diff.changed_text, "changed_text 为空（文本未变）")
    assert_empty(diff.changed_timing, "changed_timing 为空（偏移 < 0.2s 阈值）")
    assert_equal(len(diff.unchanged), 3, "所有段 unchanged（音频重对齐不触发重建）")

    # 验证 segment_key 的稳定性（与 start_time 无关）
    for old_seg, new_seg in zip(old_segs, new_segs):
        assert_equal(old_seg.segment_key, new_seg.segment_key,
                     f"segment_key 稳定: '{old_seg.text[:15]}'")
        assert_equal(old_seg.content_key, new_seg.content_key,
                     f"content_key 稳定: '{old_seg.text[:15]}'")

    print("  AC4 PASSED ✓")


# ─────────────────────────────────────────────
# EXTRA: 重复文本的 occurrence_index 处理
# ─────────────────────────────────────────────
def test_repeated_text_occurrence_index():
    print("\n" + "=" * 50)
    print("EXTRA: 重复文本 → occurrence_index 区分")
    print("=" * 50)

    text = "好的"
    ck = Segment.compute_content_key(text)
    seg1 = Segment(
        segment_key=Segment.compute_segment_key(ck, 1),
        content_key=ck,
        index=1, start=0.0, end=1.0, duration=1.0, text=text,
        audio_ref=AudioRef(),
    )
    seg2 = Segment(
        segment_key=Segment.compute_segment_key(ck, 2),
        content_key=ck,
        index=2, start=5.0, end=6.0, duration=1.0, text=text,
        audio_ref=AudioRef(),
    )

    # segment_key 应不同（occurrence_index 不同）
    assert seg1.segment_key != seg2.segment_key, "重复文本的 segment_key 应不同"
    assert seg1.content_key == seg2.content_key, "重复文本的 content_key 应相同"
    print(f"  PASS: 重复文本 segment_key 唯一性")

    # 改第 2 次出现的文本（旧段 removed，新段 added）
    text_new = "好的好的"
    ck_new = Segment.compute_content_key(text_new)
    seg2_new = Segment(
        segment_key=Segment.compute_segment_key(ck_new, 1),
        content_key=ck_new,
        index=2, start=5.0, end=6.0, duration=1.0, text=text_new,
        audio_ref=AudioRef(),
    )

    old_manifest = make_manifest([seg1, seg2])
    new_manifest = make_manifest([seg1, seg2_new])

    diff = compute_diff(old_manifest, new_manifest)

    # 第 1 次出现的 "好的" 应 unchanged
    assert_in(seg1.segment_key, diff.unchanged, "第1次出现的重复文本应 unchanged")
    # 第 2 次出现的文本变了，旧段 removed，新段 added
    assert_in(seg2.segment_key, diff.removed, "第2次出现的旧文本应 removed")
    assert_in(seg2_new.segment_key, diff.added, "第2次出现的新文本应 added")

    print("  EXTRA PASSED ✓")


# ─────────────────────────────────────────────
# STYLE: 全局样式变更
# ─────────────────────────────────────────────
def test_style_change():
    print("\n" + "=" * 50)
    print("STYLE: 全局样式变更 → 所有段 changed(STYLE)，素材不重新生成")
    print("=" * 50)

    old_segs = [
        make_segment("人工智能正在改变世界", 0.0, 3.0, 1),
        make_segment("每天都有新的突破", 3.0, 6.0, 2),
    ]
    old_style = GlobalStyle(style_version="v1", font_size=48)
    old_manifest = make_manifest(old_segs, style=old_style)

    new_segs = [
        make_segment("人工智能正在改变世界", 0.0, 3.0, 1),
        make_segment("每天都有新的突破", 3.0, 6.0, 2),
    ]
    new_style = GlobalStyle(style_version="v2", font_size=56)  # 样式变了
    new_manifest = make_manifest(new_segs, style=new_style)

    diff = compute_diff(old_manifest, new_manifest)
    print(print_diff_summary(diff))

    assert_equal(len(diff.changed_style), 2, "所有段应为 changed(STYLE)")
    assert_empty(diff.unchanged, "无 unchanged 段")

    # STYLE 变更：不重新生成素材
    plan_keys, asset_keys, render_keys = get_segments_to_rebuild(new_manifest, diff)
    assert_empty(asset_keys, "STYLE 变更不重新生成素材")
    assert_equal(len(render_keys), 2, "STYLE 变更需要重渲所有段")

    print("  STYLE PASSED ✓")


# ─────────────────────────────────────────────
# TIMING_THRESHOLD: 边界值测试
# ─────────────────────────────────────────────
def test_timing_threshold_boundary():
    print("\n" + "=" * 50)
    print("THRESHOLD: 时间阈值边界测试（0.2s）")
    print("=" * 50)

    text = "测试文本"
    seg_old = make_segment(text, 0.0, 3.0, 1)
    old_manifest = make_manifest([seg_old])

    # 偏移 0.19s（< 0.2s）→ unchanged
    seg_new_small = make_segment(text, 0.19, 3.19, 1)
    new_manifest_small = make_manifest([seg_new_small])
    diff_small = compute_diff(old_manifest, new_manifest_small)
    assert_equal(len(diff_small.unchanged), 1, "偏移 0.19s < 0.2s 阈值 → unchanged")

    # 偏移 0.21s（> 0.2s）→ TIMING
    seg_new_large = make_segment(text, 0.21, 3.21, 1)
    new_manifest_large = make_manifest([seg_new_large])
    diff_large = compute_diff(old_manifest, new_manifest_large)
    assert_equal(len(diff_large.changed_timing), 1, "偏移 0.21s > 0.2s 阈值 → TIMING")

    print("  THRESHOLD PASSED ✓")


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────
def run_all_tests():
    tests = [
        test_ac1_change_one_text,
        test_ac1b_text_change_via_segment_key,
        test_ac2_insert_two_segments,
        test_ac3_timing_only_change,
        test_ac4_segment_key_stability,
        test_repeated_text_occurrence_index,
        test_style_change,
        test_timing_threshold_boundary,
    ]

    passed = 0
    failed = 0
    errors = []

    print("\n" + "=" * 60)
    print("验收测试套件（v2 修正版）")
    print("=" * 60)

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            print(f"  ✗ {test_fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, f"EXCEPTION: {e}"))
            import traceback
            print(f"  ✗ {test_fn.__name__}: EXCEPTION")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    if errors:
        print("\n失败详情:")
        for name, msg in errors:
            print(f"  [{name}] {msg}")
    else:
        print("所有测试通过！✓")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
