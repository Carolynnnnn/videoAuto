"""
Pexels 集成测试：验证 step4_assets.py 的五级素材选择逻辑
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import shutil
import tempfile
from pathlib import Path

PEXELS_KEY = "sYaqkHNXDIvfR1snW3nuqcox1fM0sDVZQofTOd2afs7G9WU4xl1gkARB"

PASS = "\033[32m✓ PASS\033[0m"
FAIL = "\033[31m✗ FAIL\033[0m"
INFO = "\033[34mINFO\033[0m"

results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status}  {name}")
    if detail:
        print(f"         {detail}")
    results.append((name, condition))
    return condition


def make_segment(text="测试文本", keywords=None, vp_type="broll", duration=5.0):
    """创建测试用 Segment"""
    import hashlib
    from src.core.models import Segment, VisualPlan
    content_key = hashlib.md5(text.strip().lower().encode()).hexdigest()[:12]
    segment_key = f"{content_key}#0"
    seg = Segment(
        segment_key=segment_key,
        content_key=content_key,
        index=1,
        text=text,
        start=0.0,
        end=duration,
        duration=duration,
    )
    seg.visual_plan = VisualPlan(
        type=vp_type,
        keywords=keywords or ["人工智能", "科技"],
        prompt="A futuristic AI technology scene",
        use_pdf_assets=[],
    )
    return seg


# ─────────────────────────────────────────────
# Test 1: Pexels 视频优先（② 级）
# ─────────────────────────────────────────────
print("\n=== Test 1: Pexels 视频优先（② 级）===")
with tempfile.TemporaryDirectory() as tmpdir:
    generated_dir = os.path.join(tmpdir, "generated")
    library_dir = os.path.join(tmpdir, "library")
    pexels_cache = os.path.join(tmpdir, "pexels_cache")
    pexels_video_dir = os.path.join(pexels_cache, "videos")

    from src.steps.step4_assets import resolve_asset_for_segment
    seg = make_segment(keywords=["人工智能", "科技"], vp_type="broll", duration=5.0)

    seg = resolve_asset_for_segment(
        segment=seg,
        project_root=tmpdir,
        generated_dir=generated_dir,
        library_dir=library_dir,
        pexels_api_key=PEXELS_KEY,
        enable_pexels_video=True,
        enable_pexels_photo=True,
        enable_ai_image=False,
        resolution=(1080, 1920),
        aspect_ratio="9:16",
    )

    has_asset = bool(seg.asset_refs and seg.asset_refs[0].path)
    asset_kind = seg.asset_refs[0].kind if seg.asset_refs else "none"
    asset_path = seg.asset_refs[0].path if seg.asset_refs else ""
    asset_exists = os.path.exists(asset_path) if asset_path else False

    check("素材已解析", has_asset, f"kind={asset_kind}")
    check("素材来自 Pexels（视频或图片）",
          asset_kind in ("pexels_video", "pexels_photo"),
          f"actual kind={asset_kind}, path={os.path.basename(asset_path)}")
    check("素材文件存在", asset_exists, f"path={asset_path}")
    if asset_path:
        size = os.path.getsize(asset_path) if asset_exists else 0
        check("素材文件大小 > 10KB", size > 10240, f"size={size/1024:.0f}KB")


# ─────────────────────────────────────────────
# Test 2: 禁用 Pexels 视频，只用图片（③ 级）
# ─────────────────────────────────────────────
print("\n=== Test 2: 禁用 Pexels 视频，只用图片（③ 级）===")
with tempfile.TemporaryDirectory() as tmpdir:
    generated_dir = os.path.join(tmpdir, "generated")
    library_dir = os.path.join(tmpdir, "library")

    seg = make_segment(keywords=["商业", "团队"], vp_type="broll")

    seg = resolve_asset_for_segment(
        segment=seg,
        project_root=tmpdir,
        generated_dir=generated_dir,
        library_dir=library_dir,
        pexels_api_key=PEXELS_KEY,
        enable_pexels_video=False,  # 禁用视频
        enable_pexels_photo=True,
        enable_ai_image=False,
        resolution=(1080, 1920),
        aspect_ratio="9:16",
    )

    asset_kind = seg.asset_refs[0].kind if seg.asset_refs else "none"
    check("禁用视频时使用 Pexels 图片",
          asset_kind == "pexels_photo",
          f"actual kind={asset_kind}")


# ─────────────────────────────────────────────
# Test 3: 禁用 Pexels，使用模板兜底（⑤ 级）
# ─────────────────────────────────────────────
print("\n=== Test 3: 禁用 Pexels，使用模板兜底（⑤ 级）===")
with tempfile.TemporaryDirectory() as tmpdir:
    generated_dir = os.path.join(tmpdir, "generated")
    library_dir = os.path.join(tmpdir, "library")

    seg = make_segment(keywords=["测试"], vp_type="broll")

    seg = resolve_asset_for_segment(
        segment=seg,
        project_root=tmpdir,
        generated_dir=generated_dir,
        library_dir=library_dir,
        pexels_api_key="",  # 无 Pexels Key
        enable_pexels_video=False,
        enable_pexels_photo=False,
        enable_ai_image=False,
        resolution=(1080, 1920),
        aspect_ratio="9:16",
    )

    asset_kind = seg.asset_refs[0].kind if seg.asset_refs else "none"
    asset_path = seg.asset_refs[0].path if seg.asset_refs else ""
    check("无 Pexels 时使用模板兜底",
          asset_kind == "template",
          f"actual kind={asset_kind}")
    check("模板文件存在", os.path.exists(asset_path) if asset_path else False)


# ─────────────────────────────────────────────
# Test 4: 缓存复用（⓪ 级）
# ─────────────────────────────────────────────
print("\n=== Test 4: 缓存复用（⓪ 级）===")
with tempfile.TemporaryDirectory() as tmpdir:
    generated_dir = os.path.join(tmpdir, "generated")
    library_dir = os.path.join(tmpdir, "library")

    seg = make_segment(keywords=["人工智能"], vp_type="broll")

    # 第一次：下载
    import time
    t0 = time.time()
    seg = resolve_asset_for_segment(
        segment=seg,
        project_root=tmpdir,
        generated_dir=generated_dir,
        library_dir=library_dir,
        pexels_api_key=PEXELS_KEY,
        enable_pexels_video=True,
        enable_pexels_photo=True,
        enable_ai_image=False,
        resolution=(1080, 1920),
        aspect_ratio="9:16",
    )
    t1 = time.time()
    first_time = t1 - t0
    first_kind = seg.asset_refs[0].kind if seg.asset_refs else "none"

    # 第二次：应命中缓存
    seg2 = make_segment(keywords=["人工智能"], vp_type="broll")
    # 复制 plan_hash
    seg2.plan_hash = seg.plan_hash

    t2 = time.time()
    seg2 = resolve_asset_for_segment(
        segment=seg2,
        project_root=tmpdir,
        generated_dir=generated_dir,
        library_dir=library_dir,
        pexels_api_key=PEXELS_KEY,
        enable_pexels_video=True,
        enable_pexels_photo=True,
        enable_ai_image=False,
        resolution=(1080, 1920),
        aspect_ratio="9:16",
    )
    t3 = time.time()
    second_time = t3 - t2
    second_kind = seg2.asset_refs[0].kind if seg2.asset_refs else "none"

    check("第一次获取素材成功",
          first_kind in ("pexels_video", "pexels_photo", "template"),
          f"kind={first_kind}, 耗时={first_time:.1f}s")
    check("第二次命中缓存（kind=cached）",
          second_kind == "cached",
          f"kind={second_kind}, 耗时={second_time:.3f}s")
    check("缓存速度明显更快",
          second_time < first_time * 0.5 or second_time < 0.1,
          f"首次={first_time:.1f}s, 缓存={second_time:.3f}s")


# ─────────────────────────────────────────────
# Test 5: kinetic_text 类型不用视频背景
# ─────────────────────────────────────────────
print("\n=== Test 5: kinetic_text 类型跳过 Pexels 视频 ===")
with tempfile.TemporaryDirectory() as tmpdir:
    generated_dir = os.path.join(tmpdir, "generated")
    library_dir = os.path.join(tmpdir, "library")

    seg = make_segment(keywords=["人工智能"], vp_type="kinetic_text")

    seg = resolve_asset_for_segment(
        segment=seg,
        project_root=tmpdir,
        generated_dir=generated_dir,
        library_dir=library_dir,
        pexels_api_key=PEXELS_KEY,
        enable_pexels_video=True,
        enable_pexels_photo=True,
        enable_ai_image=False,
        resolution=(1080, 1920),
        aspect_ratio="9:16",
    )

    asset_kind = seg.asset_refs[0].kind if seg.asset_refs else "none"
    check("kinetic_text 不使用 Pexels 视频",
          asset_kind != "pexels_video",
          f"actual kind={asset_kind}")


# ─────────────────────────────────────────────
# Test 6: step4_assets.run_step4 统计日志
# ─────────────────────────────────────────────
print("\n=== Test 6: run_step4 接口兼容性 ===")
try:
    from src.steps.step4_assets import run_step4
    import inspect
    sig = inspect.signature(run_step4)
    params = list(sig.parameters.keys())
    check("run_step4 包含 pexels_api_key 参数", "pexels_api_key" in params, f"params={params}")
    check("run_step4 包含 enable_pexels_video 参数", "enable_pexels_video" in params)
    check("run_step4 包含 enable_pexels_photo 参数", "enable_pexels_photo" in params)
    check("run_step4 包含 enable_ai_image 参数", "enable_ai_image" in params)
except Exception as e:
    check("run_step4 接口检查", False, str(e))


# ─────────────────────────────────────────────
# 汇总
# ─────────────────────────────────────────────
print()
print("=" * 50)
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"测试结果: {passed}/{total} 通过")
if passed == total:
    print("\033[32m所有测试通过！Pexels 接入成功！\033[0m")
else:
    failed = [(n, ok) for n, ok in results if not ok]
    print(f"\033[31m失败: {[n for n, _ in failed]}\033[0m")
    sys.exit(1)
