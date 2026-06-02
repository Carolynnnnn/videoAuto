#!/usr/bin/env python3
"""
增量构建入口（v2）

功能：
  1. 对比新旧 manifest.json，使用分级 Diff 引擎识别变更类型
  2. 根据变更类型精确触发：
     - TEXT/ADDED  → 重做 visual_plan + 素材 + 渲染
     - TIMING      → 复用素材，只重渲
     - STYLE       → 复用素材，只重渲
     - UNCHANGED   → 直接复用旧产物
  3. 重新拼接 final.mp4
  4. 支持 --full-rebuild 强制全量重建

伪 API 接口（可被外部调用）：
  from build_incremental import incremental_build, IncrementalBuildResult
  result = incremental_build(
      project_root="./projects/my_video",
      new_srt_path="./projects/my_video/input/subtitle_v2.srt",
      dry_run=False,
      full_rebuild=False,
  )

使用方式：
  python3.11 build_incremental.py --project ./projects/my_video
  python3.11 build_incremental.py --project ./projects/my_video --dry-run
  python3.11 build_incremental.py --project ./projects/my_video --full-rebuild
  python3.11 build_incremental.py --project ./projects/my_video --new-srt input/subtitle_v2.srt
  python3.11 build_incremental.py --project ./projects/my_video --json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

# 确保项目根目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).parent))

from src.core.models import Manifest, DiffResult, GlobalStyle
from src.core.diff_engine import (
    compute_diff, apply_diff, get_segments_to_rebuild, print_diff_summary
)
from src.core.generation_policy import normalize_generation_policy
from src.steps.step2_manifest import run_step2
from src.steps.step3_visual_plan import run_step3
from src.steps.step4_assets import run_step4
from src.steps.step5_render import run_step5
from src.steps.step6_concat import run_step6
from src.steps.continuity_telemetry import (
    compute_quality_summary,
    format_strict_mode_failure,
    is_strict_continuity_mode_enabled,
    validate_strict_mode,
)
from src.utils.logger import get_logger

logger = get_logger("build_incremental")


def _is_test_mode_enabled() -> bool:
    raw = os.environ.get("PIXELLE_TEST_MODE", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _should_enforce_strict_continuity_gate() -> bool:
    if _is_test_mode_enabled():
        return is_strict_continuity_mode_enabled()
    return True


def _get_effective_gate_profile(requested_profile: str) -> str:
    return requested_profile


# ─────────────────────────────────────────────
# 增量构建结果（伪 API 返回值）
# ─────────────────────────────────────────────
@dataclass
class IncrementalBuildResult:
    success: bool
    project_root: str
    diff: Optional[DiffResult] = None
    final_video: Optional[str] = None
    rerendered_count: int = 0
    reused_count: int = 0
    total_segments: int = 0
    dry_run: bool = False
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "project_root": self.project_root,
            "final_video": self.final_video,
            "rerendered_count": self.rerendered_count,
            "reused_count": self.reused_count,
            "total_segments": self.total_segments,
            "dry_run": self.dry_run,
            "error": self.error,
            "diff_summary": self.diff.to_dict() if self.diff else None,
            "details": self.details,
        }


# ─────────────────────────────────────────────
# 项目路径管理
# ─────────────────────────────────────────────
class ProjectPaths:
    def __init__(self, project_root: str):
        self.root = Path(project_root)
        self.input = self.root / "input"
        self.build = self.root / "build"
        self.render = self.root / "render"
        self.segments = self.render / "segments"
        self.assets_generated = self.root / "assets" / "generated"
        self.assets_library = self.root / "assets" / "library"
        self.cache_plans = self.root / "cache" / "plans"
        self.logs = self.root / "logs"

        # 关键文件
        self.manifest_current = self.build / "manifest.json"
        self.manifest_new = self.build / "manifest_new.json"
        self.manifest_backup = self.build / "manifest_prev.json"
        self.diff_report = self.build / "diff_report.json"
        self.final_video = self.render / "final.mp4"

    def ensure_dirs(self):
        for d in [self.build, self.render, self.segments,
                  self.assets_generated, self.assets_library,
                  self.cache_plans, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

    def find_audio(self) -> Optional[str]:
        for ext in ("*.mp3", "*.wav", "*.m4a", "*.aac"):
            for f in self.input.glob(ext):
                return str(f)
        return None

    def find_srt(self) -> Optional[str]:
        # 优先 build/ 下的 subtitle.srt，其次 input/ 下
        for candidate in [self.build / "subtitle.srt", *self.input.glob("*.srt")]:
            if Path(candidate).exists():
                return str(candidate)
        return None


# ─────────────────────────────────────────────
# 核心增量构建逻辑（伪 API）
# ─────────────────────────────────────────────
def incremental_build(
    project_root: str,
    new_srt_path: Optional[str] = None,
    dry_run: bool = False,
    full_rebuild: bool = False,
    enable_ai_image: bool = True,
    llm_model: str = "gpt-4.1-mini",
    material_mode: str = "auto",
    duration_policy: Optional[Dict[str, float]] = None,
    gate_profile: str = "release",
) -> IncrementalBuildResult:
    """
    增量构建主函数（伪 API 接口）。

    :param project_root: 项目根目录
    :param new_srt_path: 新版 SRT 文件路径（None=自动查找）
    :param dry_run: 只分析 diff，不实际执行
    :param full_rebuild: 强制全量重建（忽略所有缓存）
    :param enable_ai_image: 是否启用 AI 图片生成
    :param llm_model: LLM 模型名
    :param material_mode: 素材模式 ('auto', 'ai_preferred', 'ai_only')
    :param duration_policy: 分段持续时间策略 (min/max/target 等参数)
    :return: IncrementalBuildResult
    """
    paths = ProjectPaths(project_root)
    paths.ensure_dirs()

    logger.info("=" * 60)
    logger.info(f"增量构建 v2: {project_root}")
    logger.info(f"  dry_run={dry_run}, full_rebuild={full_rebuild}")
    logger.info("=" * 60)

    # ── 1. 检查旧 manifest ──
    has_old_manifest = paths.manifest_current.exists()
    if not has_old_manifest:
        logger.warning("未找到旧 manifest.json，将执行全量构建")
        full_rebuild = True

    # ── 2. 确定 SRT 路径 ──
    srt_path = new_srt_path or paths.find_srt()
    if not srt_path or not Path(srt_path).exists():
        return IncrementalBuildResult(
            success=False,
            project_root=project_root,
            error=f"未找到 SRT 文件: {srt_path}",
        )

    # ── 3. 确定音频路径 ──
    audio_path = paths.find_audio()
    if not audio_path:
        return IncrementalBuildResult(
            success=False,
            project_root=project_root,
            error="未找到音频文件（支持 mp3/wav/m4a/aac）",
        )

    # ── 4. 加载旧 manifest ──
    old_manifest = None
    if has_old_manifest and not full_rebuild:
        try:
            old_manifest = Manifest.load(str(paths.manifest_current))
            shutil.copy2(str(paths.manifest_current), str(paths.manifest_backup))
            logger.info(f"  已加载旧 manifest: {len(old_manifest.segments)} 段")
        except Exception as e:
            logger.warning(f"  旧 manifest 加载失败: {e}，将执行全量构建")
            old_manifest = None
            full_rebuild = True

    # ── 5. 生成新 manifest（Step 2）──
    project_id = paths.root.name
    global_style = old_manifest.global_style if old_manifest else GlobalStyle()

    new_manifest = run_step2(
        srt_path=srt_path,
        audio_path=audio_path,
        project_id=project_id,
        output_manifest=str(paths.manifest_new),
        global_style=global_style,
        material_mode=material_mode,
        duration_policy=duration_policy or None,
    )

    # ── AI Workflow Default Wiring ──
    # Apply deterministic pixelle_default_workflow when material mode is AI-centric
    # and no workflow is already set (preserves explicit overrides)
    if material_mode in ("ai_only", "ai_preferred"):
        if new_manifest.pixelle_default_workflow is None:
            new_manifest.pixelle_default_workflow = "i2v"
            logger.info(f"Applied AI workflow default: pixelle_default_workflow='i2v' (source: material_mode={material_mode})")
        else:
            logger.info(f"Preserving explicit workflow: pixelle_default_workflow='{new_manifest.pixelle_default_workflow}' (source: user override)")
    else:
        if new_manifest.pixelle_default_workflow is None:
            logger.info(f"No workflow default applied (material_mode={material_mode})")
        else:
            logger.info(f"Using explicit workflow: pixelle_default_workflow='{new_manifest.pixelle_default_workflow}' (source: user override)")
    
    new_manifest.save(str(paths.manifest_new))
    logger.info(f"  Manifest saved with workflow policy: {paths.manifest_new}")

    # ── 6. 计算 Diff ──
    if old_manifest and not full_rebuild:
        diff = compute_diff(old_manifest, new_manifest)
        logger.info("\n" + print_diff_summary(diff))
    else:
        # 全量构建：所有段都是 added
        diff = DiffResult()
        diff.added = [s.segment_key for s in new_manifest.segments]
        logger.info(f"  全量构建: {len(diff.added)} 段全部重建")

    # 保存 diff 报告
    diff.save(str(paths.diff_report))

    # ── 7. dry_run 模式 ──
    if dry_run:
        logger.info("  [dry-run] 分析完成，不执行实际构建")
        return IncrementalBuildResult(
            success=True,
            project_root=project_root,
            diff=diff,
            total_segments=len(new_manifest.segments),
            rerendered_count=len(diff.need_rerender),
            reused_count=len(diff.unchanged),
            dry_run=True,
        )

    # ── 8. 应用 diff（复用旧产物）──
    if old_manifest and not full_rebuild:
        new_manifest = apply_diff(
            old_manifest=old_manifest,
            new_manifest=new_manifest,
            diff=diff,
            segments_dir=str(paths.segments),
            assets_dir=str(paths.assets_generated),
        )

    # ── 9. 获取各阶段重建列表 ──
    plan_rebuild_keys, asset_rebuild_keys, render_rebuild_keys = get_segments_to_rebuild(
        new_manifest, diff
    )

    # ── 10. Step 3：Visual Plan（TEXT + ADDED）──
    if plan_rebuild_keys:
        logger.info(f"  Step 3: 生成 visual_plan，{len(plan_rebuild_keys)} 段")
        new_manifest = run_step3(
            manifest=new_manifest,
            output_manifest=str(paths.manifest_new),
            cache_dir=str(paths.cache_plans),
            target_segment_keys=plan_rebuild_keys,
            llm_model=llm_model,
        )
    else:
        logger.info("  Step 3: 跳过（无需重新生成 visual_plan）")

    # ── 11. Step 4：素材（TEXT + ADDED）──
    if asset_rebuild_keys:
        logger.info(f"  Step 4: 处理素材，{len(asset_rebuild_keys)} 段")
        new_manifest = run_step4(
            manifest=new_manifest,
            output_manifest=str(paths.manifest_new),
            project_root=project_root,
            target_segment_keys=asset_rebuild_keys,
            enable_ai_image=enable_ai_image,
        )
    else:
        logger.info("  Step 4: 跳过（无需重新生成素材）")

    # ── 12. Step 5：渲染（所有变更类型）──
    if render_rebuild_keys:
        logger.info(f"  Step 5: 渲染片段，{len(render_rebuild_keys)} 段")
        new_manifest = run_step5(
            manifest=new_manifest,
            output_manifest=str(paths.manifest_new),
            segments_dir=str(paths.segments),
            target_segment_keys=render_rebuild_keys,
            force_rerender=full_rebuild,
        )
    else:
        logger.info("  Step 5: 跳过（无需重渲）")
        new_manifest.save(str(paths.manifest_new))

    if _should_enforce_strict_continuity_gate():
        effective_profile = _get_effective_gate_profile(gate_profile)
        logger.info(f"Continuity gate profile: {effective_profile}")
        strict_result = validate_strict_mode(compute_quality_summary(new_manifest.segments), profile=effective_profile)
        if not strict_result.passed:
            failure_report = format_strict_mode_failure(strict_result)
            if effective_profile == "release":
                for line in failure_report.splitlines():
                    logger.error(f"  {line}")
                return IncrementalBuildResult(
                    success=False,
                    project_root=project_root,
                    diff=diff,
                    rerendered_count=len(render_rebuild_keys),
                    reused_count=len(diff.unchanged),
                    total_segments=len(new_manifest.segments),
                    dry_run=False,
                    error=failure_report,
                )
            else:
                logger.warning("=" * 60)
                logger.warning("Continuity gate WARNINGS (preview mode - non-blocking)")
                for line in failure_report.splitlines():
                    logger.warning(f"  {line}")
                logger.warning("=" * 60)
        else:
            logger.info("  Strict continuity gate passed.")

    # ── 13. Step 6：拼接 final.mp4 ──
    new_manifest = run_step6(
        manifest=new_manifest,
        output_manifest=str(paths.manifest_new),
        output_video=str(paths.final_video),
        audio_path=audio_path,
    )

    # ── 14. 将新 manifest 提升为当前 manifest ──
    shutil.copy2(str(paths.manifest_new), str(paths.manifest_current))
    logger.info(f"  manifest.json 已更新: {paths.manifest_current}")

    rerendered = len(render_rebuild_keys)
    reused = len(diff.unchanged)

    logger.info("=" * 60)
    logger.info("增量构建完成！")
    logger.info(f"  重渲: {rerendered} 段，复用: {reused} 段")
    logger.info(f"  最终视频: {paths.final_video}")
    logger.info("=" * 60)

    return IncrementalBuildResult(
        success=True,
        project_root=project_root,
        diff=diff,
        final_video=str(paths.final_video),
        rerendered_count=rerendered,
        reused_count=reused,
        total_segments=len(new_manifest.segments),
        dry_run=False,
    )


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────
def _validate_and_build_duration_policy(args) -> dict:
    """
    Validate segmentation CLI args and build duration_policy dict.
    
    Returns dict with policy knobs for run_step2().
    Raises SystemExit if any duration arg is negative with clear message.
    """
    policy = {}
    
    # Validate and collect policy knobs
    duration_args = {
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "target_min_duration": args.target_min_duration,
        "target_max_duration": args.target_max_duration,
        "merge_threshold": args.merge_threshold,
        "split_threshold": args.split_threshold,
    }
    
    for key, value in duration_args.items():
        if value is not None:
            if value < 0:
                logger.error("=" * 60)
                logger.error(f"Segmentation Policy Validation Error")
                logger.error(f"  Parameter: --{key.replace('_', '-')}")
                logger.error(f"  Provided value: {value}")
                logger.error(f"  Error: Duration values must be non-negative")
                logger.error("=" * 60)
                sys.exit(1)
            policy[key] = value
    
    # Resolve generation policy via shared helper (SSOT)
    generation_policy = normalize_generation_policy(
        target_duration_minutes=args.duration_minutes,
        ai_clip_cap=None,  # AI cap remains fixed at default, not exposed to CLI
    )
    
    # Merge generation policy into duration_policy dict for Step2
    policy["target_duration_minutes"] = generation_policy["target_duration_minutes"]
    policy["ai_clip_cap"] = generation_policy["ai_clip_cap"]
    
    return policy


def main():
    parser = argparse.ArgumentParser(
        description="视频自动化工作流 - 增量构建（v2）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 基本增量构建
  python3.11 build_incremental.py --project ./projects/my_video

  # 预览变更（不执行）
  python3.11 build_incremental.py --project ./projects/my_video --dry-run

  # 强制全量重建
  python3.11 build_incremental.py --project ./projects/my_video --full-rebuild

  # 使用新版 SRT
  python3.11 build_incremental.py --project ./projects/my_video --new-srt input/v2.srt

  # 输出 JSON 结果（用于 API 集成）
  python3.11 build_incremental.py --project ./projects/my_video --json
        """,
    )
    parser.add_argument("--project", required=True, help="项目根目录路径")
    parser.add_argument("--new-srt", help="新版 SRT 文件路径（相对于项目根目录或绝对路径）")
    parser.add_argument("--dry-run", action="store_true", help="只分析 diff，不执行构建")
    parser.add_argument("--full-rebuild", action="store_true", help="强制全量重建")
    parser.add_argument("--no-ai-image", action="store_true", help="禁用 AI 图片生成")
    parser.add_argument("--llm-model", default="gpt-4.1-mini", help="LLM 模型名")
    parser.add_argument(
        "--material-mode",
        choices=["auto", "ai_preferred", "ai_only"],
        default="auto",
        help="素材模式：auto=默认（兼容），ai_preferred=优先AI，ai_only=仅AI",
    )
    parser.add_argument("--min-duration", type=float, default=None,
                        help="Minimum segment duration in seconds (default: 1.5)")
    parser.add_argument("--max-duration", type=float, default=None,
                        help="Maximum segment duration in seconds (default: 10.0)")
    parser.add_argument("--target-min-duration", type=float, default=None,
                        help="Target minimum duration for compaction (default: 8.0)")
    parser.add_argument("--target-max-duration", type=float, default=None,
                        help="Target maximum duration for windowing (default: 10.0)")
    parser.add_argument("--merge-threshold", type=float, default=None,
                        help="Pre-split merge threshold in seconds (default: 3.0)")
    parser.add_argument("--split-threshold", type=float, default=None,
                        help="Long-segment split trigger in seconds (default: max_duration)")
    parser.add_argument("--duration-minutes", type=int, default=1,
                        choices=[1, 2, 3],
                        help="Target video duration in minutes (default: 1)")
    parser.add_argument("--json", action="store_true", dest="output_json",
                        help="以 JSON 格式输出结果（用于 API 集成）")
    parser.add_argument("--gate-profile", default="release",
                        choices=["preview", "release"],
                        help="Continuity gate profile: release (default, strict blocking), "
                             "preview (warn only, non-blocking)")

    args = parser.parse_args()

    project_root = str(Path(args.project).resolve())
    new_srt = None
    if args.new_srt:
        srt_candidate = Path(project_root) / args.new_srt
        if srt_candidate.exists():
            new_srt = str(srt_candidate)
        elif Path(args.new_srt).exists():
            new_srt = str(Path(args.new_srt).resolve())
        else:
            print(f"错误：SRT 文件不存在: {args.new_srt}", file=sys.stderr)
            sys.exit(1)

    # Validate and build segmentation policy
    duration_policy = _validate_and_build_duration_policy(args)
    
    if duration_policy:
        logger.info("=" * 60)
        logger.info("Duration & Generation Policy Configuration:")
        
        # Log generation policy (target duration + AI cap)
        target_mins = duration_policy.get("target_duration_minutes")
        ai_cap = duration_policy.get("ai_clip_cap")
        if target_mins is not None:
            logger.info(f"  Target Duration: {target_mins} minutes")
        if ai_cap is not None:
            logger.info(f"  AI Clip Cap: {ai_cap} (fixed policy)")
        
        # Log segmentation overrides
        segmentation_keys = {
            "min_duration", "max_duration", "target_min_duration",
            "target_max_duration", "merge_threshold", "split_threshold"
        }
        overrides = {k: v for k, v in duration_policy.items() if k in segmentation_keys}
        if overrides:
            logger.info("  Segmentation Overrides:")
            for key, value in sorted(overrides.items()):
                logger.info(f"    {key}: {value}s")
        else:
            logger.info("  Segmentation Overrides: using Step2 defaults")
        
        logger.info("=" * 60)
    else:
        logger.info("Segmentation Policy: using Step2 defaults")

    result = incremental_build(
        project_root=project_root,
        new_srt_path=new_srt,
        dry_run=args.dry_run,
        full_rebuild=args.full_rebuild,
        enable_ai_image=not args.no_ai_image,
        llm_model=args.llm_model,
        material_mode=args.material_mode,
        duration_policy=duration_policy,
        gate_profile=args.gate_profile,
    )

    if args.output_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        if result.success:
            if result.dry_run:
                print(f"\n[dry-run] 分析完成")
                if result.diff:
                    print(f"  需重渲: {result.rerendered_count} 段")
                    print(f"  可复用: {result.reused_count} 段")
                    print(f"  详细报告: {Path(project_root) / 'build' / 'diff_report.json'}")
            else:
                print(f"\n构建成功！")
                print(f"  重渲: {result.rerendered_count} 段，复用: {result.reused_count} 段")
                print(f"  最终视频: {result.final_video}")
        else:
            print(f"\n构建失败: {result.error}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
