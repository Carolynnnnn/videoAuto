#!/usr/bin/env python3
"""
全量构建入口（v2）

从头开始执行完整流程：
  [可选] PDF → 脚本 → TTS
  Step 1: 音频 → SRT 字幕对齐
  Step 2: SRT → Manifest 生成（v2 segment_key 策略）
  Step 3: Visual Plan 自动生成（plan_hash 缓存）
  Step 4: 素材处理（content_key + plan_hash 缓存）
  Step 5: 渲染 Segment 视频（render_hash 缓存）
  Step 6: 拼接合成 final.mp4

用法：
  python3.11 build.py --project ./projects/demo
  python3.11 build.py --project ./projects/demo --from-pdf input/source.pdf
  python3.11 build.py --project ./projects/demo --skip-visual-plan
  python3.11 build.py --project ./projects/demo --no-ai-image
  python3.11 build.py --project ./projects/demo --srt build/subtitle.srt
"""
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.core.models import GlobalStyle, Manifest
from src.core.generation_policy import (
    AI_CLIP_CAP_DEFAULT,
    normalize_generation_policy,
)
from src.steps.continuity_telemetry import (
    compute_quality_summary,
    format_strict_mode_failure,
    is_strict_continuity_mode_enabled,
    validate_strict_mode,
)
from src.utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="视频自动化生产工作流 - 全量构建（v2）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--project", "-p", required=True, help="项目根目录路径")
    parser.add_argument("--project-id", default=None, help="项目 ID（默认使用目录名）")
    parser.add_argument("--from-pdf", default=None, help="从 PDF 开始（指定 PDF 路径）")
    parser.add_argument("--audio", default=None, help="指定音频文件路径")
    parser.add_argument("--script", default=None, help="指定脚本文件路径（用于字幕对齐）")
    parser.add_argument("--srt", default=None, help="直接指定已有 SRT 文件（跳过 Step 1）")
    parser.add_argument(
        "--aspect-ratio", default="9:16",
        choices=["9:16", "3:4", "16:9", "1:1"],
        help="视频画幅比例（默认 9:16）",
    )
    parser.add_argument("--skip-visual-plan", action="store_true", help="跳过 Visual Plan 生成")
    parser.add_argument("--no-ai-image", action="store_true", help="禁用 AI 图片生成")
    parser.add_argument("--pexels-api-key", default="", help="Pexels API Key（可选）")
    parser.add_argument("--bgm", default=None, help="BGM 文件路径（可选）")
    parser.add_argument("--tts-voice", default="alloy",
                        choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
    parser.add_argument("--tts-provider", default="minimax",
                        choices=["minimax", "elevenlabs", "openai"],
                        help="TTS 提供商（生产模式默认 minimax）")
    parser.add_argument("--llm-model", default="deepseek-chat", help="LLM 模型")
    parser.add_argument("--local-whisper", action="store_true", help="使用本地 Whisper")
    parser.add_argument("--no-subtitle-effects", action="store_true", help="禁用字幕渐变和高亮效果")
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--material-mode", default="auto",
                        choices=["auto", "ai_preferred", "ai_only"],
                        help="Material generation mode: auto (default, PDF→external→AI), "
                             "ai_preferred (prefer AI), ai_only (strict AI-only)")
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
    parser.add_argument("--dry-run", action="store_true", help="仅验证参数和目录，不执行实际构建")
    parser.add_argument("--gate-profile", default="release",
                        choices=["preview", "release"],
                        help="Continuity gate profile: release (default, strict blocking), "
                             "preview (warn only, non-blocking)")
    return parser.parse_args()


def get_resolution(aspect_ratio: str):
    return {
        "9:16": (1080, 1920),
        "3:4": (1080, 1440),
        "16:9": (1920, 1080),
        "1:1": (1080, 1080),
    }.get(aspect_ratio, (1080, 1920))


def _find_audio(project_root: str) -> str:
    """在 input/ 目录下查找音频文件"""
    input_dir = Path(project_root) / "input"
    for ext in ("*.mp3", "*.wav", "*.m4a", "*.aac"):
        for f in input_dir.glob(ext):
            return str(f)
    return str(input_dir / "voice_full.wav")


def _find_srt(project_root: str) -> str:
    """在 build/ 或 input/ 目录下查找 SRT 文件"""
    for candidate in [
        Path(project_root) / "build" / "subtitle.srt",
        *Path(project_root).glob("input/*.srt"),
    ]:
        if Path(candidate).exists():
            return str(candidate)
    return str(Path(project_root) / "build" / "subtitle.srt")


def _is_test_mode_enabled() -> bool:
    raw = os.environ.get("PIXELLE_TEST_MODE", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _should_enforce_strict_continuity_gate() -> bool:
    if _is_test_mode_enabled():
        return is_strict_continuity_mode_enabled()
    return True


def _get_effective_gate_profile(requested_profile: str) -> str:
    return requested_profile


def _validate_tts_provider_policy(provider: str, test_mode: bool) -> None:
    """
    Validate TTS provider against production policy.
    
    Production policy: Only 'minimax' provider allowed.
    Test mode: All providers allowed (for testing legacy paths).
    
    Raises SystemExit if provider is unsupported in production mode.
    """
    if test_mode:
        # Test mode allows all providers
        return
    
    if provider != "minimax":
        logger = get_logger("build")
        logger.error("=" * 60)
        logger.error(f"TTS Provider Policy Violation")
        logger.error(f"  Requested provider: {provider}")
        logger.error(f"  Production policy: ONLY 'minimax' allowed")
        logger.error(f"  Mode: production (PIXELLE_TEST_MODE=0)")
        logger.error("")
        logger.error("To use legacy providers in non-production:")
        logger.error("  Set PIXELLE_TEST_MODE=1")
        logger.error("=" * 60)
        sys.exit(1)


def _validate_and_build_duration_policy(args) -> dict:
    """
    Validate segmentation CLI args and build duration_policy dict.
    
    Returns dict with policy knobs for run_step2(), including:
      - Segmentation knobs: min/max/target durations, thresholds
      - Generation policy: target_duration_minutes, ai_clip_cap
    
    Raises SystemExit if any duration arg is negative with clear message.
    """
    logger = get_logger("build")
    policy = {}
    
    # Validate and collect segmentation policy knobs
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
    args = parse_args()

    project_root = str(Path(args.project).resolve())
    project_id = args.project_id or Path(project_root).name

    # 创建目录结构
    for d in ["input", "build", "render/segments", "assets/generated",
              "assets/library", "cache/plans", "logs"]:
        Path(project_root, d).mkdir(parents=True, exist_ok=True)

    logger = get_logger("build")
    logger.info("=" * 60)
    logger.info(f"全量构建 v2: project_id={project_id}")
    logger.info(f"项目目录: {project_root}")
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Validate TTS provider policy before execution
    test_mode = _is_test_mode_enabled()
    _validate_tts_provider_policy(args.tts_provider, test_mode)

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
            logger.info("  Segmentation: using Step2 defaults")
        logger.info("=" * 60)
    else:
        logger.info("Duration Policy: using all defaults")

    if args.dry_run:
        logger.info("Dry run mode enabled. Skipping execution steps.")
        logger.info("Dry run successful!")
        return

    # 全局样式
    w, h = get_resolution(args.aspect_ratio)
    style = GlobalStyle(
        aspect_ratio=args.aspect_ratio,
        resolution=f"{w}x{h}",
        enable_subtitle_effects=not args.no_subtitle_effects,
    )

    # ── PDF 前置流程 ──
    if args.from_pdf:
        from src.steps.step_pdf import run_pdf_pipeline
        pdf_path = str(Path(args.from_pdf).resolve())
        logger.info(f"执行 PDF 前置流程: {pdf_path}")
        logger.info(f"TTS Provider: {args.tts_provider} (test_mode={test_mode})")
        result = run_pdf_pipeline(
            pdf_path=pdf_path,
            project_root=project_root,
            llm_model=args.llm_model,
            tts_provider=args.tts_provider,
            tts_voice=args.tts_voice,
            allow_legacy_provider_override=test_mode,
        )
        audio_path = result["voice_path"]
    else:
        audio_path = args.audio or _find_audio(project_root)
        if not Path(audio_path).exists():
            logger.error(f"音频文件不存在: {audio_path}")
            logger.error("请提供 input/voice_full.wav 或 input/voice_full.mp3")
            sys.exit(1)

    logger.info(f"音频文件: {audio_path}")

    # ── Step 1: 音频 → SRT ──
    srt_path = str(Path(project_root) / "build" / "subtitle.srt")
    if args.srt:
        srt_path = str(Path(args.srt).resolve())
        logger.info(f"使用已有 SRT: {srt_path}")
    elif not Path(srt_path).exists():
        from src.steps.step1_align import run_step1
        run_step1(
            audio_path=audio_path,
            output_srt=srt_path,
            script_path=args.script if args.script and Path(args.script).exists() else None,
            use_local_whisper=args.local_whisper,
            whisper_model=args.whisper_model,
        )
    else:
        logger.info(f"SRT 已存在，跳过 Step 1: {srt_path}")

    manifest_path = str(Path(project_root) / "build" / "manifest.json")
    segments_dir = str(Path(project_root) / "render" / "segments")
    cache_plans = str(Path(project_root) / "cache" / "plans")
    final_video = str(Path(project_root) / "render" / "final.mp4")

    # ── Step 2: SRT → Manifest (v2) ──
    from src.steps.step2_manifest import run_step2
    manifest = run_step2(
        srt_path=srt_path,
        audio_path=audio_path,
        project_id=project_id,
        output_manifest=manifest_path,
        global_style=style,
        material_mode=args.material_mode,
        duration_policy=duration_policy or None,
    )

    # ── AI Workflow Default Wiring ──
    # Apply deterministic pixelle_default_workflow when material mode is AI-centric
    # and no workflow is already set (preserves explicit overrides)
    if args.material_mode in ("ai_only", "ai_preferred"):
        if manifest.pixelle_default_workflow is None:
            manifest.pixelle_default_workflow = "i2v"
            logger.info(f"Applied AI workflow default: pixelle_default_workflow='i2v' (source: material_mode={args.material_mode})")
        else:
            logger.info(f"Preserving explicit workflow: pixelle_default_workflow='{manifest.pixelle_default_workflow}' (source: user override)")
    else:
        if manifest.pixelle_default_workflow is None:
            logger.info(f"No workflow default applied (material_mode={args.material_mode})")
        else:
            logger.info(f"Using explicit workflow: pixelle_default_workflow='{manifest.pixelle_default_workflow}' (source: user override)")
    
    manifest.save(manifest_path)
    logger.info(f"  Manifest saved with workflow policy: {manifest_path}")

    # ── Step 3: Visual Plan ──
    if not args.skip_visual_plan:
        from src.steps.step3_visual_plan import run_step3
        manifest = run_step3(
            manifest=manifest,
            output_manifest=manifest_path,
            cache_dir=cache_plans,
            llm_model=args.llm_model,
        )
    else:
        logger.info("跳过 Visual Plan 生成（--skip-visual-plan）")

    # ── Step 4: 素材处理 ──
    from src.steps.step4_assets import run_step4
    manifest = run_step4(
        manifest=manifest,
        output_manifest=manifest_path,
        project_root=project_root,
        pexels_api_key=args.pexels_api_key,
        enable_pexels_video=True,
        enable_pexels_photo=True,
        enable_ai_image=not args.no_ai_image,
    )

    # ── Step 5: 渲染片段 ──
    from src.steps.step5_render import run_step5
    manifest = run_step5(
        manifest=manifest,
        output_manifest=manifest_path,
        segments_dir=segments_dir,
        force_rerender=False,
    )

    if _should_enforce_strict_continuity_gate():
        effective_profile = _get_effective_gate_profile(args.gate_profile)
        logger.info(f"Continuity gate profile: {effective_profile}")
        strict_result = validate_strict_mode(compute_quality_summary(manifest.segments), profile=effective_profile)
        if not strict_result.passed:
            failure_report = format_strict_mode_failure(strict_result)
            if effective_profile == "release":
                for line in failure_report.splitlines():
                    logger.error(line)
                sys.exit(1)
            else:
                logger.warning("=" * 60)
                logger.warning("Continuity gate WARNINGS (preview mode - non-blocking)")
                for line in failure_report.splitlines():
                    logger.warning(line)
                logger.warning("=" * 60)
        else:
            logger.info("Strict continuity gate passed.")

    # ── Step 6: 拼接合成 ──
    from src.steps.step6_concat import run_step6
    manifest = run_step6(
        manifest=manifest,
        output_manifest=manifest_path,
        output_video=final_video,
        audio_path=audio_path,
    )

    logger.info("=" * 60)
    logger.info("全量构建完成！")
    logger.info(f"  最终视频: {final_video}")
    logger.info(f"  Manifest: {manifest_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
