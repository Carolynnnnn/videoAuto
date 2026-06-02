"""
完整端到端测试脚本
PDF → DeepSeek 脚本 → ElevenLabs TTS → Whisper SRT → Manifest → Visual Plan → 渲染 → final.mp4
"""
import sys
import os
import shutil
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# API Keys should be set via environment variables before running
# e.g., export DEEPSEEK_API_KEY=<your-key>
# e.g., export ELEVENLABS_API_KEY=<your-key>

from src.utils.logger import get_logger
logger = get_logger("e2e_test")


def progress(msg: str):
    print(f"  [✓] {msg}")
    logger.info(msg)


def run_e2e_test(
    pdf_path: str,
    project_name: str = "ai_whitepaper",
    aspect_ratio: str = "9:16",
    progress_cb=None,
):
    """运行完整端到端测试"""
    if progress_cb is None:
        progress_cb = progress

    print("\n" + "=" * 60)
    print("  视频自动化工作流 - 端到端测试")
    print("=" * 60)
    print(f"  PDF: {pdf_path}")
    print(f"  项目: {project_name}")
    print(f"  比例: {aspect_ratio}")
    print("=" * 60 + "\n")

    project_root = str(Path(__file__).parent / "projects" / project_name)

    # ── Step 0: 初始化项目 ──
    print("【Step 0】初始化项目目录...")
    for d in ["input", "extracted/images", "build", "render/segments",
              "assets/generated", "assets/stock", "cache/visual_plans"]:
        Path(project_root, d).mkdir(parents=True, exist_ok=True)
    pdf_dest = str(Path(project_root) / "input" / "source.pdf")
    shutil.copy2(pdf_path, pdf_dest)
    progress_cb(f"项目目录已创建: {project_root}")

    # ── Step P1: 抽取 PDF 文本 ──
    print("\n【Step P1】抽取 PDF 文本...")
    from src.steps.step_pdf import (
        extract_pdf_text, extract_pdf_images,
        generate_script_from_text, generate_tts_elevenlabs
    )
    from src.core.api_config import ELEVENLABS_VOICES

    content_md = str(Path(project_root) / "extracted" / "content.md")
    text = extract_pdf_text(pdf_dest, content_md)
    progress_cb(f"PDF 文本抽取完成 ({len(text)} 字符)")

    images_dir = str(Path(project_root) / "extracted" / "images")
    extract_pdf_images(pdf_dest, images_dir)
    progress_cb("PDF 图片抽取完成")

    # ── Step P2: DeepSeek 生成脚本 ──
    print("\n【Step P2】DeepSeek 生成脚本...")
    t0 = time.time()
    script_md = str(Path(project_root) / "input" / "script.md")
    script = generate_script_from_text(
        text, script_md,
        llm_model="deepseek-chat",
        progress_cb=progress_cb,
    )
    progress_cb(f"脚本生成完成 ({len(script)} 字，耗时 {time.time()-t0:.1f}s)")
    print(f"\n  脚本预览（前200字）:\n  {'─'*40}")
    for line in script[:200].split('\n')[:6]:
        if line.strip():
            print(f"  {line}")
    print(f"  {'─'*40}\n")

    # ── Step P3: ElevenLabs TTS ──
    print("\n【Step P3】ElevenLabs TTS 生成语音...")
    t0 = time.time()
    voice_path = str(Path(project_root) / "input" / "voice_full.mp3")
    voice_id = ELEVENLABS_VOICES["rachel"]
    generate_tts_elevenlabs(
        script_md, voice_path,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        progress_cb=progress_cb,
    )
    audio_size = Path(voice_path).stat().st_size / 1024
    progress_cb(f"ElevenLabs TTS 完成 ({audio_size:.0f} KB，耗时 {time.time()-t0:.1f}s)")

    # ── Step 1: 音频对齐 → SRT ──
    print("\n【Step 1】Whisper 音频对齐 → SRT...")
    t0 = time.time()
    from src.steps.step1_align import run_step1
    srt_path = str(Path(project_root) / "build" / "subtitle.srt")
    run_step1(
        audio_path=voice_path,
        output_srt=srt_path,
        script_path=script_md,
        use_local_whisper=True,
        whisper_model="base",
    )
    srt_text = Path(srt_path).read_text(encoding="utf-8")
    srt_count = srt_text.count("\n\n")
    progress_cb(f"SRT 生成完成 ({srt_count} 条字幕，耗时 {time.time()-t0:.1f}s)")

    # ── Step 2: SRT → Manifest ──
    print("\n【Step 2】SRT → Manifest 生成...")
    from src.steps.step2_manifest import run_step2
    from src.core.models import GlobalStyle

    global_style = GlobalStyle(
        aspect_ratio=aspect_ratio,
        resolution="1080x1920" if aspect_ratio == "9:16" else "1920x1080",
        fps=30,
        font_size=48,
        subtitle_style="clean",
    )
    manifest_path = str(Path(project_root) / "build" / "manifest.json")
    manifest = run_step2(
        srt_path=srt_path,
        audio_path=voice_path,
        project_id=project_name,
        output_manifest=manifest_path,
        global_style=global_style,
    )
    progress_cb(f"Manifest 生成完成 ({len(manifest.segments)} 段)")

    # ── Step 3: Visual Plan（DeepSeek LLM）──
    print("\n【Step 3】DeepSeek Visual Plan 生成...")
    t0 = time.time()
    from src.steps.step3_visual_plan import run_step3
    cache_dir = str(Path(project_root) / "cache" / "visual_plans")
    manifest = run_step3(
        manifest=manifest,
        output_manifest=manifest_path,
        cache_dir=cache_dir,
        llm_model="deepseek-chat",
    )
    progress_cb(f"Visual Plan 生成完成 (耗时 {time.time()-t0:.1f}s)")

    # ── Step 4: 素材处理 ──
    print("\n【Step 4】素材处理（模板兜底）...")
    from src.steps.step4_assets import run_step4
    manifest = run_step4(
        manifest=manifest,
        output_manifest=manifest_path,
        project_root=project_root,
        enable_ai_image=False,  # 不调用 AI 图片生成，使用模板兜底
    )
    progress_cb("素材处理完成")

    # ── Step 5: 分段渲染 ──
    print("\n【Step 5】FFmpeg 分段视频渲染...")
    t0 = time.time()
    from src.steps.step5_render import run_step5
    segments_dir = str(Path(project_root) / "render" / "segments")
    manifest = run_step5(
        manifest=manifest,
        output_manifest=manifest_path,
        segments_dir=segments_dir,
    )
    rendered_ok = sum(1 for s in manifest.segments if s.render_ref.status == "ok")
    progress_cb(f"分段渲染完成 ({rendered_ok}/{len(manifest.segments)} 段成功，耗时 {time.time()-t0:.1f}s)")

    # ── Step 6: 拼接合成 ──
    print("\n【Step 6】FFmpeg 拼接合成 final.mp4...")
    t0 = time.time()
    from src.steps.step6_concat import run_step6
    final_path = str(Path(project_root) / "render" / "final.mp4")
    manifest = run_step6(
        manifest=manifest,
        output_manifest=manifest_path,
        output_video=final_path,
        audio_path=voice_path,
    )

    if Path(final_path).exists():
        size_mb = Path(final_path).stat().st_size / 1024 / 1024
        progress_cb(f"final.mp4 生成完成 ({size_mb:.2f} MB，耗时 {time.time()-t0:.1f}s)")
    else:
        print("  [✗] final.mp4 生成失败！")
        return None

    # ── 汇总 ──
    print("\n" + "=" * 60)
    print("  端到端测试完成！")
    print("=" * 60)
    print(f"  脚本:    {script_md}")
    print(f"  语音:    {voice_path}")
    print(f"  字幕:    {srt_path}")
    print(f"  Manifest: {manifest_path}")
    print(f"  最终视频: {final_path}")
    print("=" * 60 + "\n")

    return final_path


if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/video_pipeline/test_input.pdf"
    result = run_e2e_test(pdf_path)
    if result:
        print(f"✓ 测试成功！输出: {result}")
        sys.exit(0)
    else:
        print("✗ 测试失败！")
        sys.exit(1)
