"""
Step P1-P3：PDF 处理前置流程（v2）

P1: PDF → 文本 + 图片抽取
P2: 文本 → 脚本生成（DeepSeek LLM）
P3: 脚本 → TTS 语音生成（Minimax primary, ElevenLabs fallback）
"""
from __future__ import annotations
import os
import re
import json
import subprocess
import time
from pathlib import Path
from typing import Optional, List, Callable

from src.utils.logger import get_logger

logger = get_logger("step_pdf")


# ─────────────────────────────────────────────
# P1：PDF 文本抽取
# ─────────────────────────────────────────────
def extract_pdf_text(pdf_path: str, output_md: str) -> str:
    """使用 pdftotext 抽取 PDF 文本，输出为 Markdown 格式。"""
    logger.info(f"抽取 PDF 文本: {pdf_path}")
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)

    txt_tmp = str(Path(output_md).with_suffix(".txt"))
    try:
        rc = subprocess.run(
            ["pdftotext", "-layout", pdf_path, txt_tmp],
            capture_output=True,
        ).returncode
    except FileNotFoundError:
        logger.warning("pdftotext 命令未找到，尝试 PyPDF2")
        rc = 1

    if rc != 0 or not Path(txt_tmp).exists():
        logger.warning("pdftotext 失败，尝试 PyPDF2")
        try:
            import PyPDF2
            text_parts = []
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text_parts.append(page.extract_text() or "")
            text = "\n\n".join(text_parts)
        except ImportError:
            logger.error("PyPDF2 未安装，无法抽取 PDF 文本")
            text = ""
    else:
        text = Path(txt_tmp).read_text(encoding="utf-8", errors="replace")
        Path(txt_tmp).unlink(missing_ok=True)

    Path(output_md).write_text(f"# 文档内容\n\n{text}", encoding="utf-8")
    logger.info(f"PDF 文本已保存: {output_md} ({len(text)} 字符)")
    return text


def extract_pdf_images(pdf_path: str, output_dir: str) -> List[str]:
    """使用 pdfimages 抽取 PDF 中的图片。"""
    logger.info(f"抽取 PDF 图片: {pdf_path}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        rc = subprocess.run(
            ["pdfimages", "-png", pdf_path, str(Path(output_dir) / "page")],
            capture_output=True,
        ).returncode
    except FileNotFoundError:
        rc = 1

    if rc == 0:
        images = list(Path(output_dir).glob("page-*.png"))
        logger.info(f"  抽取图片 {len(images)} 张")
        return [str(p) for p in sorted(images)]

    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(pdf_path, dpi=150)
        paths = []
        for i, page in enumerate(pages):
            p = str(Path(output_dir) / f"page-{i:03d}.png")
            page.save(p, "PNG")
            paths.append(p)
        logger.info(f"  pdf2image 抽取图片 {len(paths)} 张")
        return paths
    except Exception as e:
        logger.warning(f"  图片抽取失败: {e}")
        return []


# ─────────────────────────────────────────────
# P2：DeepSeek LLM 生成脚本
# ─────────────────────────────────────────────
SCRIPT_SYSTEM_PROMPT = """你是一位专业的短视频脚本撰写师。
请将以下文档内容改写为适合视频号/抖音的口播脚本。

要求：
1. 总时长约 60-90 秒（约 200-300 字）
2. 每句话简短有力（10-25 字），每句单独一行
3. 开头吸引眼球，结尾有行动号召
4. 语言口语化、通俗易懂
5. 直接输出脚本文本，不需要标题和说明
"""


def generate_script_from_text(
    content_text: str,
    output_script: str,
    llm_model: str = "deepseek-chat",
    api_key: str = "",
    base_url: str = "",
    max_chars: int = 4000,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> str:
    """使用 DeepSeek LLM 将文档内容转换为口播脚本。"""
    from src.core.api_config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
    api_key = api_key or DEEPSEEK_API_KEY
    base_url = base_url or DEEPSEEK_BASE_URL
    llm_model = llm_model or DEEPSEEK_MODEL

    logger.info(f"DeepSeek 生成脚本: model={llm_model}")
    if progress_cb:
        progress_cb("正在使用 DeepSeek AI 生成视频脚本...")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)

    if len(content_text) > max_chars:
        content_text = content_text[:max_chars] + "\n...(内容已截断)"

    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": SCRIPT_SYSTEM_PROMPT},
            {"role": "user", "content": f"文档内容：\n\n{content_text}"},
        ],
        temperature=0.7,
        max_tokens=1500,
    )

    script = response.choices[0].message.content.strip()
    Path(output_script).parent.mkdir(parents=True, exist_ok=True)
    Path(output_script).write_text(script, encoding="utf-8")
    logger.info(f"脚本已生成: {output_script} ({len(script)} 字符)")
    if progress_cb:
        progress_cb(f"脚本生成完成（{len(script)} 字）")
    return script


# ─────────────────────────────────────────────
# P3：Minimax TTS 生成语音（Primary Provider）
# ─────────────────────────────────────────────
def generate_tts_minimax(
    script_path: str,
    output_audio: str,
    voice_id: str = "male-qn-qingse",
    model_id: str = "speech-01",
    api_key: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> str:
    """使用 Minimax 生成高质量 TTS 语音（Primary Provider）。"""
    from src.integrations.minimax import generate_tts_minimax as _generate_tts_minimax
    from src.integrations.minimax import MinimaxTTSError
    
    logger.info(f"Minimax TTS: voice_id={voice_id}, model={model_id}")
    if progress_cb:
        progress_cb("正在使用 Minimax 生成高质量语音...")
    
    try:
        return _generate_tts_minimax(
            script_path=script_path,
            output_audio=output_audio,
            voice_id=voice_id,
            model_id=model_id,
            api_key=api_key,
            progress_cb=progress_cb,
        )
    except MinimaxTTSError as e:
        logger.error(f"Minimax TTS 失败: {e}")
        raise


# ─────────────────────────────────────────────
# P3：ElevenLabs TTS 生成语音（Fallback Provider）
# ─────────────────────────────────────────────
def generate_tts_elevenlabs(
    script_path: str,
    output_audio: str,
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",
    model_id: str = "eleven_multilingual_v2",
    api_key: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> str:
    """使用 ElevenLabs 生成高质量 TTS 语音（Fallback Provider）。"""
    from src.core.api_config import ELEVENLABS_API_KEY
    api_key = api_key or ELEVENLABS_API_KEY

    logger.info(f"ElevenLabs TTS: voice_id={voice_id}, model={model_id}")
    if progress_cb:
        progress_cb("正在使用 ElevenLabs 生成高质量语音...")

    script_text = Path(script_path).read_text(encoding="utf-8").strip()
    script_text = re.sub(r"^#+\s+", "", script_text, flags=re.MULTILINE)
    script_text = re.sub(r"\*+", "", script_text).strip()

    logger.info(f"  脚本长度: {len(script_text)} 字符")

    from elevenlabs import ElevenLabs
    client = ElevenLabs(api_key=api_key)

    MAX_CHARS = 2000
    if len(script_text) <= MAX_CHARS:
        chunks = [script_text]
    else:
        sentences = re.split(r"([。！？\n]+)", script_text)
        chunks = []
        current = ""
        for s in sentences:
            if len(current) + len(s) <= MAX_CHARS:
                current += s
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = s
        if current.strip():
            chunks.append(current.strip())

    logger.info(f"  分为 {len(chunks)} 段生成 TTS")
    Path(output_audio).parent.mkdir(parents=True, exist_ok=True)
    audio_parts = []

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        if progress_cb:
            progress_cb(f"ElevenLabs TTS 第 {i+1}/{len(chunks)} 段...")

        tmp_path = str(Path(output_audio).parent / f"_tts_el_{i}.mp3")
        try:
            audio_generator = client.text_to_speech.convert(
                voice_id=voice_id,
                text=chunk,
                model_id=model_id,
                output_format="mp3_44100_128",
            )
            with open(tmp_path, "wb") as f:
                for audio_chunk in audio_generator:
                    f.write(audio_chunk)
            audio_parts.append(tmp_path)
            logger.info(f"  TTS 段 {i+1}/{len(chunks)} 完成")
        except Exception as e:
            logger.error(f"  ElevenLabs TTS 段 {i+1} 失败: {e}")
            raise

    if not audio_parts:
        raise ValueError("ElevenLabs TTS 生成失败：无音频输出")

    if len(audio_parts) == 1:
        import shutil
        shutil.move(audio_parts[0], output_audio)
    else:
        list_path = str(Path(output_audio).parent / "_tts_el_list.txt")
        with open(list_path, "w") as f:
            for p in audio_parts:
                f.write(f"file '{Path(p).resolve()}'\n")
        cmd = (
            f'ffmpeg -y -f concat -safe 0 -i "{list_path}" '
            f'-c:a libmp3lame -b:a 192k "{output_audio}" -loglevel error'
        )
        subprocess.run(cmd, shell=True)
        for p in audio_parts:
            Path(p).unlink(missing_ok=True)
        Path(list_path).unlink(missing_ok=True)

    logger.info(f"ElevenLabs TTS 语音已生成: {output_audio}")
    if progress_cb:
        progress_cb(f"语音生成完成")
    return output_audio


def generate_tts(
    script_path: str,
    output_audio: str,
    voice: str = "alloy",
    speed: float = 1.0,
    model: str = "tts-1",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> str:
    """OpenAI TTS 备用方案（Legacy Fallback）"""
    logger.info(f"OpenAI TTS: voice={voice}, speed={speed}")
    if progress_cb:
        progress_cb("正在生成语音（OpenAI TTS）...")

    from openai import OpenAI
    script_text = Path(script_path).read_text(encoding="utf-8").strip()
    script_text = re.sub(r"^#+\s+", "", script_text, flags=re.MULTILINE).strip()

    client = OpenAI()
    sentences = re.split(r"([。！？\n]+)", script_text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) < 4000:
            current += s
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)

    audio_parts = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        tmp_path = str(Path(output_audio).parent / f"_tts_part_{i}.mp3")
        response = client.audio.speech.create(
            model=model, voice=voice, input=chunk, speed=speed,
        )
        response.stream_to_file(tmp_path)
        audio_parts.append(tmp_path)

    Path(output_audio).parent.mkdir(parents=True, exist_ok=True)
    if len(audio_parts) == 1:
        import shutil
        shutil.move(audio_parts[0], output_audio)
    else:
        list_path = str(Path(output_audio).parent / "_tts_list.txt")
        with open(list_path, "w") as f:
            for p in audio_parts:
                f.write(f"file '{Path(p).resolve()}'\n")
        cmd = (
            f'ffmpeg -y -f concat -safe 0 -i "{list_path}" '
            f'-c:a libmp3lame -b:a 192k "{output_audio}" -loglevel error'
        )
        subprocess.run(cmd, shell=True)
        for p in audio_parts:
            Path(p).unlink(missing_ok=True)
        Path(list_path).unlink(missing_ok=True)

    logger.info(f"TTS 语音已生成: {output_audio}")
    return output_audio


# ─────────────────────────────────────────────
# 主入口：PDF 全流程
# ─────────────────────────────────────────────
def run_pdf_pipeline(
    pdf_path: str,
    project_root: str,
    llm_model: str = "deepseek-chat",
    tts_provider: str = "minimax",
    tts_voice: str = "male-qn-qingse",
    tts_speed: float = 1.0,
    progress_cb: Optional[Callable[[str], None]] = None,
    allow_legacy_provider_override: bool = False,
) -> dict:
    """
    执行 PDF 前置流程：PDF → 文本 → 脚本（DeepSeek）→ TTS

    TTS Provider Policy (Production-Hardened):
    - Production path: ONLY "minimax" provider allowed
    - Non-production override: Set allow_legacy_provider_override=True to enable
      elevenlabs/openai providers (logs explicit warning)
    - Legacy providers blocked in production unless explicit override is enabled

    :param allow_legacy_provider_override: If True, allows elevenlabs/openai providers
                                            with explicit warning logs (default: False)
    :return: {"content_md": ..., "script_md": ..., "voice_path": ...}
    """
    from src.core.api_config import ELEVENLABS_VOICES
    from src.integrations.minimax import MINIMAX_VOICES

    # ─── Policy Enforcement: Minimax-Only Production Path ───
    policy_mode = "non-production-override" if allow_legacy_provider_override else "production"
    effective_provider = tts_provider
    
    if tts_provider != "minimax" and not allow_legacy_provider_override:
        logger.warning(
            f"TTS provider '{tts_provider}' blocked by production policy. "
            f"Forcing effective provider to 'minimax'. "
            f"Policy: {policy_mode}, Requested: {tts_provider}, Effective: minimax"
        )
        effective_provider = "minimax"
    elif tts_provider != "minimax" and allow_legacy_provider_override:
        logger.warning(
            f"⚠️  LEGACY PROVIDER OVERRIDE ACTIVE: Using '{tts_provider}' provider. "
            f"Policy: {policy_mode}, Requested: {tts_provider}, Effective: {tts_provider}. "
            f"This path is NOT recommended for production use."
        )
        effective_provider = tts_provider
    else:
        logger.info(
            f"TTS provider policy enforcement: "
            f"Policy: {policy_mode}, Requested: {tts_provider}, Effective: {effective_provider}"
        )

    logger.info("=" * 50)
    logger.info(f"PDF 前置流程 v2: PDF → 脚本（DeepSeek）→ TTS（{effective_provider}）")

    extracted_dir = Path(project_root) / "extracted"
    input_dir = Path(project_root) / "input"

    if progress_cb:
        progress_cb("正在抽取 PDF 文本...")
    content_md = str(extracted_dir / "content.md")
    text = extract_pdf_text(pdf_path, content_md)

    images_dir = str(extracted_dir / "images")
    extract_pdf_images(pdf_path, images_dir)

    script_md = str(input_dir / "script.md")
    generate_script_from_text(
        text, script_md,
        llm_model=llm_model,
        progress_cb=progress_cb,
    )

    voice_path = str(input_dir / "voice_full.mp3")
    
    # Use effective_provider (policy-enforced) instead of raw tts_provider
    if effective_provider == "minimax":
        voice_id = MINIMAX_VOICES.get(tts_voice, MINIMAX_VOICES["default"])
        generate_tts_minimax(
            script_md, voice_path,
            voice_id=voice_id,
            progress_cb=progress_cb,
        )
    elif effective_provider == "elevenlabs":
        voice_id = ELEVENLABS_VOICES.get(tts_voice, ELEVENLABS_VOICES["default"])
        generate_tts_elevenlabs(
            script_md, voice_path,
            voice_id=voice_id,
            progress_cb=progress_cb,
        )
    else:
        generate_tts(
            script_md, voice_path,
            voice=tts_voice,
            speed=tts_speed,
            progress_cb=progress_cb,
        )

    return {
        "content_md": content_md,
        "script_md": script_md,
        "voice_path": voice_path,
    }
