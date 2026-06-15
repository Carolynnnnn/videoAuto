"""
Step 6：拼接合成（segments → final.mp4，v2）

关键变更：
  - 片段路径从 RenderRef.segment_video_path 读取（v2 字段名）
  - run_step6 增加 audio_path 参数（可选，覆盖 manifest.audio_path）
  - 返回值改为 Manifest（而非 bool），与其他 step 接口一致
  - 兼容旧版 status="done"/"failed" 和新版 status="ok"/"failed"
"""
from __future__ import annotations
import os
import subprocess
from pathlib import Path
from typing import List, Optional

from src.core.models import Manifest
from src.utils.logger import get_logger

logger = get_logger("step6_concat")


def _run_ffmpeg(cmd: str, timeout: int = 600) -> tuple:
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout
    )
    return result.returncode, result.stdout, result.stderr


def _write_concat_list(segment_paths: List[str], list_path: str) -> None:
    with open(list_path, "w", encoding="utf-8") as f:
        for p in segment_paths:
            abs_p = str(Path(p).resolve()).replace("'", "'\\''")
            f.write(f"file '{abs_p}'\n")


def concat_segments(
    segment_paths: List[str],
    output_path: str,
    fps: int = 30,
) -> bool:
    """使用 FFmpeg concat demuxer 拼接视频片段。"""
    if not segment_paths:
        logger.error("没有可拼接的片段")
        return False

    list_path = str(Path(output_path).parent / "_concat_list.txt")
    _write_concat_list(segment_paths, list_path)

    cmd = (
        f'ffmpeg -y '
        f'-f concat -safe 0 -i "{list_path}" '
        f'-c:v libx264 -preset fast -crf 22 '
        f'-c:a aac -b:a 192k '
        f'-pix_fmt yuv420p '
        f'-r {fps} '
        f'"{output_path}" -loglevel warning'
    )

    logger.info(f"拼接 {len(segment_paths)} 个片段...")
    rc, stdout, stderr = _run_ffmpeg(cmd, timeout=600)

    if rc == 0 and Path(output_path).exists():
        logger.info(f"拼接成功: {output_path}")
        try:
            Path(list_path).unlink()
        except Exception:
            pass
        return True
    else:
        logger.error(f"拼接失败: {stderr[-500:]}")
        return False


def _detect_initial_silence(audio_path: str, noise_db: float = -30.0) -> float:
    """
    检测音频文件开头的静音时长（秒）。
    TTS 生成的 MP3 通常有 0.1-0.3s 的无声前导，导致字幕/画面比口播早出现。
    返回静音结束的时间点（即第一个有声音内容的起始时间）。
    """
    cmd = (
        f'ffmpeg -i "{audio_path}" '
        f'-af "silencedetect=noise={noise_db}dB:d=0.05" '
        f'-f null - 2>&1 | grep silence_end | head -1'
    )
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False,
                                text=True, timeout=15,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = result.stdout or ""
        # silence_end: 0.237573 | silence_duration: 0.237573
        for line in output.splitlines():
            if "silence_end:" in line:
                parts = line.split("silence_end:")
                val = parts[1].split("|")[0].strip()
                silence_end = float(val)
                # 只处理前 1s 内的前导静音；过长则可能是正常间隙，不跳过
                if silence_end <= 1.0:
                    logger.info(f"检测到前导静音 {silence_end:.3f}s，音频将从此处开始对齐视频")
                    return silence_end
    except Exception:
        pass
    return 0.0


def mix_audio(
    video_path: str,
    voice_path: str,
    output_path: str,
    bgm_path: Optional[str] = None,
    bgm_volume: float = 0.15,
    bgm_ducking: bool = True,
    normalize_loudness: bool = True,
) -> bool:
    """将整段旁白音轨与视频合并，可选混入 BGM。"""
    # 检测 TTS 前导静音（通常 0.1-0.3s）。
    # 关键：同时裁剪视频和音频相同长度，保证 trim_start[i] 对齐不变，
    # 且 seg1 字幕与开口同步（不再有无声等待期）。
    voice_offset = _detect_initial_silence(voice_path)
    if voice_offset > 0:
        # -ss 同时作用于视频输入和音频输入：两者均从 voice_offset 处开始，对齐关系不变
        video_seek = f'-ss {voice_offset:.3f} -i "{video_path}"'
        voice_input = f'-ss {voice_offset:.3f} -i "{voice_path}"'
        logger.info(f"同步裁剪视频+音频各 {voice_offset:.3f}s，消除初始静音")
    else:
        video_seek = f'-i "{video_path}"'
        voice_input = f'-i "{voice_path}"'

    if bgm_path and Path(bgm_path).exists():
        if bgm_ducking:
            audio_filter = (
                f"[1:a]aformat=fltp,apad[voice];"
                f"[2:a]aloop=loop=-1:size=2e+09,volume={bgm_volume}[bgm];"
                f"[bgm][voice]sidechaincompress=threshold=0.02:ratio=10:attack=200:release=1000[bgm_ducked];"
                f"[voice][bgm_ducked]amix=inputs=2:duration=first:dropout_transition=3[aout]"
            )
        else:
            audio_filter = (
                f"[1:a]aformat=fltp,apad[voice];"
                f"[2:a]aloop=loop=-1:size=2e+09,volume={bgm_volume}[bgm];"
                f"[voice][bgm]amix=inputs=2:duration=first[aout]"
            )
        cmd = (
            f'ffmpeg -y '
            f'{video_seek} '
            f'{voice_input} '
            f'-i "{bgm_path}" '
            f'-filter_complex "{audio_filter}" '
            f'-map 0:v -map "[aout]" '
            f'-c:v copy -c:a aac -b:a 192k '
            f'"{output_path}" -loglevel warning'
        )
    else:
        cmd = (
            f'ffmpeg -y '
            f'{video_seek} '
            f'{voice_input} '
            f'-map 0:v -map 1:a '
            f'-c:v copy -c:a aac -b:a 192k '
            f'-shortest '
            f'"{output_path}" -loglevel warning'
        )

    logger.info("混合音轨...")
    rc, stdout, stderr = _run_ffmpeg(cmd, timeout=300)

    if rc == 0 and Path(output_path).exists():
        if normalize_loudness:
            return _normalize_loudness(output_path)
        return True
    else:
        logger.error(f"音轨混合失败: {stderr[-300:]}")
        return False


def _srt_to_ass(
    srt_path: str,
    ass_path: str,
    font_name: str = "Heiti SC",
    font_size: int = 48,
    video_width: int = 1080,
    video_height: int = 1920,
    margin_v: int = 150,
    time_offset: float = 0.0,
) -> bool:
    """将 SRT 转换为 ASS，设置正确的 PlayResX/Y 以防 libass 缩放字体。

    time_offset: 正值表示字幕时间轴向前移动 N 秒（视频被裁掉了 N 秒开头时使用）。
    """
    import re as _re

    def _parse_time(ts: str) -> float:
        h, m, s = ts.replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    def _fmt_ass_time(t: float) -> str:
        t = max(0.0, t)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h}:{m:02d}:{s:05.2f}"

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {video_width}\n"
        f"PlayResY: {video_height}\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00000000,"
        f"&H80000000,0,0,0,0,100,100,0,0,4,0,0,2,10,10,{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    try:
        with open(srt_path, encoding="utf-8") as f:
            content = f.read()

        blocks = _re.split(r"\n\s*\n", content.strip())
        lines = [header]
        for block in blocks:
            parts = block.strip().splitlines()
            if len(parts) < 3:
                continue
            m = _re.match(r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})", parts[1])
            if not m:
                continue
            start = _parse_time(m.group(1)) - time_offset
            end = _parse_time(m.group(2)) - time_offset
            if end <= 0:
                continue
            text = "\\N".join(p.strip() for p in parts[2:] if p.strip())
            lines.append(
                f"Dialogue: 0,{_fmt_ass_time(start)},{_fmt_ass_time(end)},"
                f"Default,,0,0,0,,{text}\n"
            )

        with open(ass_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return True
    except Exception as e:
        logger.error(f"SRT→ASS 转换失败: {e}")
        return False


def overlay_subtitle_srt(
    video_path: str,
    srt_path: str,
    output_path: str,
    font_size: int = 48,
    font_color: str = "white",
    srt_time_offset: float = 0.0,
    video_width: int = 1080,
    video_height: int = 1920,
) -> bool:
    """将 SRT 字幕叠加到视频上。

    先将 SRT 转成包含正确 PlayResX/Y 的 ASS 文件，再用 ass 滤镜渲染，
    避免 libass 默认 PlayResY=288 导致字体被放大 6~7 倍的问题。

    srt_time_offset: 视频起点被裁剪的秒数（正值），字幕时间轴相应前移。
    """
    ass_path = str(Path(srt_path).with_suffix(".tmp.ass"))
    ok = _srt_to_ass(
        srt_path=srt_path,
        ass_path=ass_path,
        font_size=font_size,
        video_width=video_width,
        video_height=video_height,
        time_offset=srt_time_offset,
    )
    if not ok:
        return False

    ass_escaped = str(Path(ass_path).resolve()).replace(":", "\\:")
    tmp_path = str(Path(output_path).with_suffix(".sub_tmp.mp4"))
    cmd = (
        f'ffmpeg -y -i "{video_path}" '
        f'-vf "ass={ass_escaped}" '
        f'-c:v libx264 -preset fast -crf 22 '
        f'-c:a copy '
        f'"{tmp_path}" -loglevel warning'
    )
    rc, _, stderr = _run_ffmpeg(cmd, timeout=600)
    try:
        Path(ass_path).unlink(missing_ok=True)
    except Exception:
        pass
    if rc == 0 and Path(tmp_path).exists():
        Path(output_path).unlink(missing_ok=True)
        Path(tmp_path).rename(output_path)
        logger.info(f"SRT 字幕叠加完成: {output_path}")
        return True
    else:
        logger.error(f"SRT 字幕叠加失败: {stderr[-300:]}")
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        return False


def _normalize_loudness(video_path: str, target_lufs: float = -16.0) -> bool:
    """响度归一化（EBU R128）"""
    tmp_path = str(Path(video_path).with_suffix(".tmp.mp4"))
    cmd = (
        f'ffmpeg -y -i "{video_path}" '
        f'-af "loudnorm=I={target_lufs}:TP=-1.5:LRA=11" '
        f'-c:v copy -c:a aac -b:a 192k '
        f'"{tmp_path}" -loglevel warning'
    )
    rc, _, stderr = _run_ffmpeg(cmd, timeout=300)
    if rc == 0 and Path(tmp_path).exists():
        Path(video_path).unlink()
        Path(tmp_path).rename(video_path)
        logger.info(f"响度归一化完成: {target_lufs} LUFS")
        return True
    else:
        logger.warning(f"响度归一化失败（跳过）: {stderr[-200:]}")
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        return True  # 不阻塞流程


def run_step6(
    manifest: Manifest,
    output_manifest: str,
    output_video: str,
    audio_path: Optional[str] = None,
    bgm_path: Optional[str] = None,
    bgm_volume: float = 0.15,
    bgm_ducking: bool = True,
    normalize_loudness: bool = True,
    subtitle_srt_path: Optional[str] = None,
) -> Manifest:
    """
    执行 Step 6：拼接合成 final.mp4（v2）

    :param manifest: 输入 Manifest
    :param output_manifest: 更新后 manifest.json 路径
    :param output_video: 输出 final.mp4 路径
    :param audio_path: 旁白音频路径（None=使用 manifest.audio_path）
    :param bgm_path: BGM 文件路径（可选）
    :param bgm_volume: BGM 音量
    :param bgm_ducking: 是否 ducking
    :param normalize_loudness: 是否响度归一化
    :param subtitle_srt_path: SRT 字幕文件路径（None=不叠加字幕）
    :return: 更新后的 Manifest
    """
    logger.info("=" * 50)
    logger.info("Step 6: 拼接合成 final.mp4 (v2)")

    # 收集已渲染的片段（v2：从 RenderRef.segment_video_path 读取）
    done_segments = [
        seg for seg in manifest.segments
        if seg.render_ref.status in ("ok", "done", "failed")
    ]
    done_segments.sort(key=lambda s: s.index)

    segment_paths = []
    for seg in done_segments:
        # v2 字段：segment_video_path
        seg_path = seg.render_ref.segment_video_path
        if seg_path and Path(seg_path).exists():
            segment_paths.append(seg_path)
        else:
            logger.warning(f"  [seg {seg.index}] 片段文件不存在: {seg_path}")

    if not segment_paths:
        logger.error("没有可拼接的片段文件")
        manifest.build_status = "failed"
        manifest.save(output_manifest)
        return manifest

    logger.info(f"  共 {len(segment_paths)} 个片段待拼接")

    Path(output_video).parent.mkdir(parents=True, exist_ok=True)

    # 先拼接视频
    concat_tmp = str(Path(output_video).with_suffix(".concat.mp4"))
    if not concat_segments(segment_paths, concat_tmp, fps=manifest.global_style.fps):
        manifest.build_status = "failed"
        manifest.save(output_manifest)
        return manifest

    # 混合音轨
    voice_path = audio_path or manifest.audio_path
    if voice_path and Path(voice_path).exists():
        success = mix_audio(
            video_path=concat_tmp,
            voice_path=voice_path,
            output_path=output_video,
            bgm_path=bgm_path,
            bgm_volume=bgm_volume,
            bgm_ducking=bgm_ducking,
            normalize_loudness=normalize_loudness,
        )
        try:
            Path(concat_tmp).unlink(missing_ok=True)
        except Exception:
            pass
    else:
        logger.warning("  未找到旁白音频，使用各段自带音频")
        Path(concat_tmp).rename(output_video)
        success = True

    if success and Path(output_video).exists():
        # 叠加 SRT 字幕（时间轴直接跟 Whisper 原始时间戳，天然与音频同步）
        if subtitle_srt_path and Path(subtitle_srt_path).exists():
            _audio_for_silence = audio_path or manifest.audio_path or ""
            voice_offset = _detect_initial_silence(_audio_for_silence) if _audio_for_silence and Path(_audio_for_silence).exists() else 0.0
            srt_ok = overlay_subtitle_srt(
                video_path=output_video,
                srt_path=subtitle_srt_path,
                output_path=output_video,
                font_size=manifest.global_style.font_size or 48,
                srt_time_offset=voice_offset,
                video_width=manifest.global_style.resolution_w,
                video_height=manifest.global_style.resolution_h,
            )
            if not srt_ok:
                logger.warning("SRT 叠加失败，保留无字幕版本")

        size_mb = Path(output_video).stat().st_size / 1024 / 1024
        logger.info(f"  final.mp4 生成成功: {output_video} ({size_mb:.1f} MB)")
        manifest.final_video = output_video
        manifest.build_status = "done"
    else:
        manifest.build_status = "failed"

    os.makedirs(os.path.dirname(output_manifest), exist_ok=True)
    manifest.save(output_manifest)
    logger.info(f"Manifest 已更新: {output_manifest}")

    return manifest
