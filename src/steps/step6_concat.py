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
            f'-i "{video_path}" '
            f'-i "{voice_path}" '
            f'-i "{bgm_path}" '
            f'-filter_complex "{audio_filter}" '
            f'-map 0:v -map "[aout]" '
            f'-c:v copy -c:a aac -b:a 192k '
            f'"{output_path}" -loglevel warning'
        )
    else:
        cmd = (
            f'ffmpeg -y '
            f'-i "{video_path}" '
            f'-i "{voice_path}" '
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
