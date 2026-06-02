"""
项目配置管理：路径、参数、全局设置
"""
from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProjectConfig:
    """单个项目的路径与参数配置"""
    project_root: str = ""
    project_id: str = "demo"

    # ── 输入 ──
    @property
    def input_dir(self) -> Path:
        return Path(self.project_root) / "input"

    @property
    def source_pdf(self) -> Path:
        return self.input_dir / "source.pdf"

    @property
    def script_md(self) -> Path:
        return self.input_dir / "script.md"

    @property
    def voice_full(self) -> Path:
        # 优先 wav，其次 mp3
        for ext in ("wav", "mp3"):
            p = self.input_dir / f"voice_full.{ext}"
            if p.exists():
                return p
        return self.input_dir / "voice_full.wav"

    # ── 抽取 ──
    @property
    def extracted_dir(self) -> Path:
        return Path(self.project_root) / "extracted"

    @property
    def content_md(self) -> Path:
        return self.extracted_dir / "content.md"

    @property
    def images_dir(self) -> Path:
        return self.extracted_dir / "images"

    # ── 构建产物 ──
    @property
    def build_dir(self) -> Path:
        return Path(self.project_root) / "build"

    @property
    def subtitle_srt(self) -> Path:
        return self.build_dir / "subtitle.srt"

    @property
    def manifest_json(self) -> Path:
        return self.build_dir / "manifest.json"

    @property
    def diff_json(self) -> Path:
        return self.build_dir / "diff.json"

    # ── 素材 ──
    @property
    def assets_dir(self) -> Path:
        return Path(self.project_root) / "assets"

    @property
    def generated_dir(self) -> Path:
        return self.assets_dir / "generated"

    @property
    def library_dir(self) -> Path:
        return self.assets_dir / "library"

    # ── 渲染 ──
    @property
    def render_dir(self) -> Path:
        return Path(self.project_root) / "render"

    @property
    def segments_dir(self) -> Path:
        return self.render_dir / "segments"

    @property
    def audio_dir(self) -> Path:
        return self.render_dir / "audio"

    @property
    def final_mp4(self) -> Path:
        return self.render_dir / "final.mp4"

    # ── 缓存 ──
    @property
    def cache_dir(self) -> Path:
        return Path(self.project_root) / "cache"

    @property
    def plans_cache(self) -> Path:
        return self.cache_dir / "plans"

    @property
    def search_cache(self) -> Path:
        return self.cache_dir / "search"

    # ── 日志 ──
    @property
    def logs_dir(self) -> Path:
        return Path(self.project_root) / "logs"

    def ensure_dirs(self) -> None:
        """创建所有必要目录"""
        dirs = [
            self.input_dir, self.extracted_dir, self.images_dir,
            self.extracted_dir / "tables",
            self.build_dir, self.assets_dir, self.generated_dir,
            self.library_dir, self.render_dir, self.segments_dir,
            self.audio_dir, self.cache_dir, self.plans_cache,
            self.search_cache, self.logs_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 全局运行参数
# ─────────────────────────────────────────────
@dataclass
class BuildParams:
    # Precedence: CLI > GUI/Session Config > Manifest Defaults
    # 字幕分段
    min_segment_duration: float = 1.0   # 秒，低于此值自动合并
    max_segment_duration: float = 8.0   # 秒，超过此值建议拆分
    target_segment_duration: float = 4.5

    # 增量更新阈值
    time_change_threshold: float = 0.2  # 秒，时间偏移超过此值视为 changed

    # 渲染
    max_retries: int = 3
    parallel_segments: int = 4          # 并行渲染段数
    use_gpu: bool = False

    # TTS（可选）
    tts_engine: str = "openai"          # openai / edge_tts / local
    tts_voice: str = "alloy"
    tts_speed: float = 1.0

    # 素材生成
    ai_image_model: str = "dall-e-3"
    ai_image_size: str = "1024x1792"    # 9:16

    # 视频
    aspect_ratio: str = "9:16"
    resolution_w: int = 1080
    resolution_h: int = 1920
    fps: int = 30
    video_bitrate: str = "4M"
    audio_bitrate: str = "192k"

    # BGM
    bgm_enabled: bool = False
    bgm_volume: float = 0.15
    bgm_ducking: bool = True

    # LLM
    llm_model: str = "gpt-4.1-mini"
    llm_temperature: float = 0.3


# 默认全局实例
DEFAULT_PARAMS = BuildParams()
