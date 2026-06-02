"""
核心数据模型：定义 Segment、Manifest、VisualPlan 等数据结构

Segment 唯一标识策略（v2）：
  content_key = hash(normalize(text))          # 只与文本内容相关
  segment_key = content_key + "#" + occurrence_index  # 处理重复句子
  occurrence_index: 在整条字幕中相同 content_key 出现的第几次（1,2,3…）

不使用 start_time 作为主键，避免音频重对齐时全量失效。
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
import json
import hashlib
import re
import unicodedata

from .generation_policy import (
    AI_CLIP_CAP_DEFAULT,
    TARGET_DURATION_MINUTES_ALLOWED,
    TARGET_DURATION_MINUTES_DEFAULT,
)


PIXELLE_WORKFLOWS = {"digital_human", "i2v", "action_transfer"}

# Allowed continuity policy modes
# - "frame_chain": Use end-frame from previous segment as reference (requires vendor support)
# - "seed_lock": Use consistent seed across segments (deterministic fallback)
# - "style_anchor": Use style bible anchors only (no temporal continuity)
# - "off": Disable continuity enforcement
CONTINUITY_POLICIES = {"frame_chain", "seed_lock", "style_anchor", "off"}

# Allowed material mode values
# - "auto": Default behavior - use PDF assets, then external sources, then AI generation
# - "ai_preferred": Prefer AI-generated materials, fall back to PDF/external sources if needed
# - "ai_only": Use only AI-generated materials, fail if AI generation unavailable
MATERIAL_MODES = {"auto", "ai_preferred", "ai_only"}

class ContinuityPolicyError(ValueError):
    """Raised when an invalid continuity policy mode is provided."""
    def __init__(self, value: str, allowed: set):
        self.value = value
        self.allowed = allowed
        super().__init__(
            f"Invalid continuity policy '{value}'. "
            f"Allowed values: {', '.join(sorted(allowed))}"
        )


class MaterialModeError(ValueError):
    """Raised when an invalid material mode is provided."""
    def __init__(self, value: str, allowed: set):
        self.value = value
        self.allowed = allowed
        super().__init__(
            f"Invalid material mode '{value}'. "
            f"Allowed values: {', '.join(sorted(allowed))}"
        )


class DurationCapPolicyError(ValueError):
    """Raised when an invalid target_duration_minutes or ai_clip_cap is provided."""
    def __init__(self, field: str, value: Any, allowed: Any):
        self.field = field
        self.value = value
        self.allowed = allowed
        if isinstance(allowed, set):
            allowed_str = ', '.join(str(v) for v in sorted(allowed))
        else:
            allowed_str = str(allowed)
        super().__init__(
            f"Invalid {field} '{value}'. "
            f"Allowed values: {allowed_str}"
        )


class WorkflowPolicyError(ValueError):
    """Raised when an invalid Pixelle workflow name is provided."""
    def __init__(self, value: str, allowed: set, *, field: str = "pixelle_default_workflow"):
        self.value = value
        self.allowed = allowed
        self.field = field
        super().__init__(
            f"Invalid workflow '{value}' in {field}. "
            f"Allowed values: {', '.join(sorted(allowed))}"
        )


# ─────────────────────────────────────────────
# 字幕条目
# ─────────────────────────────────────────────
@dataclass
class SRTEntry:
    index: int
    start: float        # 秒
    end: float          # 秒
    text: str

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    def to_srt_block(self) -> str:
        """输出标准 SRT 格式块"""
        def fmt(t: float) -> str:
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            ms = int(round((t - int(t)) * 1000))
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        return f"{self.index}\n{fmt(self.start)} --> {fmt(self.end)}\n{self.text}\n"


# ─────────────────────────────────────────────
# 音频引用
# ─────────────────────────────────────────────
@dataclass
class AudioRef:
    type: Literal["full", "segment"] = "full"
    path: str = ""
    trim_start: float = 0.0
    trim_end: float = 0.0


# ─────────────────────────────────────────────
# 镜头运动参数
# ─────────────────────────────────────────────
@dataclass
class MotionConfig:
    preset: str = "soft_kenburns"   # soft_kenburns / push_in / pan_left / static
    speed: float = 0.8


# ─────────────────────────────────────────────
# 叠加元素
# ─────────────────────────────────────────────
@dataclass
class OverlayItem:
    kind: str = "highlight"         # highlight / arrow / text / mask
    target: str = "center"
    strength: float = 0.6
    extra: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────
# 视觉计划
# ─────────────────────────────────────────────
@dataclass
class VisualPlan:
    type: Literal[
        "pdf_chart", "ui_mock", "broll", "ai_image",
        "ai_video_short", "kinetic_text", "template",
        "pixelle_digital_human", "pixelle_i2v", "pixelle_action_transfer"
    ] = "template"
    keywords: List[str] = field(default_factory=list)
    prompt: str = ""
    use_pdf_assets: List[Dict[str, Any]] = field(default_factory=list)
    motion: MotionConfig = field(default_factory=MotionConfig)
    overlay: List[OverlayItem] = field(default_factory=list)
    asset_path: Optional[str] = None    # 已解析的素材路径
    pixelle_workflow: Optional[Literal["digital_human", "i2v", "action_transfer"]] = None  # Pixelle workflow tag

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "keywords": self.keywords,
            "prompt": self.prompt,
            "use_pdf_assets": self.use_pdf_assets,
            "motion": asdict(self.motion),
            "overlay": [asdict(o) for o in self.overlay],
            "asset_path": self.asset_path,
            "pixelle_workflow": self.pixelle_workflow,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VisualPlan":
        motion_data = d.get("motion", {})
        if isinstance(motion_data, dict):
            motion = MotionConfig(**motion_data)
        else:
            motion = MotionConfig()
        overlay = [
            OverlayItem(**o) for o in d.get("overlay", [])
            if isinstance(o, dict)
        ]
        
        workflow = d.get("pixelle_workflow")
        if workflow is not None and workflow not in PIXELLE_WORKFLOWS:
            raise ValueError(
                f"Invalid pixelle_workflow '{workflow}'. "
                f"Must be one of: {', '.join(sorted(PIXELLE_WORKFLOWS))}"
            )
        
        return cls(
            type=d.get("type", "template"),
            keywords=d.get("keywords", []),
            prompt=d.get("prompt", ""),
            use_pdf_assets=d.get("use_pdf_assets", []),
            motion=motion,
            overlay=overlay,
            asset_path=d.get("asset_path"),
            pixelle_workflow=workflow,
        )

    def compute_plan_hash(self, global_style_asset_fields: str = "") -> str:
        """
        计算 plan_hash：用于判断素材是否需要重新生成/检索。
        plan_hash = hash(visual_plan 关键字段 + global_style 素材相关字段)
        """
        key_fields = {
            "type": self.type,
            "keywords": sorted(self.keywords),
            "prompt": self.prompt,
            "use_pdf_assets": self.use_pdf_assets,
            "pixelle_workflow": self.pixelle_workflow,
        }
        raw = json.dumps(key_fields, sort_keys=True) + "|" + global_style_asset_fields
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────
# 素材引用（v2：含 asset_hash）
# ─────────────────────────────────────────────
@dataclass
class AssetRef:
    kind: str = "template"
    path: str = ""
    asset_hash: str = ""
    fallback_reason_code: Optional[str] = None
    fallback_error_category: Optional[str] = None
    fallback_diagnostic: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "asset_hash": self.asset_hash,
            "fallback_reason_code": self.fallback_reason_code,
            "fallback_error_category": self.fallback_error_category,
            "fallback_diagnostic": self.fallback_diagnostic,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AssetRef":
        return cls(
            kind=d.get("kind", "template"),
            path=d.get("path", ""),
            asset_hash=d.get("asset_hash", ""),
            fallback_reason_code=d.get("fallback_reason_code"),
            fallback_error_category=d.get("fallback_error_category"),
            fallback_diagnostic=d.get("fallback_diagnostic"),
        )


# ─────────────────────────────────────────────
# 渲染引用（v2：含 render_hash）
# ─────────────────────────────────────────────
@dataclass
class RenderRef:
    segment_video_path: Optional[str] = None
    render_hash: Optional[str] = None
    status: Literal["pending", "ok", "failed", "skipped"] = "pending"
    error: Optional[str] = None
    retries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_video_path": self.segment_video_path,
            "render_hash": self.render_hash,
            "status": self.status,
            "error": self.error,
            "retries": self.retries,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RenderRef":
        return cls(
            segment_video_path=d.get("segment_video_path"),
            render_hash=d.get("render_hash"),
            status=d.get("status", "pending"),
            error=d.get("error"),
            retries=d.get("retries", 0),
        )


# ─────────────────────────────────────────────
# Segment（核心单元，v2）
# ─────────────────────────────────────────────
@dataclass
class Segment:
    segment_key: str
    content_key: str

    index: int
    start: float
    end: float
    duration: float
    text: str

    audio_ref: AudioRef = field(default_factory=AudioRef)
    visual_plan: Optional[VisualPlan] = None
    plan_hash: Optional[str] = None
    asset_refs: List[AssetRef] = field(default_factory=list)
    render_ref: RenderRef = field(default_factory=RenderRef)
    
    prev_last_frame_path: Optional[str] = None
    continuity_diagnostic: Optional[Dict[str, Any]] = None

    # ─────────────────────────────────────────
    # 静态工具方法
    # ─────────────────────────────────────────
    @staticmethod
    def normalize_text(text: str) -> str:
        """规范化文本（用于 content_key 计算）"""
        text = text.strip()
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\s+", " ", text)
        # 统一标点
        for src, dst in [("，", ","), ("。", "."), ("！", "!"), ("？", "?"),
                         ("：", ":"), ("；", ";"), ("、", ",")]:
            text = text.replace(src, dst)
        return text.lower()

    @staticmethod
    def compute_content_key(text: str) -> str:
        """计算 content_key = hash(normalize(text))"""
        norm = Segment.normalize_text(text)
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def compute_segment_key(content_key: str, occurrence_index: int) -> str:
        """计算 segment_key = content_key + '#' + occurrence_index"""
        return f"{content_key}#{occurrence_index}"

    def compute_render_hash(self, global_style_render_fields: str = "") -> str:
        """
        计算 render_hash：用于判断片段是否需要重渲。
        render_hash = hash(plan_hash + start + end + subtitle_style + motion + asset_hashes)
        """
        asset_hashes = "|".join(sorted(a.asset_hash for a in self.asset_refs))
        motion_str = ""
        if self.visual_plan:
            motion_str = f"{self.visual_plan.motion.preset}:{self.visual_plan.motion.speed}"
        raw = (
            f"{self.plan_hash or ''}|"
            f"{self.start:.3f}|{self.end:.3f}|"
            f"{motion_str}|"
            f"{asset_hashes}|"
            f"{global_style_render_fields}"
        )
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_key": self.segment_key,
            "content_key": self.content_key,
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "text": self.text,
            "audio_ref": asdict(self.audio_ref),
            "visual_plan": self.visual_plan.to_dict() if self.visual_plan else None,
            "plan_hash": self.plan_hash,
            "asset_refs": [a.to_dict() for a in self.asset_refs],
            "render_ref": self.render_ref.to_dict(),
            "prev_last_frame_path": self.prev_last_frame_path,
            "continuity_diagnostic": self.continuity_diagnostic,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Segment":
        audio_ref = AudioRef(**d.get("audio_ref", {}))
        vp_data = d.get("visual_plan")
        visual_plan = VisualPlan.from_dict(vp_data) if vp_data else None
        asset_refs = [AssetRef.from_dict(a) for a in d.get("asset_refs", [])]
        render_ref = RenderRef.from_dict(d.get("render_ref", {}))
        return cls(
            segment_key=d["segment_key"],
            content_key=d["content_key"],
            index=d["index"],
            start=d["start"],
            end=d["end"],
            duration=d["duration"],
            text=d["text"],
            audio_ref=audio_ref,
            visual_plan=visual_plan,
            plan_hash=d.get("plan_hash"),
            asset_refs=asset_refs,
            render_ref=render_ref,
            prev_last_frame_path=d.get("prev_last_frame_path"),
            continuity_diagnostic=d.get("continuity_diagnostic"),
        )


# ─────────────────────────────────────────────
# 风格圣经 (Style Bible)
# ─────────────────────────────────────────────
@dataclass
class StyleBible:
    tone: str = "cinematic"
    palette: str = "muted"
    camera_grammar: str = "steady"
    character_anchors: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tone": self.tone,
            "palette": self.palette,
            "camera_grammar": self.camera_grammar,
            "character_anchors": self.character_anchors,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StyleBible":
        anchors = d.get("character_anchors", {})
        if not isinstance(anchors, dict):
            raise TypeError(
                f"character_anchors must be dict, got {type(anchors).__name__}"
            )
        return cls(
            tone=d.get("tone", "cinematic"),
            palette=d.get("palette", "muted"),
            camera_grammar=d.get("camera_grammar", "steady"),
            character_anchors=anchors,
        )


# ─────────────────────────────────────────────
# 全局样式
# ─────────────────────────────────────────────
@dataclass
class GlobalStyle:
    subtitle_style: str = "clean"
    motion_preset: str = "soft"
    aspect_ratio: str = "9:16"
    resolution: str = "1080x1920"
    fps: int = 30
    font_name: str = "NotoSansCJK"
    font_size: int = 48
    font_color: str = "white"
    subtitle_bg: bool = True
    subtitle_bg_color: str = "black@0.5"
    style_version: str = "v1"
    # Precedence: CLI > GUI/Session Config > Manifest Defaults
    # Default: True (Backward compatible, effects ON by default)
    enable_subtitle_effects: bool = True
    style_bible: Optional[StyleBible] = None

    @property
    def resolution_w(self) -> int:
        return int(self.resolution.split("x")[0])

    @property
    def resolution_h(self) -> int:
        return int(self.resolution.split("x")[1])

    def asset_related_fields(self) -> str:
        """返回影响素材生成的字段（用于 plan_hash 计算）"""
        return f"{self.aspect_ratio}|{self.resolution}|{self.style_version}"

    def render_related_fields(self) -> str:
        """返回影响渲染的字段（用于 render_hash 计算）"""
        return (
            f"{self.subtitle_style}|{self.motion_preset}|"
            f"{self.fps}|{self.font_size}|{self.font_color}|"
            f"{self.subtitle_bg}|{self.subtitle_bg_color}|"
            f"{self.enable_subtitle_effects}|{self.style_version}"
        )


# ─────────────────────────────────────────────
# Budget Diagnostics (duration budget selection metadata)
# ─────────────────────────────────────────────
@dataclass
class BudgetDiagnostics:
    """Deterministic metadata for Step2 duration budget selection.
    
    All fields are computed from input only (no timestamps/random components).
    """
    requested_minutes: int                   # User-requested target_duration_minutes
    target_seconds: float                    # Converted target in seconds
    total_available_seconds: float           # Total duration of compacted segments before budget cut
    effective_selected_seconds: float        # Cumulative duration of selected segments
    selected_count: int                      # Number of segments selected
    dropped_count: int                       # Number of segments dropped due to budget
    budget_exhausted: bool                   # True if budget cutoff was applied

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested_minutes": self.requested_minutes,
            "target_seconds": self.target_seconds,
            "total_available_seconds": self.total_available_seconds,
            "effective_selected_seconds": self.effective_selected_seconds,
            "selected_count": self.selected_count,
            "dropped_count": self.dropped_count,
            "budget_exhausted": self.budget_exhausted,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BudgetDiagnostics":
        return cls(
            requested_minutes=d.get("requested_minutes", 1),
            target_seconds=d.get("target_seconds", 60.0),
            total_available_seconds=d.get("total_available_seconds", 0.0),
            effective_selected_seconds=d.get("effective_selected_seconds", 0.0),
            selected_count=d.get("selected_count", 0),
            dropped_count=d.get("dropped_count", 0),
            budget_exhausted=d.get("budget_exhausted", False),
        )


# ─────────────────────────────────────────────
# Manifest（工程总清单，v2）
# ─────────────────────────────────────────────
@dataclass
class Manifest:
    project_id: str
    build_id: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    global_style: GlobalStyle = field(default_factory=GlobalStyle)
    segments: List[Segment] = field(default_factory=list)
    pixelle_default_workflow: Optional[str] = None
    pixelle_segment_overrides: Dict[str, Optional[str]] = field(default_factory=dict)
    audio_path: str = ""
    final_video: Optional[str] = None
    build_status: Literal["pending", "building", "done", "failed"] = "pending"
    
    style_id: Optional[str] = None
    continuity_seed: Optional[int] = None
    vendor_preference: Optional[str] = None
    continuity_policy: Optional[Literal["frame_chain", "seed_lock", "style_anchor", "off"]] = None
    material_mode: Literal["auto", "ai_preferred", "ai_only"] = "auto"
    target_duration_minutes: int = TARGET_DURATION_MINUTES_DEFAULT
    ai_clip_cap: int = AI_CLIP_CAP_DEFAULT
    budget_diagnostics: Optional[BudgetDiagnostics] = None

    def __post_init__(self):
        if self.material_mode not in MATERIAL_MODES:
            raise MaterialModeError(self.material_mode, MATERIAL_MODES)
        if self.target_duration_minutes not in TARGET_DURATION_MINUTES_ALLOWED:
            raise DurationCapPolicyError(
                "target_duration_minutes",
                self.target_duration_minutes,
                TARGET_DURATION_MINUTES_ALLOWED,
            )
        if not isinstance(self.ai_clip_cap, int) or self.ai_clip_cap < 1:
            raise DurationCapPolicyError(
                "ai_clip_cap",
                self.ai_clip_cap,
                "positive integer >= 1",
            )

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "project_id": self.project_id,
            "build_id": self.build_id,
            "global_style": asdict(self.global_style),
            "pixelle_default_workflow": self.pixelle_default_workflow,
            "pixelle_segment_overrides": self.pixelle_segment_overrides,
            "audio_path": self.audio_path,
            "final_video": self.final_video,
            "build_status": self.build_status,
            "style_id": self.style_id,
            "continuity_seed": self.continuity_seed,
            "vendor_preference": self.vendor_preference,
            "continuity_policy": self.continuity_policy,
            "material_mode": self.material_mode,
            "target_duration_minutes": self.target_duration_minutes,
            "ai_clip_cap": self.ai_clip_cap,
            "segments": [s.to_dict() for s in self.segments],
        }
        if self.budget_diagnostics is not None:
            result["budget_diagnostics"] = self.budget_diagnostics.to_dict()
        return result

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Manifest":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.load_from_dict(d)

    @classmethod
    def load_from_dict(cls, d: Dict[str, Any]) -> "Manifest":
        build_id = d.get("build_id") or d.get("version", datetime.utcnow().isoformat() + "Z")
        gs_data = d.get("global_style", {})
        valid_fields = {f.name for f in GlobalStyle.__dataclass_fields__.values()}
        gs_data_clean = {k: v for k, v in gs_data.items() if k in valid_fields}
        if "style_bible" in gs_data_clean and isinstance(gs_data_clean["style_bible"], dict):
            gs_data_clean["style_bible"] = StyleBible.from_dict(gs_data_clean["style_bible"])
        gs = GlobalStyle(**gs_data_clean)
        segments = [Segment.from_dict(s) for s in d.get("segments", [])]

        pixelle_default = d.get("pixelle_default_workflow")
        if pixelle_default is not None and pixelle_default not in PIXELLE_WORKFLOWS:
            raise WorkflowPolicyError(pixelle_default, PIXELLE_WORKFLOWS, field="pixelle_default_workflow")

        raw_overrides = d.get("pixelle_segment_overrides", {})
        pixelle_overrides: Dict[str, Optional[str]] = {}
        if isinstance(raw_overrides, dict):
            for key, value in raw_overrides.items():
                if value is None or value in PIXELLE_WORKFLOWS:
                    pixelle_overrides[key] = value
                else:
                    raise WorkflowPolicyError(
                        value, PIXELLE_WORKFLOWS, field=f"pixelle_segment_overrides[{key}]"
                    )

        continuity_policy = d.get("continuity_policy")
        if continuity_policy is not None and continuity_policy not in CONTINUITY_POLICIES:
            raise ContinuityPolicyError(continuity_policy, CONTINUITY_POLICIES)

        continuity_seed = d.get("continuity_seed")
        if continuity_seed is not None and not isinstance(continuity_seed, int):
            continuity_seed = None

        material_mode = d.get("material_mode", "auto")
        if material_mode not in MATERIAL_MODES:
            raise MaterialModeError(material_mode, MATERIAL_MODES)

        target_duration_minutes = d.get("target_duration_minutes", TARGET_DURATION_MINUTES_DEFAULT)
        if target_duration_minutes not in TARGET_DURATION_MINUTES_ALLOWED:
            raise DurationCapPolicyError(
                "target_duration_minutes",
                target_duration_minutes,
                TARGET_DURATION_MINUTES_ALLOWED,
            )

        ai_clip_cap = d.get("ai_clip_cap", AI_CLIP_CAP_DEFAULT)
        if not isinstance(ai_clip_cap, int) or ai_clip_cap < 1:
            raise DurationCapPolicyError(
                "ai_clip_cap",
                ai_clip_cap,
                "positive integer >= 1",
            )

        budget_diagnostics_data = d.get("budget_diagnostics")
        budget_diagnostics: Optional[BudgetDiagnostics] = None
        if isinstance(budget_diagnostics_data, dict):
            budget_diagnostics = BudgetDiagnostics.from_dict(budget_diagnostics_data)

        return cls(
            project_id=d["project_id"],
            build_id=build_id,
            global_style=gs,
            segments=segments,
            pixelle_default_workflow=pixelle_default,
            pixelle_segment_overrides=pixelle_overrides,
            audio_path=d.get("audio_path", ""),
            final_video=d.get("final_video"),
            build_status=d.get("build_status", "pending"),
            style_id=d.get("style_id"),
            continuity_seed=continuity_seed,
            vendor_preference=d.get("vendor_preference"),
            continuity_policy=continuity_policy,
            material_mode=material_mode,
            target_duration_minutes=target_duration_minutes,
            ai_clip_cap=ai_clip_cap,
            budget_diagnostics=budget_diagnostics,
        )


# ─────────────────────────────────────────────
# 变更类型枚举
# ─────────────────────────────────────────────
class ChangeType:
    TEXT = "TEXT"       # 文本内容变化 → 必须重做 visual_plan + 素材 + 渲染
    TIMING = "TIMING"   # 时间轴变化（文本不变）→ 可复用素材，只重渲
    STYLE = "STYLE"     # 全局样式变化 → 重渲（可选是否重做 plan）
    ADDED = "ADDED"     # 新增段落 → 全部重做
    REMOVED = "REMOVED" # 删除段落 → 清理产物


# ─────────────────────────────────────────────
# Diff 结果（v2：分级变更）
# ─────────────────────────────────────────────
@dataclass
class SegmentChange:
    segment_key: str
    change_type: str    # ChangeType 中的值
    old_segment: Optional[Segment] = None
    new_segment: Optional[Segment] = None


@dataclass
class DiffResult:
    added: List[str] = field(default_factory=list)       # segment_keys
    removed: List[str] = field(default_factory=list)
    changed_text: List[str] = field(default_factory=list)    # changed(TEXT)
    changed_timing: List[str] = field(default_factory=list)  # changed(TIMING)
    changed_style: List[str] = field(default_factory=list)   # changed(STYLE)
    unchanged: List[str] = field(default_factory=list)
    changes: List[SegmentChange] = field(default_factory=list)  # 详细变更列表

    @property
    def all_changed(self) -> List[str]:
        """所有需要重建的 segment_keys"""
        return self.added + self.changed_text + self.changed_timing + self.changed_style

    @property
    def need_new_visual_plan(self) -> List[str]:
        """需要重新生成 visual_plan 的 segment_keys（TEXT + ADDED）"""
        return self.added + self.changed_text

    @property
    def need_new_assets(self) -> List[str]:
        """需要重新生成/检索素材的 segment_keys（plan_hash 变化时）"""
        return self.added + self.changed_text
        # TIMING 和 STYLE 不重做素材（plan_hash 不变）

    @property
    def need_rerender(self) -> List[str]:
        """需要重渲片段的 segment_keys（全部变更类型）"""
        return self.all_changed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "added": self.added,
            "removed": self.removed,
            "changed": {
                "TEXT": self.changed_text,
                "TIMING": self.changed_timing,
                "STYLE": self.changed_style,
            },
            "unchanged": self.unchanged,
            "summary": {
                "total_added": len(self.added),
                "total_removed": len(self.removed),
                "total_changed_text": len(self.changed_text),
                "total_changed_timing": len(self.changed_timing),
                "total_changed_style": len(self.changed_style),
                "total_unchanged": len(self.unchanged),
                "total_need_rerender": len(self.need_rerender),
                "total_reuse_assets": len(self.changed_timing) + len(self.changed_style),
            },
        }

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
