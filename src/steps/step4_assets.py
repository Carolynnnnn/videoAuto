"""
Step 4：素材执行（检索/生成/复用，v3）

素材选择优先级（从高到低）：
  ① PDF 图表    — visual_plan.type == "pdf_chart" 且 PDF 中有对应图片
  ② Pexels 视频 — 搜索匹配关键词的竖屏/横屏视频（时长匹配，enable_pexels_video=True）
  ③ Pexels 图片 — 搜索匹配关键词的图片（视频无结果时 fallback，enable_pexels_photo=True）
  ④ AI 图片生成 — DALL-E 3 生成（enable_ai_image=True 时启用）
  ⑤ 通用模板    — PIL/FFmpeg 生成纯色背景（最终兜底，始终可用）

缓存策略（v2 不变）：
  - 图片缓存路径：assets/generated/{content_key}_{plan_hash}.png
  - 视频缓存路径：assets/pexels_cache/videos/pexels_{id}_{quality}.mp4
  - 命中条件：同 content_key 且 plan_hash 相同 → 素材复用
  - asset_refs 含 asset_hash（用于 render_hash 计算）

Pexels 配置：
  - 通过 PEXELS_API_KEY 环境变量或 run_step4() 参数传入
  - 搜索结果缓存：assets/pexels_cache/search_cache.json（避免重复 API 调用）
  - enable_pexels_video: 是否优先使用 Pexels 视频（默认 True）
  - enable_pexels_photo: 是否在视频无结果时用 Pexels 图片（默认 True）
"""
from __future__ import annotations
import os
import json
import hashlib
import re
import shutil
import subprocess
import uuid
import threading
from pathlib import Path
from typing import List, Optional, Tuple, Literal

from openai import OpenAI  # type: ignore[reportMissingImports]
from pixelle_snapshot.adapters.contracts import ErrorCategory, FailureDiagnostic, normalize_error_category

from src.core.models import Manifest, Segment, VisualPlan, AssetRef
from src.core.generation_policy import index_tie_break_key
from src.steps.continuity_policy import ContinuityDirective, evaluate_continuity_policy
from src.steps.pixelle_quota_accounting import (
    QuotaExceededError,
    create_quota_diagnostic,
    get_quota_enforcement,
)
from src.steps.pixelle_reliability_controls import (
    CircuitOpenError,
    PixelleReliabilityControls,
    RateLimitExceededError,
)
from src.steps.pixelle_rollout_flags import RolloutConfig, check_rollout_eligibility
from src.utils.logger import get_logger

logger = get_logger("step4_assets")
_pixelle_rollout_config = RolloutConfig.from_env()
_pixelle_reliability_controls = PixelleReliabilityControls.from_env()
_PIXELLE_CONCURRENCY_LIMIT = max(1, int(os.environ.get("PIXELLE_CONCURRENCY_LIMIT", "4")))
_pixelle_semaphore = threading.Semaphore(_PIXELLE_CONCURRENCY_LIMIT)

AI_GENERATED_ROUTES = frozenset({"ai_image", "pixelle_video"})
AI_ONLY_ALLOWED_ROUTES = frozenset({"ai_image", "pixelle_video"})
NON_AI_ROUTES = frozenset({"pdf_chart", "pexels_video", "pexels_photo", "template"})

# Canonical route priority order for 'auto' mode (compatibility baseline).
# This documents the effective ordering when material_mode='auto':
#   0. Cache hit (always checked first)
#   1. PDF chart (if visual_plan.type == "pdf_chart")
#   2. Pexels video (if enabled and keywords present, not kinetic_text)
#   3. Pexels photo (if enabled and keywords present)
#   3.5. Pixelle video (if effective_workflow set)
#   4. AI image (if enabled and prompt present)
#   5. Template fallback (always available)
AUTO_MODE_ROUTE_PRIORITY = (
    "cached",
    "pdf_chart",
    "pexels_video",
    "pexels_photo",
    "pixelle_video",
    "ai_image",
    "template",
)


class _PixelleConstraintError(Exception):
    def __init__(self, reason_code: str, category: ErrorCategory, message: str):
        super().__init__(message)
        self.reason_code = reason_code
        self.category = category


_MINIMAX_DURATION_BY_MODEL_RESOLUTION = {
    "MiniMax-Hailuo-2.3": {"768P": 10.0, "1080P": 6.0},
    "MiniMax-Hailuo-2.3-Fast": {"768P": 10.0, "1080P": 6.0},
    "MiniMax-Hailuo-02": {"512P": 10.0, "768P": 10.0, "1080P": 6.0},
    "video-01": {"720P": 6.0},
    "T2V-01": {"720P": 6.0},
    "I2V-01": {"720P": 6.0},
}


def _load_minimax_contract_max_duration() -> float:
    contract_path = Path(__file__).resolve().parents[2] / "pixelle_snapshot" / "vendors" / "fixtures" / "minimax_media_contract.json"
    try:
        with contract_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        raw = payload.get("extensions", {}).get("limits", {}).get("max_duration")
        if isinstance(raw, (int, float)) and float(raw) > 0:
            return float(raw)
    except Exception:
        pass
    return 10.0


_MINIMAX_VENDOR_MAX_DURATION = _load_minimax_contract_max_duration()


def _resolution_profile_from_tuple(resolution: Tuple[int, int]) -> str:
    short_edge = float(min(resolution))
    if short_edge <= 512:
        return "512P"
    if short_edge <= 720:
        return "720P"
    if short_edge <= 768:
        return "768P"
    return "1080P"


def _normalize_minimax_duration_constraints(
    *,
    model: str,
    requested_duration: float,
    resolution: Tuple[int, int],
    segment_index: int,
) -> Tuple[Optional[float], dict]:
    resolution_profile = _resolution_profile_from_tuple(resolution)
    model_matrix = _MINIMAX_DURATION_BY_MODEL_RESOLUTION.get(model)
    if model_matrix is None:
        raise _PixelleConstraintError(
            reason_code="PIXELLE_MINIMAX_UNSUPPORTED_MODEL",
            category=ErrorCategory.UNSUPPORTED,
            message=f"Unsupported Minimax model '{model}' for Step4 routing constraints",
        )

    if resolution_profile not in model_matrix:
        raise _PixelleConstraintError(
            reason_code="PIXELLE_MINIMAX_UNSUPPORTED_RESOLUTION",
            category=ErrorCategory.VALIDATION,
            message=(
                f"Unsupported resolution profile '{resolution_profile}' for model '{model}' "
                f"(raw={resolution[0]}x{resolution[1]})"
            ),
        )

    combo_max_duration = min(model_matrix[resolution_profile], _MINIMAX_VENDOR_MAX_DURATION)
    if requested_duration <= combo_max_duration:
        return None, {
            "model": model,
            "resolution_profile": resolution_profile,
            "requested_duration": requested_duration,
            "effective_duration": requested_duration,
            "normalized": False,
            "combo_max_duration": combo_max_duration,
            "vendor_max_duration": _MINIMAX_VENDOR_MAX_DURATION,
        }

    if requested_duration > _MINIMAX_VENDOR_MAX_DURATION:
        logger.info(
            f"  [seg {segment_index}] Route diagnostic NORMALIZED: route=pixelle_video "
            f"reason=PIXELLE_MINIMAX_DURATION_NORMALIZED model={model} "
            f"resolution={resolution_profile} requested={requested_duration:.3f} "
            f"effective={combo_max_duration:.3f}"
        )
        return combo_max_duration, {
            "model": model,
            "resolution_profile": resolution_profile,
            "requested_duration": requested_duration,
            "effective_duration": combo_max_duration,
            "normalized": True,
            "combo_max_duration": combo_max_duration,
            "vendor_max_duration": _MINIMAX_VENDOR_MAX_DURATION,
            "reason_code": "PIXELLE_MINIMAX_DURATION_NORMALIZED",
        }

    raise _PixelleConstraintError(
        reason_code="PIXELLE_MINIMAX_UNSUPPORTED_DURATION_COMBO",
        category=ErrorCategory.VALIDATION,
        message=(
            f"Unsupported duration/model/resolution combination: model={model}, "
            f"resolution={resolution_profile}, requested_duration={requested_duration:.3f}, "
            f"max_supported_duration={combo_max_duration:.3f}"
        ),
    )


def is_route_allowed_by_mode_policy(route: str, material_mode: str) -> bool:
    """
    Determine if a route is allowed under the given material mode policy.

    Mode behaviors:
    - 'auto': All routes allowed in canonical priority order (baseline behavior).
    - 'ai_preferred': All routes allowed (AI routes prioritized by caller logic).
    - 'ai_only': Only AI-generated routes allowed (ai_image, pixelle_video). Non-AI routes
      (pdf_chart, pexels_video, pexels_photo, template) are blocked. Cache hits are only allowed
      when cache metadata indicates the cached asset kind is AI-generated.

    Args:
        route: The asset route being considered (e.g., 'pdf_chart', 'pexels_video').
        material_mode: The material mode policy ('auto', 'ai_preferred', 'ai_only').

    Returns:
        True if the route is allowed under the policy, False otherwise.
    """
    if material_mode == "auto":
        # Auto mode: preserve canonical route order, all routes allowed.
        # This branch is explicit to document the baseline behavior.
        return True
    if material_mode == "ai_only":
        return route not in NON_AI_ROUTES
    # ai_preferred and any other mode: all routes allowed
    return True


def _with_legacy_aliases(ref: AssetRef) -> AssetRef:
    setattr(ref, "reason_code", ref.fallback_reason_code)
    setattr(ref, "error_category", ref.fallback_error_category)
    return ref


SEMANTIC_HINTS = {
    "ai": ["artificial intelligence", "machine learning", "digital technology"],
    "人工智能": ["artificial intelligence", "machine learning", "digital technology"],
    "chatgpt": ["chatbot interface", "large language model"],
    "sora": ["ai video generation", "digital media"],
    "医疗": ["medical diagnosis", "hospital doctor"],
    "诊断": ["medical diagnosis", "healthcare technology"],
    "教育": ["classroom learning", "education technology"],
    "金融": ["financial market", "business analytics"],
    "数据": ["data analytics", "business chart"],
    "图表": ["data visualization", "statistics chart"],
    "机器人": ["robot automation", "smart factory"],
    "自动驾驶": ["self driving car", "autonomous vehicle"],
}

PROMPT_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "about", "vertical", "quality",
    "high", "clean", "modern", "visual", "scene", "style", "background", "cinematic",
}


def _normalize_keywords(raw_keywords: List[str], max_count: int = 6) -> List[str]:
    result: List[str] = []
    seen = set()
    for raw in raw_keywords:
        if not isinstance(raw, str):
            continue
        kw = re.sub(r"\s+", " ", raw.strip().lower()).replace("_", " ")
        if not kw or kw in seen:
            continue
        seen.add(kw)
        result.append(kw)
        if len(result) >= max_count:
            break
    return result


def _extract_prompt_terms(prompt: str, max_count: int = 3) -> List[str]:
    if not prompt:
        return []
    terms = []
    for token in re.findall(r"[a-zA-Z]{4,}", prompt.lower()):
        if token in PROMPT_STOPWORDS:
            continue
        terms.append(token)
        if len(terms) >= max_count:
            break
    return terms


def _expand_semantic_keywords(base_keywords: List[str], prompt: str, segment_text: str) -> List[str]:
    corpus = f"{' '.join(base_keywords)} {prompt} {segment_text}".lower()
    expanded = list(base_keywords)

    for trigger, mapped in SEMANTIC_HINTS.items():
        if trigger in corpus:
            expanded.extend(mapped)

    if re.search(r"\d+|万亿|增长|下降|%", corpus):
        expanded.extend(["data dashboard", "growth chart"])

    expanded.extend(_extract_prompt_terms(prompt))
    return _normalize_keywords(expanded)


def compute_segment_semantic_priority_score(segment: Segment) -> dict:
    vp = segment.visual_plan
    base_keywords = _normalize_keywords((vp.keywords if vp else []) or [])
    prompt = (vp.prompt if vp else "") or ""

    prompt_terms = _extract_prompt_terms(prompt, max_count=6)
    expanded_keywords = _expand_semantic_keywords(base_keywords, prompt, segment.text)

    expanded_only_count = max(len(expanded_keywords) - len(base_keywords), 0)
    text_tokens = re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", (segment.text or "").lower())
    unique_token_count = len(set(text_tokens))
    text_char_count = len((segment.text or "").strip())

    keyword_weight = len(base_keywords) * 30
    expanded_weight = expanded_only_count * 12
    prompt_weight = len(prompt_terms) * 10
    richness_weight = min(unique_token_count, 20) * 3 + min(text_char_count, 120) // 20

    score = float(keyword_weight + expanded_weight + prompt_weight + richness_weight)
    tie_break = index_tie_break_key(segment.index)

    return {
        "score": score,
        "score_components": {
            "visual_keywords": keyword_weight,
            "expanded_keywords": expanded_weight,
            "prompt_terms": prompt_weight,
            "text_richness": richness_weight,
        },
        "inputs": {
            "visual_keywords": base_keywords,
            "expanded_keywords": expanded_keywords,
            "prompt_terms": prompt_terms,
            "text_unique_tokens": unique_token_count,
            "text_char_count": text_char_count,
        },
        "tie_break": {
            "index": segment.index,
            "stable_index_key": tie_break,
            "rank_key": (-score, tie_break),
        },
    }


def build_top6_ai_allocation_map(
    segments: List[Segment],
    target_segment_keys: Optional[List[str]] = None,
    max_ai_segments: int = 6,
) -> dict[str, bool]:
    capped_n = max(0, min(max_ai_segments, 6))
    target_keys = set(target_segment_keys) if target_segment_keys else None

    allocation: dict[str, bool] = {seg.segment_key: False for seg in segments}
    ranked_candidates: List[Tuple[Tuple[float, int], str]] = []

    for seg in segments:
        if target_keys is not None and seg.segment_key not in target_keys:
            continue
        semantic_priority = compute_segment_semantic_priority_score(seg)
        tie_break = semantic_priority.get("tie_break", {})
        raw_rank_key = tie_break.get("rank_key")

        if (
            isinstance(raw_rank_key, tuple)
            and len(raw_rank_key) == 2
            and isinstance(raw_rank_key[0], (float, int))
            and isinstance(raw_rank_key[1], int)
        ):
            rank_key: Tuple[float, int] = (float(raw_rank_key[0]), int(raw_rank_key[1]))
        else:
            score = float(semantic_priority.get("score", 0.0))
            rank_key = (-score, index_tie_break_key(seg.index))

        ranked_candidates.append((rank_key, seg.segment_key))

    ranked_candidates.sort(key=lambda item: item[0])
    for _, segment_key in ranked_candidates[:capped_n]:
        allocation[segment_key] = True

    return allocation


def _effective_visual_type(vp_type: str, keywords: List[str]) -> str:
    if vp_type != "template":
        return vp_type

    joined = " ".join(keywords)
    if any(token in joined for token in ("chart", "analytics", "report", "statistics")):
        return "pdf_chart"
    if any(token in joined for token in ("medical", "education", "finance", "technology", "robot")):
        return "broll"
    return "template"


# ─────────────────────────────────────────────
# 缓存工具
# ─────────────────────────────────────────────


def _asset_cache_meta_path(asset_path: str) -> str:
    return asset_path + ".meta.json"


def _read_asset_cache_meta(asset_path: str) -> Optional[dict]:
    meta_path = _asset_cache_meta_path(asset_path)
    try:
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def _write_asset_cache_meta_if_in_generated_dir(
    *, asset_path: Optional[str], generated_dir: str, kind: str, material_mode: str
) -> None:
    if not asset_path:
        return
    try:
        gen_abs = os.path.abspath(generated_dir)
        asset_abs = os.path.abspath(asset_path)
        if os.path.commonpath([gen_abs, asset_abs]) != gen_abs:
            return
        meta_path = _asset_cache_meta_path(asset_path)
        payload = {
            "kind": kind,
            "material_mode": material_mode,
            "is_ai_generated": kind in AI_GENERATED_ROUTES,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        # Cache metadata is best-effort; never break Step4 on write issues.
        return
def _compute_effective_cache_hash(plan_hash: str, workflow: Optional[str]) -> str:
    """
    Compute effective cache hash incorporating Pixelle workflow.
    When workflow is None, returns plan_hash unchanged.
    When workflow is set, appends workflow suffix to make cache key unique.
    """
    if workflow is None:
        return plan_hash
    combined = f"{plan_hash}_{workflow}"
    return hashlib.md5(combined.encode("utf-8")).hexdigest()[:12]


def _resolve_effective_pixelle_workflow(
    segment: Segment,
    manifest_default: Optional[str],
    manifest_overrides: Optional[dict],
) -> Optional[str]:
    """
    Resolve effective Pixelle workflow for a segment.
    Priority: segment.visual_plan.pixelle_workflow > manifest_overrides > manifest_default > type inference
    """
    # Priority 1: Explicit workflow in visual_plan
    if segment.visual_plan and segment.visual_plan.pixelle_workflow:
        return segment.visual_plan.pixelle_workflow
    
    # Priority 2: Segment-level override in manifest
    if manifest_overrides and segment.segment_key in manifest_overrides:
        return manifest_overrides[segment.segment_key]
    
    # Priority 3: Manifest default workflow
    if manifest_default:
        return manifest_default
    
    # Priority 4: Infer from visual_plan type
    if segment.visual_plan:
        vp_type = segment.visual_plan.type
        if vp_type == "pixelle_digital_human":
            return "digital_human"
        elif vp_type == "pixelle_i2v":
            return "i2v"
        elif vp_type == "pixelle_action_transfer":
            return "action_transfer"
    
    return None


def _resolve_effective_pixelle_workflow_with_source(
    segment: Segment,
    manifest_default: Optional[str],
    manifest_overrides: Optional[dict],
) -> Tuple[Optional[str], str]:
    """
    Resolve effective Pixelle workflow for a segment AND return its source.
    Returns: (workflow, source) where source is one of:
      - "visual_plan" (segment.visual_plan.pixelle_workflow)
      - "segment_override" (manifest_overrides[segment_key])
      - "manifest_default" (pixelle_default_workflow)
      - "type_inference" (inferred from visual_plan.type)
      - "none" (no workflow resolved)
    """
    # Priority 1: Explicit workflow in visual_plan
    if segment.visual_plan and segment.visual_plan.pixelle_workflow:
        return segment.visual_plan.pixelle_workflow, "visual_plan"
    
    # Priority 2: Segment-level override in manifest
    if manifest_overrides and segment.segment_key in manifest_overrides:
        return manifest_overrides[segment.segment_key], "segment_override"
    
    # Priority 3: Manifest default workflow
    if manifest_default:
        return manifest_default, "manifest_default"
    
    # Priority 4: Infer from visual_plan type
    if segment.visual_plan:
        vp_type = segment.visual_plan.type
        if vp_type == "pixelle_digital_human":
            return "digital_human", "type_inference"
        elif vp_type == "pixelle_i2v":
            return "i2v", "type_inference"
        elif vp_type == "pixelle_action_transfer":
            return "action_transfer", "type_inference"
    
    return None, "none"


def _resolve_pixelle_backend_mode() -> str:
    try:
        from pixelle_snapshot.config_loader import ProviderConfigError, load_provider_config
    except Exception:
        fallback = os.environ.get("PIXELLE_BACKEND_MODE", "direct")
        return fallback if fallback in ("legacy", "direct") else "direct"

    try:
        return load_provider_config().backend_mode
    except ProviderConfigError:
        fallback = os.environ.get("PIXELLE_BACKEND_MODE", "direct")
        return fallback if fallback in ("legacy", "direct") else "direct"


def _resolve_routed_pixelle_capability(effective_capability: str) -> str:
    backend_mode = _resolve_pixelle_backend_mode()
    if backend_mode == "direct":
        return "minimax_video"
    return effective_capability


def _build_ai_mode_provider_chain(effective_capability: str) -> List[Tuple[str, str]]:
    chain: List[Tuple[str, str]] = [("minimax_primary", "minimax_video")]
    if effective_capability != "minimax_video":
        chain.append(("legacy_secondary", effective_capability))
    return chain


def _resolve_pixelle_asset(
    segment: Segment,
    project_root: str,
    generated_dir: str,
    effective_capability: Optional[str] = None,
    material_mode: str = "auto",
    vendor_preference: Optional[str] = None,
    continuity_directive: Optional[ContinuityDirective] = None,
    routed_capability_override: Optional[str] = None,
    resolution: Tuple[int, int] = (1080, 1920),
):
    """
    Resolve Pixelle asset with retry policy.
    Returns AssetRef on success or failure info on error.
    """
    if not effective_capability:
        diagnostic = FailureDiagnostic.from_error(
            category=ErrorCategory.UNSUPPORTED,
            reason_code="PIXELLE_CAPABILITY_UNAVAILABLE",
        )
        return _with_legacy_aliases(AssetRef(
            kind="pixelle_failed",
            path="",
            asset_hash="000000000000",
            fallback_reason_code="PIXELLE_CAPABILITY_UNAVAILABLE",
            fallback_error_category=ErrorCategory.UNSUPPORTED.value,
            fallback_diagnostic=diagnostic.to_dict(),
        ))
    
    try:
        from pixelle_snapshot.adapters import is_capability_available, get_adapter
        from src.steps.pixelle_retry_policy import PixelleRetryPolicy, classify_provider_error

        routed_capability = routed_capability_override or _resolve_routed_pixelle_capability(effective_capability)

        # Check if capability is available
        if not is_capability_available(routed_capability, vendor_preference=vendor_preference):
            diagnostic = FailureDiagnostic.from_error(
                category=ErrorCategory.UNSUPPORTED,
                reason_code="PIXELLE_CAPABILITY_UNAVAILABLE",
            )
            return _with_legacy_aliases(AssetRef(
                kind="pixelle_failed",
                path="",
                asset_hash="000000000000",
                fallback_reason_code="PIXELLE_CAPABILITY_UNAVAILABLE",
                fallback_error_category=ErrorCategory.UNSUPPORTED.value,
                fallback_diagnostic=diagnostic.to_dict(),
            ))

        try:
            _pixelle_reliability_controls.before_provider_call(
                capability=effective_capability,
                segment_key=segment.segment_key,
            )
        except RateLimitExceededError:
            diagnostic = FailureDiagnostic.from_error(
                category=ErrorCategory.RESOURCE,
                reason_code="PIXELLE_RATE_LIMITED",
            )
            return _with_legacy_aliases(AssetRef(
                kind="pixelle_failed",
                path="",
                asset_hash="000000000000",
                fallback_reason_code="PIXELLE_RATE_LIMITED",
                fallback_error_category=ErrorCategory.RESOURCE.value,
                fallback_diagnostic=diagnostic.to_dict(),
            ))
        except CircuitOpenError:
            diagnostic = FailureDiagnostic.from_error(
                category=ErrorCategory.PROVIDER,
                reason_code="PIXELLE_CIRCUIT_OPEN",
            )
            return _with_legacy_aliases(AssetRef(
                kind="pixelle_failed",
                path="",
                asset_hash="000000000000",
                fallback_reason_code="PIXELLE_CIRCUIT_OPEN",
                fallback_error_category=ErrorCategory.PROVIDER.value,
                fallback_diagnostic=diagnostic.to_dict(),
            ))

        quota = get_quota_enforcement()
        try:
            quota.check_before_request(
                segment_key=segment.segment_key,
                capability=effective_capability,
            )
        except QuotaExceededError as quota_error:
            return _with_legacy_aliases(AssetRef(
                kind="pixelle_failed",
                path="",
                asset_hash="000000000000",
                fallback_reason_code=quota_error.reason_code,
                fallback_error_category=quota_error.category,
                fallback_diagnostic=create_quota_diagnostic(quota_error),
            ))
        
        # Get retry policy from environment
        max_retries = int(os.environ.get("PIXELLE_PROVIDER_MAX_RETRIES", "3"))
        base_delay = float(os.environ.get("PIXELLE_PROVIDER_RETRY_BASE_DELAY", "1.0"))
        
        policy = PixelleRetryPolicy(
            max_attempts=max_retries,
            base_delay_seconds=base_delay,
            max_delay_seconds=10.0,
        )
        
        adapter = get_adapter(routed_capability, vendor_preference=vendor_preference)
        if adapter is None:
            diagnostic = FailureDiagnostic.from_error(
                category=ErrorCategory.UNSUPPORTED,
                reason_code="PIXELLE_ADAPTER_NOT_FOUND",
            )
            return _with_legacy_aliases(AssetRef(
                kind="pixelle_failed",
                path="",
                asset_hash="000000000000",
                fallback_reason_code="PIXELLE_ADAPTER_NOT_FOUND",
                fallback_error_category=ErrorCategory.UNSUPPORTED.value,
                fallback_diagnostic=diagnostic.to_dict(),
            ))
        output_dir = Path(generated_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            request = _build_pixelle_request(
                capability=routed_capability,
                segment=segment,
                project_root=project_root,
                output_dir=str(output_dir),
                continuity_directive=continuity_directive,
                pixelle_workflow=effective_capability,
                resolution=resolution,
            )
        except _PixelleConstraintError as request_error:
            diagnostic = FailureDiagnostic.from_error(
                category=request_error.category,
                reason_code=request_error.reason_code,
                custom_guidance=str(request_error),
            )
            return _with_legacy_aliases(AssetRef(
                kind="pixelle_failed",
                path="",
                asset_hash="000000000000",
                fallback_reason_code=request_error.reason_code,
                fallback_error_category=request_error.category.value,
                fallback_diagnostic=diagnostic.to_dict(),
            ))
        except Exception as request_error:
            diagnostic = FailureDiagnostic.from_error(
                category=ErrorCategory.VALIDATION,
                reason_code="PIXELLE_REQUEST_BUILD_FAILED",
                custom_guidance=str(request_error),
            )
            return _with_legacy_aliases(AssetRef(
                kind="pixelle_failed",
                path="",
                asset_hash="000000000000",
                fallback_reason_code="PIXELLE_REQUEST_BUILD_FAILED",
                fallback_error_category=ErrorCategory.VALIDATION.value,
                fallback_diagnostic=diagnostic.to_dict(),
            ))

        assert adapter is not None
        
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                with _pixelle_semaphore:
                    response = adapter.invoke(request)
                
                if response.success and response.output_path:
                    from pixelle_snapshot import test_doubles

                    final_output_path = response.output_path
                    if test_doubles.is_test_mode_enabled() and effective_capability == "digital_human":
                        deterministic = test_doubles.create_digital_human_test_output(
                            segment_key=segment.segment_key,
                            segment_duration=segment.duration,
                            output_dir=str(output_dir),
                            avatar_id="default_avatar",
                            voice_id="default_voice",
                        )
                        final_output_path = deterministic.output_path

                    metadata = getattr(response, "metadata", {}) or {}
                    if metadata.get("mvp_placeholder") and not metadata.get("test_mode"):
                        diagnostic = FailureDiagnostic.from_error(
                            category=ErrorCategory.EXECUTION,
                            reason_code="PIXELLE_INVOCATION_FAILED",
                        )
                        return _with_legacy_aliases(AssetRef(
                            kind="pixelle_failed",
                            path="",
                            asset_hash="000000000000",
                            fallback_reason_code="PIXELLE_INVOCATION_FAILED",
                                fallback_error_category=ErrorCategory.EXECUTION.value,
                                fallback_diagnostic=diagnostic.to_dict(),
                        ))

                    if (
                        material_mode == "auto"
                        and effective_capability == "digital_human"
                        and metadata.get("test_mode")
                    ):
                        diagnostic = FailureDiagnostic.from_error(
                            category=ErrorCategory.EXECUTION,
                            reason_code="PIXELLE_INVOCATION_FAILED",
                        )
                        return _with_legacy_aliases(AssetRef(
                            kind="pixelle_failed",
                            path="",
                            asset_hash="000000000000",
                            fallback_reason_code="PIXELLE_INVOCATION_FAILED",
                            fallback_error_category=ErrorCategory.EXECUTION.value,
                            fallback_diagnostic=diagnostic.to_dict(),
                        ))

                    if (
                        test_doubles.is_test_mode_enabled()
                        and effective_capability == "digital_human"
                        and metadata.get("test_mode")
                    ):
                        deterministic = test_doubles.create_digital_human_test_output(
                            segment_key=segment.segment_key,
                            segment_duration=segment.duration,
                            output_dir=str(output_dir),
                            avatar_id="default_avatar",
                            voice_id="default_voice",
                        )
                        final_output_path = deterministic.output_path

                    _pixelle_reliability_controls.record_success()
                    request_id = str(getattr(response, "metadata", {}).get("request_id") or f"req-{uuid.uuid4().hex[:12]}")
                    cost_usd = float(getattr(response, "metadata", {}).get("cost_usd") or 0.0)
                    quota.accounting.record_usage(
                        request_id=request_id,
                        segment_key=segment.segment_key,
                        capability=effective_capability,
                        cost_usd=cost_usd,
                    )
                    # Success
                    return _with_legacy_aliases(AssetRef(
                        kind="pixelle_video",
                        path=final_output_path,
                        asset_hash=_file_hash(final_output_path),
                    ))
                
                # Failed response
                error = response.error
                if error:
                    # Classify error
                    category = normalize_error_category(getattr(error, "category", None))
                    _pixelle_reliability_controls.record_failure(category=category)
                    details = getattr(error, "details", {})
                    reason_code = "PIXELLE_INVOCATION_FAILED"
                    if isinstance(details, dict) and details.get("reason_code"):
                        reason_code = str(details.get("reason_code"))

                    if error.category == ErrorCategory.PROVIDER:
                        classified = classify_provider_error(error)
                        if not classified.retryable:
                            # Non-retryable error, stop immediately
                            diagnostic = FailureDiagnostic.from_error(
                                category=ErrorCategory(classified.category),
                                reason_code=reason_code,
                                custom_guidance=str(getattr(error, "message", "")) or None,
                            )
                            return _with_legacy_aliases(AssetRef(
                                kind="pixelle_failed",
                                path="",
                                asset_hash="000000000000",
                                fallback_reason_code=reason_code,
                                fallback_error_category=classified.category,
                                fallback_diagnostic=diagnostic.to_dict(),
                            ))
                    
                    last_error = error
                    
                    # Check if retryable
                    from src.steps.pixelle_retry_policy import ERROR_RETRY_MATRIX
                    if not ERROR_RETRY_MATRIX.get(category, False):
                        # Non-retryable
                        diagnostic = FailureDiagnostic.from_error(
                            category=ErrorCategory(category),
                            reason_code=reason_code,
                            custom_guidance=str(getattr(error, "message", "")) or None,
                        )
                        return _with_legacy_aliases(AssetRef(
                            kind="pixelle_failed",
                            path="",
                            asset_hash="000000000000",
                            fallback_reason_code=reason_code,
                            fallback_error_category=category,
                            fallback_diagnostic=diagnostic.to_dict(),
                        ))
                    
                    # Retryable error - wait and retry
                    if attempt < max_retries:
                        import time
                        time.sleep(policy.backoff_seconds(attempt))
                        continue
                
            except Exception as e:
                last_error = e
                category = normalize_error_category(getattr(e, "category", ErrorCategory.EXECUTION))
                _pixelle_reliability_controls.record_failure(category=category)
                if attempt < max_retries:
                    import time
                    time.sleep(policy.backoff_seconds(attempt))
                    continue
        
        # Exhausted retries
        error_category = normalize_error_category(getattr(last_error, "category", ErrorCategory.EXECUTION))
        reason_code = "PIXELLE_INVOCATION_FAILED"
        details = getattr(last_error, "details", {})
        if isinstance(details, dict) and details.get("reason_code"):
            reason_code = str(details.get("reason_code"))
        diagnostic = FailureDiagnostic.from_error(
            category=ErrorCategory(error_category),
            reason_code=reason_code,
            custom_guidance=str(getattr(last_error, "message", "")) or None,
        )
        return _with_legacy_aliases(AssetRef(
            kind="pixelle_failed",
            path="",
            asset_hash="000000000000",
            fallback_reason_code=reason_code,
            fallback_error_category=error_category,
            fallback_diagnostic=diagnostic.to_dict(),
        ))
        
    except ImportError:
        # Pixelle not available
        diagnostic = FailureDiagnostic.from_error(
            category=ErrorCategory.UNSUPPORTED,
            reason_code="PIXELLE_IMPORT_ERROR",
        )
        return _with_legacy_aliases(AssetRef(
            kind="pixelle_failed",
            path="",
            asset_hash="000000000000",
            fallback_reason_code="PIXELLE_IMPORT_ERROR",
            fallback_error_category=ErrorCategory.UNSUPPORTED.value,
            fallback_diagnostic=diagnostic.to_dict(),
        ))


def _build_pixelle_request(
    capability: str,
    segment: Segment,
    project_root: str,
    output_dir: str,
    resolution: Tuple[int, int],
    continuity_directive: Optional[ContinuityDirective] = None,
    pixelle_workflow: Optional[str] = None,
):
    from pixelle_snapshot.adapters import ActionTransferRequest, DigitalHumanRequest, I2VRequest, MinimaxVideoRequest

    common = {
        "segment_key": segment.segment_key,
        "segment_text": segment.text,
        "segment_duration": segment.duration,
        "project_root": project_root,
        "output_dir": output_dir,
        "metadata": {
            "continuity": continuity_directive.request_metadata() if continuity_directive else None,
        },
    }

    if capability == "digital_human":
        return DigitalHumanRequest(
            **common,
            avatar_id="default_avatar",
            voice_id="default_voice",
        )
    if capability == "i2v":
        input_image_path = str(Path(project_root) / "assets" / "inputs" / "placeholder.png")
        if continuity_directive and continuity_directive.start_frame_path:
            input_image_path = continuity_directive.start_frame_path
        return I2VRequest(
            **common,
            input_image_path=input_image_path,
        )
    if capability == "minimax_video":
        model = os.environ.get("PIXELLE_MINIMAX_MODEL", "MiniMax-Hailuo-02")
        normalized_duration, constraint_diagnostic = _normalize_minimax_duration_constraints(
            model=model,
            requested_duration=float(segment.duration),
            resolution=resolution,
            segment_index=segment.index,
        )

        minimax_metadata = {
            "continuity": continuity_directive.request_metadata() if continuity_directive else None,
            "constraint_diagnostic": constraint_diagnostic,
        }
        request_common = dict(common)
        request_common["metadata"] = minimax_metadata
        input_image_path = None
        if pixelle_workflow == "i2v":
            input_image_path = str(Path(project_root) / "assets" / "inputs" / "placeholder.png")
            if continuity_directive and continuity_directive.start_frame_path:
                input_image_path = continuity_directive.start_frame_path
        return MinimaxVideoRequest(
            **request_common,
            input_image_path=input_image_path,
            model=model,
            target_duration=normalized_duration,
        )
    if capability == "action_transfer":
        return ActionTransferRequest(
            **common,
            reference_video_path=str(Path(project_root) / "assets" / "inputs" / "reference.mp4"),
            target_image_path=str(Path(project_root) / "assets" / "inputs" / "target.png"),
        )
    raise ValueError(f"Unsupported Pixelle capability: {capability}")


def _file_hash(path: str) -> str:
    """计算文件内容哈希（用于 asset_hash）"""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:12]
    except Exception:
        return "000000000000"


def _asset_cache_path(generated_dir: str, content_key: str, plan_hash: str, ext: str = "png") -> str:
    """返回素材缓存路径：assets/generated/{content_key}_{plan_hash}.{ext}"""
    return os.path.join(generated_dir, f"{content_key}_{plan_hash}.{ext}")


def _find_cached_asset(generated_dir: str, content_key: str, plan_hash: str) -> Optional[str]:
    """
    检查素材缓存：同 content_key 且 plan_hash 相同 → 命中
    视频优先（mp4），再找图片（png/jpg/jpeg）
    """
    for ext in ("mp4", "png", "jpg", "jpeg"):
        path = _asset_cache_path(generated_dir, content_key, plan_hash, ext)
        if os.path.exists(path) and os.path.getsize(path) > 1024:
            return path
    return None


# ─────────────────────────────────────────────
# ① PDF 图表素材
# ─────────────────────────────────────────────
def resolve_pdf_asset(vp: VisualPlan, project_root: str) -> Optional[str]:
    """从 visual_plan.use_pdf_assets 中找到对应图片"""
    for ref in getattr(vp, "use_pdf_assets", []):
        img_path = Path(project_root) / ref.get("image", "")
        if img_path.exists():
            return str(img_path)
    return None


# ─────────────────────────────────────────────
# 本地素材库检索（辅助）
# ─────────────────────────────────────────────
def search_library_asset(keywords: List[str], library_dir: str) -> Optional[str]:
    """在本地素材库中按关键词检索图片/视频"""
    lib = Path(library_dir)
    if not lib.exists():
        return None
    for kw in keywords:
        kw_lower = kw.lower()
        for ext in ("*.mp4", "*.mov", "*.jpg", "*.jpeg", "*.png"):
            for f in lib.glob(f"**/{ext}"):
                if kw_lower in f.stem.lower():
                    return str(f)
    return None


# ─────────────────────────────────────────────
# ② Pexels 视频
# ─────────────────────────────────────────────
def fetch_pexels_video(
    keywords: List[str],
    visual_type: str,
    segment_duration: float,
    download_dir: str,
    cache_dir: str,
    api_key: str,
    aspect_ratio: str = "9:16",
) -> Optional[str]:
    """
    从 Pexels 搜索并下载最佳视频，返回本地 mp4 路径。
    失败时返回 None（不抛异常，让上层 fallback 到下一级）。
    """
    if not api_key:
        return None
    try:
        from src.utils.pexels_client import PexelsClient
        client = PexelsClient(
            api_key=api_key,
            cache_dir=cache_dir,
            aspect_ratio=aspect_ratio,
            preferred_quality="hd",
        )
        return client.fetch_best_video(
            keywords=keywords,
            visual_type=visual_type,
            segment_duration=segment_duration,
            download_dir=download_dir,
        )
    except Exception as e:
        logger.warning(f"  Pexels 视频获取失败: {e}")
        return None


# ─────────────────────────────────────────────
# ③ Pexels 图片
# ─────────────────────────────────────────────
def fetch_pexels_photo(
    keywords: List[str],
    visual_type: str,
    download_dir: str,
    cache_dir: str,
    api_key: str,
    aspect_ratio: str = "9:16",
) -> Optional[str]:
    """
    从 Pexels 搜索并下载最佳图片，返回本地 jpg 路径。
    失败时返回 None。
    """
    if not api_key:
        return None
    try:
        from src.utils.pexels_client import PexelsClient
        client = PexelsClient(
            api_key=api_key,
            cache_dir=cache_dir,
            aspect_ratio=aspect_ratio,
        )
        return client.fetch_best_photo(
            keywords=keywords,
            visual_type=visual_type,
            download_dir=download_dir,
        )
    except Exception as e:
        logger.warning(f"  Pexels 图片获取失败: {e}")
        return None


# ─────────────────────────────────────────────
# ④ AI 图片生成（DALL-E 3）
# ─────────────────────────────────────────────
def generate_ai_image(
    prompt: str,
    output_path: str,
    size: str = "1024x1792",
    model: str = "dall-e-3",
) -> Optional[str]:
    """调用 DALL-E 生成图片并保存到 output_path"""
    try:
        import requests as req
        client = OpenAI()
        logger.info(f"  AI 图片生成: {prompt[:60]}...")
        final_size: Literal["1024x1024", "1024x1792", "1792x1024"]
        if size == "1024x1024":
            final_size = "1024x1024"
        elif size == "1792x1024":
            final_size = "1792x1024"
        else:
            final_size = "1024x1792"

        response = client.images.generate(
            model=model,
            prompt=prompt,
            size=final_size,
            quality="standard",
            n=1,
        )
        data = response.data
        if not data:
            return None
        image_url = data[0].url
        if not image_url:
            return None
        img_data = req.get(image_url, timeout=30).content
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(img_data)
        logger.info(f"  AI 图片已保存: {output_path}")
        return output_path
    except Exception as e:
        logger.warning(f"  AI 图片生成失败: {e}")
        return None


# ─────────────────────────────────────────────
# ⑤ 通用模板兜底
# ─────────────────────────────────────────────
def generate_template_asset(
    output_path: str,
    width: int = 1080,
    height: int = 1920,
    text: str = "",
) -> str:
    """生成通用背景图（深色渐变 + 网格线），用 PIL 优先，FFmpeg 兜底"""
    try:
        from PIL import Image, ImageDraw  # type: ignore[reportMissingImports]
        img = Image.new("RGB", (width, height), color=(10, 10, 26))
        draw = ImageDraw.Draw(img)
        for y in range(height):
            ratio = y / height
            r = int(10 + ratio * 20)
            g = int(10 + ratio * 5)
            b = int(26 + ratio * 40)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        for x in range(0, width, 80):
            draw.line([(x, 0), (x, height)], fill=(30, 30, 50), width=1)
        for y in range(0, height, 80):
            draw.line([(0, y), (width, y)], fill=(30, 30, 50), width=1)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, "PNG")
        return output_path
    except ImportError:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = (
            f'ffmpeg -y -f lavfi -i color=c=0x0a0a1a:size={width}x{height}:rate=1 '
            f'-frames:v 1 "{output_path}" -loglevel error'
        )
        subprocess.run(cmd, shell=True)
        return output_path


# ─────────────────────────────────────────────
# 主素材解析逻辑（五级优先级）
# ─────────────────────────────────────────────
def resolve_asset_for_segment(
    segment: Segment,
    project_root: str,
    generated_dir: str,
    library_dir: str,
    pexels_api_key: str = "",
    enable_pexels_video: bool = True,
    enable_pexels_photo: bool = True,
    enable_ai_image: bool = False,
    resolution: Tuple[int, int] = (1080, 1920),
    aspect_ratio: str = "9:16",
    pixelle_default_workflow: Optional[str] = None,
    pixelle_segment_overrides: Optional[dict] = None,
    vendor_preference: Optional[str] = None,
    continuity_policy: Optional[str] = None,
    continuity_seed: Optional[int] = None,
    style_id: Optional[str] = None,
    project_id: str = "",
    previous_segment: Optional[Segment] = None,
    material_mode: str = "auto",
) -> Segment:
    """
    为单个 Segment 解析素材，更新 segment.asset_refs 和 segment.visual_plan.asset_path。

    五级优先级（详见模块文档）。
    缓存命中时直接返回，不重新下载/生成。
    """
    vp = segment.visual_plan
    plan_hash = segment.plan_hash or "nohash"
    content_key = segment.content_key

    # Resolve effective Pixelle workflow if applicable (with source tracking)
    effective_workflow, workflow_source = _resolve_effective_pixelle_workflow_with_source(
        segment, pixelle_default_workflow, pixelle_segment_overrides
    )
    
    # Compute effective cache hash (includes workflow if applicable)
    effective_cache_hash = _compute_effective_cache_hash(plan_hash, effective_workflow)

    ai_selected = bool(getattr(segment, "step4_ai_selected", False))
    ai_allocation_map = getattr(segment, "step4_ai_allocation_map", None)
    ai_cap_enforced = isinstance(ai_allocation_map, dict) and segment.segment_key in ai_allocation_map
    ai_routes_allowed = (not ai_cap_enforced) or ai_selected

    # ── ⓪ 缓存检查（content_key + effective_cache_hash 完全命中）──
    cached = _find_cached_asset(generated_dir, content_key, effective_cache_hash)
    if cached:
        meta = _read_asset_cache_meta(cached)
        cached_kind = meta.get("kind") if isinstance(meta, dict) else None
        cached_is_ai = bool(meta.get("is_ai_generated")) if isinstance(meta, dict) else False

        if not ai_routes_allowed and (cached_is_ai or cached_kind in AI_GENERATED_ROUTES):
            logger.info(
                f"  [seg {segment.index}] Route diagnostic BLOCKED: route=cached "
                f"mode={material_mode} reason=AI_ROUTE_BLOCKED_BY_ALLOCATION_CAP category=POLICY"
            )
            cached = None

        # In ai_only mode, never silently accept cached non-AI assets.
        if cached and material_mode == "ai_only" and ai_routes_allowed:
            if cached_kind not in AI_GENERATED_ROUTES:
                logger.info(
                    f"  [seg {segment.index}] Route diagnostic BLOCKED: route=cached "
                    f"mode={material_mode} reason=CACHE_NON_AI_BLOCKED category=POLICY"
                )
                cached = None

    if cached:
        logger.debug(f"  [seg {segment.index}] ⓪ 缓存命中: {os.path.basename(cached)}")
        asset_hash = _file_hash(cached)
        if vp:
            vp.asset_path = cached
        segment.asset_refs = [AssetRef(kind="cached", path=cached, asset_hash=asset_hash)]
        return segment

    asset_path: Optional[str] = None
    asset_kind = "template"
    pixelle_fallback_reason: Optional[str] = None
    pixelle_fallback_category: Optional[str] = None
    pixelle_fallback_diagnostic: Optional[dict] = None
    vp_type = vp.type if vp else "template"
    keywords: List[str] = (vp.keywords if vp else []) or []
    prompt = vp.prompt if vp else ""
    search_keywords = _expand_semantic_keywords(keywords, prompt, segment.text)
    search_visual_type = _effective_visual_type(vp_type, search_keywords)
    semantic_priority = compute_segment_semantic_priority_score(segment)
    segment_duration = getattr(segment, "duration", 5.0)
    continuity_directive: Optional[ContinuityDirective] = None

    if effective_workflow:
        continuity_directive = evaluate_continuity_policy(
            segment=segment,
            previous_segment=previous_segment,
            policy_mode=continuity_policy,
            continuity_seed=continuity_seed,
            style_id=style_id,
            project_id=project_id,
            vendor_preference=vendor_preference,
            project_root=project_root,
            resolution=resolution,
        )
        segment.prev_last_frame_path = continuity_directive.start_frame_path
        segment.continuity_diagnostic = continuity_directive.diagnostic

    # Pexels 缓存目录
    pexels_cache_dir = str(Path(project_root) / "assets" / "pexels_cache")
    pexels_video_dir = str(Path(pexels_cache_dir) / "videos")
    pexels_photo_dir = str(Path(pexels_cache_dir) / "photos")

    pdf_chart_blocked_by_policy = False

    if vp_type == "pdf_chart" and vp:
        if is_route_allowed_by_mode_policy("pdf_chart", material_mode):
            asset_path = resolve_pdf_asset(vp, project_root)
            if asset_path:
                asset_kind = "pdf_chart"
                logger.info(
                    f"  [seg {segment.index}] Route diagnostic SUCCESS: route=pdf_chart mode={material_mode}"
                )
                logger.info(f"  [seg {segment.index}] ① PDF 图表: {os.path.basename(asset_path)}")
        else:
            pdf_chart_blocked_by_policy = True
            logger.info(
                f"  [seg {segment.index}] Route diagnostic BLOCKED: route=pdf_chart "
                f"mode={material_mode} reason=PDF_CHART_BLOCKED_BY_MODE_POLICY category=POLICY"
            )
            logger.info(f"  [seg {segment.index}] ① PDF 图表: blocked by mode_policy={material_mode}")

    # ── Route order depends on material_mode ──
    # ai_preferred: Pixelle first, then non-AI fallbacks
    # auto/ai_only: Standard order (Pexels first, then Pixelle)
    
    # Structured diagnostics for success-rate tracking
    route_diagnostic = {
        "segment_key": segment.segment_key,
        "material_mode": material_mode,
        "ai_selected": ai_selected,
        "ai_routes_allowed": ai_routes_allowed,
        "ai_allocation_map": ai_allocation_map if isinstance(ai_allocation_map, dict) else None,
        "workflow": effective_workflow,
        "workflow_source": workflow_source,
        "route_attempted": None,
        "route_selected": None,
        "provider_stage": None,
        "provider_attempts": [],
        "constraint_decisions": [],
        "semantic_priority": semantic_priority,
        "reason_code": None,
        "error_category": None,
    }

    def _return_ai_only_exhausted(
        *,
        reason_code: str,
        category: str,
        guidance: str,
        include_original_failure: bool,
        precondition_failure: Optional[dict] = None,
    ) -> Segment:
        nonlocal route_diagnostic

        route_diagnostic["route_attempted"] = "ai_only_exhausted"
        route_diagnostic["reason_code"] = reason_code
        route_diagnostic["error_category"] = category
        logger.info(
            f"  [seg {segment.index}] Route diagnostic EXHAUSTED: route=ai_only_exhausted "
            f"mode={material_mode} reason={reason_code} category={category}"
        )
        logger.info(f"  [seg {segment.index}] ⑤ ai_only 模式: AI 路由已耗尽，返回失败")

        ai_only_diagnostic = {
            "category": category,
            "reason_code": reason_code,
            "retryable": False,
            "guidance": guidance,
            "fallback_hint": None,
            "ai_selected": ai_selected,
            "ai_allocation_map": ai_allocation_map if isinstance(ai_allocation_map, dict) else None,
        }
        if precondition_failure:
            ai_only_diagnostic["precondition_failure"] = precondition_failure
        if include_original_failure and pixelle_fallback_diagnostic:
            ai_only_diagnostic["original_failure"] = pixelle_fallback_diagnostic
        ai_only_diagnostic["semantic_priority"] = semantic_priority

        segment.asset_refs = [
            _with_legacy_aliases(AssetRef(
                kind="ai_only_exhausted",
                path="",
                asset_hash="000000000000",
                fallback_reason_code=reason_code,
                fallback_error_category=category,
                fallback_diagnostic=ai_only_diagnostic,
            ))
        ]
        setattr(segment, "step4_route_diagnostic", route_diagnostic)
        return segment
    
    def _attempt_pixelle_route() -> bool:
        """Attempt Pixelle route, returns True if successful."""
        nonlocal asset_path, asset_kind, pixelle_fallback_reason, pixelle_fallback_category, pixelle_fallback_diagnostic
        nonlocal route_diagnostic

        def _invoke_resolve_pixelle_asset_compat(**kwargs):
            try:
                return _resolve_pixelle_asset(**kwargs)
            except TypeError as exc:
                msg = str(exc)
                if "unexpected keyword argument" in msg and "resolution" in msg and "resolution" in kwargs:
                    compat_kwargs = dict(kwargs)
                    compat_kwargs.pop("resolution", None)
                    return _resolve_pixelle_asset(**compat_kwargs)
                raise
        
        route_diagnostic["route_attempted"] = "pixelle_video"
        
        if not effective_workflow:
            route_diagnostic["reason_code"] = "NO_WORKFLOW_CONFIGURED"
            route_diagnostic["error_category"] = "CONFIG"
            logger.info(
                f"  [seg {segment.index}] Route diagnostic: route=pixelle_video "
                f"mode={material_mode} reason=NO_WORKFLOW_CONFIGURED"
            )
            return False
            
        rollout = check_rollout_eligibility(segment.segment_key, _pixelle_rollout_config)
        if not rollout.eligible:
            diagnostic = FailureDiagnostic.from_error(
                category=ErrorCategory.PROVIDER,
                reason_code="PIXELLE_ROLLOUT_INELIGIBLE",
            )
            pixelle_fallback_reason = "PIXELLE_ROLLOUT_INELIGIBLE"
            pixelle_fallback_category = ErrorCategory.PROVIDER.value
            pixelle_fallback_diagnostic = diagnostic.to_dict()
            route_diagnostic["reason_code"] = "PIXELLE_ROLLOUT_INELIGIBLE"
            route_diagnostic["error_category"] = ErrorCategory.PROVIDER.value
            logger.info(
                f"  [seg {segment.index}] Route diagnostic: route=pixelle_video "
                f"mode={material_mode} reason=PIXELLE_ROLLOUT_INELIGIBLE category=PROVIDER"
            )
            return False
            
        provider_chain: List[Tuple[str, str]]
        if material_mode in ("ai_only", "ai_preferred"):
            provider_chain = _build_ai_mode_provider_chain(effective_workflow)
        else:
            provider_chain = [
                ("backend_selected", _resolve_routed_pixelle_capability(effective_workflow))
            ]

        provider_attempts: List[dict] = []

        for provider_stage, routed_capability in provider_chain:
            route_diagnostic["provider_stage"] = provider_stage
            logger.info(
                f"  [seg {segment.index}] Route diagnostic: route=pixelle_video "
                f"mode={material_mode} stage={provider_stage} capability={routed_capability}"
            )
            try:
                pixelle_ref = _invoke_resolve_pixelle_asset_compat(
                    segment=segment,
                    project_root=project_root,
                    generated_dir=generated_dir,
                    effective_capability=effective_workflow,
                    material_mode=material_mode,
                    vendor_preference=vendor_preference,
                    continuity_directive=continuity_directive,
                    routed_capability_override=routed_capability,
                    resolution=resolution,
                )
            except TypeError as exc:
                if "unexpected keyword argument" not in str(exc) or "routed_capability_override" not in str(exc):
                    raise
                fallback_capability = effective_workflow if material_mode == "auto" else routed_capability
                pixelle_ref = _invoke_resolve_pixelle_asset_compat(
                    segment=segment,
                    project_root=project_root,
                    generated_dir=generated_dir,
                    effective_capability=fallback_capability,
                    material_mode=material_mode,
                    vendor_preference=vendor_preference,
                    continuity_directive=continuity_directive,
                    resolution=resolution,
                )
            # Handle both AssetRef and tuple returns (for test mocking compatibility)
            if isinstance(pixelle_ref, tuple):
                # Old tuple format: (path, asset_hash, reason_code, error_category)
                if len(pixelle_ref) >= 4:
                    pixelle_fallback_reason = pixelle_ref[2]
                    pixelle_fallback_category = pixelle_ref[3]
                provider_attempts.append(
                    {
                        "provider_stage": provider_stage,
                        "routed_capability": routed_capability,
                        "reason_code": pixelle_fallback_reason,
                        "error_category": pixelle_fallback_category,
                    }
                )
                continue

            if pixelle_ref and hasattr(pixelle_ref, "kind") and pixelle_ref.kind == "pixelle_video" and pixelle_ref.path:
                asset_path = pixelle_ref.path
                asset_kind = "pixelle_video"
                route_diagnostic["route_selected"] = "pixelle_video"
                route_diagnostic["provider_stage"] = provider_stage
                route_diagnostic["reason_code"] = None
                route_diagnostic["error_category"] = None
                if provider_attempts:
                    provider_attempts.append(
                        {
                            "provider_stage": provider_stage,
                            "routed_capability": routed_capability,
                            "reason_code": None,
                            "error_category": None,
                            "success": True,
                        }
                    )
                    pixelle_fallback_diagnostic = dict(pixelle_fallback_diagnostic or {})
                    pixelle_fallback_diagnostic["provider_attempts"] = provider_attempts
                    pixelle_fallback_diagnostic["provider_chain_mode"] = material_mode
                    pixelle_fallback_diagnostic["provider_selected"] = {
                        "provider_stage": provider_stage,
                        "routed_capability": routed_capability,
                    }
                    route_diagnostic["provider_attempts"] = provider_attempts
                logger.info(
                    f"  [seg {segment.index}] Route diagnostic SUCCESS: route=pixelle_video "
                    f"mode={material_mode} stage={provider_stage} capability={routed_capability}"
                )
                logger.info(f"  [seg {segment.index}] Pixelle 视频: {os.path.basename(pixelle_ref.path)}")
                return True

            if pixelle_ref and hasattr(pixelle_ref, "fallback_reason_code"):
                pixelle_fallback_reason = pixelle_ref.fallback_reason_code
                pixelle_fallback_category = pixelle_ref.fallback_error_category
                pixelle_fallback_diagnostic = pixelle_ref.fallback_diagnostic

            provider_attempts.append(
                {
                    "provider_stage": provider_stage,
                    "routed_capability": routed_capability,
                    "reason_code": pixelle_fallback_reason,
                    "error_category": pixelle_fallback_category,
                }
            )

        if provider_attempts:
            pixelle_fallback_diagnostic = dict(pixelle_fallback_diagnostic or {})
            pixelle_fallback_diagnostic["provider_attempts"] = provider_attempts
            pixelle_fallback_diagnostic["provider_chain_mode"] = material_mode
            pixelle_fallback_diagnostic["provider_chain_exhausted"] = True
            route_diagnostic["provider_attempts"] = provider_attempts
            route_diagnostic["reason_code"] = pixelle_fallback_reason or "PROVIDER_CHAIN_EXHAUSTED"
            route_diagnostic["error_category"] = pixelle_fallback_category or "PROVIDER"
            logger.info(
                f"  [seg {segment.index}] Route diagnostic EXHAUSTED: route=pixelle_video "
                f"mode={material_mode} attempts={len(provider_attempts)} "
                f"reason={route_diagnostic['reason_code']} category={route_diagnostic['error_category']}"
            )
        return False

    def _attempt_pexels_video_route() -> bool:
        """Attempt Pexels video route, returns True if successful."""
        nonlocal asset_path, asset_kind
        nonlocal route_diagnostic
        
        route_diagnostic["route_attempted"] = "pexels_video"
        
        if not (enable_pexels_video and pexels_api_key and search_keywords):
            route_diagnostic["reason_code"] = "PEXELS_VIDEO_NOT_ENABLED"
            route_diagnostic["error_category"] = "CONFIG"
            return False
        if vp_type in ("kinetic_text",):
            route_diagnostic["reason_code"] = "KINETIC_TEXT_INCOMPATIBLE"
            route_diagnostic["error_category"] = "POLICY"
            return False
            
        logger.info(f"  [seg {segment.index}] Pexels 视频搜索: {search_keywords[:2]}")
        pexels_video = fetch_pexels_video(
            keywords=search_keywords,
            visual_type=search_visual_type,
            segment_duration=segment_duration,
            download_dir=pexels_video_dir,
            cache_dir=pexels_cache_dir,
            api_key=pexels_api_key,
            aspect_ratio=aspect_ratio,
        )
        if pexels_video:
            asset_path = pexels_video
            asset_kind = "pexels_video"
            route_diagnostic["route_selected"] = "pexels_video"
            route_diagnostic["reason_code"] = None
            route_diagnostic["error_category"] = None
            logger.info(
                f"  [seg {segment.index}] Route diagnostic SUCCESS: route=pexels_video mode={material_mode}"
            )
            logger.info(f"  [seg {segment.index}] Pexels 视频: {os.path.basename(pexels_video)}")
            return True
        route_diagnostic["reason_code"] = "PEXELS_VIDEO_NO_RESULTS"
        route_diagnostic["error_category"] = "PROVIDER"
        return False

    def _attempt_pexels_photo_route() -> bool:
        """Attempt Pexels photo route, returns True if successful."""
        nonlocal asset_path, asset_kind
        nonlocal route_diagnostic
        
        route_diagnostic["route_attempted"] = "pexels_photo"
        
        if not (enable_pexels_photo and pexels_api_key and search_keywords):
            route_diagnostic["reason_code"] = "PEXELS_PHOTO_NOT_ENABLED"
            route_diagnostic["error_category"] = "CONFIG"
            return False
            
        logger.info(f"  [seg {segment.index}] Pexels 图片搜索: {search_keywords[:2]}")
        pexels_photo = fetch_pexels_photo(
            keywords=search_keywords,
            visual_type=search_visual_type,
            download_dir=pexels_photo_dir,
            cache_dir=pexels_cache_dir,
            api_key=pexels_api_key,
            aspect_ratio=aspect_ratio,
        )
        if pexels_photo:
            asset_path = pexels_photo
            asset_kind = "pexels_photo"
            route_diagnostic["route_selected"] = "pexels_photo"
            route_diagnostic["reason_code"] = None
            route_diagnostic["error_category"] = None
            logger.info(
                f"  [seg {segment.index}] Route diagnostic SUCCESS: route=pexels_photo mode={material_mode}"
            )
            logger.info(f"  [seg {segment.index}] Pexels 图片: {os.path.basename(pexels_photo)}")
            return True
        route_diagnostic["reason_code"] = "PEXELS_PHOTO_NO_RESULTS"
        route_diagnostic["error_category"] = "PROVIDER"
        return False

    # ── ai_preferred mode: Try Pixelle FIRST, then fall back to non-AI routes ──
    if not ai_routes_allowed:
        route_diagnostic["constraint_decisions"].append(
            {
                "type": "ai_allocation_cap",
                "reason_code": "AI_ROUTE_BLOCKED_BY_ALLOCATION_CAP",
                "category": "POLICY",
                "ai_selected": ai_selected,
            }
        )
        logger.info(
            f"  [seg {segment.index}] Route diagnostic POLICY: mode={material_mode} "
            f"reason=AI_ROUTE_BLOCKED_BY_ALLOCATION_CAP ai_selected={ai_selected}"
        )
        if not asset_path:
            _attempt_pexels_video_route()
        if not asset_path:
            _attempt_pexels_photo_route()
    elif material_mode == "ai_preferred":
        # In ai_preferred mode:
        # 1. Try Pixelle/AI first (if workflow configured)
        # 2. On failure, fall back to Pexels video/photo (non-AI sources remain available)
        if not asset_path:
            _attempt_pixelle_route()
        
        # Non-AI fallbacks after AI attempt (preserve fallback chain)
        if not asset_path:
            _attempt_pexels_video_route()
        if not asset_path:
            _attempt_pexels_photo_route()
    elif material_mode == "ai_only":
        # ── ai_only mode: STRICT AI-only routes, no Pexels fallback ──
        # In ai_only mode:
        # 1. Only AI routes allowed: pixelle_video, ai_image
        # 2. Pexels routes (pexels_video, pexels_photo) are BLOCKED by policy
        # 3. On AI failure, return explicit failure - NOT template fallback
        if not effective_workflow:
            return _return_ai_only_exhausted(
                reason_code="AI_ONLY_MISSING_WORKFLOW",
                category="CONFIG",
                guidance=(
                    "ai_only mode requires a Pixelle workflow. Set visual_plan.pixelle_workflow, "
                    "pixelle_segment_overrides[segment_key], or pixelle_default_workflow."
                ),
                include_original_failure=False,
                precondition_failure={
                    "category": "CONFIG",
                    "reason_code": "AI_ONLY_MISSING_WORKFLOW",
                    "required": "effective_workflow",
                },
            )
        if not asset_path:
            _attempt_pixelle_route()
    else:
        # ── auto mode: Standard route order (Pexels first, then Pixelle) ──
        # ── ② Pexels 视频 ──
        # kinetic_text 不适合用视频背景（文字动画需要干净背景），跳过
        if not asset_path:
            _attempt_pexels_video_route()

        # ── ③ Pexels 图片 ──
        if not asset_path:
            _attempt_pexels_photo_route()

        # ── ③.5 Pixelle 视频生成 ──
        if not asset_path:
            _attempt_pixelle_route()

    # ── ④ AI 图片生成 ──
    if (
        not asset_path
        and ai_routes_allowed
        and enable_ai_image
        and vp
        and vp.prompt
        and vp_type in ("ai_image", "broll")
    ):
        route_diagnostic["route_attempted"] = "ai_image"
        ai_output = _asset_cache_path(generated_dir, content_key, effective_cache_hash, "png")
        asset_path = generate_ai_image(
            prompt=vp.prompt,
            output_path=ai_output,
            size=f"{resolution[0]}x{resolution[1]}",
        )
        if asset_path:
            asset_kind = "ai_image"
            route_diagnostic["route_selected"] = "ai_image"
            route_diagnostic["reason_code"] = None
            route_diagnostic["error_category"] = None
            logger.info(
                f"  [seg {segment.index}] Route diagnostic SUCCESS: route=ai_image mode={material_mode}"
            )
        else:
            route_diagnostic["reason_code"] = "AI_IMAGE_GENERATION_FAILED"
            route_diagnostic["error_category"] = "PROVIDER"

    # ── ⑤ 通用模板兜底 ──
    if not asset_path:
        if material_mode == "ai_only" and ai_routes_allowed:
            # Strict ai_only failure: AI routes were allowed but ALL providers exhausted
            return _return_ai_only_exhausted(
                reason_code="AI_ONLY_ROUTES_EXHAUSTED",
                category="POLICY",
                guidance="ai_only mode requires AI-generated assets; all AI routes failed",
                include_original_failure=True,
            )
        
        # Determine if this is a cap-policy fallback in ai_only mode
        is_cap_policy_fallback = material_mode == "ai_only" and not ai_routes_allowed
        
        if is_cap_policy_fallback:
            # Cap-driven fallback in ai_only mode: segment not in top-6 AI allocation
            # This is a POLICY decision (cap exhausted), not a PROVIDER failure
            route_diagnostic["route_attempted"] = "template"
            route_diagnostic["route_selected"] = "template"
            route_diagnostic["reason_code"] = "AI_ONLY_CAP_POLICY_FALLBACK"
            route_diagnostic["error_category"] = "POLICY"
            logger.info(
                f"  [seg {segment.index}] Route diagnostic CAP_POLICY: mode={material_mode} "
                f"reason=AI_ONLY_CAP_POLICY_FALLBACK ai_selected={ai_selected} "
                f"(segment not in top-6 AI allocation, policy-driven downgrade to non-AI route)"
            )
        else:
            route_diagnostic["route_attempted"] = "template"
            route_diagnostic["route_selected"] = "template"
            route_diagnostic["reason_code"] = "TEMPLATE_FALLBACK"
            route_diagnostic["error_category"] = None
            logger.info(
                f"  [seg {segment.index}] Route diagnostic FALLBACK: route=template mode={material_mode} "
                f"reason=TEMPLATE_FALLBACK"
            )
        logger.info(f"  [seg {segment.index}] ⑤ 通用模板兜底")
        template_path = _asset_cache_path(generated_dir, content_key, effective_cache_hash, "png")
        asset_path = generate_template_asset(
            output_path=template_path,
            width=resolution[0],
            height=resolution[1],
            text=segment.text,
        )
        asset_kind = "template"

    # ── 写入缓存（图片类统一复制到 generated/{key}_{effective_hash}.png）──
    if asset_path and asset_kind not in ("pexels_video", "pixelle_video"):
        cache_path = _asset_cache_path(generated_dir, content_key, effective_cache_hash, "png")
        if asset_path != cache_path:
            os.makedirs(generated_dir, exist_ok=True)
            shutil.copy2(asset_path, cache_path)
            asset_path = cache_path
    # 视频类：直接使用原路径（不复制大文件），但在 generated 目录写一个 .mp4 软链接
    elif asset_path and asset_kind in ("pexels_video", "pixelle_video"):
        cache_path = _asset_cache_path(generated_dir, content_key, effective_cache_hash, "mp4")
        if asset_path != cache_path:
            os.makedirs(generated_dir, exist_ok=True)
            try:
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                os.symlink(os.path.abspath(asset_path), cache_path)
                asset_path = cache_path
            except Exception:
                # 软链接失败则直接用原路径
                pass

    _write_asset_cache_meta_if_in_generated_dir(
        asset_path=asset_path,
        generated_dir=generated_dir,
        kind=asset_kind,
        material_mode=material_mode,
    )

    asset_hash = _file_hash(asset_path) if asset_path else "000000000000"
    if vp:
        vp.asset_path = asset_path

    fallback_reason = pixelle_fallback_reason
    fallback_category = pixelle_fallback_category
    fallback_diagnostic = pixelle_fallback_diagnostic

    if pdf_chart_blocked_by_policy and asset_kind == "template":
        fallback_reason = "PDF_CHART_BLOCKED_BY_MODE_POLICY"
        fallback_category = "POLICY"
        fallback_diagnostic = {
            "category": "POLICY",
            "reason_code": "PDF_CHART_BLOCKED_BY_MODE_POLICY",
            "retryable": False,
            "guidance": f"pdf_chart route blocked by material_mode={material_mode}",
            "fallback_hint": None,
        }

    if material_mode == "ai_only" and not ai_routes_allowed and asset_kind == "template":
        fallback_reason = "AI_ONLY_CAP_POLICY_FALLBACK"
        fallback_category = "POLICY"
        fallback_diagnostic = {
            "category": "POLICY",
            "reason_code": "AI_ONLY_CAP_POLICY_FALLBACK",
            "retryable": False,
            "guidance": "ai_only segment not in top-6 AI allocation; cap-policy downgrade to non-AI route",
            "ai_selected": ai_selected,
            "fallback_hint": None,
        }

    segment.asset_refs = [
        AssetRef(
            kind=asset_kind,
            path=asset_path or "",
            asset_hash=asset_hash,
            fallback_reason_code=fallback_reason,
            fallback_error_category=fallback_category,
            fallback_diagnostic=fallback_diagnostic,
        )
    ]
    
    if not route_diagnostic["route_selected"]:
        route_diagnostic["route_selected"] = asset_kind
    setattr(segment, "step4_route_diagnostic", route_diagnostic)
    logger.info(
        f"  [seg {segment.index}] Route diagnostic FINAL: "
        f"mode={route_diagnostic['material_mode']} "
        f"workflow={route_diagnostic['workflow']} "
        f"workflow_source={route_diagnostic['workflow_source']} "
        f"selected={route_diagnostic['route_selected']} "
        f"reason={route_diagnostic['reason_code']} "
        f"category={route_diagnostic['error_category']}"
    )
    
    return segment


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────
def run_step4(
    manifest: Manifest,
    output_manifest: str,
    project_root: str,
    target_segment_keys: Optional[List[str]] = None,
    pexels_api_key: str = "",
    enable_pexels_video: bool = True,
    enable_pexels_photo: bool = True,
    enable_ai_image: bool = False,
) -> Manifest:
    """
    执行 Step 4：为 Segment 解析素材（v3，接入 Pexels）

    :param manifest:              输入 Manifest
    :param output_manifest:       更新后 manifest.json 路径
    :param project_root:          项目根目录
    :param target_segment_keys:   只处理这些 segment_key（None=全部）
    :param pexels_api_key:        Pexels API Key（空字符串=禁用 Pexels）
    :param enable_pexels_video:   是否启用 Pexels 视频搜索（优先级 ②）
    :param enable_pexels_photo:   是否启用 Pexels 图片搜索（优先级 ③）
    :param enable_ai_image:       是否启用 AI 图片生成（优先级 ④）
    :return: 更新后的 Manifest
    """
    logger.info("=" * 50)
    logger.info("Step 4: 素材执行（v3，五级优先级）")

    #TT|    # 从环境变量获取 Pexels API Key（优先使用参数传入的）
    if not pexels_api_key:
        from src.core.api_config import PEXELS_API_KEY
        pexels_api_key = os.environ.get("PEXELS_API_KEY", PEXELS_API_KEY)
    if not pexels_api_key:
        pexels_api_key = os.environ.get("PEXELS_API_KEY", "")

    logger.info(f"  ① PDF 图表:   始终检查")
    logger.info(f"  ② Pexels 视频: {'✓ 启用' if enable_pexels_video and pexels_api_key else '✗ 禁用'}")
    logger.info(f"  ③ Pexels 图片: {'✓ 启用' if enable_pexels_photo and pexels_api_key else '✗ 禁用'}")
    logger.info(f"  ④ AI 图片:    {'✓ 启用' if enable_ai_image else '✗ 禁用'}")
    logger.info(f"  ⑤ 模板兜底:   始终启用")

    generated_dir = str(Path(project_root) / "assets" / "generated")
    library_dir = str(Path(project_root) / "assets" / "library")
    resolution = (manifest.global_style.resolution_w, manifest.global_style.resolution_h)
    aspect_ratio = getattr(manifest.global_style, "aspect_ratio", "9:16")

    target_keys = set(target_segment_keys) if target_segment_keys else None
    if target_keys is not None:
        all_segment_keys = {seg.segment_key for seg in manifest.segments}
        invalid_keys = sorted(target_keys - all_segment_keys)
        if invalid_keys:
            raise ValueError(
                "Invalid target_segment_keys for Step4: " + ", ".join(invalid_keys)
            )

    ai_allocation_map = build_top6_ai_allocation_map(
        manifest.segments,
        target_segment_keys=target_segment_keys,
        max_ai_segments=6,
    )
    setattr(manifest, "step4_ai_allocation_map", ai_allocation_map)
    processed = 0
    skipped = 0

    # 素材来源统计
    source_counts: dict = {}

    # Cap policy telemetry counters (T14)
    cap_telemetry_ai_selected_count = 0
    cap_telemetry_ai_routed_count = 0
    cap_telemetry_non_ai_replacement_count = 0

    for idx, seg in enumerate(manifest.segments):
        if target_keys and seg.segment_key not in target_keys:
            skipped += 1
            continue

        previous_segment = manifest.segments[idx - 1] if idx > 0 else None
        setattr(seg, "step4_ai_selected", bool(ai_allocation_map.get(seg.segment_key, False)))
        setattr(seg, "step4_ai_allocation_map", ai_allocation_map)

        logger.info(f"  处理素材 [{seg.index}/{len(manifest.segments)}]: {seg.text[:30]}...")
        resolve_asset_for_segment(
            segment=seg,
            project_root=project_root,
            generated_dir=generated_dir,
            library_dir=library_dir,
            pexels_api_key=pexels_api_key,
            enable_pexels_video=enable_pexels_video,
            enable_pexels_photo=enable_pexels_photo,
            enable_ai_image=enable_ai_image,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            pixelle_default_workflow=manifest.pixelle_default_workflow,
            pixelle_segment_overrides=manifest.pixelle_segment_overrides,
            vendor_preference=manifest.vendor_preference,
            continuity_policy=manifest.continuity_policy,
            continuity_seed=manifest.continuity_seed,
            style_id=manifest.style_id,
            project_id=manifest.project_id,
            previous_segment=previous_segment,
            material_mode=manifest.material_mode,
        )

        if seg.asset_refs:
            kind = seg.asset_refs[0].kind
            source_counts[kind] = source_counts.get(kind, 0) + 1
            
            seg_ai_selected = bool(getattr(seg, "step4_ai_selected", False))
            if seg_ai_selected:
                cap_telemetry_ai_selected_count += 1
                if kind in AI_GENERATED_ROUTES:
                    cap_telemetry_ai_routed_count += 1
                else:
                    cap_telemetry_non_ai_replacement_count += 1

        processed += 1

    # 输出素材来源统计
    logger.info(f"Step 4 完成: 处理 {processed} 段，跳过 {skipped} 段")
    if source_counts:
        logger.info("  素材来源统计:")
        label_map = {
            "pdf_chart":    "① PDF 图表",
            "pexels_video": "② Pexels 视频",
            "pexels_photo": "③ Pexels 图片",
            "ai_image":     "④ AI 生成",
            "template":     "⑤ 模板兜底",
            "cached":       "⓪ 缓存复用",
        }
        for kind, count in sorted(source_counts.items()):
            logger.info(f"    {label_map.get(kind, kind)}: {count} 段")

    total_segments = len(manifest.segments)
    ai_selected_in_map = sum(1 for v in ai_allocation_map.values() if v)
    ai_skipped_over_cap = total_segments - ai_selected_in_map
    cap_telemetry = {
        "total_segments": total_segments,
        "ai_selected_count": ai_selected_in_map,
        "ai_skipped_over_cap_count": ai_skipped_over_cap,
        "ai_routed_count": cap_telemetry_ai_routed_count,
        "non_ai_replacement_count": cap_telemetry_non_ai_replacement_count,
        "processed": processed,
        "skipped": skipped,
        "source_counts": dict(source_counts),
    }
    setattr(manifest, "step4_cap_telemetry", cap_telemetry)

    logger.info("  AI Cap Policy 统计:")
    logger.info(f"    total_segments={cap_telemetry['total_segments']}")
    logger.info(f"    ai_selected_count={cap_telemetry['ai_selected_count']}")
    logger.info(f"    ai_skipped_over_cap_count={cap_telemetry['ai_skipped_over_cap_count']}")
    logger.info(f"    ai_routed_count={cap_telemetry['ai_routed_count']}")
    logger.info(f"    non_ai_replacement_count={cap_telemetry['non_ai_replacement_count']}")

    os.makedirs(os.path.dirname(output_manifest), exist_ok=True)
    manifest.save(output_manifest)
    logger.info(f"Manifest 已更新: {output_manifest}")
    return manifest
