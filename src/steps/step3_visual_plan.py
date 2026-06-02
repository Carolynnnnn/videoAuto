"""
Step 3：字幕段 → 镜头/素材计划（Visual Plan 自动生成，v2）

关键变更：
  - 使用 plan_hash = hash(visual_plan 关键字段 + global_style 素材相关字段)
  - 缓存 key 改为 plan_hash（不再是 md5(text|style_version)）
  - 支持只处理指定 segment_keys（增量更新时只处理 TEXT changed + ADDED）
  - plan_hash 不变 → 直接复用旧 asset_refs，不重新生成素材
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

from openai import OpenAI

from src.core.models import Manifest, Segment, VisualPlan, MotionConfig, OverlayItem
from src.utils.logger import get_logger

logger = get_logger("step3_visual_plan")


SEMANTIC_KEYWORD_RULES = [
    (("ai", "人工智能", "chatgpt", "sora", "大模型", "智能体"), ["artificial intelligence", "machine learning", "digital technology"]),
    (("自动驾驶", "无人驾驶", "autonomous", "self driving"), ["self driving car", "autonomous vehicle", "smart traffic"]),
    (("医疗", "医院", "诊断", "healthcare", "medical"), ["medical diagnosis", "hospital doctor", "healthcare technology"]),
    (("教育", "学习", "学校", "education", "learning"), ["classroom learning", "education technology", "online study"]),
    (("金融", "投资", "银行", "finance", "investment"), ["financial market", "business analytics", "investment strategy"]),
    (("机器人", "robot", "automation", "自动化"), ["robot automation", "industrial robot", "smart factory"]),
    (("数据", "图表", "统计", "报告", "data", "chart", "report"), ["data analytics", "business chart", "statistics report"]),
]

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "about", "your", "you", "are", "not",
    "have", "has", "will", "can", "our", "out", "all", "new", "more", "less", "than", "been",
}


def _extract_emphasis_tokens(plan_dict: Dict[str, Any]) -> List[str]:
    tokens: List[str] = []
    for key in ("subtitle_emphasis", "emphasis", "emphasis_words", "emphasis_phrases"):
        raw = plan_dict.get(key)
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str) and item.strip():
                    tokens.append(item.strip())
                elif isinstance(item, dict):
                    token = item.get("text") or item.get("phrase") or item.get("word")
                    if isinstance(token, str) and token.strip():
                        tokens.append(token.strip())
        elif isinstance(raw, str) and raw.strip():
            tokens.extend([part.strip() for part in raw.split(",") if part.strip()])

    return _normalize_keywords(tokens, max_count=6)


def _normalize_keywords(raw_keywords: List[str], max_count: int = 5) -> List[str]:
    result: List[str] = []
    seen = set()
    for raw in raw_keywords:
        if not isinstance(raw, str):
            continue
        kw = re.sub(r"\s+", " ", raw.strip().lower())
        kw = kw.replace("_", " ")
        if not kw or kw in seen:
            continue
        seen.add(kw)
        result.append(kw)
        if len(result) >= max_count:
            break
    return result


def _extract_semantic_keywords(text: str, prev_text: str = "", next_text: str = "", max_count: int = 5) -> List[str]:
    corpus = f"{prev_text} {text} {next_text}".lower()
    keywords: List[str] = []

    for triggers, mapped_keywords in SEMANTIC_KEYWORD_RULES:
        if any(trigger in corpus for trigger in triggers):
            keywords.extend(mapped_keywords)

    if re.search(r"\d+|百分之|%|万亿|增长|下降", corpus):
        keywords.extend(["data dashboard", "growth chart"])

    ascii_terms = re.findall(r"[a-zA-Z]{3,}", corpus)
    keywords.extend(ascii_terms[:2])

    normalized = _normalize_keywords(keywords, max_count=max_count)
    if normalized:
        return normalized
    return ["technology background", "digital abstract", "modern business"]


# ─────────────────────────────────────────────
# Sticker overlay generation (deterministic)
# ─────────────────────────────────────────────
DEFAULT_STICKER_PATHS = [
    "assets/stickers/default.gif",
    ".sisyphus/evidence/task-7-sticker.gif",
]


def _get_fallback_sticker_path() -> Optional[str]:
    """Return first available sticker asset path from fallback list."""
    for path in DEFAULT_STICKER_PATHS:
        if Path(path).exists():
            return str(Path(path).resolve())
    return None


def _should_add_sticker(segment_index: int, segment_text: str) -> bool:
    """
    Deterministic rule for adding sticker overlays.
    Rule: Add sticker every 3rd segment AND segment has sufficient content.
    For ASCII: 2+ words. For CJK: 4+ characters.
    No randomness - fully reproducible.
    """
    if (segment_index + 1) % 3 != 0:
        return False
    
    # Check for CJK characters (Chinese/Japanese/Korean)
    text = segment_text.strip()
    cjk_count = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff' or '\u3040' <= ch <= '\u30ff' or '\uac00' <= ch <= '\ud7af')
    
    if cjk_count >= 4:
        # CJK text: 4+ CJK characters is sufficient
        return True
    
    # ASCII fallback: 2+ words
    word_count = len(text.split())
    return word_count >= 2


def _generate_sticker_overlay(
    segment_index: int,
    segment_text: str,
    segment_duration: float,
) -> Optional[OverlayItem]:
    """
    Generate a sticker overlay for qualifying segments.
    
    Returns OverlayItem with kind='sticker' or None if not applicable.
    Sticker timing STRICTLY bounded by segment duration.
    """
    if not _should_add_sticker(segment_index, segment_text):
        return None
    
    sticker_path = _get_fallback_sticker_path()
    if not sticker_path:
        logger.debug(f"  [seg {segment_index}] No sticker asset available, skipping")
        return None
    
    # Skip stickers for very short segments (< 0.5s)
    if segment_duration < 0.5:
        logger.debug(f"  [seg {segment_index}] Segment too short for sticker ({segment_duration:.2f}s)")
        return None
    
    # Deterministic anchor selection based on segment index
    anchors = ["center", "top-right", "bottom-left", "top-left", "bottom-right"]
    anchor = anchors[segment_index % len(anchors)]
    
    # Sticker timing: 10% start offset, 80% duration, STRICT segment bound
    start_offset = segment_duration * 0.1
    available_duration = segment_duration - start_offset
    sticker_duration = min(segment_duration * 0.8, available_duration)
    
    # Minimum duration 0.3s, but NEVER exceed segment bound
    sticker_duration = max(0.3, min(sticker_duration, segment_duration - start_offset))
    
    return OverlayItem(
        kind="sticker",
        target="video",
        strength=0.8,
        extra={
            "asset_path": sticker_path,
            "anchor": anchor,
            "scale": 0.25,
            "transparency": 0.9,
            "start_time": start_offset,
            "duration": sticker_duration,
        },
    )


def _build_asset_prompt(base_prompt: str, segment_text: str, visual_type: str, keywords: List[str]) -> str:
    if isinstance(base_prompt, str) and base_prompt.strip():
        return base_prompt.strip()

    keyword_text = ", ".join(keywords[:3])
    if visual_type == "pdf_chart":
        return f"clean data visualization, infographic style, {keyword_text}, vertical 9:16"
    if visual_type == "kinetic_text":
        return f"minimal background with strong typography, {keyword_text}, vertical 9:16"
    if visual_type == "broll":
        return f"cinematic b-roll scene about {segment_text[:60]}, {keyword_text}, vertical 9:16"
    return f"high quality visual about {segment_text[:60]}, {keyword_text}, vertical 9:16"

# ─────────────────────────────────────────────
# LLM 提示词模板
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """你是一位专业的短视频导演和视觉策划师。
你的任务是为每一条字幕文本生成一个结构化的"视觉计划"（visual_plan），用于指导视频制作。

输出必须是严格的 JSON 格式，包含以下字段：
{
  "type": "画面类型（从以下选择：pdf_chart/broll/ai_image/kinetic_text/template）",
  "keywords": ["搜索关键词1", "搜索关键词2", "搜索关键词3"],
  "prompt": "AI 图片生成提示词（英文，简洁描述画面内容和风格）",
  "motion": {
    "preset": "镜头运动（soft_kenburns/push_in/pan_left/pan_right/zoom_out/static）",
    "speed": 0.8
  },
  "overlay": [
    {"kind": "highlight/arrow/text", "target": "center/top/bottom", "strength": 0.6}
  ],
  "subtitle_emphasis": ["需要强调的词或短语（可选，最多3个）"]
}

**字幕分段与长度要求**：
- 每段字幕应保持短小精悍，理想长度在 40 个字符以内。
- 如果输入字幕过长，请确保视觉规划（prompt/keywords）聚焦于最核心的视觉主体。

**画面类型选择规则**：
- pdf_chart：字幕涉及数据、图表、统计、报告内容。
- broll：字幕涉及真实场景、人物、活动、地点、具体事件。
- ai_image：字幕涉及概念、抽象、情感、品牌、隐喻。
- kinetic_text：字幕是口号、标题、数字强调、纯文字展示。
- template：通用背景+字幕（兜底选项）。

**关键词生成要求**：
- keywords 必须精准匹配字幕内容的视觉场景，使用英文。
- 例如"人工智能"→["artificial intelligence", "AI technology", "machine learning"]
- 例如"自动驾驶"→["self driving car", "autonomous vehicle", "road traffic"]
- 关键词应描述：主体场景 + 动作/状态 + 画面风格。
- 优先选择 Pexels 等素材库常见的描述词。

**字幕强调规则（subtitle_emphasis）**：
- 仅在有明显重点词时输出，最多 3 个。
- **硬性约束**：每个词必须是 1-6 个字符的短词。
- **精确匹配**：必须是原字幕文本中的精确子串，严禁改写、翻译或编造。
- **去噪去重**：严禁输出重复词，严禁输出无意义的虚词（如：的、了、和、是）。

视频风格：科技感、简洁、9:16 竖屏，适合微信视频号/抖音。
"""

USER_PROMPT_TEMPLATE = """字幕文本："{text}"
时长：{duration:.1f}秒
上下文（前一句）："{prev_text}"
上下文（后一句）："{next_text}"

请生成该字幕的 visual_plan JSON："""


# ─────────────────────────────────────────────
# 缓存管理（以 plan_hash 为 key）
# ─────────────────────────────────────────────
def _load_plan_cache(cache_dir: str, plan_hash: str) -> Optional[Dict[str, Any]]:
    path = Path(cache_dir) / f"{plan_hash}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_plan_cache(cache_dir: str, plan_hash: str, plan_dict: Dict[str, Any]) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = Path(cache_dir) / f"{plan_hash}.json"
    path.write_text(json.dumps(plan_dict, ensure_ascii=False, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────
# 单段 Visual Plan 生成
# ─────────────────────────────────────────────
def generate_visual_plan_for_segment(
    segment: Segment,
    global_style_asset_fields: str = "",
    prev_text: str = "",
    next_text: str = "",
    cache_dir: Optional[str] = None,
    llm_model: str = "deepseek-chat",
) -> tuple[VisualPlan, str]:
    """
    为单个 Segment 生成 VisualPlan（带 plan_hash 缓存）。

    返回 (VisualPlan, plan_hash)
    """
    # 先尝试用 content_key 查找缓存（基于文本内容的缓存）
    text_cache_key = segment.content_key
    if cache_dir:
        cached = _load_plan_cache(cache_dir, text_cache_key)
        if cached:
            vp = VisualPlan.from_dict(cached)
            plan_hash = vp.compute_plan_hash(global_style_asset_fields)
            logger.debug(f"  [seg {segment.index}] 命中 plan 缓存: {text_cache_key[:12]}")
            return vp, plan_hash

    # 调用 LLM（支持 DeepSeek / OpenAI 兼容接口）
    from src.core.api_config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
    _api_key = os.environ.get("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY)
    _base_url = DEEPSEEK_BASE_URL
    _model = llm_model if llm_model not in ("gpt-4.1-mini", "gpt-4o-mini", None, "") else DEEPSEEK_MODEL
    client = OpenAI(api_key=_api_key, base_url=_base_url)
    user_msg = USER_PROMPT_TEMPLATE.format(
        text=segment.text,
        duration=segment.duration,
        prev_text=prev_text,
        next_text=next_text,
    )

    try:
        response = client.chat.completions.create(
            model=_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        raw_content = response.choices[0].message.content or "{}"
        plan_dict = json.loads(raw_content)
        logger.debug(f"  [seg {segment.index}] LLM 生成 visual_plan: type={plan_dict.get('type')}")
    except Exception as e:
        logger.warning(f"  [seg {segment.index}] LLM 调用失败: {e}，使用 template 兜底")
        plan_dict = {
            "type": "template",
            "keywords": segment.text[:20].split(),
            "prompt": f"clean background with text: {segment.text[:50]}",
            "motion": {"preset": "static", "speed": 1.0},
            "overlay": [],
        }

    # 构建 VisualPlan 对象
    motion = MotionConfig(
        preset=plan_dict.get("motion", {}).get("preset", "soft_kenburns"),
        speed=plan_dict.get("motion", {}).get("speed", 0.8),
    )
    overlay = [
        OverlayItem(**o) for o in plan_dict.get("overlay", [])
        if isinstance(o, dict) and "kind" in o
    ]
    emphasis_tokens = _extract_emphasis_tokens(plan_dict)
    if emphasis_tokens:
        overlay.append(
            OverlayItem(
                kind="subtitle_emphasis",
                target="subtitle",
                strength=0.8,
                extra={"tokens": emphasis_tokens},
            )
        )
    # Add sticker overlay for qualifying segments (deterministic rule)
    sticker_overlay = _generate_sticker_overlay(
        segment_index=segment.index,
        segment_text=segment.text,
        segment_duration=segment.duration,
    )
    if sticker_overlay:
        overlay.append(sticker_overlay)
        logger.debug(f"  [seg {segment.index}] Added sticker overlay: anchor={sticker_overlay.extra.get('anchor')}")

    llm_keywords = plan_dict.get("keywords", [])
    semantic_keywords = _extract_semantic_keywords(segment.text, prev_text=prev_text, next_text=next_text)
    merged_keywords = _normalize_keywords((llm_keywords if isinstance(llm_keywords, list) else []) + semantic_keywords)

    vp = VisualPlan(
        type=plan_dict.get("type", "template"),
        keywords=merged_keywords,
        prompt=_build_asset_prompt(
            base_prompt=plan_dict.get("prompt", ""),
            segment_text=segment.text,
            visual_type=plan_dict.get("type", "template"),
            keywords=merged_keywords,
        ),
        motion=motion,
        overlay=overlay,
    )

    # 计算 plan_hash
    plan_hash = vp.compute_plan_hash(global_style_asset_fields)

    # 保存缓存（以 content_key 为 key，方便相同文本复用）
    if cache_dir:
        _save_plan_cache(cache_dir, text_cache_key, vp.to_dict())

    return vp, plan_hash


# ─────────────────────────────────────────────
# 批量生成（支持只处理指定 segment_keys）
# ─────────────────────────────────────────────
def run_step3(
    manifest: Manifest,
    output_manifest: str,
    cache_dir: Optional[str] = None,
    target_segment_keys: Optional[List[str]] = None,
    llm_model: str = "gpt-4.1-mini",
) -> Manifest:
    """
    执行 Step 3：为 Manifest 中的 Segment 生成 Visual Plan（v2）

    :param manifest: 输入 Manifest
    :param output_manifest: 更新后 manifest.json 输出路径
    :param cache_dir: 缓存目录（可选）
    :param target_segment_keys: 只处理这些 segment_key（None=全部）
    :param llm_model: LLM 模型名
    :return: 更新后的 Manifest
    """
    logger.info("=" * 50)
    logger.info("Step 3: 字幕段 → Visual Plan 生成 (v2)")
    logger.info(f"  LLM 模型: {llm_model}")

    global_style_asset_fields = manifest.global_style.asset_related_fields()
    segments = manifest.segments

    target_keys = set(target_segment_keys) if target_segment_keys is not None else None
    if target_keys is not None:
        manifest_keys = {seg.segment_key for seg in segments}
        unknown_keys = sorted(target_keys - manifest_keys)
        if unknown_keys:
            joined = ", ".join(unknown_keys)
            raise ValueError(f"Invalid target_segment_keys for Step3: {joined}")

    processed = 0
    skipped = 0

    for i, seg in enumerate(segments):
        # 如果指定了目标 keys，只处理这些
        if target_keys is not None and seg.segment_key not in target_keys:
            skipped += 1
            continue

        # 全量模式下，已有 visual_plan 的跳过
        if target_keys is None and seg.visual_plan is not None:
            skipped += 1
            continue

        prev_text = segments[i - 1].text if i > 0 else ""
        next_text = segments[i + 1].text if i < len(segments) - 1 else ""

        logger.info(f"  生成 visual_plan [{i+1}/{len(segments)}]: {seg.text[:30]}...")
        vp, plan_hash = generate_visual_plan_for_segment(
            segment=seg,
            global_style_asset_fields=global_style_asset_fields,
            prev_text=prev_text,
            next_text=next_text,
            cache_dir=cache_dir,
            llm_model=llm_model,
        )
        seg.visual_plan = vp
        seg.plan_hash = plan_hash
        processed += 1

    logger.info(f"Step 3 完成: 处理 {processed} 段, 跳过 {skipped} 段")

    os.makedirs(os.path.dirname(output_manifest), exist_ok=True)
    manifest.save(output_manifest)
    logger.info(f"Manifest 已更新: {output_manifest}")

    return manifest
