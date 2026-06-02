from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.core.api_config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


POSITIVE_HINTS = {
    "happy",
    "joy",
    "great",
    "awesome",
    "love",
    "excited",
    "excellent",
    "wonderful",
    "amazing",
    "开心",
    "高兴",
    "太棒",
    "喜悦",
    "喜欢",
    "成功",
}

NEGATIVE_HINTS = {
    "sad",
    "terrible",
    "bad",
    "worried",
    "angry",
    "pain",
    "upset",
    "fear",
    "stress",
    "难过",
    "糟糕",
    "担心",
    "焦虑",
    "失败",
    "痛苦",
}

SENTIMENT_ALIASES = {
    "happy": "positive",
    "excited": "positive",
    "love": "positive",
    "surprise": "positive",
    "surprised": "positive",
    "joy": "positive",
    "sad": "negative",
    "angry": "negative",
    "fear": "negative",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}


@dataclass
class EffectPlan:
    segment_index: int
    text: str
    sentiment: str
    effect_type: str
    reason: str = ""
    asset_id: str = ""
    asset_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "text": self.text,
            "sentiment": self.sentiment,
            "effect_type": self.effect_type,
            "reason": self.reason,
            "asset_id": self.asset_id,
            "asset_tags": self.asset_tags,
        }


def _normalize_sentiment(value: str, text: str) -> str:
    normalized = SENTIMENT_ALIASES.get((value or "").strip().lower())
    if normalized:
        return normalized
    return infer_sentiment(text)


def infer_sentiment(text: str) -> str:
    lowered = text.lower()
    positive_score = sum(1 for token in POSITIVE_HINTS if token in lowered)
    negative_score = sum(1 for token in NEGATIVE_HINTS if token in lowered)
    if positive_score > negative_score:
        return "positive"
    if negative_score > positive_score:
        return "negative"
    return "neutral"


def _default_effect_for_sentiment(sentiment: str) -> str:
    if sentiment == "positive":
        return "joyful"
    if sentiment == "negative":
        return "calming"
    return "subtle"


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = text.removesuffix("```").strip()
    return text


def _extract_json_payload(raw_response: str) -> Any:
    stripped = _strip_json_fence(raw_response)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def parse_llm_response(raw_response: str, segments: list[str], library_path: str) -> list[EffectPlan]:
    try:
        payload = _extract_json_payload(raw_response)
        rows = payload.get("recommendations", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("LLM response does not contain recommendation list")
    except Exception:
        rows = []

    indexed_rows: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        idx = row.get("segment_index")
        if isinstance(idx, int) and 0 <= idx < len(segments):
            indexed_rows[idx] = row

    plans: list[EffectPlan] = []
    for idx, text in enumerate(segments):
        row = indexed_rows.get(idx, {})
        sentiment = _normalize_sentiment(str(row.get("sentiment", "")), text)
        effect_type = str(row.get("effect_type") or _default_effect_for_sentiment(sentiment)).strip().lower()
        if not effect_type:
            effect_type = _default_effect_for_sentiment(sentiment)
        plans.append(
            EffectPlan(
                segment_index=idx,
                text=text,
                sentiment=sentiment,
                effect_type=effect_type,
                reason=str(row.get("reason", "")).strip(),
            )
        )

    return map_to_assets(plans, library_path)


def _read_library(library_path: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(Path(library_path).read_text(encoding="utf-8"))
        stickers = data.get("stickers", []) if isinstance(data, dict) else []
        return stickers if isinstance(stickers, list) else []
    except Exception:
        return []


def _sentiment_targets(sentiment: str) -> list[str]:
    if sentiment == "positive":
        return ["happy", "excited", "love", "surprised", "generic"]
    if sentiment == "negative":
        return ["sad", "angry", "generic"]
    return ["neutral", "generic", "happy"]


def _pick_asset_id(stickers: list[dict[str, Any]], sentiment: str, effect_type: str) -> tuple[str, list[str]]:
    targets = _sentiment_targets(sentiment)
    for target in targets:
        for sticker in stickers:
            sticker_sentiment = str(sticker.get("sentiment", "")).lower()
            keywords = [str(k).lower() for k in sticker.get("keywords", []) if isinstance(k, str)]
            if sticker_sentiment != target:
                continue
            if effect_type == "joyful" and any(k in {"joy", "celebrate", "happy", "party"} for k in keywords):
                return str(sticker.get("id", "")), keywords
            if effect_type == "calming" and any(k in {"calm", "comfort", "sad", "breathe"} for k in keywords):
                return str(sticker.get("id", "")), keywords
            if effect_type == "subtle":
                return str(sticker.get("id", "")), keywords
    for sticker in stickers:
        asset_id = str(sticker.get("id", ""))
        if asset_id:
            keywords = [str(k).lower() for k in sticker.get("keywords", []) if isinstance(k, str)]
            return asset_id, keywords
    return "", []


def map_to_assets(plans: list[EffectPlan], library_path: str) -> list[EffectPlan]:
    stickers = _read_library(library_path)
    mapped: list[EffectPlan] = []
    for plan in plans:
        asset_id, tags = _pick_asset_id(stickers, plan.sentiment, plan.effect_type)
        plan.asset_id = asset_id
        plan.asset_tags = tags
        mapped.append(plan)
    return mapped


def _load_prompt_and_schema(prompt_path: str, schema_path: str) -> tuple[str, str]:
    prompt = Path(prompt_path).read_text(encoding="utf-8")
    schema = Path(schema_path).read_text(encoding="utf-8")
    return prompt, schema


def recommend_effects(
    segments: list[str],
    client: Any | None = None,
    prompt_path: str = "src/effects/prompts/effect_recommendation.txt",
    schema_path: str = "src/effects/schemas/effect_recommendation.json",
    library_path: str = "assets/builtin_library/index.json",
    llm_model: str = "gpt-4.1-mini",
) -> list[EffectPlan]:
    if not segments:
        return []

    prompt_template, schema_text = _load_prompt_and_schema(prompt_path, schema_path)
    payload = {
        "segments": [{"segment_index": i, "text": text} for i, text in enumerate(segments)],
        "schema": json.loads(schema_text),
    }
    user_message = (
        f"{prompt_template.strip()}\n\n"
        "Return JSON only.\n"
        f"Input payload:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

    if client is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY)
        client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    model = llm_model if llm_model not in {"", None, "gpt-4.1-mini", "gpt-4o-mini"} else DEEPSEEK_MODEL

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an effect recommendation engine."},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
    except Exception:
        raw = "{bad-json"

    return parse_llm_response(raw, segments, library_path)
