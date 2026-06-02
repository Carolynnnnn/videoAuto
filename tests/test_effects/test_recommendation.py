from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.effects.recommendation import (
    EffectPlan,
    map_to_assets,
    parse_llm_response,
    recommend_effects,
)


FIXTURE_CASES = Path("tests/fixtures/llm_test_cases/tc001-tc010.json")


def _mock_openai_client(payload: str) -> Any:
    message = MagicMock()
    message.content = payload
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


def test_parse_llm_response_parses_valid_json() -> None:
    raw = json.dumps(
        {
            "recommendations": [
                {
                    "segment_index": 0,
                    "sentiment": "positive",
                    "effect_type": "joyful",
                    "reason": "celebration tone",
                    "asset_hint": "party",
                },
                {
                    "segment_index": 1,
                    "sentiment": "negative",
                    "effect_type": "calming",
                    "reason": "comfort mood",
                    "asset_hint": "breathe",
                },
            ]
        }
    )

    plans = parse_llm_response(raw, ["Great news!", "This is hard."], "assets/builtin_library/index.json")

    assert [p.sentiment for p in plans] == ["positive", "negative"]
    assert [p.effect_type for p in plans] == ["joyful", "calming"]
    assert all(isinstance(p, EffectPlan) for p in plans)


def test_parse_llm_response_accepts_markdown_fenced_json() -> None:
    raw = """```json
{\"recommendations\":[{\"segment_index\":0,\"sentiment\":\"neutral\",\"effect_type\":\"subtle\"}]}
```"""

    plans = parse_llm_response(raw, ["We review the facts."], "assets/builtin_library/index.json")

    assert len(plans) == 1
    assert plans[0].sentiment == "neutral"
    assert plans[0].effect_type == "subtle"


def test_parse_llm_response_invalid_json_returns_fallback_defaults() -> None:
    plans = parse_llm_response("invalid json{", ["I am so happy!", "I feel terrible", "Statement only"], "assets/builtin_library/index.json")

    assert [p.sentiment for p in plans] == ["positive", "negative", "neutral"]
    assert [p.effect_type for p in plans] == ["joyful", "calming", "subtle"]


def test_map_to_assets_matches_sentiment_to_builtin_library(tmp_path: Path) -> None:
    library = {
        "stickers": [
            {"id": "joy_001", "filename": "joy.gif", "sentiment": "happy", "keywords": ["joy"]},
            {"id": "calm_001", "filename": "calm.gif", "sentiment": "sad", "keywords": ["calm"]},
            {"id": "neutral_001", "filename": "neutral.gif", "sentiment": "neutral", "keywords": ["neutral"]},
        ]
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(library), encoding="utf-8")

    plans = [
        EffectPlan(segment_index=0, text="good", sentiment="positive", effect_type="joyful", reason="", asset_id="", asset_tags=[]),
        EffectPlan(segment_index=1, text="bad", sentiment="negative", effect_type="calming", reason="", asset_id="", asset_tags=[]),
        EffectPlan(segment_index=2, text="ok", sentiment="neutral", effect_type="subtle", reason="", asset_id="", asset_tags=[]),
    ]

    mapped = map_to_assets(plans, str(index_path))

    assert [p.asset_id for p in mapped] == ["joy_001", "calm_001", "neutral_001"]


def test_recommend_effects_uses_single_batch_api_call() -> None:
    payload = json.dumps(
        {
            "recommendations": [
                {"segment_index": 0, "sentiment": "positive", "effect_type": "joyful"},
                {"segment_index": 1, "sentiment": "negative", "effect_type": "calming"},
            ]
        }
    )
    client = _mock_openai_client(payload)

    plans = recommend_effects(
        ["I got promoted today!", "I am worried about tomorrow."],
        client=client,
        prompt_path="src/effects/prompts/effect_recommendation.txt",
        schema_path="src/effects/schemas/effect_recommendation.json",
        library_path="assets/builtin_library/index.json",
    )

    assert len(plans) == 2
    assert client.chat.completions.create.call_count == 1


def test_task6_ten_cases_sentiment_and_effect_alignment() -> None:
    cases = json.loads(FIXTURE_CASES.read_text(encoding="utf-8"))

    plans = parse_llm_response("{bad-json", [c["text"] for c in cases], "assets/builtin_library/index.json")

    assert len(plans) == 10
    for plan, case in zip(plans, cases):
        assert plan.sentiment == case["expected_sentiment"]
        if plan.sentiment == "positive":
            assert plan.effect_type == "joyful"
        elif plan.sentiment == "negative":
            assert plan.effect_type == "calming"
        else:
            assert plan.effect_type == "subtle"
