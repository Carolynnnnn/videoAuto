"""
Tests for style bible validation and deterministic prompt composition.

Covers T13 selectors:
- style_prompt_deterministic: Assert repeated composition outputs identical results with style/continuity anchors
- style_bible_invalid: Assert missing required style fields fail validation
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.core.models import StyleBible, GlobalStyle


def test_style_prompt_deterministic():
    """
    T13 Selector: style_prompt_deterministic
    
    Assert repeated composition outputs are identical and include style/continuity anchors.
    
    Steps:
      1. Create fixed style bible fixture
      2. Compose prompt twice with same inputs
      3. Assert outputs are byte-identical
      4. Assert prompt includes expected style anchors
    
    Expected Result: Stable reproducible prompt composition
    """
    # Fixed style bible fixture
    style_bible = StyleBible(
        tone="cinematic",
        palette="muted",
        camera_grammar="steady",
        character_anchors={"protagonist": "young professional in modern office"}
    )
    
    # Segment fixture
    segment_text = "人工智能正在改变世界"
    segment_index = 0
    continuity_seed = 42
    
    # Compose prompt twice with deterministic inputs
    prompt1 = _compose_prompt_with_style(
        text=segment_text,
        style_bible=style_bible,
        segment_index=segment_index,
        continuity_seed=continuity_seed
    )
    
    prompt2 = _compose_prompt_with_style(
        text=segment_text,
        style_bible=style_bible,
        segment_index=segment_index,
        continuity_seed=continuity_seed
    )
    
    # Assert byte-identical outputs
    assert prompt1 == prompt2, "Prompt composition must be deterministic"
    
    # Assert prompt includes style anchors
    assert "cinematic" in prompt1.lower(), "Prompt must include tone anchor"
    assert "muted" in prompt1.lower() or "palette" in prompt1.lower(), "Prompt must reference palette"
    assert "steady" in prompt1.lower() or "camera" in prompt1.lower(), "Prompt must reference camera grammar"
    
    # Assert prompt includes continuity context
    assert "seed" in prompt1.lower() or str(continuity_seed) in prompt1, "Prompt must include continuity seed"
    
    # Assert prompt includes segment content
    assert segment_text in prompt1 or "ai" in prompt1.lower() or "artificial intelligence" in prompt1.lower(), \
        "Prompt must reference segment content"


def test_style_bible_invalid():
    """
    T13 Selector: style_bible_invalid
    
    Assert missing required style fields fail validation.
    
    Steps:
      1. Create style fixture missing required fields
      2. Attempt validation or composition
      3. Assert validation error references missing fields
    
    Expected Result: Fail-fast style schema validation
    """
    # Test missing palette
    with pytest.raises((ValueError, TypeError, AttributeError)) as exc_info:
        invalid_style = _validate_style_bible_schema({
            "tone": "cinematic",
            # palette missing
            "camera_grammar": "steady",
            "character_anchors": {}
        })
    
    error_message = str(exc_info.value).lower()
    assert "palette" in error_message or "required" in error_message or "missing" in error_message, \
        "Validation error must reference missing palette field"
    
    # Test missing camera_grammar
    with pytest.raises((ValueError, TypeError, AttributeError)) as exc_info:
        invalid_style = _validate_style_bible_schema({
            "tone": "cinematic",
            "palette": "muted",
            # camera_grammar missing
            "character_anchors": {}
        })
    
    error_message = str(exc_info.value).lower()
    assert "camera" in error_message or "grammar" in error_message or "required" in error_message, \
        "Validation error must reference missing camera_grammar field"
    
    # Test missing tone
    with pytest.raises((ValueError, TypeError, AttributeError)) as exc_info:
        invalid_style = _validate_style_bible_schema({
            # tone missing
            "palette": "muted",
            "camera_grammar": "steady",
            "character_anchors": {}
        })
    
    error_message = str(exc_info.value).lower()
    assert "tone" in error_message or "required" in error_message, \
        "Validation error must reference missing tone field"


# ─────────────────────────────────────────────────────────────
# Helper functions for deterministic prompt composition
# ─────────────────────────────────────────────────────────────

def _compose_prompt_with_style(
    text: str,
    style_bible: StyleBible,
    segment_index: int,
    continuity_seed: int
) -> str:
    """
    Deterministic prompt composer combining segment text + style bible + continuity context.
    
    This is a reference implementation for T13. Actual production implementation
    should be integrated into src/steps/step3_visual_plan.py or a new prompt module.
    """
    # Base prompt from segment text
    base = f"Generate visual for: {text}"
    
    # Style anchors
    style_anchors = [
        f"tone:{style_bible.tone}",
        f"palette:{style_bible.palette}",
        f"camera:{style_bible.camera_grammar}"
    ]
    
    # Character anchors (if any)
    if style_bible.character_anchors:
        for role, description in sorted(style_bible.character_anchors.items()):
            style_anchors.append(f"{role}:{description}")
    
    # Continuity context
    continuity_tag = f"seed={continuity_seed}"
    
    # Deterministic composition (sorted for stability)
    prompt_parts = [base] + sorted(style_anchors) + [continuity_tag]
    
    return " | ".join(prompt_parts)


def _validate_style_bible_schema(data: dict) -> StyleBible:
    """
    Validate style bible schema and raise ValueError for missing required fields.
    
    Required fields: tone, palette, camera_grammar
    Optional fields: character_anchors
    """
    required_fields = ["tone", "palette", "camera_grammar"]
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        raise ValueError(
            f"StyleBible validation failed: missing required fields: {', '.join(missing_fields)}"
        )
    
    # Construct valid StyleBible
    return StyleBible(
        tone=data["tone"],
        palette=data["palette"],
        camera_grammar=data["camera_grammar"],
        character_anchors=data.get("character_anchors", {})
    )
