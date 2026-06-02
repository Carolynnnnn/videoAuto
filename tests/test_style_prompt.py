"""
T13 Selector Tests: Style Prompt Coverage

Test selectors:
- test_style_prompt_deterministic: stable hash from same inputs
- test_style_bible_invalid: validation error on bad payload
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.models import VisualPlan, StyleBible


def test_style_prompt_deterministic():
    """
    Deterministic prompt composition: same inputs => same plan_hash.
    Uses VisualPlan.compute_plan_hash with fixed style inputs.
    """
    plan = VisualPlan(
        type="ai_video_short",
        keywords=["sunset", "ocean"],
        prompt="A serene coastal sunset with gentle waves",
        pixelle_workflow="i2v",
    )
    
    style_fields = "9:16|1080x1920|v1"
    hash1 = plan.compute_plan_hash(style_fields)
    hash2 = plan.compute_plan_hash(style_fields)
    
    assert hash1 == hash2, "Same inputs must produce same hash"
    assert len(hash1) == 16, "Hash should be 16 hex chars"


def test_style_bible_invalid():
    """
    Invalid style bible payload is rejected with explicit error.
    Test: character_anchors must be dict, not list.
    """
    try:
        # Invalid: character_anchors should be dict, not list
        StyleBible.from_dict({
            "tone": "cinematic",
            "palette": "vibrant",
            "camera_grammar": "dynamic",
            "character_anchors": ["invalid", "list", "type"],  # Wrong type
        })
        assert False, "Should have raised TypeError for invalid character_anchors"
    except TypeError as e:
        # Python dataclass will raise TypeError for wrong type
        assert "character_anchors" in str(e) or "dict" in str(e).lower()


if __name__ == "__main__":
    test_style_prompt_deterministic()
    test_style_bible_invalid()
    print("✓ Both selector tests passed")
