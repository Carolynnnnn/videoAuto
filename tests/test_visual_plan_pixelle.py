"""
Tests for VisualPlan Pixelle workflow extensions (T3)

Validates:
- Pixelle segment types (pixelle_digital_human, pixelle_i2v, pixelle_action_transfer)
- pixelle_workflow field serialization/deserialization
- Backward compatibility with existing non-Pixelle types
- Validation of workflow values
- plan_hash includes pixelle_workflow when present
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.core.models import VisualPlan, MotionConfig


class TestPixelleSegmentTypes:
    """Test new pixelle_* type literals in VisualPlan"""

    def test_pixelle_digital_human_type(self):
        plan = VisualPlan(type="pixelle_digital_human", pixelle_workflow="digital_human")
        assert plan.type == "pixelle_digital_human"
        assert plan.pixelle_workflow == "digital_human"

    def test_pixelle_i2v_type(self):
        plan = VisualPlan(type="pixelle_i2v", pixelle_workflow="i2v")
        assert plan.type == "pixelle_i2v"
        assert plan.pixelle_workflow == "i2v"

    def test_pixelle_action_transfer_type(self):
        plan = VisualPlan(type="pixelle_action_transfer", pixelle_workflow="action_transfer")
        assert plan.type == "pixelle_action_transfer"
        assert plan.pixelle_workflow == "action_transfer"


class TestPixelleWorkflowField:
    """Test pixelle_workflow field behavior"""

    def test_workflow_field_optional(self):
        plan = VisualPlan(type="template")
        assert plan.pixelle_workflow is None

    def test_workflow_field_with_value(self):
        plan = VisualPlan(type="pixelle_digital_human", pixelle_workflow="digital_human")
        assert plan.pixelle_workflow == "digital_human"

    def test_workflow_serialization(self):
        plan = VisualPlan(type="pixelle_i2v", pixelle_workflow="i2v", keywords=["test"])
        d = plan.to_dict()
        assert d["type"] == "pixelle_i2v"
        assert d["pixelle_workflow"] == "i2v"
        assert d["keywords"] == ["test"]

    def test_workflow_deserialization(self):
        data = {
            "type": "pixelle_action_transfer",
            "pixelle_workflow": "action_transfer",
            "keywords": ["motion", "transfer"],
            "prompt": "test prompt",
        }
        plan = VisualPlan.from_dict(data)
        assert plan.type == "pixelle_action_transfer"
        assert plan.pixelle_workflow == "action_transfer"
        assert plan.keywords == ["motion", "transfer"]
        assert plan.prompt == "test prompt"

    def test_workflow_deserialization_missing_field(self):
        data = {"type": "template", "keywords": []}
        plan = VisualPlan.from_dict(data)
        assert plan.type == "template"
        assert plan.pixelle_workflow is None


class TestBackwardCompatibility:
    """Ensure existing non-Pixelle segment types remain valid"""

    def test_existing_types_unchanged(self):
        from typing import get_args, Literal
        
        type_hints = get_args(VisualPlan.__dataclass_fields__['type'].type)
        existing_types = [t for t in type_hints if not t.startswith("pixelle_")]
        
        for type_str in existing_types:
            plan = VisualPlan(type=type_str)
            assert plan.type == type_str
            assert plan.pixelle_workflow is None

    def test_legacy_manifest_parsing(self):
        legacy_data = {
            "type": "ai_image",
            "keywords": ["technology", "innovation"],
            "prompt": "a futuristic cityscape",
            "motion": {"preset": "soft_kenburns", "speed": 0.8},
        }
        plan = VisualPlan.from_dict(legacy_data)
        assert plan.type == "ai_image"
        assert plan.keywords == ["technology", "innovation"]
        assert plan.prompt == "a futuristic cityscape"
        assert plan.pixelle_workflow is None

    def test_round_trip_legacy_plan(self):
        original = VisualPlan(
            type="broll",
            keywords=["nature", "landscape"],
            prompt="beautiful mountain scene",
        )
        serialized = original.to_dict()
        deserialized = VisualPlan.from_dict(serialized)
        assert deserialized.type == original.type
        assert deserialized.keywords == original.keywords
        assert deserialized.prompt == original.prompt
        assert deserialized.pixelle_workflow is None


class TestPlanHashWithWorkflow:
    """Test plan_hash computation includes pixelle_workflow"""

    def test_plan_hash_includes_workflow(self):
        plan1 = VisualPlan(
            type="pixelle_digital_human",
            pixelle_workflow="digital_human",
            keywords=["speaker"],
        )
        plan2 = VisualPlan(
            type="pixelle_digital_human",
            pixelle_workflow="i2v",
            keywords=["speaker"],
        )
        hash1 = plan1.compute_plan_hash()
        hash2 = plan2.compute_plan_hash()
        assert hash1 != hash2

    def test_plan_hash_with_none_workflow(self):
        plan1 = VisualPlan(type="template", keywords=["test"])
        plan2 = VisualPlan(type="template", keywords=["test"], pixelle_workflow=None)
        hash1 = plan1.compute_plan_hash()
        hash2 = plan2.compute_plan_hash()
        assert hash1 == hash2

    def test_plan_hash_different_workflows(self):
        base_kwargs = {"type": "pixelle_digital_human", "keywords": ["test"]}
        plan_dh = VisualPlan(**base_kwargs, pixelle_workflow="digital_human")
        plan_i2v = VisualPlan(**base_kwargs, pixelle_workflow="i2v")
        plan_at = VisualPlan(**base_kwargs, pixelle_workflow="action_transfer")

        hash_dh = plan_dh.compute_plan_hash()
        hash_i2v = plan_i2v.compute_plan_hash()
        hash_at = plan_at.compute_plan_hash()

        assert hash_dh != hash_i2v
        assert hash_i2v != hash_at
        assert hash_dh != hash_at


class TestWorkflowValidation:
    """Test workflow field validation"""

    def test_valid_workflow_digital_human(self):
        plan = VisualPlan(type="template", pixelle_workflow="digital_human")
        assert plan.pixelle_workflow == "digital_human"

    def test_valid_workflow_i2v(self):
        plan = VisualPlan(type="template", pixelle_workflow="i2v")
        assert plan.pixelle_workflow == "i2v"
    
    def test_valid_workflow_action_transfer(self):
        plan = VisualPlan(type="template", pixelle_workflow="action_transfer")
        assert plan.pixelle_workflow == "action_transfer"

    def test_workflow_none_allowed(self):
        plan = VisualPlan(type="template", pixelle_workflow=None)
        assert plan.pixelle_workflow is None
    
    def test_invalid_workflow_rejected(self):
        invalid_data = {
            "type": "template",
            "pixelle_workflow": "unknown_workflow"
        }
        with pytest.raises(ValueError) as exc_info:
            VisualPlan.from_dict(invalid_data)
        assert "Invalid pixelle_workflow" in str(exc_info.value)
        assert "unknown_workflow" in str(exc_info.value)
        assert "digital_human" in str(exc_info.value)


class TestRoundTripPixellePlans:
    """Test full serialization/deserialization cycles for Pixelle plans"""

    def test_digital_human_round_trip(self):
        original = VisualPlan(
            type="pixelle_digital_human",
            pixelle_workflow="digital_human",
            keywords=["presenter", "avatar"],
            prompt="professional speaker in business attire",
        )
        d = original.to_dict()
        restored = VisualPlan.from_dict(d)
        assert restored.type == original.type
        assert restored.pixelle_workflow == original.pixelle_workflow
        assert restored.keywords == original.keywords
        assert restored.prompt == original.prompt

    def test_i2v_round_trip(self):
        original = VisualPlan(
            type="pixelle_i2v",
            pixelle_workflow="i2v",
            keywords=["animation", "motion"],
            prompt="animate static image with smooth camera movement",
            asset_path="/path/to/source.png",
        )
        d = original.to_dict()
        restored = VisualPlan.from_dict(d)
        assert restored.type == original.type
        assert restored.pixelle_workflow == original.pixelle_workflow
        assert restored.asset_path == original.asset_path

    def test_action_transfer_round_trip(self):
        original = VisualPlan(
            type="pixelle_action_transfer",
            pixelle_workflow="action_transfer",
            keywords=["dance", "motion", "transfer"],
            prompt="transfer dance moves to target character",
        )
        d = original.to_dict()
        restored = VisualPlan.from_dict(d)
        assert restored.type == original.type
        assert restored.pixelle_workflow == original.pixelle_workflow
        assert restored.keywords == original.keywords
