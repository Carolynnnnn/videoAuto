import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from pixelle_snapshot.adapters.broker import VendorBroker
from pixelle_snapshot.adapters.digital_human import DigitalHumanAdapter
from pixelle_snapshot.adapters.contracts import DigitalHumanRequest, ErrorCategory
from pixelle_snapshot import test_doubles

def test_broker_test_mode_routing(tmp_path):
    # Setup
    test_doubles.enable_test_mode()
    try:
        raw_adapter = DigitalHumanAdapter()
        broker = VendorBroker("digital_human", raw_adapter, vendor_preference="test")
        
        request = DigitalHumanRequest(
            segment_key="test_seg_001#1",
            segment_text="Hello from digital human",
            segment_duration=2.0,
            project_root=str(tmp_path),
            output_dir=str(tmp_path / "output"),
            avatar_id="default_avatar",
            voice_id="default_voice",
        )
        
        # Execute
        response = broker.invoke(request)
        
        # Verify
        assert response.success is True
        assert response.output_path is not None
        assert "test_mode" in response.metadata or response.metadata.get("test_mode") is True or response.metadata.get("deterministic") is True
    finally:
        test_doubles.disable_test_mode()

def test_broker_unknown_vendor(tmp_path):
    # Setup
    raw_adapter = DigitalHumanAdapter()
    broker = VendorBroker("digital_human", raw_adapter, vendor_preference="unknown_vendor")
    
    request = DigitalHumanRequest(
        segment_key="test_seg_001#1",
        segment_text="Hello from digital human",
        segment_duration=2.0,
        project_root=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        avatar_id="default_avatar",
        voice_id="default_voice",
    )
    
    # Execute
    response = broker.invoke(request)
    
    # Verify
    assert response.success is False
    assert response.error is not None
    assert response.error.category == ErrorCategory.VALIDATION
    assert "Unknown vendor" in response.error.message
    assert response.error.details.get("reason_code") == "UNKNOWN_VENDOR"
