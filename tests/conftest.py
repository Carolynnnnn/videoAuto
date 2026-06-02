"""
Pytest configuration for videoauto test suite.

This conftest ensures repo root modules (pixelle_snapshot, src) are discoverable
without external PYTHONPATH manipulation.
"""
import sys
from pathlib import Path

import pytest

# Add repository root to sys.path to make pixelle_snapshot and src importable
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


@pytest.fixture(autouse=True)
def reset_pixelle_reliability_controls(monkeypatch):
    """
    Reset Pixelle reliability controls before each test to prevent circuit breaker
    state contamination between tests.
    
    The circuit breaker is a module-level singleton in step4_assets.py that accumulates
    failure records. Tests that simulate Pixelle failures can open the circuit, causing
    subsequent tests to see PIXELLE_CIRCUIT_OPEN errors.
    """
    try:
        from src.steps.pixelle_reliability_controls import (
            PixelleReliabilityControls,
            ReliabilityConfig,
        )
        # Create a fresh instance with default config for each test
        fresh_controls = PixelleReliabilityControls(ReliabilityConfig())
        monkeypatch.setattr(
            "src.steps.step4_assets._pixelle_reliability_controls",
            fresh_controls,
        )
    except ImportError:
        # Module not available in all test contexts; skip silently
        pass
