from pathlib import Path

import pytest

from pixelle_snapshot.adapters.contracts import ValidationError
from pixelle_snapshot.vendors import (
    ContinuityRequirements,
    build_runtime_capability_matrix,
    deserialize_runtime_capability_matrix,
    load_vendor_contract_file,
    serialize_runtime_capability_matrix,
)


FIXTURES_DIR = Path(__file__).parent.parent / "pixelle_snapshot" / "vendors" / "fixtures"


def test_capability_matrix_build_round_trip_includes_required_flags() -> None:
    contracts = [
        load_vendor_contract_file(FIXTURES_DIR / "future_vendor_placeholder_contract.json"),
        load_vendor_contract_file(FIXTURES_DIR / "minimax_media_contract.json"),
    ]

    runtime = build_runtime_capability_matrix(contracts)
    serialized = serialize_runtime_capability_matrix(runtime)
    deserialized = deserialize_runtime_capability_matrix(serialized)

    assert runtime == deserialized
    assert set(serialized.keys()) == {"future_vendor:media", "minimax:media"}

    future_vendor = serialized["future_vendor:media"]
    minimax = serialized["minimax:media"]

    assert future_vendor["supports_end_frame"] is True
    assert future_vendor["supports_seed"] is True
    assert future_vendor["supports_reference_image"] is True
    assert future_vendor["supports_reference_video"] is True
    assert future_vendor["max_duration"] == 30.0

    assert minimax["supports_end_frame"] is False
    assert minimax["supports_seed"] is True
    assert minimax["supports_reference_image"] is True
    assert minimax["supports_reference_video"] is False
    assert minimax["max_duration"] == 10.0


def test_capability_matrix_invalid_continuity_rejects_unsupported_end_frame() -> None:
    contracts = [load_vendor_contract_file(FIXTURES_DIR / "minimax_media_contract.json")]
    runtime = build_runtime_capability_matrix(contracts)

    requirements = ContinuityRequirements(
        require_end_frame=True,
        require_reference_image=True,
    )

    with pytest.raises(ValidationError) as exc_info:
        requirements.validate_against(runtime["minimax:media"])

    error = exc_info.value
    assert error.details.get("field") == "require_end_frame"
    assert error.details.get("reason_code") == "CAPABILITY_MATRIX_UNSUPPORTED_CONTINUITY"


def test_capability_matrix_invalid_continuity_rejects_invalid_requirement_combo() -> None:
    requirements = ContinuityRequirements(require_end_frame=True)

    with pytest.raises(ValidationError) as exc_info:
        requirements.validate()

    error = exc_info.value
    assert error.details.get("field") == "require_end_frame"
    assert error.details.get("reason_code") == "CAPABILITY_MATRIX_INVALID_CONTINUITY_COMBINATION"
