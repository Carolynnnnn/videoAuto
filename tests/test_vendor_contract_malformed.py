from pathlib import Path

import pytest

from pixelle_snapshot.adapters.contracts import ValidationError
from pixelle_snapshot.vendors import load_vendor_contract_file


FIXTURES_DIR = Path(__file__).parent.parent / "pixelle_snapshot" / "vendors" / "fixtures"


def test_vendor_contract_malformed_missing_auth_spec_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        load_vendor_contract_file(FIXTURES_DIR / "malformed_missing_auth_contract.json")

    error = exc_info.value
    assert error.category.value == "VALIDATION"
    assert error.details.get("field") == "auth"
    assert error.details.get("reason_code") == "VENDOR_CONTRACT_MISSING_AUTH_SPEC"
    assert "auth spec is required" in error.message
