import json
from pathlib import Path

from pixelle_snapshot.vendors import LIFECYCLE_CAPABILITIES, load_vendor_contract, load_vendor_contract_file


FIXTURES_DIR = Path(__file__).parent.parent / "pixelle_snapshot" / "vendors" / "fixtures"


def test_vendor_contract_normalization_minimax_voice_fixture() -> None:
    normalized = load_vendor_contract_file(FIXTURES_DIR / "minimax_voice_contract.json")
    payload = normalized.to_dict()

    assert payload["vendor_id"] == "minimax"
    assert payload["domain"] == "voice"
    assert list(payload["capabilities"].keys()) == list(LIFECYCLE_CAPABILITIES)
    assert payload["capabilities"] == {
        "submit": True,
        "poll": True,
        "fetch": True,
        "cancel": False,
        "sync": True,
    }
    assert payload["endpoints"]["sync"] == {"method": "POST", "path": "/v1/t2a_v2"}
    assert payload["endpoints"]["submit"] == {"method": "POST", "path": "/v1/t2a_async_v2"}
    assert payload["endpoints"]["poll"] == {"method": "GET", "path": "/v1/query/t2a_async_query_v2"}
    assert payload["endpoints"]["fetch"] == {"method": "GET", "path": "/v1/files/retrieve"}


def test_vendor_contract_normalization_minimax_media_fixture() -> None:
    normalized = load_vendor_contract_file(FIXTURES_DIR / "minimax_media_contract.json")
    payload = normalized.to_dict()

    assert payload["vendor_id"] == "minimax"
    assert payload["domain"] == "media"
    assert payload["capabilities"] == {
        "submit": True,
        "poll": True,
        "fetch": True,
        "cancel": False,
        "sync": False,
    }
    assert payload["endpoints"]["submit"]["path"] == "/v1/video_generation"
    assert payload["endpoints"]["poll"]["path"] == "/v1/query/video_generation"
    assert payload["endpoints"]["fetch"]["path"] == "/v1/files/retrieve"


def test_vendor_contract_normalization_future_vendor_placeholder() -> None:
    normalized = load_vendor_contract_file(FIXTURES_DIR / "future_vendor_placeholder_contract.json")
    payload = normalized.to_dict()

    assert payload["vendor_id"] == "future_vendor"
    assert payload["capabilities"] == {
        "submit": True,
        "poll": True,
        "fetch": True,
        "cancel": True,
        "sync": False,
    }
    assert payload["endpoints"]["cancel"] == {
        "method": "POST",
        "path": "/v1/jobs/{job_id}/cancel",
    }


def test_vendor_contract_normalization_deterministic_output_order() -> None:
    source_path = FIXTURES_DIR / "minimax_media_contract.json"
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    normalized_once = load_vendor_contract(source_payload).to_dict()
    normalized_twice = load_vendor_contract(source_payload).to_dict()

    assert normalized_once == normalized_twice
    assert list(normalized_once["capabilities"].keys()) == list(LIFECYCLE_CAPABILITIES)
