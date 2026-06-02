from pathlib import Path

pytest = __import__("pytest")

from pixelle_snapshot.adapters.artifact_validation import validate_downloaded_artifact
from pixelle_snapshot.adapters.contracts import ErrorCategory


def _mp4_header_bytes() -> bytes:
    return b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"


def test_validate_downloaded_artifact_accepts_valid_mp4_with_duration_metadata(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.mp4"
    artifact.write_bytes(_mp4_header_bytes())

    result = validate_downloaded_artifact(
        file_path=str(artifact),
        artifact_url="https://cdn.example/artifact.mp4",
        artifact_payload={"duration": 3.2},
    )

    assert result.artifact_bytes == len(_mp4_header_bytes())
    assert result.artifact_format == "mp4"
    assert result.artifact_duration == 3.2


def test_validate_downloaded_artifact_rejects_invalid_signature(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.mp4"
    artifact.write_bytes(b"not-video-content")

    with pytest.raises(Exception) as exc:
        validate_downloaded_artifact(
            file_path=str(artifact),
            artifact_url="https://cdn.example/artifact.mp4",
            artifact_payload={},
        )

    assert getattr(exc.value, "category", None) == ErrorCategory.EXECUTION
    assert getattr(exc.value, "details", {}).get("reason_code") == "PIXELLE_ARTIFACT_CORRUPTED"


def test_validate_downloaded_artifact_rejects_missing_duration(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.mp4"
    artifact.write_bytes(_mp4_header_bytes())

    with pytest.raises(Exception) as exc:
        validate_downloaded_artifact(
            file_path=str(artifact),
            artifact_url="https://cdn.example/artifact.mp4",
            artifact_payload={},
        )

    assert getattr(exc.value, "category", None) == ErrorCategory.EXECUTION
    assert getattr(exc.value, "details", {}).get("reason_code") == "PIXELLE_ARTIFACT_INVALID_DURATION"
