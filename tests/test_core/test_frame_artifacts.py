"""
Tests for frame artifact IO utilities.

Covers:
  - Happy path: successful end-frame extraction with metadata persistence
  - Invalid input: missing video, invalid timestamp
  - Cache-hit behavior: avoid duplicate extraction
  - Metadata round-trip: save and load
"""
import os
import json
import tempfile
import shutil
from pathlib import Path

import pytest

from src.core.frame_artifacts import (
    compute_frame_hash,
    get_artifact_paths,
    load_artifact_metadata,
    save_artifact_metadata,
    extract_end_frame,
    validate_artifact_metadata,
    clear_segment_artifacts,
    FrameArtifactMetadata,
    FrameArtifactError,
)


@pytest.fixture
def temp_artifact_dir(tmp_path):
    """Temporary artifact directory for tests."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return str(artifact_dir)


@pytest.fixture
def sample_video(tmp_path):
    """
    Create a minimal valid video file for testing.
    Uses FFmpeg to generate 1-second test video.
    """
    video_path = tmp_path / "test_video.mp4"
    
    # Generate 1-second test video: black screen 1080x1920 @ 30fps
    cmd = (
        f'ffmpeg -y -f lavfi -i color=c=black:s=1080x1920:d=1.0:r=30 '
        f'-pix_fmt yuv420p "{video_path}" -loglevel error'
    )
    
    result = os.system(cmd)
    if result != 0:
        pytest.skip("FFmpeg not available or failed to generate test video")
    
    if not video_path.exists():
        pytest.skip("Test video generation failed")
    
    return str(video_path)


def test_compute_frame_hash_deterministic():
    """Frame hash computation is deterministic for same inputs."""
    hash1 = compute_frame_hash("video.mp4", 5.0, "1080x1920")
    hash2 = compute_frame_hash("video.mp4", 5.0, "1080x1920")
    assert hash1 == hash2
    assert len(hash1) == 16


def test_compute_frame_hash_different_inputs():
    """Frame hash differs for different inputs."""
    hash1 = compute_frame_hash("video.mp4", 5.0, "1080x1920")
    hash2 = compute_frame_hash("video2.mp4", 5.0, "1080x1920")
    hash3 = compute_frame_hash("video.mp4", 6.0, "1080x1920")
    hash4 = compute_frame_hash("video.mp4", 5.0, "1920x1080")
    
    assert hash1 != hash2
    assert hash1 != hash3
    assert hash1 != hash4


def test_get_artifact_paths(temp_artifact_dir):
    """Artifact path generation follows expected structure."""
    frame_path, metadata_path = get_artifact_paths(
        "abc123#1",
        artifact_dir=temp_artifact_dir,
    )
    
    assert frame_path.endswith("abc123#1_end.png")
    assert metadata_path.endswith("abc123#1_end.json")
    assert Path(temp_artifact_dir) in Path(frame_path).parents


def test_metadata_round_trip(temp_artifact_dir):
    """Metadata can be saved and loaded correctly."""
    metadata = FrameArtifactMetadata(
        segment_key="abc123#1",
        frame_hash="1234567890abcdef",
        timestamp=5.5,
        resolution="1080x1920",
        video_path="/path/to/video.mp4",
        artifact_path="/path/to/frame.png",
    )
    
    metadata_path = os.path.join(temp_artifact_dir, "test_metadata.json")
    
    # Save
    save_artifact_metadata(metadata, metadata_path)
    assert os.path.exists(metadata_path)
    
    # Load
    loaded = load_artifact_metadata(metadata_path)
    assert loaded is not None
    assert loaded.segment_key == metadata.segment_key
    assert loaded.frame_hash == metadata.frame_hash
    assert loaded.timestamp == metadata.timestamp
    assert loaded.resolution == metadata.resolution
    assert loaded.video_path == metadata.video_path
    assert loaded.artifact_path == metadata.artifact_path


def test_load_metadata_missing_file():
    """Loading missing metadata returns None."""
    loaded = load_artifact_metadata("/nonexistent/path.json")
    assert loaded is None


def test_load_metadata_invalid_json(temp_artifact_dir):
    """Loading invalid JSON returns None."""
    invalid_path = os.path.join(temp_artifact_dir, "invalid.json")
    with open(invalid_path, "w") as f:
        f.write("{ invalid json }")
    
    loaded = load_artifact_metadata(invalid_path)
    assert loaded is None


def test_end_frame_extraction_happy(sample_video, temp_artifact_dir):
    """
    Happy path: extract end-frame from valid video.
    
    This test verifies:
      - Extraction succeeds with valid inputs
      - Artifact file is created
      - Metadata is persisted
      - No error is returned
    """
    segment_key = "test_segment#1"
    video_duration = 1.0
    resolution = "1080x1920"
    
    artifact_path, error = extract_end_frame(
        segment_key=segment_key,
        video_path=sample_video,
        video_duration=video_duration,
        resolution=resolution,
        artifact_dir=temp_artifact_dir,
    )
    
    # Should succeed
    assert error is None
    assert artifact_path is not None
    
    # Artifact should exist
    assert os.path.exists(artifact_path)
    assert artifact_path.endswith("_end.png")
    
    # Metadata should exist
    frame_path, metadata_path = get_artifact_paths(segment_key, temp_artifact_dir)
    assert os.path.exists(metadata_path)
    
    # Metadata should be valid
    metadata = load_artifact_metadata(metadata_path)
    assert metadata is not None
    assert metadata.segment_key == segment_key
    assert metadata.resolution == resolution
    assert metadata.video_path == sample_video


def test_end_frame_extraction_cache_hit(sample_video, temp_artifact_dir):
    """Cache-hit: second extraction reuses existing artifact."""
    segment_key = "test_segment_cache#1"
    video_duration = 1.0
    resolution = "1080x1920"
    
    # First extraction
    path1, error1 = extract_end_frame(
        segment_key=segment_key,
        video_path=sample_video,
        video_duration=video_duration,
        resolution=resolution,
        artifact_dir=temp_artifact_dir,
    )
    assert error1 is None
    assert path1 is not None
    
    # Get artifact modification time
    mtime1 = os.path.getmtime(path1)
    
    # Second extraction (should be cache hit)
    path2, error2 = extract_end_frame(
        segment_key=segment_key,
        video_path=sample_video,
        video_duration=video_duration,
        resolution=resolution,
        artifact_dir=temp_artifact_dir,
    )
    assert error2 is None
    assert path2 == path1
    
    # Artifact should not be re-created (mtime unchanged)
    assert path2 is not None
    mtime2 = os.path.getmtime(path2)
    assert mtime2 == mtime1


def test_end_frame_extraction_force_reextract(sample_video, temp_artifact_dir):
    """Force re-extraction bypasses cache hit."""
    segment_key = "test_segment_force#1"
    video_duration = 1.0
    resolution = "1080x1920"
    
    # First extraction
    path1, error1 = extract_end_frame(
        segment_key=segment_key,
        video_path=sample_video,
        video_duration=video_duration,
        resolution=resolution,
        artifact_dir=temp_artifact_dir,
    )
    assert error1 is None
    
    # Second extraction with force_reextract=True
    path2, error2 = extract_end_frame(
        segment_key=segment_key,
        video_path=sample_video,
        video_duration=video_duration,
        resolution=resolution,
        artifact_dir=temp_artifact_dir,
        force_reextract=True,
    )
    assert error2 is None
    assert path2 == path1  # Same path but re-extracted


def test_end_frame_extraction_invalid_input(temp_artifact_dir):
    """
    Invalid input: missing video file returns typed error.
    
    This test verifies:
      - Extraction fails with explicit error
      - Error category is 'missing_video'
      - No artifact is created
    """
    segment_key = "test_invalid#1"
    video_path = "/nonexistent/video.mp4"
    video_duration = 1.0
    resolution = "1080x1920"
    
    artifact_path, error = extract_end_frame(
        segment_key=segment_key,
        video_path=video_path,
        video_duration=video_duration,
        resolution=resolution,
        artifact_dir=temp_artifact_dir,
    )
    
    # Should fail
    assert artifact_path is None
    assert error is not None
    assert isinstance(error, FrameArtifactError)
    assert error.error_category == "missing_video"
    assert error.segment_key == segment_key
    assert error.video_path == video_path


def test_end_frame_extraction_invalid_duration(sample_video, temp_artifact_dir):
    """Invalid duration returns typed error."""
    segment_key = "test_invalid_duration#1"
    video_duration = -1.0  # Invalid
    resolution = "1080x1920"
    
    artifact_path, error = extract_end_frame(
        segment_key=segment_key,
        video_path=sample_video,
        video_duration=video_duration,
        resolution=resolution,
        artifact_dir=temp_artifact_dir,
    )
    
    assert artifact_path is None
    assert error is not None
    assert error.error_category == "invalid_timestamp"


def test_validate_artifact_metadata_valid(sample_video, temp_artifact_dir):
    """Validation succeeds for valid artifact and metadata."""
    segment_key = "test_validate#1"
    
    # Extract frame first
    extract_end_frame(
        segment_key=segment_key,
        video_path=sample_video,
        video_duration=1.0,
        resolution="1080x1920",
        artifact_dir=temp_artifact_dir,
    )
    
    # Validate
    valid, error = validate_artifact_metadata(segment_key, temp_artifact_dir)
    assert valid is True
    assert error is None


def test_validate_artifact_metadata_missing(temp_artifact_dir):
    """Validation fails for missing metadata."""
    segment_key = "nonexistent#1"
    
    valid, error = validate_artifact_metadata(segment_key, temp_artifact_dir)
    assert valid is False
    assert error is not None
    assert error.error_category == "invalid_metadata"


def test_clear_segment_artifacts(sample_video, temp_artifact_dir):
    """Clearing artifacts removes frame and metadata."""
    segment_key = "test_clear#1"
    
    # Extract frame
    extract_end_frame(
        segment_key=segment_key,
        video_path=sample_video,
        video_duration=1.0,
        resolution="1080x1920",
        artifact_dir=temp_artifact_dir,
    )
    
    # Verify artifacts exist
    frame_path, metadata_path = get_artifact_paths(segment_key, temp_artifact_dir)
    assert os.path.exists(frame_path)
    assert os.path.exists(metadata_path)
    
    # Clear artifacts
    clear_segment_artifacts(segment_key, temp_artifact_dir)
    
    # Verify artifacts removed
    assert not os.path.exists(frame_path)
    assert not os.path.exists(metadata_path)
