"""
Frame Artifact IO Utilities: End-frame extraction and metadata persistence.

Provides vendor-agnostic utilities for extracting segment end-frame images
and persisting metadata (segment key, hash, timestamp, resolution) with
cache-hit behavior to avoid duplicate extraction when reusable.

Key features:
  - Extract end-frame from rendered segment video
  - Persist metadata as JSON sidecar (segment_key, frame_hash, timestamp, resolution)
  - Cache-hit: Skip extraction if metadata hash matches and artifact exists
  - Typed failure handling for invalid/missing inputs
  - Consumable by continuity policy adapters (Task 14/15/19)

Artifact path structure:
  artifacts/continuity/frames/{segment_key}_end.png
  artifacts/continuity/frames/{segment_key}_end.json
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple, Literal, Dict, Any

from src.utils.logger import get_logger

logger = get_logger("frame_artifacts")


# Default artifact directory for continuity frames
DEFAULT_ARTIFACT_DIR = "artifacts/continuity/frames"


@dataclass
class FrameArtifactError:
    """Typed error descriptor for frame artifact failures."""
    error_category: Literal[
        "missing_video",
        "invalid_timestamp",
        "extraction_failed",
        "invalid_metadata",
        "io_error",
    ]
    message: str
    segment_key: Optional[str] = None
    video_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_category": self.error_category,
            "message": self.message,
            "segment_key": self.segment_key,
            "video_path": self.video_path,
        }


@dataclass
class FrameArtifactMetadata:
    """
    Metadata descriptor for extracted end-frame artifact.
    
    Fields:
        segment_key: Unique segment identifier (content_key + occurrence)
        frame_hash: Hash of extraction inputs (video_path + timestamp + resolution)
        timestamp: Extraction timestamp within video (seconds)
        resolution: Frame resolution as "WxH" (e.g., "1080x1920")
        video_path: Source video path
        artifact_path: Output frame image path
    """
    segment_key: str
    frame_hash: str
    timestamp: float
    resolution: str
    video_path: str
    artifact_path: str
    metadata_version: str = "v1"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameArtifactMetadata":
        return cls(
            segment_key=data["segment_key"],
            frame_hash=data["frame_hash"],
            timestamp=data["timestamp"],
            resolution=data["resolution"],
            video_path=data["video_path"],
            artifact_path=data["artifact_path"],
            metadata_version=data.get("metadata_version", "v1"),
        )


def compute_frame_hash(
    video_path: str,
    timestamp: float,
    resolution: str,
) -> str:
    """
    Compute deterministic hash for frame extraction inputs.
    
    Used for cache-hit detection: same inputs → same hash → reuse artifact.
    
    Args:
        video_path: Source video path
        timestamp: Extraction timestamp (seconds)
        resolution: Target resolution "WxH"
        
    Returns:
        16-char hex hash
    """
    raw = f"{video_path}|{timestamp:.3f}|{resolution}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def get_artifact_paths(
    segment_key: str,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
) -> Tuple[str, str]:
    """
    Get artifact paths for segment end-frame.
    
    Args:
        segment_key: Segment identifier
        artifact_dir: Base directory for artifacts
        
    Returns:
        (frame_path, metadata_path) tuple:
            - frame_path: PNG image path
            - metadata_path: JSON sidecar path
    """
    base_dir = Path(artifact_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    frame_path = base_dir / f"{segment_key}_end.png"
    metadata_path = base_dir / f"{segment_key}_end.json"
    
    return str(frame_path), str(metadata_path)


def load_artifact_metadata(
    metadata_path: str,
) -> Optional[FrameArtifactMetadata]:
    """
    Load artifact metadata from JSON sidecar.
    
    Args:
        metadata_path: Path to JSON metadata file
        
    Returns:
        FrameArtifactMetadata if valid, None if missing/invalid
    """
    if not os.path.exists(metadata_path):
        return None
    
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return FrameArtifactMetadata.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Invalid metadata at {metadata_path}: {e}")
        return None


def save_artifact_metadata(
    metadata: FrameArtifactMetadata,
    metadata_path: str,
) -> None:
    """
    Persist artifact metadata to JSON sidecar.
    
    Args:
        metadata: Metadata descriptor
        metadata_path: Output JSON path
    """
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, ensure_ascii=False, indent=2)
    logger.debug(f"Saved metadata: {metadata_path}")


def extract_end_frame(
    segment_key: str,
    video_path: str,
    video_duration: float,
    resolution: str,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    force_reextract: bool = False,
    timeout: int = 10,
) -> Tuple[Optional[str], Optional[FrameArtifactError]]:
    """
    Extract end-frame from segment video with cache-hit behavior.
    
    Extraction strategy:
      1. Compute frame_hash from inputs (video_path, timestamp, resolution)
      2. Check existing metadata: if hash matches + artifact exists → cache hit
      3. Otherwise: extract frame via FFmpeg, persist metadata
      4. Return artifact path or typed error
    
    Args:
        segment_key: Unique segment identifier
        video_path: Path to rendered segment video
        video_duration: Total duration (seconds) for end-frame timestamp
        resolution: Target resolution "WxH" (e.g., "1080x1920")
        artifact_dir: Base directory for artifacts
        force_reextract: Skip cache-hit check and force new extraction
        timeout: FFmpeg extraction timeout (seconds)
        
    Returns:
        (artifact_path, error) tuple:
            - artifact_path: PNG path if successful
            - error: FrameArtifactError if failed
    """
    # Validate inputs
    if not os.path.exists(video_path):
        return None, FrameArtifactError(
            error_category="missing_video",
            message=f"Video path does not exist: {video_path}",
            segment_key=segment_key,
            video_path=video_path,
        )
    
    if video_duration <= 0:
        return None, FrameArtifactError(
            error_category="invalid_timestamp",
            message=f"Invalid video duration: {video_duration}",
            segment_key=segment_key,
            video_path=video_path,
        )
    
    # Compute extraction timestamp (end-frame: 1 frame before end)
    # Use duration - 0.1s to avoid edge-case end-of-stream issues
    end_timestamp = max(0.0, video_duration - 0.1)
    
    # Compute frame hash for cache-hit detection
    frame_hash = compute_frame_hash(video_path, end_timestamp, resolution)
    
    # Get artifact paths
    frame_path, metadata_path = get_artifact_paths(segment_key, artifact_dir)
    
    # Check cache hit
    if not force_reextract:
        existing_metadata = load_artifact_metadata(metadata_path)
        if existing_metadata is not None:
            if (
                existing_metadata.frame_hash == frame_hash
                and os.path.exists(frame_path)
            ):
                logger.debug(
                    f"Cache hit for segment {segment_key}: "
                    f"frame_hash={frame_hash}, path={frame_path}"
                )
                return frame_path, None
    
    # Extract frame via FFmpeg
    logger.info(
        f"Extracting end-frame for segment {segment_key}: "
        f"timestamp={end_timestamp:.3f}s, resolution={resolution}"
    )
    
    # Ensure parent directory exists
    Path(frame_path).parent.mkdir(parents=True, exist_ok=True)
    
    # FFmpeg command: extract single frame at end_timestamp
    # Use -ss before -i for fast seeking
    cmd = (
        f'ffmpeg -y -ss {end_timestamp:.3f} -i "{video_path}" '
        f'-frames:v 1 -q:v 2 "{frame_path}" -loglevel error'
    )
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        if result.returncode != 0:
            return None, FrameArtifactError(
                error_category="extraction_failed",
                message=f"FFmpeg extraction failed: {result.stderr}",
                segment_key=segment_key,
                video_path=video_path,
            )
        
        if not os.path.exists(frame_path):
            return None, FrameArtifactError(
                error_category="extraction_failed",
                message=f"Extraction succeeded but artifact missing: {frame_path}",
                segment_key=segment_key,
                video_path=video_path,
            )
        
    except subprocess.TimeoutExpired:
        return None, FrameArtifactError(
            error_category="extraction_failed",
            message=f"Extraction timeout after {timeout}s",
            segment_key=segment_key,
            video_path=video_path,
        )
    except Exception as e:
        return None, FrameArtifactError(
            error_category="io_error",
            message=f"Extraction exception: {str(e)}",
            segment_key=segment_key,
            video_path=video_path,
        )
    
    # Persist metadata
    metadata = FrameArtifactMetadata(
        segment_key=segment_key,
        frame_hash=frame_hash,
        timestamp=end_timestamp,
        resolution=resolution,
        video_path=video_path,
        artifact_path=frame_path,
    )
    
    try:
        save_artifact_metadata(metadata, metadata_path)
    except Exception as e:
        logger.warning(f"Failed to save metadata for {segment_key}: {e}")
    
    logger.info(
        f"End-frame extracted: segment={segment_key}, "
        f"frame_hash={frame_hash}, path={frame_path}"
    )
    
    return frame_path, None


def validate_artifact_metadata(
    segment_key: str,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
) -> Tuple[bool, Optional[FrameArtifactError]]:
    """
    Validate artifact metadata and frame existence.
    
    Args:
        segment_key: Segment identifier
        artifact_dir: Base directory for artifacts
        
    Returns:
        (valid, error) tuple:
            - valid: True if metadata and artifact both exist and valid
            - error: FrameArtifactError if validation failed
    """
    frame_path, metadata_path = get_artifact_paths(segment_key, artifact_dir)
    
    # Check metadata existence
    if not os.path.exists(metadata_path):
        return False, FrameArtifactError(
            error_category="invalid_metadata",
            message=f"Metadata missing: {metadata_path}",
            segment_key=segment_key,
        )
    
    # Load and validate metadata
    metadata = load_artifact_metadata(metadata_path)
    if metadata is None:
        return False, FrameArtifactError(
            error_category="invalid_metadata",
            message=f"Metadata invalid or corrupted: {metadata_path}",
            segment_key=segment_key,
        )
    
    # Check frame artifact existence
    if not os.path.exists(frame_path):
        return False, FrameArtifactError(
            error_category="io_error",
            message=f"Frame artifact missing: {frame_path}",
            segment_key=segment_key,
        )
    
    # Validate frame path matches metadata
    if metadata.artifact_path != frame_path:
        return False, FrameArtifactError(
            error_category="invalid_metadata",
            message=(
                f"Metadata artifact_path mismatch: "
                f"expected={frame_path}, got={metadata.artifact_path}"
            ),
            segment_key=segment_key,
        )
    
    return True, None


def clear_segment_artifacts(
    segment_key: str,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
) -> None:
    """
    Remove all artifacts for a segment (frame + metadata).
    
    Args:
        segment_key: Segment identifier
        artifact_dir: Base directory for artifacts
    """
    frame_path, metadata_path = get_artifact_paths(segment_key, artifact_dir)
    
    for path in [frame_path, metadata_path]:
        if os.path.exists(path):
            try:
                os.remove(path)
                logger.debug(f"Removed artifact: {path}")
            except Exception as e:
                logger.warning(f"Failed to remove {path}: {e}")
