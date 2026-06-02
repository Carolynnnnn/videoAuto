"""Tests for preview_generator module."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.effects.preview_generator import (
    _compute_preview_hash,
    _get_cache_dir,
    clear_preview_cache,
    extract_frame,
    extract_middle_frame,
    generate_effect_preview,
    generate_sticker_preview,
    generate_text_preview,
    get_cached_preview,
)


def _create_test_image(path: Path, size: tuple[int, int] = (200, 200)) -> None:
    """Create a test image for preview tests."""
    image = Image.new("RGBA", size, (100, 150, 200, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def _create_test_gif(path: Path, size: tuple[int, int] = (100, 100)) -> None:
    """Create a test GIF for sticker tests."""
    image = Image.new("RGBA", size, (255, 0, 0, 180))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="GIF")


class TestPreviewHash:
    """Tests for preview hash computation."""

    def test_compute_hash_returns_string(self) -> None:
        result = _compute_preview_hash("video.mp4", 1.5)
        assert isinstance(result, str)
        assert len(result) == 16

    def test_same_inputs_produce_same_hash(self) -> None:
        hash1 = _compute_preview_hash("video.mp4", 1.5)
        hash2 = _compute_preview_hash("video.mp4", 1.5)
        assert hash1 == hash2

    def test_different_paths_produce_different_hash(self) -> None:
        hash1 = _compute_preview_hash("video1.mp4", 1.5)
        hash2 = _compute_preview_hash("video2.mp4", 1.5)
        assert hash1 != hash2

    def test_different_timestamps_produce_different_hash(self) -> None:
        hash1 = _compute_preview_hash("video.mp4", 1.0)
        hash2 = _compute_preview_hash("video.mp4", 2.0)
        assert hash1 != hash2

    def test_effects_change_hash(self) -> None:
        effects = [{"type": "sticker", "anchor": "center"}]
        hash1 = _compute_preview_hash("video.mp4", 1.0, None)
        hash2 = _compute_preview_hash("video.mp4", 1.0, effects)
        assert hash1 != hash2


class TestCacheDirectory:
    """Tests for cache directory management."""

    def test_get_cache_dir_creates_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.effects.preview_generator.PREVIEW_CACHE_DIR", str(tmp_path / "test_cache"))
        cache_dir = _get_cache_dir()
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_clear_preview_cache_removes_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cache_subdir = tmp_path / "preview_cache"
        cache_subdir.mkdir(parents=True)
        (cache_subdir / "test1.png").touch()
        (cache_subdir / "test2.png").touch()
        
        monkeypatch.setattr("src.effects.preview_generator.PREVIEW_CACHE_DIR", str(cache_subdir))
        deleted = clear_preview_cache()
        
        assert deleted == 2
        assert not (cache_subdir / "test1.png").exists()


class TestExtractFrame:
    """Tests for frame extraction functions."""

    def test_extract_frame_returns_false_for_missing_file(self, tmp_path: Path) -> None:
        result = extract_frame(
            "/nonexistent/video.mp4",
            str(tmp_path / "output.png"),
            1.0,
        )
        assert result is False

    def test_extract_middle_frame_returns_false_for_missing_file(self, tmp_path: Path) -> None:
        result = extract_middle_frame(
            "/nonexistent/video.mp4",
            str(tmp_path / "output.png"),
        )
        assert result is False

    def test_extract_frame_creates_output_directory(self, tmp_path: Path) -> None:
        # Even though ffmpeg will fail, the directory should be created
        output_path = tmp_path / "subdir" / "output.png"
        extract_frame(
            "/nonexistent/video.mp4",
            str(output_path),
            1.0,
        )
        # Directory should be created even if extraction fails
        assert output_path.parent.exists()


class TestGenerateStickerPreview:
    """Tests for sticker preview generation."""

    def test_returns_false_for_missing_background(self, tmp_path: Path) -> None:
        sticker_path = tmp_path / "sticker.gif"
        _create_test_gif(sticker_path)
        
        result = generate_sticker_preview(
            "/nonexistent/bg.png",
            str(sticker_path),
            str(tmp_path / "output.png"),
            anchor="center",
        )
        assert result is False

    def test_returns_false_for_missing_sticker(self, tmp_path: Path) -> None:
        bg_path = tmp_path / "bg.png"
        _create_test_image(bg_path)
        
        result = generate_sticker_preview(
            str(bg_path),
            "/nonexistent/sticker.gif",
            str(tmp_path / "output.png"),
            anchor="center",
        )
        assert result is False

    def test_returns_false_for_invalid_anchor(self, tmp_path: Path) -> None:
        bg_path = tmp_path / "bg.png"
        sticker_path = tmp_path / "sticker.gif"
        _create_test_image(bg_path)
        _create_test_gif(sticker_path)
        
        result = generate_sticker_preview(
            str(bg_path),
            str(sticker_path),
            str(tmp_path / "output.png"),
            anchor="invalid-anchor",
        )
        assert result is False

    def test_generates_preview_with_valid_inputs(self, tmp_path: Path) -> None:
        bg_path = tmp_path / "bg.png"
        sticker_path = tmp_path / "sticker.gif"
        output_path = tmp_path / "output.png"
        _create_test_image(bg_path, (1080, 1920))
        _create_test_gif(sticker_path, (100, 100))
        
        result = generate_sticker_preview(
            str(bg_path),
            str(sticker_path),
            str(output_path),
            anchor="center",
            scale=0.5,
            video_width=1080,
            video_height=1920,
        )
        
        assert result is True
        assert output_path.exists()

    def test_sticker_positioned_at_anchor(self, tmp_path: Path) -> None:
        bg_path = tmp_path / "bg.png"
        sticker_path = tmp_path / "sticker.gif"
        output_path = tmp_path / "output.png"
        
        # Create white background
        bg = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
        bg.save(bg_path, "PNG")
        
        # Create red sticker
        sticker = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
        sticker.save(sticker_path, "GIF")
        
        result = generate_sticker_preview(
            str(bg_path),
            str(sticker_path),
            str(output_path),
            anchor="top-left",
            scale=1.0,
            video_width=100,
            video_height=100,
        )
        
        assert result is True
        output = Image.open(output_path)
        # Check top-left pixel should be red
        pixel = output.getpixel((0, 0))
        assert pixel[0] == 255  # Red channel


class TestGenerateTextPreview:
    """Tests for text preview generation."""

    def test_returns_false_for_missing_background(self, tmp_path: Path) -> None:
        result = generate_text_preview(
            "/nonexistent/bg.png",
            str(tmp_path / "output.png"),
            "Test text",
            x=100,
            y=100,
        )
        assert result is False

    def test_generates_preview_with_valid_inputs(self, tmp_path: Path) -> None:
        bg_path = tmp_path / "bg.png"
        output_path = tmp_path / "output.png"
        _create_test_image(bg_path, (1080, 1920))
        
        result = generate_text_preview(
            str(bg_path),
            str(output_path),
            text="Hello World",
            x=100,
            y=100,
            font_size=48,
            color="white",
        )
        
        assert result is True
        assert output_path.exists()

    def test_text_preview_handles_string_positions(self, tmp_path: Path) -> None:
        bg_path = tmp_path / "bg.png"
        output_path = tmp_path / "output.png"
        _create_test_image(bg_path, (1080, 1920))
        
        result = generate_text_preview(
            str(bg_path),
            str(output_path),
            text="Centered",
            x="(w-text_w)/2",
            y="h*0.78",
            video_width=1080,
            video_height=1920,
        )
        
        assert result is True
        assert output_path.exists()


class TestGenerateEffectPreview:
    """Tests for comprehensive effect preview generation."""

    def test_returns_error_for_missing_video(self, tmp_path: Path) -> None:
        result = generate_effect_preview(
            "/nonexistent/video.mp4",
            str(tmp_path / "output.png"),
        )
        assert result["success"] is False
        assert "not exist" in result["error"].lower()

    def test_generates_preview_from_image(self, tmp_path: Path) -> None:
        bg_path = tmp_path / "bg.png"
        output_path = tmp_path / "output.png"
        _create_test_image(bg_path, (1080, 1920))
        
        result = generate_effect_preview(
            str(bg_path),
            str(output_path),
            use_cache=False,
        )
        
        assert result["success"] is True
        assert result["cached"] is False
        assert output_path.exists()

    def test_applies_sticker_effect(self, tmp_path: Path) -> None:
        bg_path = tmp_path / "bg.png"
        sticker_path = tmp_path / "sticker.gif"
        output_path = tmp_path / "output.png"
        _create_test_image(bg_path, (1080, 1920))
        _create_test_gif(sticker_path, (100, 100))
        
        effects = [
            {
                "type": "sticker",
                "asset_path": str(sticker_path),
                "anchor": "center",
                "scale": 0.5,
            }
        ]
        
        result = generate_effect_preview(
            str(bg_path),
            str(output_path),
            effects=effects,
            use_cache=False,
        )
        
        assert result["success"] is True
        assert output_path.exists()

    def test_applies_text_effect(self, tmp_path: Path) -> None:
        bg_path = tmp_path / "bg.png"
        output_path = tmp_path / "output.png"
        _create_test_image(bg_path, (1080, 1920))
        
        effects = [
            {
                "type": "text",
                "text": "Hello Preview",
                "x": 100,
                "y": 100,
                "font_size": 32,
                "color": "yellow",
            }
        ]
        
        result = generate_effect_preview(
            str(bg_path),
            str(output_path),
            effects=effects,
            use_cache=False,
        )
        
        assert result["success"] is True
        assert output_path.exists()

    def test_uses_cache_when_enabled(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cache_dir = tmp_path / "cache"
        monkeypatch.setattr("src.effects.preview_generator.PREVIEW_CACHE_DIR", str(cache_dir))
        
        bg_path = tmp_path / "bg.png"
        output_path = tmp_path / "output.png"
        _create_test_image(bg_path, (1080, 1920))
        
        # First call - should not be cached
        result1 = generate_effect_preview(
            str(bg_path),
            str(output_path),
            timestamp=1.0,
            use_cache=True,
        )
        assert result1["success"] is True
        assert result1["cached"] is False
        
        # Second call - should use cache
        result2 = generate_effect_preview(
            str(bg_path),
            str(tmp_path / "output2.png"),
            timestamp=1.0,
            use_cache=True,
        )
        assert result2["success"] is True
        assert result2["cached"] is True

    def test_skips_invalid_sticker(self, tmp_path: Path) -> None:
        bg_path = tmp_path / "bg.png"
        output_path = tmp_path / "output.png"
        _create_test_image(bg_path, (1080, 1920))
        
        effects = [
            {
                "type": "sticker",
                "asset_path": "/nonexistent/sticker.gif",
                "anchor": "center",
            }
        ]
        
        result = generate_effect_preview(
            str(bg_path),
            str(output_path),
            effects=effects,
            use_cache=False,
        )
        
        # Should still succeed, just skip invalid sticker
        assert result["success"] is True

    def test_handles_multiple_effects(self, tmp_path: Path) -> None:
        bg_path = tmp_path / "bg.png"
        sticker_path = tmp_path / "sticker.gif"
        output_path = tmp_path / "output.png"
        _create_test_image(bg_path, (1080, 1920))
        _create_test_gif(sticker_path, (100, 100))
        
        effects = [
            {
                "type": "sticker",
                "asset_path": str(sticker_path),
                "anchor": "top-right",
                "scale": 0.8,
            },
            {
                "type": "text",
                "text": "Multiple Effects",
                "x": 100,
                "y": 200,
            },
        ]
        
        result = generate_effect_preview(
            str(bg_path),
            str(output_path),
            effects=effects,
            use_cache=False,
        )
        
        assert result["success"] is True
        assert output_path.exists()


class TestGetCachedPreview:
    """Tests for cache lookup function."""

    def test_returns_none_when_not_cached(self) -> None:
        result = get_cached_preview("/nonexistent/video.mp4", 1.0)
        assert result is None

    def test_returns_path_when_cached(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr("src.effects.preview_generator.PREVIEW_CACHE_DIR", str(cache_dir))
        
        # Create a cached file with known hash
        bg_path = tmp_path / "bg.png"
        _create_test_image(bg_path)
        
        # Generate preview to create cache
        output_path = tmp_path / "output.png"
        generate_effect_preview(
            str(bg_path),
            str(output_path),
            timestamp=1.0,
            use_cache=True,
        )
        
        # Check cache lookup
        cached = get_cached_preview(str(bg_path), 1.0)
        assert cached is not None
        assert Path(cached).exists()


class TestPreviewGenerationSpeed:
    """Tests for preview generation performance."""

    def test_image_preview_under_one_second(self, tmp_path: Path) -> None:
        bg_path = tmp_path / "bg.png"
        output_path = tmp_path / "output.png"
        _create_test_image(bg_path, (1920, 1080))
        
        start = time.time()
        result = generate_effect_preview(
            str(bg_path),
            str(output_path),
            use_cache=False,
        )
        elapsed = time.time() - start
        
        assert result["success"] is True
        assert elapsed < 1.0, f"Preview generation took {elapsed:.2f}s, exceeds 1s limit"

    def test_sticker_overlay_under_one_second(self, tmp_path: Path) -> None:
        bg_path = tmp_path / "bg.png"
        sticker_path = tmp_path / "sticker.gif"
        output_path = tmp_path / "output.png"
        _create_test_image(bg_path, (1080, 1920))
        _create_test_gif(sticker_path, (200, 200))
        
        effects = [
            {
                "type": "sticker",
                "asset_path": str(sticker_path),
                "anchor": "center",
                "scale": 1.0,
            }
        ]
        
        start = time.time()
        result = generate_effect_preview(
            str(bg_path),
            str(output_path),
            effects=effects,
            use_cache=False,
        )
        elapsed = time.time() - start
        
        assert result["success"] is True
        assert elapsed < 1.0, f"Sticker preview took {elapsed:.2f}s, exceeds 1s limit"

    def test_cached_preview_very_fast(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cache_dir = tmp_path / "cache"
        monkeypatch.setattr("src.effects.preview_generator.PREVIEW_CACHE_DIR", str(cache_dir))
        
        bg_path = tmp_path / "bg.png"
        output_path = tmp_path / "output.png"
        _create_test_image(bg_path, (1920, 1080))
        
        # First call creates cache
        generate_effect_preview(str(bg_path), str(output_path), use_cache=True)
        
        # Second call should use cache - should be very fast
        start = time.time()
        result = generate_effect_preview(
            str(bg_path),
            str(tmp_path / "output2.png"),
            use_cache=True,
        )
        elapsed = time.time() - start
        
        assert result["success"] is True
        assert result["cached"] is True
        assert elapsed < 0.1, f"Cached preview took {elapsed:.3f}s, should be <0.1s"
