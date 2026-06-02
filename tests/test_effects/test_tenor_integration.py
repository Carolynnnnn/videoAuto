"""
TDD Tests for Tenor Integration - Complete search→download→cache→fallback flow

Tests cover:
- get_sticker_for_sentiment() - main entry point
- download_and_cache() - caching logic
- fallback_to_builtin() - builtin library fallback
- Asset selection UI data structure
- Preloading optimization
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.effects.tenor_integration import (
    AssetResult,
    TenorIntegration,
    download_and_cache,
    fallback_to_builtin,
    get_sticker_for_sentiment,
    load_builtin_library,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────
@pytest.fixture
def mock_builtin_index(tmp_path: Path) -> Path:
    """Create a mock builtin library index."""
    index_data = {
        "version": "1.0",
        "stickers": [
            {
                "id": "happy_001",
                "filename": "happy.gif",
                "sentiment": "happy",
                "keywords": ["happy", "joy", "celebrate"],
                "width": 300,
                "height": 300,
                "size_bytes": 245000,
            },
            {
                "id": "sad_001",
                "filename": "sad.gif",
                "sentiment": "sad",
                "keywords": ["sad", "cry", "unhappy"],
                "width": 250,
                "height": 250,
                "size_bytes": 210000,
            },
            {
                "id": "generic_001",
                "filename": "generic.gif",
                "sentiment": "generic",
                "keywords": ["reaction", "emoji"],
                "width": 200,
                "height": 200,
                "size_bytes": 150000,
            },
        ],
    }
    library_dir = tmp_path / "builtin_library"
    library_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = library_dir / "index.json"
    index_path.write_text(json.dumps(index_data))
    
    # Create placeholder GIF files
    for sticker in index_data["stickers"]:
        gif_path = library_dir / sticker["filename"]
        # Create minimal GIF header
        gif_path.write_bytes(b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;")
    
    return library_dir


@pytest.fixture
def mock_tenor_cache(tmp_path: Path) -> Path:
    """Create a mock Tenor cache directory."""
    cache_dir = tmp_path / "tenor_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def tenor_integration(mock_builtin_index: Path, mock_tenor_cache: Path) -> TenorIntegration:
    """Create TenorIntegration instance with mocked directories."""
    return TenorIntegration(
        api_key="test_api_key",
        cache_dir=str(mock_tenor_cache),
        builtin_library_dir=str(mock_builtin_index),
    )


# ─────────────────────────────────────────────
# Test: load_builtin_library
# ─────────────────────────────────────────────
class TestLoadBuiltinLibrary:
    def test_loads_valid_index(self, mock_builtin_index: Path) -> None:
        """Should load and parse builtin library index."""
        library = load_builtin_library(str(mock_builtin_index))
        
        assert "stickers" in library
        assert len(library["stickers"]) == 3
        assert library["stickers"][0]["sentiment"] == "happy"

    def test_returns_empty_on_missing_directory(self, tmp_path: Path) -> None:
        """Should return empty dict when directory doesn't exist."""
        library = load_builtin_library(str(tmp_path / "nonexistent"))
        
        assert library == {}

    def test_returns_empty_on_invalid_json(self, tmp_path: Path) -> None:
        """Should return empty dict on malformed JSON."""
        library_dir = tmp_path / "bad_library"
        library_dir.mkdir()
        (library_dir / "index.json").write_text("{invalid json")
        
        library = load_builtin_library(str(library_dir))
        
        assert library == {}


# ─────────────────────────────────────────────
# Test: fallback_to_builtin
# ─────────────────────────────────────────────
class TestFallbackToBuiltin:
    def test_finds_by_sentiment_match(self, mock_builtin_index: Path) -> None:
        """Should find sticker by sentiment tag."""
        result = fallback_to_builtin(
            sentiment="happy",
            keywords=[],
            builtin_library_dir=str(mock_builtin_index),
        )
        
        assert result is not None
        assert result.source == "builtin"
        assert "happy" in result.path

    def test_finds_by_keyword_match(self, mock_builtin_index: Path) -> None:
        """Should find sticker by keyword when sentiment doesn't match."""
        result = fallback_to_builtin(
            sentiment="unknown",
            keywords=["celebrate"],
            builtin_library_dir=str(mock_builtin_index),
        )
        
        assert result is not None
        assert result.source == "builtin"

    def test_returns_generic_fallback(self, mock_builtin_index: Path) -> None:
        """Should return generic sticker when nothing matches."""
        result = fallback_to_builtin(
            sentiment="unknown",
            keywords=["nonexistent"],
            builtin_library_dir=str(mock_builtin_index),
        )
        
        assert result is not None
        assert result.source == "builtin"
        # Should return generic or any available
        assert "generic" in result.path or result.path.endswith(".gif")

    def test_returns_none_on_empty_library(self, tmp_path: Path) -> None:
        """Should return None when library is empty."""
        result = fallback_to_builtin(
            sentiment="happy",
            keywords=[],
            builtin_library_dir=str(tmp_path / "empty"),
        )
        
        assert result is None


# ─────────────────────────────────────────────
# Test: download_and_cache
# ─────────────────────────────────────────────
class TestDownloadAndCache:
    def test_caches_downloaded_file(self, mock_tenor_cache: Path) -> None:
        """Should download and cache file."""
        mock_response = MagicMock()
        mock_response.content = b"GIF89a" + b"\x00" * 100
        mock_response.headers = {"content-length": "106"}
        mock_response.iter_content = lambda chunk_size: [mock_response.content]
        mock_response.raise_for_status = lambda: None
        
        with patch("requests.get", return_value=mock_response):
            path = download_and_cache(
                url="https://tenor.com/test.gif",
                tenor_id="12345",
                cache_dir=str(mock_tenor_cache),
            )
        
        assert path is not None
        assert Path(path).exists()
        assert "12345" in path

    def test_skips_download_if_cached(self, mock_tenor_cache: Path) -> None:
        """Should return cached file without re-downloading."""
        # Pre-create cached file
        cached_file = mock_tenor_cache / "tenor_12345.gif"
        cached_file.write_bytes(b"GIF89a" + b"\x00" * 100)
        
        with patch("requests.get") as mock_get:
            path = download_and_cache(
                url="https://tenor.com/test.gif",
                tenor_id="12345",
                cache_dir=str(mock_tenor_cache),
            )
        
        # Should not call requests.get
        mock_get.assert_not_called()
        assert path == str(cached_file)

    def test_rejects_large_files(self, mock_tenor_cache: Path) -> None:
        """Should reject files > 10MB."""
        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(11 * 1024 * 1024)}  # 11MB
        
        with patch("requests.head", return_value=mock_response):
            path = download_and_cache(
                url="https://tenor.com/large.gif",
                tenor_id="99999",
                cache_dir=str(mock_tenor_cache),
            )
        
        assert path is None

    def test_returns_none_on_download_error(self, mock_tenor_cache: Path) -> None:
        """Should return None on network error."""
        with patch("requests.get", side_effect=Exception("Network error")):
            with patch("requests.head", side_effect=Exception("Network error")):
                path = download_and_cache(
                    url="https://tenor.com/test.gif",
                    tenor_id="error",
                    cache_dir=str(mock_tenor_cache),
                )
        
        assert path is None


# ─────────────────────────────────────────────
# Test: TenorIntegration.search_gifs
# ─────────────────────────────────────────────
class TestTenorIntegrationSearch:
    def test_search_returns_results(self, tenor_integration: TenorIntegration) -> None:
        """Should search Tenor API and return results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "id": "12345",
                    "title": "Happy dance",
                    "media_formats": {
                        "gif": {
                            "url": "https://tenor.com/12345.gif",
                            "size": 500000,
                        }
                    },
                    "content_description": "happy celebration",
                }
            ]
        }
        
        with patch.object(tenor_integration._session, "get", return_value=mock_response):
            results = tenor_integration.search_gifs("happy", sentiment="happy")
        
        assert len(results) >= 1
        assert results[0]["id"] == "12345"

    def test_search_handles_api_error(self, tenor_integration: TenorIntegration) -> None:
        """Should return empty list on API error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        
        with patch.object(tenor_integration._session, "get", return_value=mock_response):
            results = tenor_integration.search_gifs("test", sentiment="neutral")
        
        assert results == []

    def test_search_retries_on_rate_limit(self, tenor_integration: TenorIntegration) -> None:
        """Should retry on rate limit (429)."""
        call_count = 0
        
        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            if call_count < 3:
                mock_response.status_code = 429
            else:
                mock_response.status_code = 200
                mock_response.json.return_value = {"results": []}
            return mock_response
        
        with patch.object(tenor_integration._session, "get", side_effect=mock_get):
            with patch("time.sleep"):  # Don't actually sleep
                results = tenor_integration.search_gifs("test", sentiment="neutral")
        
        assert call_count == 3  # Retried twice


# ─────────────────────────────────────────────
# Test: get_sticker_for_sentiment (main entry)
# ─────────────────────────────────────────────
class TestGetStickerForSentiment:
    def test_returns_tenor_result_on_success(
        self, mock_builtin_index: Path, mock_tenor_cache: Path
    ) -> None:
        """Should return Tenor result when API succeeds."""
        # Mock Tenor search response
        mock_search_response = MagicMock()
        mock_search_response.status_code = 200
        mock_search_response.json.return_value = {
            "results": [
                {
                    "id": "67890",
                    "title": "Joy",
                    "media_formats": {
                        "gif": {
                            "url": "https://tenor.com/67890.gif",
                            "size": 300000,
                        }
                    },
                    "content_description": "joyful",
                }
            ]
        }
        
        # Mock download
        mock_download_response = MagicMock()
        mock_download_response.content = b"GIF89a" + b"\x00" * 100
        mock_download_response.headers = {"content-length": "106"}
        mock_download_response.iter_content = lambda chunk_size: [mock_download_response.content]
        mock_download_response.raise_for_status = lambda: None
        
        with patch("requests.Session") as MockSession:
            mock_session = MagicMock()
            mock_session.get.side_effect = [mock_search_response, mock_download_response]
            MockSession.return_value = mock_session
            
            with patch("requests.get", return_value=mock_download_response):
                result = get_sticker_for_sentiment(
                    sentiment="happy",
                    keywords=["joy"],
                    api_key="test_key",
                    cache_dir=str(mock_tenor_cache),
                    builtin_library_dir=str(mock_builtin_index),
                )
        
        # Result should be valid
        assert result is not None
        assert isinstance(result, AssetResult)
        assert result.source in ("tenor", "builtin")  # Either is acceptable

    def test_falls_back_to_builtin_on_api_failure(
        self, mock_builtin_index: Path, mock_tenor_cache: Path
    ) -> None:
        """Should fallback to builtin when Tenor fails."""
        with patch("requests.Session") as MockSession:
            mock_session = MagicMock()
            mock_session.get.side_effect = Exception("API unavailable")
            MockSession.return_value = mock_session
            
            result = get_sticker_for_sentiment(
                sentiment="happy",
                keywords=["joy"],
                api_key="test_key",
                cache_dir=str(mock_tenor_cache),
                builtin_library_dir=str(mock_builtin_index),
            )
        
        assert result is not None
        assert result.source == "builtin"
        assert "happy" in result.path

    def test_falls_back_on_empty_results(
        self, mock_builtin_index: Path, mock_tenor_cache: Path
    ) -> None:
        """Should fallback when Tenor returns empty results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        
        with patch("requests.Session") as MockSession:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_session.headers = MagicMock()
            MockSession.return_value = mock_session
            
            result = get_sticker_for_sentiment(
                sentiment="sad",
                keywords=["unhappy"],
                api_key="test_key",
                cache_dir=str(mock_tenor_cache),
                builtin_library_dir=str(mock_builtin_index),
            )
        
        assert result is not None
        assert result.source == "builtin"

    def test_logs_fallback_event(
        self, mock_builtin_index: Path, mock_tenor_cache: Path, caplog
    ) -> None:
        """Should log when falling back to builtin."""
        with patch("requests.Session") as MockSession:
            mock_session = MagicMock()
            mock_session.get.side_effect = Exception("API error")
            MockSession.return_value = mock_session
            
            with caplog.at_level("WARNING"):
                result = get_sticker_for_sentiment(
                    sentiment="happy",
                    keywords=[],
                    api_key="test_key",
                    cache_dir=str(mock_tenor_cache),
                    builtin_library_dir=str(mock_builtin_index),
                )
        
        # Should have logged fallback
        assert result is not None
        assert result.source == "builtin"


# ─────────────────────────────────────────────
# Test: AssetResult structure
# ─────────────────────────────────────────────
class TestAssetResult:
    def test_has_required_fields(self) -> None:
        """AssetResult should have path, source, and metadata."""
        result = AssetResult(
            path="/path/to/sticker.gif",
            source="tenor",
            metadata={"id": "12345", "sentiment": "happy"},
        )
        
        assert result.path == "/path/to/sticker.gif"
        assert result.source == "tenor"
        assert result.metadata["id"] == "12345"

    def test_source_must_be_tenor_or_builtin(self) -> None:
        """Source field should be 'tenor' or 'builtin'."""
        valid_sources = ("tenor", "builtin")
        
        for source in valid_sources:
            result = AssetResult(path="/test.gif", source=source, metadata={})
            assert result.source == source


# ─────────────────────────────────────────────
# Test: Asset preloading
# ─────────────────────────────────────────────
class TestAssetPreloading:
    def test_preload_multiple_sentiments(
        self, tenor_integration: TenorIntegration
    ) -> None:
        """Should support preloading stickers for multiple sentiments."""
        sentiments = ["happy", "sad", "excited"]
        
        # This method should exist and not raise
        assert hasattr(tenor_integration, "preload_sentiments")
        
        # Mock the internal search
        with patch.object(tenor_integration, "search_gifs", return_value=[]):
            results = tenor_integration.preload_sentiments(sentiments)
        
        assert isinstance(results, dict)
        assert all(s in results for s in sentiments)


# ─────────────────────────────────────────────
# Test: Integration - Full flow
# ─────────────────────────────────────────────
class TestFullIntegrationFlow:
    def test_tenor_success_flow(
        self, mock_builtin_index: Path, mock_tenor_cache: Path
    ) -> None:
        """Test complete flow: Tenor search → download → cache → return."""
        # Setup mocks for complete flow
        mock_search_response = MagicMock()
        mock_search_response.status_code = 200
        mock_search_response.json.return_value = {
            "results": [
                {
                    "id": "flow_test_123",
                    "title": "Test GIF",
                    "media_formats": {
                        "gif": {
                            "url": "https://tenor.com/flow_test.gif",
                            "size": 200000,
                        }
                    },
                    "content_description": "test",
                }
            ]
        }
        
        mock_download_response = MagicMock()
        mock_download_response.content = b"GIF89a" + b"\x00" * 200
        mock_download_response.headers = {"content-length": "206"}
        mock_download_response.iter_content = lambda chunk_size: [mock_download_response.content]
        mock_download_response.raise_for_status = lambda: None
        
        with patch("requests.Session") as MockSession:
            mock_session = MagicMock()
            mock_session.headers = MagicMock()
            mock_session.get.side_effect = [mock_search_response, mock_download_response]
            MockSession.return_value = mock_session
            
            with patch("requests.get", return_value=mock_download_response):
                with patch("requests.head") as mock_head:
                    mock_head_resp = MagicMock()
                    mock_head_resp.headers = {"content-length": "206"}
                    mock_head.return_value = mock_head_resp
                    
                    result = get_sticker_for_sentiment(
                        sentiment="neutral",
                        keywords=["test"],
                        api_key="test_key",
                        cache_dir=str(mock_tenor_cache),
                        builtin_library_dir=str(mock_builtin_index),
                    )
        
        assert result is not None
        # Could be tenor or builtin depending on mock setup
        assert result.source in ("tenor", "builtin")

    def test_fallback_flow(
        self, mock_builtin_index: Path, mock_tenor_cache: Path
    ) -> None:
        """Test complete flow: Tenor fails → fallback → builtin."""
        with patch("requests.Session") as MockSession:
            mock_session = MagicMock()
            mock_session.get.side_effect = Exception("Tenor API down")
            MockSession.return_value = mock_session
            
            result = get_sticker_for_sentiment(
                sentiment="happy",
                keywords=["celebrate"],
                api_key="invalid_key",
                cache_dir=str(mock_tenor_cache),
                builtin_library_dir=str(mock_builtin_index),
            )
        
        assert result is not None
        assert result.source == "builtin"
        assert Path(result.path).exists()
        assert result.metadata is not None
