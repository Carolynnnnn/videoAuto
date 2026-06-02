"""
Tenor Integration - Complete search→download→cache→fallback flow for stickers

Features:
- Tenor API search with sentiment-aware queries
- Download with caching (assets/tenor_cache/)
- Automatic fallback to builtin library on failure
- Asset selection UI data structure
- Preloading optimization

Flow:
1. get_sticker_for_sentiment(sentiment, keywords)
2. Search Tenor API with retry logic (max 3 retries)
3. Download best result to cache
4. On failure → fallback_to_builtin()
5. Return AssetResult with source field (tenor/builtin)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from src.utils.logger import get_logger

logger = get_logger("tenor_integration")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
TENOR_SEARCH_URL = "https://tenor.googleapis.com/v2/search"
TENOR_FEATURED_URL = "https://tenor.googleapis.com/v2/featured"

MAX_GIF_BYTES = 10 * 1024 * 1024  # 10MB
MAX_RETRIES = 3
RATE_LIMIT_BACKOFF = 2  # seconds
DOWNLOAD_TIMEOUT = 30
CACHE_TTL_HOURS = 24

# Default paths
DEFAULT_CACHE_DIR = "assets/tenor_cache"
DEFAULT_BUILTIN_LIBRARY_DIR = "assets/builtin_library"


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────
@dataclass
class AssetResult:
    """Result from sticker fetch operation."""
    path: str
    source: str  # "tenor" or "builtin"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.source not in ("tenor", "builtin"):
            raise ValueError(f"source must be 'tenor' or 'builtin', got '{self.source}'")


@dataclass
class TenorSearchResult:
    """Single result from Tenor search."""
    id: str
    title: str
    gif_url: str
    size_bytes: int
    description: str = ""
    sentiment: str = ""


# ─────────────────────────────────────────────
# Builtin Library Functions
# ─────────────────────────────────────────────
def load_builtin_library(builtin_library_dir: str) -> Dict[str, Any]:
    """
    Load builtin library index.json.
    
    Returns empty dict if directory doesn't exist or JSON is invalid.
    """
    library_path = Path(builtin_library_dir)
    index_path = library_path / "index.json"
    
    if not index_path.exists():
        logger.debug(f"Builtin library not found: {index_path}")
        return {}
    
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load builtin library: {e}")
        return {}


def fallback_to_builtin(
    sentiment: str,
    keywords: List[str],
    builtin_library_dir: str = DEFAULT_BUILTIN_LIBRARY_DIR,
) -> Optional[AssetResult]:
    """
    Fallback to builtin library when Tenor fails.
    
    Search order:
    1. Match by sentiment tag
    2. Match by keyword
    3. Return any GIF (generic fallback)
    
    Returns None if library is empty.
    """
    logger.info(f"Falling back to builtin library for sentiment='{sentiment}'")
    
    library = load_builtin_library(builtin_library_dir)
    if not library or "stickers" not in library:
        logger.warning("Builtin library is empty or invalid")
        return None
    
    stickers = library["stickers"]
    library_dir = Path(builtin_library_dir)
    
    # 1. Match by sentiment
    for sticker in stickers:
        if sticker.get("sentiment", "").lower() == sentiment.lower():
            gif_path = library_dir / sticker["filename"]
            if gif_path.exists():
                logger.info(f"Builtin fallback: matched by sentiment '{sentiment}'")
                return AssetResult(
                    path=str(gif_path),
                    source="builtin",
                    metadata={
                        "id": sticker.get("id", ""),
                        "sentiment": sticker.get("sentiment", ""),
                        "keywords": sticker.get("keywords", []),
                    },
                )
    
    # 2. Match by keyword
    keywords_lower = [k.lower() for k in keywords]
    for sticker in stickers:
        sticker_keywords = [k.lower() for k in sticker.get("keywords", [])]
        for kw in keywords_lower:
            if kw in sticker_keywords:
                gif_path = library_dir / sticker["filename"]
                if gif_path.exists():
                    logger.info(f"Builtin fallback: matched by keyword '{kw}'")
                    return AssetResult(
                        path=str(gif_path),
                        source="builtin",
                        metadata={
                            "id": sticker.get("id", ""),
                            "sentiment": sticker.get("sentiment", ""),
                            "keywords": sticker.get("keywords", []),
                        },
                    )
    
    # 3. Return generic or first available
    for sticker in stickers:
        if sticker.get("sentiment", "").lower() == "generic":
            gif_path = library_dir / sticker["filename"]
            if gif_path.exists():
                logger.info("Builtin fallback: using generic sticker")
                return AssetResult(
                    path=str(gif_path),
                    source="builtin",
                    metadata={
                        "id": sticker.get("id", ""),
                        "sentiment": sticker.get("sentiment", ""),
                        "keywords": sticker.get("keywords", []),
                    },
                )
    
    # Last resort: return any available
    for sticker in stickers:
        gif_path = library_dir / sticker["filename"]
        if gif_path.exists():
            logger.info("Builtin fallback: using first available sticker")
            return AssetResult(
                path=str(gif_path),
                source="builtin",
                metadata={
                    "id": sticker.get("id", ""),
                    "sentiment": sticker.get("sentiment", ""),
                    "keywords": sticker.get("keywords", []),
                },
            )
    
    logger.warning("No builtin stickers available")
    return None


# ─────────────────────────────────────────────
# Download and Cache Functions
# ─────────────────────────────────────────────
def download_and_cache(
    url: str,
    tenor_id: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Optional[str]:
    """
    Download GIF from URL and cache locally.
    
    - Checks cache first, skips download if exists
    - Rejects files > 10MB
    - Returns local path on success, None on failure
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"tenor_{tenor_id}.gif"
    local_path = cache_path / filename
    
    # Check cache
    if local_path.exists() and local_path.stat().st_size > 100:
        logger.debug(f"Cache hit: {filename}")
        return str(local_path)
    
    # Check file size before downloading
    try:
        head_resp = requests.head(url, timeout=10)
        content_length = int(head_resp.headers.get("content-length", 0))
        if content_length > MAX_GIF_BYTES:
            logger.warning(f"GIF too large ({content_length} bytes), skipping")
            return None
    except Exception as e:
        logger.debug(f"HEAD request failed: {e}")
        # Continue anyway, we'll check size during download
    
    # Download
    try:
        logger.info(f"Downloading Tenor GIF: {tenor_id}")
        resp = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
        
        total_size = int(resp.headers.get("content-length", 0))
        if total_size > MAX_GIF_BYTES:
            logger.warning(f"GIF too large ({total_size} bytes), skipping")
            return None
        
        downloaded = 0
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded > MAX_GIF_BYTES:
                        logger.warning("Download exceeded 10MB limit, aborting")
                        f.close()
                        local_path.unlink(missing_ok=True)
                        return None
        
        logger.info(f"Downloaded: {filename} ({downloaded/1024:.1f} KB)")
        return str(local_path)
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        if local_path.exists():
            local_path.unlink(missing_ok=True)
        return None


# ─────────────────────────────────────────────
# Tenor Integration Class
# ─────────────────────────────────────────────
class TenorIntegration:
    """
    Tenor API client with caching and fallback support.
    
    Features:
    - Search with sentiment-aware queries
    - Rate limiting with backoff
    - Local caching
    - Automatic fallback to builtin library
    """
    
    def __init__(
        self,
        api_key: str,
        cache_dir: str = DEFAULT_CACHE_DIR,
        builtin_library_dir: str = DEFAULT_BUILTIN_LIBRARY_DIR,
        max_retries: int = MAX_RETRIES,
    ):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.builtin_library_dir = builtin_library_dir
        self.max_retries = max_retries
        
        self._session = requests.Session()
        self._search_cache: Dict[str, Any] = {}
        
        # Load disk cache
        self._disk_cache_path = self.cache_dir / "search_cache.json"
        self._disk_cache: Dict[str, Any] = {}
        if self._disk_cache_path.exists():
            try:
                self._disk_cache = json.loads(
                    self._disk_cache_path.read_text(encoding="utf-8")
                )
            except Exception:
                pass
    
    def _save_disk_cache(self) -> None:
        """Save search cache to disk."""
        try:
            self._disk_cache_path.write_text(
                json.dumps(self._disk_cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug(f"Failed to save disk cache: {e}")
    
    def _get(self, url: str, params: dict, retries: int = 0) -> Optional[dict]:
        """GET request with retry logic."""
        try:
            resp = self._session.get(url, params=params, timeout=15)
            
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                logger.warning(f"Rate limited (429), waiting {RATE_LIMIT_BACKOFF}s")
                time.sleep(RATE_LIMIT_BACKOFF)
                if retries < self.max_retries:
                    return self._get(url, params, retries + 1)
            elif resp.status_code == 401:
                logger.error("Invalid Tenor API key")
            else:
                logger.warning(f"Tenor API error: {resp.status_code}")
                if retries < self.max_retries:
                    time.sleep(1)
                    return self._get(url, params, retries + 1)
                    
        except requests.RequestException as e:
            logger.warning(f"Request failed: {e}")
            if retries < self.max_retries:
                time.sleep(2)
                return self._get(url, params, retries + 1)
        
        return None
    
    def search_gifs(
        self,
        query: str,
        sentiment: str = "",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search Tenor for GIFs matching query and sentiment.
        
        Returns list of result dicts with id, url, size info.
        """
        # Build sentiment-aware query
        search_query = query
        if sentiment and sentiment not in query.lower():
            search_query = f"{sentiment} {query}"
        
        # Check disk cache
        cache_key = f"search:{search_query}:{limit}"
        if cache_key in self._disk_cache:
            cached = self._disk_cache[cache_key]
            # Check TTL (24 hours)
            cache_time = cached.get("_cache_time", 0)
            if time.time() - cache_time < CACHE_TTL_HOURS * 3600:
                logger.debug(f"Search cache hit: {search_query}")
                return cached.get("results", [])
        
        params = {
            "key": self.api_key,
            "q": search_query,
            "limit": limit,
            "media_filter": "gif",
            "contentfilter": "medium",
        }
        
        data = self._get(TENOR_SEARCH_URL, params)
        if not data or "results" not in data:
            logger.debug(f"No Tenor results for: {search_query}")
            return []
        
        results = []
        for item in data.get("results", []):
            media = item.get("media_formats", {})
            gif_info = media.get("gif", {})
            
            if not gif_info.get("url"):
                continue
            
            results.append({
                "id": str(item.get("id", "")),
                "title": item.get("title", ""),
                "url": gif_info.get("url", ""),
                "size": gif_info.get("size", 0),
                "description": item.get("content_description", ""),
            })
        
        # Cache results
        if results:
            self._disk_cache[cache_key] = {
                "results": results,
                "_cache_time": time.time(),
            }
            self._save_disk_cache()
        
        logger.debug(f"Tenor search '{search_query}': {len(results)} results")
        return results
    
    def fetch_sticker(
        self,
        sentiment: str,
        keywords: List[str],
    ) -> Optional[AssetResult]:
        """
        Fetch sticker for sentiment with full flow.
        
        1. Search Tenor
        2. Download best result
        3. On failure → fallback to builtin
        """
        # Build search query from keywords
        query = " ".join(keywords[:2]) if keywords else sentiment
        
        try:
            results = self.search_gifs(query, sentiment=sentiment, limit=5)
            
            if not results:
                logger.info(f"No Tenor results, falling back for '{sentiment}'")
                return fallback_to_builtin(
                    sentiment=sentiment,
                    keywords=keywords,
                    builtin_library_dir=self.builtin_library_dir,
                )
            
            # Try to download first valid result
            for result in results:
                # Skip if too large
                if result.get("size", 0) > MAX_GIF_BYTES:
                    continue
                
                path = download_and_cache(
                    url=result["url"],
                    tenor_id=result["id"],
                    cache_dir=str(self.cache_dir),
                )
                
                if path:
                    return AssetResult(
                        path=path,
                        source="tenor",
                        metadata={
                            "id": result["id"],
                            "title": result.get("title", ""),
                            "sentiment": sentiment,
                            "keywords": keywords,
                        },
                    )
            
            # All downloads failed
            logger.info("All Tenor downloads failed, falling back")
            return fallback_to_builtin(
                sentiment=sentiment,
                keywords=keywords,
                builtin_library_dir=self.builtin_library_dir,
            )
            
        except Exception as e:
            logger.warning(f"Tenor fetch failed: {e}")
            return fallback_to_builtin(
                sentiment=sentiment,
                keywords=keywords,
                builtin_library_dir=self.builtin_library_dir,
            )
    
    def preload_sentiments(
        self,
        sentiments: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Preload search results for multiple sentiments.
        
        Useful for GUI to show asset selection options.
        Returns dict mapping sentiment → results list.
        """
        preloaded: Dict[str, List[Dict[str, Any]]] = {}
        
        for sentiment in sentiments:
            try:
                results = self.search_gifs(sentiment, sentiment=sentiment, limit=5)
                preloaded[sentiment] = results
            except Exception as e:
                logger.debug(f"Preload failed for '{sentiment}': {e}")
                preloaded[sentiment] = []
        
        return preloaded


# ─────────────────────────────────────────────
# Convenience Function (Main Entry Point)
# ─────────────────────────────────────────────
def get_sticker_for_sentiment(
    sentiment: str,
    keywords: List[str],
    api_key: str = "",
    cache_dir: str = DEFAULT_CACHE_DIR,
    builtin_library_dir: str = DEFAULT_BUILTIN_LIBRARY_DIR,
) -> Optional[AssetResult]:
    """
    Main entry point: Get sticker for sentiment with full flow.
    
    Flow:
    1. recommendation → Tenor search → download → cache
    2. On any failure → auto fallback to builtin assets
    
    Args:
        sentiment: Emotion/sentiment tag (e.g., "happy", "sad")
        keywords: Additional keywords for search
        api_key: Tenor API key (or from TENOR_API_KEY env var)
        cache_dir: Directory for cached downloads
        builtin_library_dir: Directory for builtin stickers
    
    Returns:
        AssetResult with path, source, and metadata
        None if all fallbacks fail
    """
    # Get API key from env if not provided
    api_key = api_key or os.environ.get("TENOR_API_KEY", "")
    
    if not api_key:
        logger.info("No Tenor API key, using builtin library only")
        return fallback_to_builtin(
            sentiment=sentiment,
            keywords=keywords,
            builtin_library_dir=builtin_library_dir,
        )
    
    try:
        integration = TenorIntegration(
            api_key=api_key,
            cache_dir=cache_dir,
            builtin_library_dir=builtin_library_dir,
        )
        
        return integration.fetch_sticker(sentiment, keywords)
        
    except Exception as e:
        logger.warning(f"TenorIntegration failed: {e}, falling back to builtin")
        return fallback_to_builtin(
            sentiment=sentiment,
            keywords=keywords,
            builtin_library_dir=builtin_library_dir,
        )
