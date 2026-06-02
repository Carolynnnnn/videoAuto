"""
Pexels 素材客户端
支持：
  - 视频搜索（优先竖屏/匹配比例）
  - 图片搜索（fallback）
  - 本地磁盘缓存（避免重复下载）
  - 关键词翻译（中文 → 英文，供 Pexels 搜索）
  - 速率限制感知（自动退避）
  - 多关键词轮询（提高命中率）
"""
from __future__ import annotations
import os
import json
import time
import hashlib
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

from src.utils.logger import get_logger

logger = get_logger("pexels_client")

# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────
PEXELS_VIDEO_SEARCH_URL = "https://api.pexels.com/videos/search"
PEXELS_PHOTO_SEARCH_URL = "https://api.pexels.com/v1/search"
PEXELS_VIDEO_GET_URL    = "https://api.pexels.com/videos/videos/{id}"
PEXELS_PHOTO_GET_URL    = "https://api.pexels.com/v1/photos/{id}"

# 目标分辨率偏好（宽x高）
PREFERRED_RESOLUTIONS = {
    "9:16": [(1080, 1920), (720, 1280), (540, 960)],
    "16:9": [(1920, 1080), (1280, 720), (960, 540)],
    "1:1":  [(1080, 1080), (720, 720)],
}

# 视频时长约束（秒）
MIN_VIDEO_DURATION = 3
MAX_VIDEO_DURATION = 30

# 下载超时
DOWNLOAD_TIMEOUT = 60

# 速率限制退避（秒）
RATE_LIMIT_BACKOFF = 5


# ─────────────────────────────────────────────
# 数据类
# ─────────────────────────────────────────────
@dataclass
class PexelsVideoResult:
    """Pexels 视频搜索结果"""
    pexels_id: int
    duration: int
    width: int
    height: int
    download_url: str          # 最佳质量的下载链接
    quality: str               # sd / hd / uhd
    fps: float
    photographer: str
    pexels_url: str
    keywords: List[str] = field(default_factory=list)


@dataclass
class PexelsPhotoResult:
    """Pexels 图片搜索结果"""
    pexels_id: int
    width: int
    height: int
    download_url: str          # 最佳尺寸的下载链接
    photographer: str
    pexels_url: str
    keywords: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# 关键词翻译（中文 → 英文）
# ─────────────────────────────────────────────
# 常用中文关键词映射表（离线，避免额外 API 调用）
CN_EN_KEYWORD_MAP = {
    # 科技
    "人工智能": "artificial intelligence",
    "AI": "artificial intelligence technology",
    "机器学习": "machine learning",
    "深度学习": "deep learning neural network",
    "大模型": "large language model AI",
    "算法": "algorithm technology",
    "数据": "data analytics",
    "芯片": "computer chip semiconductor",
    "机器人": "robot automation",
    "自动驾驶": "autonomous driving car",
    "区块链": "blockchain technology",
    "云计算": "cloud computing server",
    "量子计算": "quantum computing",
    "编程": "programming coding",
    "代码": "code programming",
    "互联网": "internet technology",
    "5G": "5G network technology",
    # 商业
    "商业": "business office",
    "企业": "corporate business",
    "创业": "startup entrepreneur",
    "投资": "investment finance",
    "经济": "economy finance",
    "市场": "market business",
    "团队": "team collaboration",
    "会议": "business meeting",
    "办公": "office work",
    "领导": "leadership business",
    "增长": "business growth",
    "数字化": "digital transformation",
    # 医疗
    "医疗": "medical healthcare",
    "健康": "health wellness",
    "医院": "hospital medical",
    "药物": "medicine pharmaceutical",
    "手术": "surgery medical",
    "诊断": "medical diagnosis",
    # 教育
    "教育": "education learning",
    "学习": "learning study",
    "学校": "school education",
    "知识": "knowledge learning",
    "培训": "training education",
    # 自然
    "城市": "city urban",
    "自然": "nature landscape",
    "未来": "future technology",
    "创新": "innovation technology",
    "全球": "global world",
    "社会": "society people",
    "文化": "culture diversity",
    "能源": "energy power",
    "环境": "environment nature",
    "太空": "space universe",
    "科学": "science research",
    "研究": "research laboratory",
    # 媒体/内容
    "视频": "video content creation",
    "音乐": "music audio",
    "艺术": "art creative",
    "设计": "design creative",
    "品牌": "brand marketing",
    "广告": "advertising marketing",
    # 人物
    "人": "people person",
    "女性": "woman female",
    "男性": "man male",
    "年轻人": "young people",
    "老人": "elderly senior",
    "儿童": "children kids",
    "家庭": "family home",
}


def translate_keywords_to_en(keywords: List[str]) -> List[str]:
    """
    将中文关键词翻译为英文（用于 Pexels 搜索）。
    优先使用本地映射表，未命中则直接使用原词（Pexels 也支持部分中文）。
    """
    result = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        # 直接映射
        if kw in CN_EN_KEYWORD_MAP:
            result.append(CN_EN_KEYWORD_MAP[kw])
            continue
        # 部分匹配
        matched = False
        for cn, en in CN_EN_KEYWORD_MAP.items():
            if cn in kw or kw in cn:
                result.append(en)
                matched = True
                break
        if not matched:
            # 如果是纯 ASCII，直接使用
            if kw.isascii():
                result.append(kw)
            else:
                # 中文但未命中，尝试 LLM 翻译（可选）
                # 这里先直接跳过，用英文通用词兜底
                logger.debug(f"  关键词未命中映射: {kw}，跳过")
    # 去重
    seen = set()
    deduped = []
    for r in result:
        if r not in seen:
            seen.add(r)
            deduped.append(r)
    return deduped


def build_search_query(keywords: List[str], visual_type: str = "broll") -> str:
    """
    根据关键词和视觉类型构建 Pexels 搜索词。
    """
    en_keywords = translate_keywords_to_en(keywords)
    if not en_keywords:
        # 根据 visual_type 使用通用词
        fallback_map = {
            "broll": "technology abstract background",
            "ai_image": "abstract digital art",
            "kinetic_text": "clean minimal background",
            "pdf_chart": "data analytics business",
            "template": "clean background minimal",
        }
        return fallback_map.get(visual_type, "technology background")
    # 取前 2 个关键词拼接（避免过长导致无结果）
    return " ".join(en_keywords[:2])


# ─────────────────────────────────────────────
# Pexels 客户端
# ─────────────────────────────────────────────
class PexelsClient:
    """
    Pexels API 客户端，支持视频/图片搜索、下载、本地缓存。
    """

    def __init__(
        self,
        api_key: str,
        cache_dir: str,
        aspect_ratio: str = "9:16",
        preferred_quality: str = "hd",   # sd / hd / uhd
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.aspect_ratio = aspect_ratio
        self.preferred_quality = preferred_quality
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({"Authorization": api_key})
        self._rate_limit_remaining = 25000
        self._search_cache: Dict[str, Any] = {}  # 内存级搜索缓存

        # 加载磁盘搜索缓存
        self._disk_cache_path = self.cache_dir / "search_cache.json"
        self._disk_cache: Dict[str, Any] = {}
        if self._disk_cache_path.exists():
            try:
                self._disk_cache = json.loads(
                    self._disk_cache_path.read_text(encoding="utf-8")
                )
            except Exception:
                pass

    def _save_disk_cache(self):
        self._disk_cache_path.write_text(
            json.dumps(self._disk_cache, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def _get(self, url: str, params: dict, retries: int = 0) -> Optional[dict]:
        """带重试和速率限制感知的 GET 请求"""
        if self._rate_limit_remaining <= 2:
            logger.warning("  Pexels 速率限制接近，等待退避...")
            time.sleep(RATE_LIMIT_BACKOFF)

        try:
            resp = self._session.get(url, params=params, timeout=15)
            # 更新速率限制计数
            remaining = resp.headers.get("x-ratelimit-remaining")
            if remaining:
                self._rate_limit_remaining = int(remaining)

            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                logger.warning(f"  Pexels 速率限制 (429)，等待 {RATE_LIMIT_BACKOFF}s...")
                time.sleep(RATE_LIMIT_BACKOFF)
                if retries < self.max_retries:
                    return self._get(url, params, retries + 1)
            elif resp.status_code == 401:
                logger.error("  Pexels API Key 无效！")
            else:
                logger.warning(f"  Pexels API 错误: {resp.status_code}")
                if retries < self.max_retries:
                    time.sleep(1)
                    return self._get(url, params, retries + 1)
        except requests.RequestException as e:
            logger.warning(f"  Pexels 请求异常: {e}")
            if retries < self.max_retries:
                time.sleep(2)
                return self._get(url, params, retries + 1)
        return None

    # ── 视频搜索 ──
    def search_videos(
        self,
        query: str,
        per_page: int = 10,
        min_duration: int = MIN_VIDEO_DURATION,
        max_duration: int = MAX_VIDEO_DURATION,
    ) -> List[PexelsVideoResult]:
        """
        搜索 Pexels 视频，自动选择最佳分辨率文件。
        """
        # 确定方向
        orientation_map = {"9:16": "portrait", "16:9": "landscape", "1:1": "square"}
        orientation = orientation_map.get(self.aspect_ratio, "portrait")

        # 检查磁盘缓存
        cache_key = f"video:{query}:{orientation}:{per_page}"
        if cache_key in self._disk_cache:
            logger.debug(f"  Pexels 视频搜索命中缓存: {query}")
            return [PexelsVideoResult(**r) for r in self._disk_cache[cache_key]]

        params = {
            "query": query,
            "per_page": per_page,
            "orientation": orientation,
            "size": "medium",  # small/medium/large
        }
        data = self._get(PEXELS_VIDEO_SEARCH_URL, params)
        if not data or not data.get("videos"):
            logger.debug(f"  Pexels 视频搜索无结果: {query}")
            return []

        results = []
        for v in data["videos"]:
            duration = v.get("duration", 0)
            if duration < min_duration or duration > max_duration:
                continue

            # 选择最佳视频文件
            best_file = self._pick_best_video_file(v.get("video_files", []))
            if not best_file:
                continue

            results.append(PexelsVideoResult(
                pexels_id=v["id"],
                duration=duration,
                width=best_file.get("width", 0),
                height=best_file.get("height", 0),
                download_url=best_file["link"],
                quality=best_file.get("quality", "hd"),
                fps=float(best_file.get("fps", 25)),
                photographer=v.get("user", {}).get("name", ""),
                pexels_url=v.get("url", ""),
                keywords=[query],
            ))

        # 保存缓存
        if results:
            self._disk_cache[cache_key] = [
                {k: v for k, v in r.__dict__.items()} for r in results
            ]
            self._save_disk_cache()

        logger.debug(f"  Pexels 视频搜索: '{query}' → {len(results)} 个结果")
        return results

    def _pick_best_video_file(self, video_files: List[dict]) -> Optional[dict]:
        """
        从视频文件列表中选择最佳文件。
        策略：
          1. 优先匹配目标比例（如 9:16 → 竖屏）
          2. 在匹配比例中选 preferred_quality（hd 优先）
          3. 如无匹配比例，选任意 hd 文件
        """
        if not video_files:
            return None

        preferred_res = PREFERRED_RESOLUTIONS.get(self.aspect_ratio, [(1080, 1920)])

        # 按质量优先级排序
        quality_order = {"uhd": 3, "hd": 2, "sd": 1}
        sorted_files = sorted(
            video_files,
            key=lambda f: (
                quality_order.get(f.get("quality", "sd"), 0),
                f.get("width", 0) * f.get("height", 0),
            ),
            reverse=True,
        )

        # 优先找匹配比例的 hd 文件
        for target_w, target_h in preferred_res:
            for f in sorted_files:
                w, h = f.get("width", 0), f.get("height", 0)
                if w == 0 or h == 0:
                    continue
                # 比例匹配（允许 ±5% 误差）
                target_ratio = target_w / target_h
                actual_ratio = w / h
                if abs(actual_ratio - target_ratio) / target_ratio < 0.05:
                    if f.get("quality") in ("hd", "uhd"):
                        return f

        # 退而求其次：找任意 hd 文件
        for f in sorted_files:
            if f.get("quality") in ("hd", "uhd"):
                return f

        # 最后兜底：返回第一个
        return sorted_files[0] if sorted_files else None

    # ── 图片搜索 ──
    def search_photos(
        self,
        query: str,
        per_page: int = 10,
    ) -> List[PexelsPhotoResult]:
        """
        搜索 Pexels 图片。
        """
        orientation_map = {"9:16": "portrait", "16:9": "landscape", "1:1": "square"}
        orientation = orientation_map.get(self.aspect_ratio, "portrait")

        cache_key = f"photo:{query}:{orientation}:{per_page}"
        if cache_key in self._disk_cache:
            logger.debug(f"  Pexels 图片搜索命中缓存: {query}")
            return [PexelsPhotoResult(**r) for r in self._disk_cache[cache_key]]

        params = {
            "query": query,
            "per_page": per_page,
            "orientation": orientation,
        }
        data = self._get(PEXELS_PHOTO_SEARCH_URL, params)
        if not data or not data.get("photos"):
            logger.debug(f"  Pexels 图片搜索无结果: {query}")
            return []

        results = []
        for p in data["photos"]:
            src = p.get("src", {})
            # 选择合适尺寸：portrait > large > medium
            url = src.get("portrait") or src.get("large2x") or src.get("large") or src.get("medium", "")
            if not url:
                continue
            results.append(PexelsPhotoResult(
                pexels_id=p["id"],
                width=p.get("width", 0),
                height=p.get("height", 0),
                download_url=url,
                photographer=p.get("photographer", ""),
                pexels_url=p.get("url", ""),
                keywords=[query],
            ))

        if results:
            self._disk_cache[cache_key] = [
                {k: v for k, v in r.__dict__.items()} for r in results
            ]
            self._save_disk_cache()

        logger.debug(f"  Pexels 图片搜索: '{query}' → {len(results)} 个结果")
        return results

    # ── 多关键词搜索（提高命中率）──
    def search_videos_multi_query(
        self,
        keywords: List[str],
        visual_type: str = "broll",
        segment_duration: float = 5.0,
        max_results: int = 5,
    ) -> List[PexelsVideoResult]:
        """
        使用多个关键词轮询搜索，返回最相关的视频列表。
        """
        en_keywords = translate_keywords_to_en(keywords)
        if not en_keywords:
            en_keywords = [build_search_query(keywords, visual_type)]

        all_results: List[PexelsVideoResult] = []
        seen_ids = set()

        # 策略1：用前两个关键词组合搜索
        if len(en_keywords) >= 2:
            combined = f"{en_keywords[0]} {en_keywords[1]}"
            results = self.search_videos(combined, per_page=5)
            for r in results:
                if r.pexels_id not in seen_ids:
                    seen_ids.add(r.pexels_id)
                    all_results.append(r)

        # 策略2：逐个关键词搜索
        for kw in en_keywords[:3]:
            if len(all_results) >= max_results:
                break
            results = self.search_videos(kw, per_page=5)
            for r in results:
                if r.pexels_id not in seen_ids:
                    seen_ids.add(r.pexels_id)
                    all_results.append(r)

        # 策略3：如果还没有结果，用 visual_type 通用词
        if not all_results:
            fallback_query = build_search_query([], visual_type)
            results = self.search_videos(fallback_query, per_page=5)
            for r in results:
                if r.pexels_id not in seen_ids:
                    seen_ids.add(r.pexels_id)
                    all_results.append(r)

        # 按时长与目标时长的匹配度排序
        def duration_score(r: PexelsVideoResult) -> float:
            # 时长接近 segment_duration 的排前面，但不超过太多
            diff = abs(r.duration - segment_duration)
            return diff

        all_results.sort(key=duration_score)
        return all_results[:max_results]

    def search_photos_multi_query(
        self,
        keywords: List[str],
        visual_type: str = "broll",
        max_results: int = 5,
    ) -> List[PexelsPhotoResult]:
        """
        使用多个关键词轮询搜索图片。
        """
        en_keywords = translate_keywords_to_en(keywords)
        if not en_keywords:
            en_keywords = [build_search_query(keywords, visual_type)]

        all_results: List[PexelsPhotoResult] = []
        seen_ids = set()

        for kw in en_keywords[:3]:
            if len(all_results) >= max_results:
                break
            results = self.search_photos(kw, per_page=5)
            for r in results:
                if r.pexels_id not in seen_ids:
                    seen_ids.add(r.pexels_id)
                    all_results.append(r)

        if not all_results:
            fallback_query = build_search_query([], visual_type)
            results = self.search_photos(fallback_query, per_page=5)
            for r in results:
                if r.pexels_id not in seen_ids:
                    seen_ids.add(r.pexels_id)
                    all_results.append(r)

        return all_results[:max_results]

    # ── 下载 ──
    def download_video(
        self,
        result: PexelsVideoResult,
        output_dir: str,
    ) -> Optional[str]:
        """
        下载 Pexels 视频到本地，支持断点续传（文件已存在则跳过）。
        返回本地文件路径。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 文件名：pexels_{id}_{quality}.mp4
        filename = f"pexels_{result.pexels_id}_{result.quality}.mp4"
        local_path = output_dir / filename

        if local_path.exists() and local_path.stat().st_size > 10240:
            logger.debug(f"  Pexels 视频已缓存: {filename}")
            return str(local_path)

        logger.info(f"  下载 Pexels 视频: {filename} ({result.quality} {result.width}x{result.height})")
        try:
            resp = self._session.get(result.download_url, stream=True, timeout=DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            logger.info(f"  下载完成: {filename} ({downloaded/1024:.0f} KB)")
            return str(local_path)
        except Exception as e:
            logger.error(f"  Pexels 视频下载失败: {e}")
            if local_path.exists():
                local_path.unlink()
            return None

    def download_photo(
        self,
        result: PexelsPhotoResult,
        output_dir: str,
    ) -> Optional[str]:
        """
        下载 Pexels 图片到本地。
        返回本地文件路径。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"pexels_{result.pexels_id}.jpg"
        local_path = output_dir / filename

        if local_path.exists() and local_path.stat().st_size > 1024:
            logger.debug(f"  Pexels 图片已缓存: {filename}")
            return str(local_path)

        logger.info(f"  下载 Pexels 图片: {filename}")
        try:
            resp = self._session.get(result.download_url, stream=True, timeout=DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            return str(local_path)
        except Exception as e:
            logger.error(f"  Pexels 图片下载失败: {e}")
            if local_path.exists():
                local_path.unlink()
            return None

    # ── 一站式：搜索并下载最佳视频 ──
    def fetch_best_video(
        self,
        keywords: List[str],
        visual_type: str,
        segment_duration: float,
        download_dir: str,
    ) -> Optional[str]:
        """
        搜索并下载最佳 Pexels 视频，返回本地路径。
        """
        results = self.search_videos_multi_query(
            keywords=keywords,
            visual_type=visual_type,
            segment_duration=segment_duration,
        )
        if not results:
            logger.debug(f"  Pexels 视频搜索无结果，关键词: {keywords}")
            return None

        # 取第一个结果下载
        best = results[0]
        return self.download_video(best, download_dir)

    def fetch_best_photo(
        self,
        keywords: List[str],
        visual_type: str,
        download_dir: str,
    ) -> Optional[str]:
        """
        搜索并下载最佳 Pexels 图片，返回本地路径。
        """
        results = self.search_photos_multi_query(
            keywords=keywords,
            visual_type=visual_type,
        )
        if not results:
            logger.debug(f"  Pexels 图片搜索无结果，关键词: {keywords}")
            return None

        best = results[0]
        return self.download_photo(best, download_dir)

    @property
    def rate_limit_remaining(self) -> int:
        return self._rate_limit_remaining
