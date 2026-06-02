"""
API 配置管理

支持：
  - DeepSeek API（LLM，OpenAI 兼容接口）
  - ElevenLabs API（高质量 TTS）
  - OpenAI API（备用 LLM + TTS）
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional


DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")

PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")

ELEVENLABS_VOICES = {
    "rachel": "21m00Tcm4TlvDq8ikWAM",
    "adam": "pNInz6obpgDQGcFmaJgB",
    "bella": "EXAVITQu4vr4xnSDxMaL",
    "elli": "MF3mGyEYCl7XYWbV9V6O",
    "josh": "TxGEqnHWrfWFTfGW9XjX",
    "arnold": "VR6AewLTigWG4xSOukaG",
    "default": "21m00Tcm4TlvDq8ikWAM",
}

MINIMAX_VOICES = {
    "male-qn-qingse": "male-qn-qingse",
    "female-tianmei": "Tianmei",
    "female-qnshaonv": "female-qnshaonv",
    "male-qn-jingying": "male-qn-jingying",
    "default": "male-qn-qingse",
}


@dataclass
class StabilityConfig:
    api_timeout: float = 30.0
    api_connect_timeout: float = 10.0
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    retry_exponential_base: float = 2.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_recovery_timeout: float = 60.0
    download_timeout: float = 120.0
    download_chunk_size: int = 65536


DEFAULT_STABILITY = StabilityConfig()


@dataclass
class APIConfig:
    llm_provider: str = "deepseek"
    llm_model: str = DEEPSEEK_MODEL
    llm_api_key: str = DEEPSEEK_API_KEY
    llm_base_url: str = DEEPSEEK_BASE_URL
    
    tts_provider: str = "elevenlabs"
    elevenlabs_api_key: str = ELEVENLABS_API_KEY
    elevenlabs_voice_id: str = ELEVENLABS_VOICES["rachel"]
    elevenlabs_model: str = "eleven_multilingual_v2"
    
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    openai_tts_voice: str = "alloy"
    
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    
    def get_llm_client(self):
        from openai import OpenAI
        return OpenAI(
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            timeout=self.stability.api_timeout,
        )
    
    def get_elevenlabs_client(self):
        from elevenlabs import ElevenLabs
        return ElevenLabs(api_key=self.elevenlabs_api_key)


DEFAULT_CONFIG = APIConfig()
