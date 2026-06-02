"""
Minimax Integration Module

Provides unified client layer for Minimax voice (T2A) and media (video) endpoints.
Supports both synchronous and asynchronous lifecycle patterns normalized to
canonical broker contracts.

Key Components:
- MinimaxUnifiedClient: Production-grade client implementing ProviderLifecycleClient
- MinimaxConfig: Typed configuration for Minimax API access
- MockMinimaxTransport: Deterministic mock transport for testing
- MinimaxTTSAdapter: High-level TTS adapter for PDF pipeline integration
- generate_tts_minimax: Drop-in TTS function for step_pdf.py

Usage:
    from src.integrations.minimax import MinimaxUnifiedClient, MinimaxConfig
    
    config = MinimaxConfig.from_env()
    client = MinimaxUnifiedClient(config=config)
    
    # Voice (T2A) endpoint - sync
    result = client.submit("voice", request)
    
    # Video endpoint - async with polling
    result = client.submit("video", request)
    poll_result = client.poll(result.job_id)
    fetch_result = client.fetch(result.job_id, output_dir)
    
    # TTS adapter (for PDF pipeline)
    from src.integrations.minimax import generate_tts_minimax
    
    output = generate_tts_minimax(
        script_path="input/script.md",
        output_audio="input/voice_full.mp3",
    )
"""
from src.integrations.minimax.client import (
    MinimaxUnifiedClient,
    MinimaxConfig,
    MinimaxEndpointType,
)
from src.integrations.minimax.transport import (
    MinimaxTransport,
    MockMinimaxTransport,
    TransportResponse,
)
from src.integrations.minimax.tts import (
    MinimaxTTSAdapter,
    MinimaxTTSError,
    generate_tts_minimax,
    MockMinimaxTTSTransportFactory,
)
from src.core.api_config import MINIMAX_VOICES

__all__ = [
    "MinimaxUnifiedClient",
    "MinimaxConfig",
    "MinimaxEndpointType",
    "MinimaxTransport",
    "MockMinimaxTransport", 
    "TransportResponse",
    "MinimaxTTSAdapter",
    "MinimaxTTSError",
    "generate_tts_minimax",
    "MockMinimaxTTSTransportFactory",
    "MINIMAX_VOICES",
]
