"""
Template Vendor Transport

Provides placeholder transport layer for Template Vendor integration.
Follows the same pattern as MinimaxTransport for consistency.

Design Goals:
- Demonstrate transport abstraction pattern
- Enable testing without external dependencies
- Keep structure consistent with other vendor integrations
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class TransportResponse:
    """
    Normalized transport response.
    
    Attributes:
        status_code: HTTP status code
        data: Response payload
        headers: Response headers
    """
    status_code: int
    data: Dict[str, Any]
    headers: Dict[str, str]


class TemplateVendorTransport:
    """
    Placeholder transport interface for Template Vendor.
    
    Real implementations would handle HTTP requests, retries,
    and error handling. This is a structural placeholder only.
    """
    
    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0):
        """
        Initialize transport.
        
        Args:
            base_url: Vendor API base URL
            api_key: Vendor API key
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
    
    def post(self, endpoint: str, payload: Dict[str, Any]) -> TransportResponse:
        """
        Send POST request.
        
        Args:
            endpoint: API endpoint path
            payload: Request body
            
        Returns:
            TransportResponse
            
        Raises:
            NotImplementedError: This is a template placeholder
        """
        raise NotImplementedError(
            "TemplateVendorTransport.post() is a placeholder."
        )


class MockTemplateVendorTransport(TemplateVendorTransport):
    """
    Deterministic mock transport for testing.
    
    Returns predictable responses without making real API calls.
    Follows the pattern of MockMinimaxTransport.
    """
    
    def __init__(self, base_url: str = "http://mock.template-vendor.test", api_key: str = "mock_key", timeout: float = 30.0):
        """
        Initialize mock transport.
        
        Args:
            base_url: Mock base URL
            api_key: Mock API key
            timeout: Mock timeout
        """
        super().__init__(base_url, api_key, timeout)
        self.call_count = 0
    
    def post(self, endpoint: str, payload: Dict[str, Any]) -> TransportResponse:
        """
        Return mock POST response.
        
        Args:
            endpoint: API endpoint path
            payload: Request body
            
        Returns:
            Deterministic mock response
        """
        self.call_count += 1
        return TransportResponse(
            status_code=200,
            data={
                "mock": True,
                "endpoint": endpoint,
                "call_count": self.call_count,
            },
            headers={"Content-Type": "application/json"},
        )
