"""
Template Vendor Integration Module

Provides a minimal scaffold for future vendor integrations.
This module demonstrates the expected structure and contract patterns
for new vendor adapters without implementing real functionality.

Key Components:
- TemplateVendorConfig: Typed configuration for future vendor API access
- TemplateVendorClient: Placeholder client implementing future lifecycle patterns
- MockTemplateVendorTransport: Deterministic mock transport for testing

Usage:
    from src.integrations.template_vendor import TemplateVendorClient, TemplateVendorConfig
    
    config = TemplateVendorConfig.from_env()
    client = TemplateVendorClient(config=config)
    
    # Example endpoint pattern (not implemented)
    # result = client.submit("example_endpoint", request)

Design Goals:
- Demonstrate vendor integration structure
- Provide import-safe placeholder for testing
- Enable registry discovery without full implementation
- Keep extension-safe for future vendors

Note:
    This is a TEMPLATE ONLY. Real vendor integrations should follow
    the patterns established in src/integrations/minimax/.
"""
from src.integrations.template_vendor.client import (
    TemplateVendorClient,
    TemplateVendorConfig,
)
from src.integrations.template_vendor.transport import (
    TemplateVendorTransport,
    MockTemplateVendorTransport,
)

__all__ = [
    "TemplateVendorClient",
    "TemplateVendorConfig",
    "TemplateVendorTransport",
    "MockTemplateVendorTransport",
]
