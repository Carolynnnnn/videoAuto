"""
Test: vendor_template_registration

Validates that the template_vendor integration package can be discovered
and imported without errors. This test ensures the scaffold structure is
correct and ready for future vendor implementations.

Test Goals:
- Verify template_vendor package imports cleanly
- Validate config and client classes are accessible
- Confirm transport layer is available
- Check that the scaffold follows minimax integration patterns

Note:
    This test validates the STRUCTURE only. Real vendor integrations
    should add functional tests for their specific API contracts.
"""
import pytest
from src.integrations.template_vendor import (
    TemplateVendorClient,
    TemplateVendorConfig,
    TemplateVendorTransport,
    MockTemplateVendorTransport,
)


def test_vendor_template_registration_imports():
    """
    Test that template_vendor package exports expected components.
    
    This validates:
    - Package __init__.py is correctly structured
    - All key components are importable
    - No import-time errors occur
    """
    # Verify all expected exports are present
    assert TemplateVendorClient is not None
    assert TemplateVendorConfig is not None
    assert TemplateVendorTransport is not None
    assert MockTemplateVendorTransport is not None


def test_vendor_template_registration_config_instantiation():
    """
    Test that TemplateVendorConfig can be instantiated.
    
    This validates:
    - Config dataclass is properly defined
    - from_env() class method works
    - Default values are set correctly
    """
    # Test direct instantiation
    config = TemplateVendorConfig()
    assert config.api_key == "template_vendor_key_placeholder"
    assert config.base_url == "https://api.template-vendor.example"
    assert config.timeout == 30.0
    
    # Test from_env() class method
    config_from_env = TemplateVendorConfig.from_env()
    assert config_from_env.api_key == "template_vendor_key_placeholder"
    assert config_from_env.base_url == "https://api.template-vendor.example"


def test_vendor_template_registration_client_instantiation():
    """
    Test that TemplateVendorClient can be instantiated.
    
    This validates:
    - Client class is properly defined
    - Constructor accepts config parameter
    - is_available() method exists and returns False (placeholder)
    """
    # Test with explicit config
    config = TemplateVendorConfig()
    client = TemplateVendorClient(config=config)
    assert client.config == config
    assert client.is_available() is False
    
    # Test with default config
    client_default = TemplateVendorClient()
    assert client_default.config is not None
    assert client_default.is_available() is False


def test_vendor_template_registration_transport_structure():
    """
    Test that transport classes follow expected patterns.
    
    This validates:
    - Base transport class is properly defined
    - Mock transport class extends base transport
    - Mock transport provides deterministic responses
    """
    # Test mock transport instantiation
    mock_transport = MockTemplateVendorTransport()
    assert mock_transport.base_url == "http://mock.template-vendor.test"
    assert mock_transport.api_key == "mock_key"
    assert mock_transport.timeout == 30.0
    assert mock_transport.call_count == 0
    
    # Test mock transport provides deterministic response
    response = mock_transport.post("/test/endpoint", {"test": "payload"})
    assert response.status_code == 200
    assert response.data["mock"] is True
    assert response.data["endpoint"] == "/test/endpoint"
    assert response.data["call_count"] == 1
    assert mock_transport.call_count == 1
    
    # Verify call count increments
    response2 = mock_transport.post("/another/endpoint", {})
    assert response2.data["call_count"] == 2
    assert mock_transport.call_count == 2


def test_vendor_template_registration_client_submit_not_implemented():
    """
    Test that client.submit() raises NotImplementedError.
    
    This validates:
    - Placeholder methods are properly marked as unimplemented
    - Error messages are clear and actionable
    """
    client = TemplateVendorClient()
    
    with pytest.raises(NotImplementedError) as exc_info:
        client.submit("example_endpoint", {"test": "request"})
    
    error_message = str(exc_info.value)
    assert "TemplateVendorClient.submit()" in error_message
    assert "placeholder" in error_message.lower()


def test_vendor_template_registration_mimics_minimax_structure():
    """
    Test that template_vendor structure mirrors minimax integration.
    
    This validates:
    - Config class uses dataclass pattern
    - Client class has similar constructor signature
    - Transport abstraction follows same pattern
    - Mock transport provides test doubles
    """
    # Config pattern validation
    config = TemplateVendorConfig.from_env()
    assert hasattr(config, "api_key")
    assert hasattr(config, "base_url")
    assert hasattr(config, "timeout")
    
    # Client pattern validation
    client = TemplateVendorClient(config=config)
    assert hasattr(client, "config")
    assert hasattr(client, "is_available")
    assert hasattr(client, "submit")
    
    # Transport pattern validation
    transport = MockTemplateVendorTransport()
    assert hasattr(transport, "base_url")
    assert hasattr(transport, "api_key")
    assert hasattr(transport, "timeout")
    assert hasattr(transport, "post")
    assert hasattr(transport, "call_count")
