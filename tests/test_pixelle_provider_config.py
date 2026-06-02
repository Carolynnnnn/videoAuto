#!/usr/bin/env python3
"""
Tests for pixelle_snapshot.config_loader - Provider configuration validation

Covers:
- Happy path: valid config with all required secrets
- Error path: missing required secrets with actionable diagnostics
- Test mode: deterministic mode bypasses secret requirements
- Default values and boundary conditions
"""
import os
import pytest
from pixelle_snapshot.config_loader import (
    ProviderConfig,
    ProviderConfigError,
    load_provider_config,
)


class TestProviderConfigHappyPath:
    """Test valid configuration scenarios."""
    
    def test_production_config_with_all_secrets(self, monkeypatch):
        """Valid production config loads successfully with typed object and defaults."""
        monkeypatch.setenv("PIXELLE_PROVIDER_URL", "https://api.example.com")
        monkeypatch.setenv("PIXELLE_PROVIDER_API_KEY", "test_key_12345")
        monkeypatch.setenv("MINIMAX_API_KEY", "minimax_test_key")
        monkeypatch.setenv("PIXELLE_TEST_MODE", "0")
        
        config = load_provider_config()
        
        assert config.provider_url == "https://api.example.com"
        assert config.provider_api_key == "test_key_12345"
        assert config.minimax_api_key == "minimax_test_key"
        assert config.test_mode is False
        assert config.timeout_seconds == 300.0
        assert config.max_retries == 3
        assert config.retry_base_delay == 1.0
        assert config.rollout_enabled is False
        assert config.rollout_percentage == 0
    
    def test_config_with_custom_values(self, monkeypatch):
        """Custom timeout, retry, and rollout values are parsed correctly."""
        monkeypatch.setenv("PIXELLE_PROVIDER_URL", "https://api.example.com")
        monkeypatch.setenv("PIXELLE_PROVIDER_API_KEY", "test_key_12345")
        monkeypatch.setenv("MINIMAX_API_KEY", "minimax_test_key")
        monkeypatch.setenv("PIXELLE_TEST_MODE", "0")
        monkeypatch.setenv("PIXELLE_PROVIDER_TIMEOUT", "600.0")
        monkeypatch.setenv("PIXELLE_PROVIDER_MAX_RETRIES", "5")
        monkeypatch.setenv("PIXELLE_PROVIDER_RETRY_BASE_DELAY", "2.0")
        monkeypatch.setenv("PIXELLE_ROLLOUT_ENABLED", "1")
        monkeypatch.setenv("PIXELLE_ROLLOUT_PERCENTAGE", "10")
        
        config = load_provider_config()
        
        assert config.timeout_seconds == 600.0
        assert config.max_retries == 5
        assert config.retry_base_delay == 2.0
        assert config.rollout_enabled is True
        assert config.rollout_percentage == 10
    
    def test_test_mode_bypasses_secret_validation(self, monkeypatch):
        """Test mode does not require provider URL or API key."""
        monkeypatch.setenv("PIXELLE_TEST_MODE", "1")
        # Intentionally do not set PIXELLE_PROVIDER_URL or PIXELLE_PROVIDER_API_KEY
        
        config = load_provider_config()
        
        assert config.test_mode is True
        assert config.provider_url is None
        assert config.provider_api_key is None
        # Should not raise ProviderConfigError


class TestProviderConfigErrorPath:
    """Test validation failures with actionable error messages."""
    
    def test_missing_provider_url_fails_fast(self, monkeypatch):
        """Missing PIXELLE_PROVIDER_URL in legacy mode raises actionable error."""
        monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
        monkeypatch.setenv("PIXELLE_TEST_MODE", "0")
        monkeypatch.setenv("PIXELLE_PROVIDER_API_KEY", "test_key_12345")
        # Intentionally do not set PIXELLE_PROVIDER_URL
        
        with pytest.raises(ProviderConfigError) as exc_info:
            load_provider_config()
        
        error_msg = str(exc_info.value)
        assert "PIXELLE_PROVIDER_URL" in error_msg
        assert "ACTION REQUIRED" in error_msg
        assert "export PIXELLE_PROVIDER_URL" in error_msg
        assert "PIXELLE_TEST_MODE=1" in error_msg
    
    def test_missing_provider_api_key_fails_fast(self, monkeypatch):
        """Missing PIXELLE_PROVIDER_API_KEY in legacy mode raises actionable error."""
        monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
        monkeypatch.setenv("PIXELLE_TEST_MODE", "0")
        monkeypatch.setenv("PIXELLE_PROVIDER_URL", "https://api.example.com")
        # Intentionally do not set PIXELLE_PROVIDER_API_KEY
        
        with pytest.raises(ProviderConfigError) as exc_info:
            load_provider_config()
        
        error_msg = str(exc_info.value)
        assert "PIXELLE_PROVIDER_API_KEY" in error_msg
        assert "ACTION REQUIRED" in error_msg
        assert "export PIXELLE_PROVIDER_API_KEY" in error_msg
        assert "PIXELLE_TEST_MODE=1" in error_msg
    
    def test_invalid_timeout_fails(self, monkeypatch):
        """Negative or zero timeout raises validation error."""
        monkeypatch.setenv("PIXELLE_PROVIDER_URL", "https://api.example.com")
        monkeypatch.setenv("PIXELLE_PROVIDER_API_KEY", "test_key_12345")
        monkeypatch.setenv("MINIMAX_API_KEY", "minimax_test_key")
        monkeypatch.setenv("PIXELLE_TEST_MODE", "0")
        monkeypatch.setenv("PIXELLE_PROVIDER_TIMEOUT", "-1.0")
        
        with pytest.raises(ProviderConfigError) as exc_info:
            load_provider_config()
        
        error_msg = str(exc_info.value)
        assert "PIXELLE_PROVIDER_TIMEOUT" in error_msg
        assert "Timeout must be positive" in error_msg
    
    def test_invalid_max_retries_fails(self, monkeypatch):
        """Negative max retries raises validation error."""
        monkeypatch.setenv("PIXELLE_PROVIDER_URL", "https://api.example.com")
        monkeypatch.setenv("PIXELLE_PROVIDER_API_KEY", "test_key_12345")
        monkeypatch.setenv("MINIMAX_API_KEY", "minimax_test_key")
        monkeypatch.setenv("PIXELLE_TEST_MODE", "0")
        monkeypatch.setenv("PIXELLE_PROVIDER_MAX_RETRIES", "-1")
        
        with pytest.raises(ProviderConfigError) as exc_info:
            load_provider_config()
        
        error_msg = str(exc_info.value)
        assert "PIXELLE_PROVIDER_MAX_RETRIES" in error_msg
        assert "Max retries must be non-negative" in error_msg
    
    def test_invalid_rollout_percentage_fails(self, monkeypatch):
        """Rollout percentage outside 0-100 range raises validation error."""
        monkeypatch.setenv("PIXELLE_PROVIDER_URL", "https://api.example.com")
        monkeypatch.setenv("PIXELLE_PROVIDER_API_KEY", "test_key_12345")
        monkeypatch.setenv("MINIMAX_API_KEY", "minimax_test_key")
        monkeypatch.setenv("PIXELLE_TEST_MODE", "0")
        monkeypatch.setenv("PIXELLE_ROLLOUT_PERCENTAGE", "150")
        
        with pytest.raises(ProviderConfigError) as exc_info:
            load_provider_config()
        
        error_msg = str(exc_info.value)
        assert "PIXELLE_ROLLOUT_PERCENTAGE" in error_msg
        assert "Rollout percentage must be 0-100" in error_msg


class TestProviderConfigDirectInstantiation:
    """Test ProviderConfig dataclass validation directly."""
    
    def test_valid_production_config_validates(self):
        """Valid production config passes validation."""
        config = ProviderConfig(
            provider_url="https://api.example.com",
            provider_api_key="test_key_12345",
            minimax_api_key="minimax_test_key",
            test_mode=False,
        )
        
        config.validate()
    
    def test_test_mode_config_validates_without_secrets(self):
        """Test mode config validates even without provider secrets."""
        config = ProviderConfig(
            test_mode=True,
        )
        
        # Should not raise
        config.validate()
    
    def test_production_config_without_url_fails(self):
        """Legacy mode config without URL fails validation."""
        config = ProviderConfig(
            backend_mode="legacy",
            provider_api_key="test_key_12345",
            minimax_api_key="minimax_test_key",
            test_mode=False,
        )
        
        with pytest.raises(ProviderConfigError) as exc_info:
            config.validate()
        
        assert "PIXELLE_PROVIDER_URL" in str(exc_info.value)
    
    def test_production_config_without_api_key_fails(self):
        """Legacy mode config without API key fails validation."""
        config = ProviderConfig(
            backend_mode="legacy",
            provider_url="https://api.example.com",
            minimax_api_key="minimax_test_key",
            test_mode=False,
        )
        
        with pytest.raises(ProviderConfigError) as exc_info:
            config.validate()
        
        assert "PIXELLE_PROVIDER_API_KEY" in str(exc_info.value)


class TestConfigDefaults:
    """Test default value handling."""
    
    def test_default_values_applied(self, monkeypatch):
        """Default values are applied when env vars not set."""
        monkeypatch.setenv("PIXELLE_PROVIDER_URL", "https://api.example.com")
        monkeypatch.setenv("PIXELLE_PROVIDER_API_KEY", "test_key_12345")
        monkeypatch.setenv("MINIMAX_API_KEY", "minimax_test_key")
        monkeypatch.setenv("PIXELLE_TEST_MODE", "0")
        
        config = load_provider_config()
        
        assert config.timeout_seconds == 300.0
        assert config.max_retries == 3
        assert config.retry_base_delay == 1.0
        assert config.rollout_enabled is False
        assert config.rollout_percentage == 0


class TestBackendModeValidation:
    """Test PIXELLE_BACKEND_MODE validation and legacy mode support."""
    
    @pytest.mark.step4
    @pytest.mark.legacy_provider_mode
    def test_legacy_provider_mode_happy(self, monkeypatch):
        """Backend mode 'legacy' is accepted and config loads with provider path expectations."""
        monkeypatch.setenv("PIXELLE_PROVIDER_URL", "https://api.example.com")
        monkeypatch.setenv("PIXELLE_PROVIDER_API_KEY", "test_key_12345")
        monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
        monkeypatch.setenv("PIXELLE_TEST_MODE", "0")
        
        config = load_provider_config()
        
        assert config.backend_mode == "legacy"
        assert config.provider_url == "https://api.example.com"
        assert config.provider_api_key == "test_key_12345"
        assert config.test_mode is False
    
    @pytest.mark.step4
    @pytest.mark.legacy_provider_mode
    def test_legacy_mode_invalid_flag(self, monkeypatch):
        """Invalid PIXELLE_BACKEND_MODE fails fast and message lists allowed values."""
        monkeypatch.setenv("PIXELLE_PROVIDER_URL", "https://api.example.com")
        monkeypatch.setenv("PIXELLE_PROVIDER_API_KEY", "test_key_12345")
        monkeypatch.setenv("PIXELLE_BACKEND_MODE", "invalid_mode")
        monkeypatch.setenv("PIXELLE_TEST_MODE", "0")
        
        with pytest.raises(ProviderConfigError) as exc_info:
            load_provider_config()
        
        error_msg = str(exc_info.value)
        assert "PIXELLE_BACKEND_MODE" in error_msg
        assert "invalid_mode" in error_msg
        assert "legacy" in error_msg
        assert "direct" in error_msg
        assert "ACTION REQUIRED" in error_msg
    
    @pytest.mark.step4
    def test_direct_mode_happy_without_provider_credentials(self, monkeypatch):
        """Direct mode does NOT require provider url/key, only minimax credentials."""
        monkeypatch.setenv("MINIMAX_API_KEY", "minimax_test_key")
        monkeypatch.setenv("PIXELLE_BACKEND_MODE", "direct")
        monkeypatch.setenv("PIXELLE_TEST_MODE", "0")
        
        config = load_provider_config()
        
        assert config.backend_mode == "direct"
        assert config.minimax_api_key == "minimax_test_key"
        assert config.minimax_base_url == "https://api.minimaxi.com"
        assert config.provider_url is None
        assert config.provider_api_key is None
    
    @pytest.mark.step4
    def test_direct_mode_requires_minimax_api_key(self, monkeypatch):
        """Direct mode still requires MINIMAX_API_KEY."""
        monkeypatch.setenv("PIXELLE_BACKEND_MODE", "direct")
        monkeypatch.setenv("PIXELLE_TEST_MODE", "0")
        
        with pytest.raises(ProviderConfigError) as exc_info:
            load_provider_config()
        
        error_msg = str(exc_info.value)
        assert "MINIMAX_API_KEY" in error_msg
        assert "ACTION REQUIRED" in error_msg
    
    @pytest.mark.step4
    @pytest.mark.legacy_provider_mode
    def test_legacy_mode_requires_provider_url(self, monkeypatch):
        """Legacy mode requires PIXELLE_PROVIDER_URL."""
        monkeypatch.setenv("PIXELLE_PROVIDER_API_KEY", "test_key_12345")
        monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
        monkeypatch.setenv("PIXELLE_TEST_MODE", "0")
        
        with pytest.raises(ProviderConfigError) as exc_info:
            load_provider_config()
        
        error_msg = str(exc_info.value)
        assert "PIXELLE_PROVIDER_URL" in error_msg
        assert "ACTION REQUIRED" in error_msg
        assert "PIXELLE_BACKEND_MODE=direct" in error_msg
    
    @pytest.mark.step4
    @pytest.mark.legacy_provider_mode
    def test_legacy_mode_requires_provider_api_key(self, monkeypatch):
        """Legacy mode requires PIXELLE_PROVIDER_API_KEY."""
        monkeypatch.setenv("PIXELLE_PROVIDER_URL", "https://api.example.com")
        monkeypatch.setenv("PIXELLE_BACKEND_MODE", "legacy")
        monkeypatch.setenv("PIXELLE_TEST_MODE", "0")
        
        with pytest.raises(ProviderConfigError) as exc_info:
            load_provider_config()
        
        error_msg = str(exc_info.value)
        assert "PIXELLE_PROVIDER_API_KEY" in error_msg
        assert "ACTION REQUIRED" in error_msg
        assert "PIXELLE_BACKEND_MODE=direct" in error_msg


class TestVendorConfigPreflight:
    """Test vendor configuration preflight validation."""
    
    def test_vendor_config_preflight_pass(self, monkeypatch):
        """All required vendor config present passes validation."""
        monkeypatch.setenv("PIXELLE_PROVIDER_URL", "https://api.example.com")
        monkeypatch.setenv("PIXELLE_PROVIDER_API_KEY", "test_key_12345")
        monkeypatch.setenv("MINIMAX_API_KEY", "minimax_test_key_valid")
        monkeypatch.setenv("MINIMAX_BASE_URL", "https://api.minimaxi.com")
        monkeypatch.setenv("PIXELLE_TEST_MODE", "0")
        
        config = load_provider_config()
        
        assert config.minimax_api_key == "minimax_test_key_valid"
        assert config.minimax_base_url == "https://api.minimaxi.com"
        assert config.test_mode is False
    
    def test_vendor_config_preflight_missing_key(self, monkeypatch):
        """Missing MINIMAX_API_KEY in production mode raises preflight error."""
        monkeypatch.setenv("PIXELLE_PROVIDER_URL", "https://api.example.com")
        monkeypatch.setenv("PIXELLE_PROVIDER_API_KEY", "test_key_12345")
        monkeypatch.setenv("PIXELLE_TEST_MODE", "0")
        
        with pytest.raises(ProviderConfigError) as exc_info:
            load_provider_config()
        
        error_msg = str(exc_info.value)
        assert "MINIMAX_API_KEY" in error_msg
        assert "ACTION REQUIRED" in error_msg
        assert "export MINIMAX_API_KEY" in error_msg
        assert "PIXELLE_TEST_MODE=1" in error_msg
    
    def test_vendor_config_default_base_url(self, monkeypatch):
        """MINIMAX_BASE_URL defaults to production endpoint."""
        monkeypatch.setenv("PIXELLE_PROVIDER_URL", "https://api.example.com")
        monkeypatch.setenv("PIXELLE_PROVIDER_API_KEY", "test_key_12345")
        monkeypatch.setenv("MINIMAX_API_KEY", "minimax_test_key")
        monkeypatch.setenv("PIXELLE_TEST_MODE", "0")
        
        config = load_provider_config()
        
        assert config.minimax_base_url == "https://api.minimaxi.com"
    
    def test_vendor_config_empty_base_url_fails(self, monkeypatch):
        """Empty MINIMAX_BASE_URL in production mode raises validation error."""
        monkeypatch.setenv("PIXELLE_PROVIDER_URL", "https://api.example.com")
        monkeypatch.setenv("PIXELLE_PROVIDER_API_KEY", "test_key_12345")
        monkeypatch.setenv("MINIMAX_API_KEY", "minimax_test_key")
        monkeypatch.setenv("MINIMAX_BASE_URL", "")
        monkeypatch.setenv("PIXELLE_TEST_MODE", "0")
        
        with pytest.raises(ProviderConfigError) as exc_info:
            load_provider_config()
        
        error_msg = str(exc_info.value)
        assert "MINIMAX_BASE_URL" in error_msg
        assert "must be non-empty" in error_msg
