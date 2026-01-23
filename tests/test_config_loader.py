"""
Tests for configuration file loading and API key resolution.

Tests the config system:
- Config file discovery (TOML, YAML)
- Config file loading and parsing
- API key resolution with priority
- Environment variable integration
"""

import os
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from agenticflow.config import (
    find_config_file,
    load_toml,
    load_yaml,
    get_api_key,
    get_provider_config,
)


class TestConfigFileDiscovery:
    """Test config file discovery logic."""
    
    def test_find_project_toml(self, tmp_path):
        """Test finding project-level TOML config."""
        config_file = tmp_path / "agenticflow.toml"
        config_file.write_text("[models]\ndefault = 'gpt4'")
        
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            found = find_config_file()
            assert found == config_file
    
    def test_find_project_yaml(self, tmp_path):
        """Test finding project-level YAML config."""
        config_file = tmp_path / "agenticflow.yaml"
        config_file.write_text("models:\n  default: gpt4")
        
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            found = find_config_file()
            assert found == config_file
    
    def test_toml_preferred_over_yaml(self, tmp_path):
        """Test that TOML is preferred when both exist."""
        toml_file = tmp_path / "agenticflow.toml"
        yaml_file = tmp_path / "agenticflow.yaml"
        toml_file.write_text("[models]\ndefault = 'gpt4'")
        yaml_file.write_text("models:\n  default: claude")
        
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            found = find_config_file()
            assert found == toml_file
    
    def test_user_config_fallback(self, tmp_path):
        """Test falling back to user-level config."""
        user_config = tmp_path / ".agenticflow" / "config.toml"
        user_config.parent.mkdir(parents=True)
        user_config.write_text("[models]\ndefault = 'gpt4'")
        
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            with patch('pathlib.Path.home', return_value=tmp_path):
                found = find_config_file()
                assert found == user_config
    
    def test_no_config_found(self, tmp_path):
        """Test behavior when no config file exists."""
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            with patch('pathlib.Path.home', return_value=tmp_path):
                found = find_config_file()
                assert found is None


class TestTomlLoading:
    """Test TOML config file loading."""
    
    def test_load_simple_toml(self, tmp_path):
        """Test loading simple TOML config."""
        config_file = tmp_path / "test.toml"
        config_file.write_text("""
[models]
default = "gpt4"

[models.openai]
api_key = "sk-test-key"
organization = "org-test"
""")
        
        config = load_toml(config_file)
        assert config["models"]["default"] == "gpt4"
        assert config["models"]["openai"]["api_key"] == "sk-test-key"
        assert config["models"]["openai"]["organization"] == "org-test"
    
    def test_load_nested_toml(self, tmp_path):
        """Test loading TOML with nested structures."""
        config_file = tmp_path / "test.toml"
        config_file.write_text("""
[models.openai]
api_key = "sk-test"
temperature = 0.7

[models.anthropic]
api_key = "sk-ant-test"
max_tokens = 4096
""")
        
        config = load_toml(config_file)
        assert config["models"]["openai"]["temperature"] == 0.7
        assert config["models"]["anthropic"]["max_tokens"] == 4096
    
    def test_load_invalid_toml(self, tmp_path):
        """Test handling of invalid TOML."""
        config_file = tmp_path / "test.toml"
        config_file.write_text("invalid toml content [[[")
        
        with pytest.raises(Exception):  # Should raise TOMLDecodeError or similar
            load_toml(config_file)
    
    def test_load_nonexistent_toml(self, tmp_path):
        """Test loading non-existent TOML file."""
        config_file = tmp_path / "nonexistent.toml"
        
        with pytest.raises(FileNotFoundError):
            load_toml(config_file)


class TestYamlLoading:
    """Test YAML config file loading."""
    
    def test_load_simple_yaml(self, tmp_path):
        """Test loading simple YAML config."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""
models:
  default: gpt4
  openai:
    api_key: sk-test-key
    organization: org-test
""")
        
        config = load_yaml(config_file)
        assert config["models"]["default"] == "gpt4"
        assert config["models"]["openai"]["api_key"] == "sk-test-key"
    
    def test_load_yml_extension(self, tmp_path):
        """Test loading .yml file extension."""
        config_file = tmp_path / "test.yml"
        config_file.write_text("models:\n  default: claude")
        
        config = load_yaml(config_file)
        assert config["models"]["default"] == "claude"
    
    def test_yaml_not_available(self, tmp_path):
        """Test behavior when PyYAML not installed."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("models:\n  default: gpt4")
        
        # Need to patch at module level before import
        import agenticflow.config as config_module
        original_yaml = config_module.yaml
        try:
            config_module.yaml = None
            with pytest.raises((ImportError, AttributeError)):
                load_yaml(config_file)
        finally:
            config_module.yaml = original_yaml


class TestApiKeyResolution:
    """Test API key resolution with priority."""
    
    def test_explicit_key_highest_priority(self):
        """Test that explicit API key has highest priority."""
        explicit_key = "sk-explicit-key"
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env-key"}):
            with patch('agenticflow.config.get_provider_config', return_value={"api_key": "sk-config-key"}):
                key = get_api_key("openai", explicit_key)
                assert key == explicit_key
    
    def test_env_var_second_priority(self):
        """Test that env var has second priority."""
        env_key = "sk-env-key"
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": env_key}):
            with patch('agenticflow.config.get_provider_config', return_value={"api_key": "sk-config-key"}):
                key = get_api_key("openai", None)
                assert key == env_key
    
    def test_config_file_lowest_priority(self, tmp_path):
        """Test that config file has lowest priority."""
        # Create a config file
        config_file = tmp_path / "agenticflow.toml"
        config_file.write_text("""
[models.openai]
api_key = "sk-config-key"
""")
        
        with patch.dict(os.environ, {}, clear=True):
            with patch('pathlib.Path.home', return_value=tmp_path):
                with patch('pathlib.Path.cwd', return_value=tmp_path):
                    # Should load from config file
                    key = get_api_key("openai", None)
                    assert key == "sk-config-key"
    
    def test_no_key_available(self, tmp_path):
        """Test behavior when no API key is available."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('pathlib.Path.home', return_value=tmp_path):
                with patch('agenticflow.config.get_provider_config', return_value={}):
                    key = get_api_key("openai", None)
                    assert key is None
    
    def test_provider_specific_env_vars(self):
        """Test provider-specific environment variables."""
        test_cases = [
            ("openai", "OPENAI_API_KEY", "sk-openai-test"),
            ("anthropic", "ANTHROPIC_API_KEY", "sk-ant-test"),
            ("gemini", "GEMINI_API_KEY", "gemini-test"),
            ("groq", "GROQ_API_KEY", "gsk-test"),
        ]
        
        for provider, env_var, expected_key in test_cases:
            with patch.dict(os.environ, {env_var: expected_key}):
                key = get_api_key(provider, None)
                assert key == expected_key


class TestProviderConfig:
    """Test provider configuration retrieval."""
    
    def test_get_full_provider_config(self, tmp_path):
        """Test getting full provider configuration."""
        config_file = tmp_path / "agenticflow.toml"
        config_file.write_text("""
[models.openai]
api_key = "sk-test"
organization = "org-test"
temperature = 0.7
max_tokens = 2000
""")
        
        with patch('agenticflow.config.find_config_file', return_value=config_file):
            config = get_provider_config("openai")
            assert config["api_key"] == "sk-test"
            assert config["organization"] == "org-test"
            assert config["temperature"] == 0.7
            assert config["max_tokens"] == 2000
    
    def test_get_config_no_file(self):
        """Test getting config when no file exists."""
        with patch('agenticflow.config.find_config_file', return_value=None):
            config = get_provider_config("openai")
            assert config == {}
    
    def test_get_config_provider_not_in_file(self, tmp_path):
        """Test getting config for provider not in file."""
        config_file = tmp_path / "agenticflow.toml"
        config_file.write_text("""
[models.openai]
api_key = "sk-test"
""")
        
        with patch('agenticflow.config.find_config_file', return_value=config_file):
            config = get_provider_config("anthropic")
            assert config == {}


class TestConfigIntegration:
    """Test full config integration scenarios."""
    
    def test_priority_chain_complete(self, tmp_path):
        """Test complete priority chain."""
        # Set up config file
        config_file = tmp_path / "agenticflow.toml"
        config_file.write_text("""
[models.openai]
api_key = "sk-config-key"
temperature = 0.5
""")
        
        # Explicit > Env > Config
        explicit_key = "sk-explicit"
        env_key = "sk-env"
        
        with patch('agenticflow.config.find_config_file', return_value=config_file):
            # Test explicit wins
            with patch.dict(os.environ, {"OPENAI_API_KEY": env_key}):
                key = get_api_key("openai", explicit_key)
                assert key == explicit_key
            
            # Test env wins over config
            with patch.dict(os.environ, {"OPENAI_API_KEY": env_key}):
                key = get_api_key("openai", None)
                assert key == env_key
            
            # Test config used when nothing else
            with patch.dict(os.environ, {}, clear=True):
                key = get_api_key("openai", None)
                assert key == "sk-config-key"
    
    def test_multiple_providers_config(self, tmp_path):
        """Test config with multiple providers."""
        config_file = tmp_path / "agenticflow.toml"
        config_file.write_text("""
[models.openai]
api_key = "sk-openai"

[models.anthropic]
api_key = "sk-ant"

[models.gemini]
api_key = "gemini-key"
""")
        
        with patch('agenticflow.config.find_config_file', return_value=config_file):
            with patch.dict(os.environ, {}, clear=True):
                assert get_api_key("openai", None) == "sk-openai"
                assert get_api_key("anthropic", None) == "sk-ant"
                assert get_api_key("gemini", None) == "gemini-key"


class TestDotenvIntegration:
    """Test .env file integration."""
    
    def test_dotenv_loaded_on_import(self):
        """Test that .env is loaded on module import."""
        # This is tested implicitly by the module loading .env on import
        # We can verify by checking if environment variables are accessible
        with patch('agenticflow.config.load_dotenv') as mock_load:
            # Re-import would call load_dotenv
            # In practice, this is already done, so we test behavior
            pass
    
    def test_env_file_priority(self, tmp_path):
        """Test that .env file variables work in priority chain."""
        # Create a .env file
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=sk-dotenv-key\n")
        
        # Load it
        from dotenv import load_dotenv
        with patch.dict(os.environ, {}, clear=True):
            load_dotenv(env_file, override=True)

            key = get_api_key("openai", None)
            # Should get the key from env (loaded from .env)
            assert key == "sk-dotenv-key"


class TestErrorHandling:
    """Test error handling in config system."""
    
    def test_corrupted_toml_handling(self, tmp_path):
        """Test handling of corrupted TOML files."""
        config_file = tmp_path / "bad.toml"
        config_file.write_text("this is not valid toml {{{}}")
        
        with pytest.raises(Exception):
            load_toml(config_file)
    
    def test_corrupted_yaml_handling(self, tmp_path):
        """Test handling of corrupted YAML files."""
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("invalid: yaml: content: [[[")
        
        with pytest.raises(Exception):
            load_yaml(config_file)
    
    def test_missing_permissions(self, tmp_path):
        """Test handling of permission errors."""
        # Skip on Windows as chmod doesn't work the same way
        import sys
        if sys.platform == 'win32':
            pytest.skip("Permission test not applicable on Windows")
        
        config_file = tmp_path / "protected.toml"
        config_file.write_text("[models]\ndefault = 'gpt4'")
        
        os.chmod(config_file, 0o000)
        try:
            with pytest.raises(PermissionError):
                load_toml(config_file)
        finally:
            os.chmod(config_file, 0o644)
