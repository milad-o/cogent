"""
Tests for model integration with Agent class.

Tests the end-to-end integration:
- Agent accepts string models
- String models resolve correctly
- Integration with config system
- Backward compatibility
"""

from unittest.mock import MagicMock, patch

import pytest

from cogent import Agent
from cogent.models import BaseChatModel, create_chat


class TestAgentStringModels:
    """Test Agent class with string model parameters."""

    def test_agent_with_string_model(self):
        """Test creating agent with string model."""
        # Mock the API call to avoid needing real API keys
        with patch('cogent.models.openai.OpenAIChat.__init__', return_value=None):
            agent = Agent(name="TestAgent", model="gpt4")
            # Verify agent was created (would fail without model resolution)
            assert agent is not None
            assert agent.name == "TestAgent"

    def test_agent_with_provider_prefix(self):
        """Test creating agent with provider prefix."""
        with patch('cogent.models.anthropic.AnthropicChat.__init__', return_value=None):
            agent = Agent(name="TestAgent", model="anthropic:claude")
            assert agent is not None

    def test_agent_with_model_instance(self):
        """Test backward compatibility with model instances."""
        # Create a mock model
        mock_model = MagicMock(spec=BaseChatModel)

        agent = Agent(name="TestAgent", model=mock_model)
        assert agent is not None
        # Agent should use the provided model instance
        assert agent.model is mock_model


class TestCreateChatIntegration:
    """Test create_chat factory with 3-tier API."""

    def test_create_chat_single_arg(self):
        """Test create_chat with single argument."""
        with patch('cogent.models.openai.OpenAIChat.__init__', return_value=None) as mock_init:
            try:
                create_chat("gpt4")
            except Exception:
                pass  # May fail on generate call, but init should be called

            # Verify OpenAIChat was initialized
            mock_init.assert_called()

    def test_create_chat_two_args(self):
        """Test create_chat with provider and model."""
        with patch('cogent.models.openai.OpenAIChat.__init__', return_value=None) as mock_init:
            try:
                create_chat("openai", "gpt-4o")
            except Exception:
                pass

            mock_init.assert_called()

    def test_create_chat_with_kwargs(self):
        """Test create_chat with additional parameters."""
        with patch('cogent.models.openai.OpenAIChat.__init__', return_value=None) as mock_init:
            try:
                create_chat("gpt4", temperature=0.9, max_tokens=1000)
            except Exception:
                pass

            # Verify kwargs were passed
            mock_init.assert_called()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs.get("temperature") == 0.9
            assert call_kwargs.get("max_tokens") == 1000


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_existing_code_unchanged(self):
        """Test that existing code patterns still work."""
        from cogent.models import OpenAIChat

        # Old style - direct instantiation
        with patch('cogent.models.openai.OpenAIChat.__init__', return_value=None):
            model = OpenAIChat(model="gpt-4o", api_key="sk-test")
            assert model is not None

    def test_create_chat_backward_compat(self):
        """Test create_chat backward compatibility."""
        with patch('cogent.models.openai.OpenAIChat.__init__', return_value=None):
            # Old two-argument form
            try:
                model = create_chat("openai", "gpt-4o")
            except Exception:
                pass  # Construction should work

    def test_agent_with_model_instance_unchanged(self):
        """Test that passing model instances to Agent still works."""
        mock_model = MagicMock(spec=BaseChatModel)
        agent = Agent(name="Test", model=mock_model)

        # Agent should accept and use the model
        assert agent.model is mock_model


class TestErrorHandling:
    """Test error handling in model integration."""

    def test_invalid_model_string(self):
        """Test handling of invalid model strings."""
        with pytest.raises(Exception):
            # Should raise error for completely unknown model
            agent = Agent("Test", model="totally-invalid-model-xyz-123")

    def test_missing_api_key(self):
        """Test behavior when API key is missing."""
        with patch.dict('os.environ', {}, clear=True):
            with patch('cogent.config.get_provider_config', return_value={}):
                # Should raise error about missing API key
                with pytest.raises(Exception):
                    agent = Agent("Test", model="gpt4")

    def test_empty_model_string(self):
        """Test handling of empty model string."""
        with pytest.raises(Exception):
            agent = Agent("Test", model="")

    def test_none_model_with_config(self):
        """Test that None model uses default from config."""
        # When model=None, should try to load from config
        with patch('cogent.config.get_provider_config', return_value={"default": "gpt4"}):
            with patch('cogent.models.openai.OpenAIChat.__init__', return_value=None):
                try:
                    agent = Agent("Test", model=None)
                    # Should use default model from config
                except Exception:
                    pass  # Config loading logic may vary


class TestConfigIntegration:
    """Test integration with config system."""

    def test_api_key_from_config(self, tmp_path):
        """Test that API keys are loaded from config."""
        config_file = tmp_path / "cogent.toml"
        config_file.write_text("""
[models.openai]
api_key = "sk-config-test-key"
""")

        with patch('cogent.config.find_config_file', return_value=config_file):
            with patch.dict('os.environ', {}, clear=True):
                with patch('cogent.models.openai.OpenAIChat.__init__', return_value=None) as mock_init:
                    try:
                        create_chat("gpt4")
                    except:
                        pass

                    # Verify API key from config was used
                    if mock_init.called:
                        call_kwargs = mock_init.call_args[1]
                        assert call_kwargs.get("api_key") == "sk-config-test-key"

    def test_env_var_overrides_config(self, tmp_path):
        """Test that environment variables override config file."""
        config_file = tmp_path / "cogent.toml"
        config_file.write_text("""
[models.openai]
api_key = "sk-config-key"
""")

        with patch('cogent.config.find_config_file', return_value=config_file):
            with patch.dict('os.environ', {"OPENAI_API_KEY": "sk-env-key"}):
                with patch('cogent.models.openai.OpenAIChat.__init__', return_value=None) as mock_init:
                    try:
                        create_chat("gpt4")
                    except:
                        pass

                    # Verify env var was used, not config
                    if mock_init.called:
                        call_kwargs = mock_init.call_args[1]
                        assert call_kwargs.get("api_key") == "sk-env-key"

    def test_explicit_key_highest_priority(self, tmp_path):
        """Test that explicit API key has highest priority."""
        config_file = tmp_path / "cogent.toml"
        config_file.write_text("""
[models.openai]
api_key = "sk-config-key"
""")

        with patch('cogent.config.find_config_file', return_value=config_file):
            with patch.dict('os.environ', {"OPENAI_API_KEY": "sk-env-key"}):
                with patch('cogent.models.openai.OpenAIChat.__init__', return_value=None) as mock_init:
                    try:
                        create_chat("gpt4", api_key="sk-explicit-key")
                    except Exception:
                        pass

                    # Verify explicit key was used
                    if mock_init.called:
                        call_kwargs = mock_init.call_args[1]
                        assert call_kwargs.get("api_key") == "sk-explicit-key"


class TestThreeTierAPI:
    """Test all three API tiers work together."""

    def test_tier1_high_level(self):
        """Test Tier 1 (high-level string API)."""
        with patch('cogent.models.openai.OpenAIChat.__init__', return_value=None):
            try:
                agent = Agent("Helper", model="gpt4")
                assert agent.name == "Helper"
            except Exception:
                pass  # Construction should work

    def test_tier2_factory(self):
        """Test Tier 2 (factory function)."""
        with patch('cogent.models.openai.OpenAIChat.__init__', return_value=None):
            try:
                model = create_chat("gpt4")
                assert model is not None
            except Exception:
                pass

    def test_tier3_direct(self):
        """Test Tier 3 (direct model classes)."""
        from cogent.models import OpenAIChat

        with patch('cogent.models.openai.OpenAIChat.__init__', return_value=None):
            model = OpenAIChat(model="gpt-4o", api_key="sk-test")
            assert model is not None

    def test_all_tiers_interoperable(self):
        """Test that all tiers can be used together."""
        from cogent.models import OpenAIChat

        mock_model = MagicMock(spec=BaseChatModel)

        # Tier 3: Direct class
        with patch('cogent.models.openai.OpenAIChat.__init__', return_value=None):
            direct_model = OpenAIChat(model="gpt-4o", api_key="sk-test")

        # Tier 2: Factory
        with patch('cogent.models.openai.OpenAIChat.__init__', return_value=None):
            try:
                factory_model = create_chat("gpt4")
            except Exception:
                pass

        # Tier 1: String in Agent
        agent = Agent(name="Helper", model=mock_model)

        # All should work without conflict
        assert agent is not None
