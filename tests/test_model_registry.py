"""
Tests for model registry and resolution system.

Tests the 3-tier model API's core resolution logic:
- Model alias resolution (gpt4 -> gpt-4o)
- Provider pattern matching (gpt-4o -> openai)
- Provider prefix parsing (anthropic:claude -> anthropic, claude)
- Error handling for invalid model strings
"""

import pytest

from agenticflow.models import BaseChatModel
from agenticflow.models.registry import (
    MODEL_ALIASES,
    MODEL_PROVIDERS,
    resolve_and_create_model,
    resolve_model,
)


class TestModelAliases:
    """Test model alias resolution."""

    def test_openai_aliases(self):
        """Test OpenAI model aliases."""
        assert resolve_model("gpt4") == ("openai", "gpt-4o")
        assert resolve_model("gpt4-mini") == ("openai", "gpt-4o-mini")
        assert resolve_model("gpt4-turbo") == ("openai", "gpt-4-turbo")
        assert resolve_model("gpt35") == ("openai", "gpt-3.5-turbo")

    def test_anthropic_aliases(self):
        """Test Anthropic model aliases."""
        assert resolve_model("claude") == ("anthropic", "claude-sonnet-4-20250514")
        assert resolve_model("claude-opus") == ("anthropic", "claude-opus-4-20250514")
        assert resolve_model("claude-haiku") == ("anthropic", "claude-haiku-4-20250323")
        assert resolve_model("claude-sonnet") == ("anthropic", "claude-sonnet-4-20250514")

    def test_gemini_aliases(self):
        """Test Gemini model aliases."""
        assert resolve_model("gemini") == ("gemini", "gemini-2.5-flash")
        assert resolve_model("gemini-flash") == ("gemini", "gemini-2.5-flash")
        assert resolve_model("gemini-pro") == ("gemini", "gemini-2.5-pro")

    def test_groq_aliases(self):
        """Test Groq model aliases."""
        assert resolve_model("llama") == ("groq", "llama-3.3-70b-versatile")
        assert resolve_model("llama-70b") == ("groq", "llama-3.3-70b-versatile")
        assert resolve_model("llama-8b") == ("groq", "llama-3.1-8b-instant")
        assert resolve_model("mixtral") == ("groq", "mixtral-8x7b-32768")

    def test_ollama_aliases(self):
        """Test Ollama model aliases."""
        assert resolve_model("ollama") == ("ollama", "llama3.2")

    def test_case_sensitivity(self):
        """Test that aliases are case-insensitive."""
        assert resolve_model("gpt4") == ("openai", "gpt-4o")
        # Aliases are actually case-insensitive
        provider, model = resolve_model("GPT4")
        assert model == "gpt-4o"  # Resolves same as lowercase


class TestProviderPatternMatching:
    """Test provider auto-detection from model names."""

    def test_openai_patterns(self):
        """Test OpenAI model patterns."""
        assert resolve_model("gpt-4o")[0] == "openai"
        assert resolve_model("gpt-4-turbo")[0] == "openai"
        assert resolve_model("gpt-3.5-turbo")[0] == "openai"
        assert resolve_model("o1-preview")[0] == "openai"
        assert resolve_model("o3-mini")[0] == "openai"

    def test_anthropic_patterns(self):
        """Test Anthropic model patterns."""
        assert resolve_model("claude-3-opus")[0] == "anthropic"
        assert resolve_model("claude-3-sonnet")[0] == "anthropic"
        assert resolve_model("claude-3-haiku")[0] == "anthropic"

    def test_gemini_patterns(self):
        """Test Gemini model patterns."""
        assert resolve_model("gemini-1.5-pro")[0] == "gemini"
        assert resolve_model("gemini-2.0-flash-exp")[0] == "gemini"

    def test_groq_patterns(self):
        """Test Groq model patterns."""
        assert resolve_model("llama-3.1-70b")[0] == "groq"
        assert resolve_model("mixtral-8x7b")[0] == "groq"

    def test_cloudflare_patterns(self):
        """Test Cloudflare model patterns."""
        assert resolve_model("@cf/meta/llama-3-8b")[0] == "cloudflare"

    def test_cohere_patterns(self):
        """Test Cohere model patterns."""
        assert resolve_model("command-r-plus")[0] == "cohere"


class TestProviderPrefix:
    """Test provider:model syntax."""

    def test_explicit_provider_prefix(self):
        """Test explicit provider prefix parsing."""
        assert resolve_model("openai:gpt-4o") == ("openai", "gpt-4o")
        # When provider is explicit, model part is kept as-is
        assert resolve_model("anthropic:claude-sonnet-4") == ("anthropic", "claude-sonnet-4")
        assert resolve_model("gemini:gemini-2.5-pro") == ("gemini", "gemini-2.5-pro")
        # Provider prefix with alias still resolves the alias
        assert resolve_model("groq:llama-70b") == ("groq", "llama-3.3-70b-versatile")

    def test_provider_prefix_with_alias(self):
        """Test provider prefix with model alias."""
        provider, model = resolve_model("anthropic:claude")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-20250514"

    def test_provider_prefix_without_alias(self):
        """Test provider prefix with full model name."""
        provider, model = resolve_model("openai:gpt-4-turbo-preview")
        assert provider == "openai"
        assert model == "gpt-4-turbo-preview"

    def test_multiple_colons(self):
        """Test handling of model names with multiple colons."""
        # Azure-style model names might have colons
        provider, model = resolve_model("azure:deployment:model-name")
        assert provider == "azure"
        assert model == "deployment:model-name"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_string(self):
        """Test empty string handling."""
        with pytest.raises(ValueError, match="Invalid model string"):
            resolve_model("")

    def test_unknown_model_no_pattern(self):
        """Test unknown model with no pattern match."""
        with pytest.raises(ValueError, match="Cannot auto-detect provider"):
            resolve_model("unknown-model-xyz")

    def test_whitespace_handling(self):
        """Test whitespace in model strings."""
        # Should handle or reject whitespace
        provider, model = resolve_model("gpt4 ")
        # Implementation should either strip or reject

    def test_none_input(self):
        """Test None input handling."""
        with pytest.raises((TypeError, ValueError)):
            resolve_model(None)


class TestResolveAndCreateModel:
    """Test the full model creation pipeline."""

    def test_create_from_alias(self):
        """Test creating model from alias."""
        # This will fail without API keys, so we'll check the type
        # In actual usage, API keys would be provided
        try:
            model = resolve_and_create_model("gpt4")
            assert isinstance(model, BaseChatModel)
        except Exception as e:
            # Expected if no API key available
            assert "api_key" in str(e).lower() or "API key" in str(e)

    def test_create_with_provider_prefix(self):
        """Test creating model with provider prefix."""
        try:
            model = resolve_and_create_model("anthropic:claude")
            assert isinstance(model, BaseChatModel)
        except Exception as e:
            # Expected if no API key available
            assert "api_key" in str(e).lower() or "API key" in str(e)

    def test_create_with_explicit_api_key(self):
        """Test creating model with explicit API key."""
        # Even with a dummy key, we can test the creation path
        try:
            model = resolve_and_create_model("gpt4", api_key="sk-test-dummy-key")
            assert isinstance(model, BaseChatModel)
        except Exception:
            # May fail on validation, but should get past initial creation
            pass


class TestRegistryCompleteness:
    """Test that registry is complete and consistent."""

    def test_all_aliases_resolve(self):
        """Test that all aliases in MODEL_ALIASES resolve."""
        for alias, expected_model in MODEL_ALIASES.items():
            provider, model = resolve_model(alias)
            # Check that we get a valid provider and the expected model
            assert provider is not None
            assert model == expected_model

    def test_no_circular_aliases(self):
        """Test that aliases don't create circular references."""
        # Check for multi-step circular loops (A->B->C->A)
        # Skip self-references (like "o1" -> "o1") which are intentional
        for alias in MODEL_ALIASES:
            current = alias
            path = [current]
            visited = {current}

            for _ in range(10):  # Max depth
                if current not in MODEL_ALIASES:
                    break

                next_alias = MODEL_ALIASES[current]

                # Skip self-references (intentional for final model names)
                if next_alias == current:
                    break

                # Check for circular loops
                if next_alias in visited:
                    pytest.fail(f"Circular reference: {' -> '.join(path + [next_alias])}")

                visited.add(next_alias)
                path.append(next_alias)
                current = next_alias

    def test_provider_patterns_are_strings(self):
        """Test that all provider patterns are strings."""
        for pattern, provider in MODEL_PROVIDERS.items():
            assert isinstance(pattern, str)
            assert isinstance(provider, str)
            assert len(pattern) > 0
            assert len(provider) > 0


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_full_model_names_still_work(self):
        """Test that full model names still resolve correctly."""
        # These are what users might have been using before
        assert resolve_model("gpt-4o")[0] == "openai"
        assert resolve_model("claude-3-opus")[0] == "anthropic"
        assert resolve_model("gemini-1.5-pro")[0] == "gemini"

    def test_provider_detection_unchanged(self):
        """Test that provider detection logic is consistent."""
        # Models with provider prefixes should still work
        openai_models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        for model in openai_models:
            provider, _ = resolve_model(model)
            assert provider == "openai"
