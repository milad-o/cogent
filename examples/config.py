"""
Central configuration for AgenticFlow examples.

Uses Pydantic Settings to automatically load from .env file and environment variables.
All examples should import settings from here instead of manually handling env vars.

Usage:
    from config import settings, get_model
    
    # Get the default model (based on available API keys)
    model = get_model()
    
    # Or get a specific provider
    model = get_model("gemini")
    model = get_model("openai")
    model = get_model("anthropic")
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Find project root (where .env is located)
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """Central settings for all examples.
    
    Automatically loads from:
    1. Environment variables
    2. .env file in project root
    
    API keys are optional - examples will use whichever is available.
    """
    
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra env vars
        case_sensitive=False,  # Allow GEMINI_API_KEY or gemini_api_key
        populate_by_name=True,  # Allow both field name and alias
    )
    
    # ==========================================================================
    # API Keys (all optional - examples use what's available)
    # ==========================================================================
    
    gemini_api_key: str | None = Field(default=None, description="Google Gemini API key")
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    groq_api_key: str | None = Field(default=None, description="Groq API key")
    
    # Azure OpenAI (optional)
    azure_openai_api_key: str | None = Field(default=None)
    azure_openai_endpoint: str | None = Field(default=None)
    azure_openai_api_version: str = Field(default="2024-02-15-preview")
    azure_openai_deployment: str | None = Field(default=None)
    
    # Ollama (local models - no API key needed, just set DEFAULT_PROVIDER=ollama)
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    ollama_model: str = Field(default="qwen2.5:7b", alias="OLLAMA_MODEL")
    
    # ==========================================================================
    # Default Model Settings
    # ==========================================================================
    
    default_provider: Literal["gemini", "openai", "anthropic", "groq", "azure", "ollama"] | None = Field(
        default=None,
        alias="DEFAULT_PROVIDER",
        description="Preferred model provider. If not set, uses first available.",
    )
    
    # Model names per provider (can be overridden in .env)
    gemini_model: str = Field(default="gemini-2.0-flash-exp", alias="GEMINI_MODEL")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    anthropic_model: str = Field(default="claude-sonnet-4-20250514", alias="ANTHROPIC_MODEL")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    
    # ==========================================================================
    # Example Settings
    # ==========================================================================
    
    verbose_level: Literal["minimal", "verbose", "debug", "trace"] = Field(
        default="verbose",
        description="Default verbosity for examples",
    )
    
    # ==========================================================================
    # Helper Properties
    # ==========================================================================
    
    @property
    def has_gemini(self) -> bool:
        """Check if Gemini API key is configured."""
        return bool(self.gemini_api_key)
    
    @property
    def has_openai(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.openai_api_key)
    
    @property
    def has_anthropic(self) -> bool:
        """Check if Anthropic API key is configured."""
        return bool(self.anthropic_api_key)
    
    @property
    def has_groq(self) -> bool:
        """Check if Groq API key is configured."""
        return bool(self.groq_api_key)
    
    @property
    def has_azure(self) -> bool:
        """Check if Azure OpenAI is configured."""
        return bool(self.azure_openai_api_key and self.azure_openai_endpoint)
    
    @property
    def has_ollama(self) -> bool:
        """Check if Ollama is selected as the default provider."""
        return self.default_provider == "ollama"
    
    @property
    def available_providers(self) -> list[str]:
        """List of providers with configured API keys."""
        providers = []
        if self.has_gemini:
            providers.append("gemini")
        if self.has_openai:
            providers.append("openai")
        if self.has_anthropic:
            providers.append("anthropic")
        if self.has_groq:
            providers.append("groq")
        if self.has_azure:
            providers.append("azure")
        if self.has_ollama:
            providers.append("ollama")
        return providers
    
    def get_preferred_provider(self) -> str | None:
        """Get the preferred provider based on settings and availability."""
        if self.default_provider and self.default_provider in self.available_providers:
            return self.default_provider
        # Return first available
        return self.available_providers[0] if self.available_providers else None


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Singleton instance for easy import
settings = get_settings()


def get_model(provider: str | None = None):
    """Get a chat model for the specified or default provider.
    
    Args:
        provider: Specific provider to use. If None, uses default/first available.
        
    Returns:
        A configured chat model instance.
        
    Raises:
        ValueError: If no API keys are configured or provider not available.
        
    Example:
        from config import get_model
        
        model = get_model()  # Uses default/first available
        model = get_model("gemini")  # Specifically use Gemini
    """
    s = settings
    
    # Determine which provider to use
    if provider is None:
        provider = s.get_preferred_provider()
    
    if provider is None:
        raise ValueError(
            "No API keys configured! Please set at least one of:\n"
            "  - GEMINI_API_KEY\n"
            "  - OPENAI_API_KEY\n"
            "  - ANTHROPIC_API_KEY\n"
            "  - GROQ_API_KEY\n"
            "in your .env file or environment variables."
        )
    
    if provider == "gemini":
        if not s.has_gemini:
            raise ValueError("GEMINI_API_KEY not configured")
        from agenticflow.models.gemini import GeminiChat
        return GeminiChat(model=s.gemini_model, api_key=s.gemini_api_key)
    
    elif provider == "openai":
        if not s.has_openai:
            raise ValueError("OPENAI_API_KEY not configured")
        from agenticflow.models import ChatModel
        return ChatModel(model=s.openai_model, api_key=s.openai_api_key)
    
    elif provider == "anthropic":
        if not s.has_anthropic:
            raise ValueError("ANTHROPIC_API_KEY not configured")
        from agenticflow.models.anthropic import AnthropicChat
        return AnthropicChat(model=s.anthropic_model, api_key=s.anthropic_api_key)
    
    elif provider == "groq":
        if not s.has_groq:
            raise ValueError("GROQ_API_KEY not configured")
        from agenticflow.models.groq import GroqChat
        return GroqChat(model=s.groq_model, api_key=s.groq_api_key)
    
    elif provider == "azure":
        if not s.has_azure:
            raise ValueError("Azure OpenAI not configured (need API key and endpoint)")
        from agenticflow.models.azure import AzureChat
        return AzureChat(
            deployment=s.azure_openai_deployment or "gpt-4o",
            api_key=s.azure_openai_api_key,
            endpoint=s.azure_openai_endpoint,
            api_version=s.azure_openai_api_version,
        )
    
    elif provider == "ollama":
        from agenticflow.models.ollama import OllamaChat
        return OllamaChat(
            model=s.ollama_model,
            host=s.ollama_host,
        )
    
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Available: gemini, openai, anthropic, groq, azure, ollama"
        )


def print_config():
    """Print current configuration (useful for debugging)."""
    s = settings
    print("=" * 50)
    print("AgenticFlow Example Configuration")
    print("=" * 50)
    print(f"Available providers: {', '.join(s.available_providers) or 'None!'}")
    print(f"Default provider: {s.get_preferred_provider() or 'None'}")
    print(f"Verbose level: {s.verbose_level}")
    print()
    print("Configured Models:")
    print(f"  Gemini:    {'✓ ' + s.gemini_model if s.has_gemini else '✗ not set'}")
    print(f"  OpenAI:    {'✓ ' + s.openai_model if s.has_openai else '✗ not set'}")
    print(f"  Anthropic: {'✓ ' + s.anthropic_model if s.has_anthropic else '✗ not set'}")
    print(f"  Groq:      {'✓ ' + s.groq_model if s.has_groq else '✗ not set'}")
    print(f"  Azure:     {'✓ configured' if s.has_azure else '✗ not set'}")
    print(f"  Ollama:    {'✓ ' + s.ollama_model + ' @ ' + s.ollama_host if s.default_provider == 'ollama' else '(set DEFAULT_PROVIDER=ollama to use)'}")
    print()
    print("To change provider, add to .env:")
    print("  DEFAULT_PROVIDER=ollama  # or: gemini, openai, anthropic, groq, azure")
    print("  OLLAMA_MODEL=qwen2.5:7b  # optional, defaults to qwen2.5:7b")
    print("=" * 50)


if __name__ == "__main__":
    # When run directly, show configuration
    print_config()
