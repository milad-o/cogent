"""
Central configuration for AgenticFlow examples.

Uses Pydantic Settings to automatically load from .env file and environment variables.
All examples should import settings from here instead of manually handling env vars.

Usage:
    from config import settings, get_model, get_embeddings
    
    model = get_model()       # Uses LLM_PROVIDER from .env
    embeddings = get_embeddings()  # Uses EMBEDDING_PROVIDER from .env

Required .env configuration:
    LLM_PROVIDER=openai|gemini|anthropic|groq|azure|ollama
    EMBEDDING_PROVIDER=openai|azure|ollama
    
    # Plus the appropriate API key for your chosen provider
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Find project root (where .env is located)
PROJECT_ROOT = Path(__file__).parent.parent

# Valid provider choices
LLMProvider = Literal["gemini", "openai", "anthropic", "groq", "azure", "ollama"]
EmbeddingProvider = Literal["openai", "azure", "ollama"]


class Settings(BaseSettings):
    """Central settings for all examples.
    
    Automatically loads from:
    1. Environment variables
    2. .env file in project root
    
    You MUST set LLM_PROVIDER and EMBEDDING_PROVIDER explicitly in .env.
    """
    
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )
    
    # ==========================================================================
    # Required: Provider Selection
    # ==========================================================================
    
    llm_provider: LLMProvider = Field(
        alias="LLM_PROVIDER",
        description="LLM provider to use. Required.",
    )
    
    embedding_provider: EmbeddingProvider = Field(
        alias="EMBEDDING_PROVIDER",
        description="Embedding provider to use. Required.",
    )
    
    # ==========================================================================
    # API Keys
    # ==========================================================================
    
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    
    # ==========================================================================
    # Azure OpenAI Configuration
    # ==========================================================================
    
    azure_openai_endpoint: str | None = Field(default=None, alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment: str | None = Field(default=None, alias="AZURE_OPENAI_DEPLOYMENT")
    azure_openai_api_version: str = Field(default="2024-02-15-preview", alias="AZURE_OPENAI_API_VERSION")
    
    # Azure auth type: api_key, managed_identity, or default (DefaultAzureCredential)
    azure_auth_type: Literal["api_key", "managed_identity", "default"] = Field(
        default="api_key",
        alias="AZURE_AUTH_TYPE",
    )
    
    # For api_key auth
    azure_openai_api_key: str | None = Field(default=None, alias="AZURE_OPENAI_API_KEY")
    
    # For managed_identity auth (optional client ID for user-assigned identity)
    azure_managed_identity_client_id: str | None = Field(
        default=None,
        alias="AZURE_MANAGED_IDENTITY_CLIENT_ID",
    )
    
    # Azure embedding deployment (separate from chat deployment)
    azure_openai_embedding_deployment: str | None = Field(
        default=None,
        alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    )
    
    # ==========================================================================
    # Ollama Configuration (local models)
    # ==========================================================================
    
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    ollama_model: str = Field(default="qwen2.5:7b", alias="OLLAMA_MODEL")
    ollama_embedding_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBEDDING_MODEL")
    
    # ==========================================================================
    # Model Names (can override defaults per provider)
    # ==========================================================================
    
    gemini_model: str = Field(default="gemini-2.0-flash-exp", alias="GEMINI_MODEL")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    anthropic_model: str = Field(default="claude-sonnet-4-20250514", alias="ANTHROPIC_MODEL")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    
    # ==========================================================================
    # Embedding Model Names
    # ==========================================================================
    
    openai_embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")
    
    # ==========================================================================
    # Example Settings
    # ==========================================================================
    
    verbose_level: Literal["minimal", "verbose", "debug", "trace"] = Field(
        default="verbose",
        alias="VERBOSE_LEVEL",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Singleton instance for easy import
settings = get_settings()


def get_model(provider: LLMProvider | None = None):
    """Get a chat model for the specified or configured provider.
    
    Args:
        provider: Override provider (uses LLM_PROVIDER from .env if None).
        
    Returns:
        A configured chat model instance.
        
    Raises:
        ValueError: If provider not configured properly.
    """
    s = settings
    provider = provider or s.llm_provider
    
    if provider == "gemini":
        if not s.gemini_api_key:
            raise ValueError("LLM_PROVIDER=gemini requires GEMINI_API_KEY")
        from agenticflow.models.gemini import GeminiChat
        return GeminiChat(model=s.gemini_model, api_key=s.gemini_api_key)
    
    elif provider == "openai":
        if not s.openai_api_key:
            raise ValueError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
        from agenticflow.models.openai import OpenAIChat
        return OpenAIChat(model=s.openai_model, api_key=s.openai_api_key)
    
    elif provider == "anthropic":
        if not s.anthropic_api_key:
            raise ValueError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY")
        from agenticflow.models.anthropic import AnthropicChat
        return AnthropicChat(model=s.anthropic_model, api_key=s.anthropic_api_key)
    
    elif provider == "groq":
        if not s.groq_api_key:
            raise ValueError("LLM_PROVIDER=groq requires GROQ_API_KEY")
        from agenticflow.models.groq import GroqChat
        return GroqChat(model=s.groq_model, api_key=s.groq_api_key)
    
    elif provider == "azure":
        return _create_azure_chat(s)
    
    elif provider == "ollama":
        from agenticflow.models.ollama import OllamaChat
        return OllamaChat(model=s.ollama_model, host=s.ollama_host)
    
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def _create_azure_chat(s: Settings):
    """Create Azure OpenAI chat model with appropriate auth method."""
    if not s.azure_openai_endpoint:
        raise ValueError("LLM_PROVIDER=azure requires AZURE_OPENAI_ENDPOINT")
    if not s.azure_openai_deployment:
        raise ValueError("LLM_PROVIDER=azure requires AZURE_OPENAI_DEPLOYMENT")
    
    from agenticflow.models.azure import AzureChat
    
    if s.azure_auth_type == "api_key":
        if not s.azure_openai_api_key:
            raise ValueError("AZURE_AUTH_TYPE=api_key requires AZURE_OPENAI_API_KEY")
        return AzureChat(
            deployment=s.azure_openai_deployment,
            azure_endpoint=s.azure_openai_endpoint,
            api_version=s.azure_openai_api_version,
            api_key=s.azure_openai_api_key,
        )
    
    elif s.azure_auth_type == "managed_identity":
        return AzureChat(
            deployment=s.azure_openai_deployment,
            azure_endpoint=s.azure_openai_endpoint,
            api_version=s.azure_openai_api_version,
            use_managed_identity=True,
            managed_identity_client_id=s.azure_managed_identity_client_id,
        )
    
    elif s.azure_auth_type == "default":
        return AzureChat(
            deployment=s.azure_openai_deployment,
            azure_endpoint=s.azure_openai_endpoint,
            api_version=s.azure_openai_api_version,
            use_azure_ad=True,
        )
    
    else:
        raise ValueError(f"Unknown AZURE_AUTH_TYPE: {s.azure_auth_type}")


def get_embeddings(provider: EmbeddingProvider | None = None):
    """Get an embedding provider for the specified or configured provider.
    
    Args:
        provider: Override provider (uses EMBEDDING_PROVIDER from .env if None).
        
    Returns:
        A configured embedding provider instance.
        
    Raises:
        ValueError: If provider not configured properly.
    """
    s = settings
    provider = provider or s.embedding_provider
    
    if provider == "openai":
        if not s.openai_api_key:
            raise ValueError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")
        from agenticflow.vectorstore import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=s.openai_embedding_model,
            api_key=s.openai_api_key,
        )
    
    elif provider == "azure":
        return _create_azure_embeddings(s)
    
    elif provider == "ollama":
        from agenticflow.vectorstore import OllamaEmbeddings
        return OllamaEmbeddings(
            model=s.ollama_embedding_model,
            base_url=s.ollama_host,
        )
    
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}")


def _create_azure_embeddings(s: Settings):
    """Create Azure OpenAI embeddings with appropriate auth method."""
    if not s.azure_openai_endpoint:
        raise ValueError("EMBEDDING_PROVIDER=azure requires AZURE_OPENAI_ENDPOINT")
    if not s.azure_openai_embedding_deployment:
        raise ValueError("EMBEDDING_PROVIDER=azure requires AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    
    from agenticflow.models.azure import AzureEmbedding
    
    if s.azure_auth_type == "api_key":
        if not s.azure_openai_api_key:
            raise ValueError("AZURE_AUTH_TYPE=api_key requires AZURE_OPENAI_API_KEY")
        return AzureEmbedding(
            deployment=s.azure_openai_embedding_deployment,
            azure_endpoint=s.azure_openai_endpoint,
            api_version=s.azure_openai_api_version,
            api_key=s.azure_openai_api_key,
        )
    
    elif s.azure_auth_type == "managed_identity":
        return AzureEmbedding(
            deployment=s.azure_openai_embedding_deployment,
            azure_endpoint=s.azure_openai_endpoint,
            api_version=s.azure_openai_api_version,
            use_managed_identity=True,
            managed_identity_client_id=s.azure_managed_identity_client_id,
        )
    
    elif s.azure_auth_type == "default":
        return AzureEmbedding(
            deployment=s.azure_openai_embedding_deployment,
            azure_endpoint=s.azure_openai_endpoint,
            api_version=s.azure_openai_api_version,
            use_azure_ad=True,
        )
    
    else:
        raise ValueError(f"Unknown AZURE_AUTH_TYPE: {s.azure_auth_type}")


def print_config():
    """Print current configuration (useful for debugging)."""
    s = settings
    print("=" * 50)
    print("AgenticFlow Configuration")
    print("=" * 50)
    print(f"LLM Provider:       {s.llm_provider}")
    print(f"Embedding Provider: {s.embedding_provider}")
    print(f"Verbose Level:      {s.verbose_level}")
    print()
    
    # Show provider-specific config
    if s.llm_provider == "gemini":
        print(f"  Model: {s.gemini_model}")
        print(f"  API Key: {'✓ set' if s.gemini_api_key else '✗ missing!'}")
    elif s.llm_provider == "openai":
        print(f"  Model: {s.openai_model}")
        print(f"  API Key: {'✓ set' if s.openai_api_key else '✗ missing!'}")
    elif s.llm_provider == "anthropic":
        print(f"  Model: {s.anthropic_model}")
        print(f"  API Key: {'✓ set' if s.anthropic_api_key else '✗ missing!'}")
    elif s.llm_provider == "groq":
        print(f"  Model: {s.groq_model}")
        print(f"  API Key: {'✓ set' if s.groq_api_key else '✗ missing!'}")
    elif s.llm_provider == "azure":
        print(f"  Deployment: {s.azure_openai_deployment or '✗ missing!'}")
        print(f"  Endpoint: {s.azure_openai_endpoint or '✗ missing!'}")
        print(f"  Auth Type: {s.azure_auth_type}")
        if s.azure_auth_type == "api_key":
            print(f"  API Key: {'✓ set' if s.azure_openai_api_key else '✗ missing!'}")
        elif s.azure_auth_type == "managed_identity":
            print(f"  Client ID: {s.azure_managed_identity_client_id or '(system-assigned)'}")
    elif s.llm_provider == "ollama":
        print(f"  Model: {s.ollama_model}")
        print(f"  Host: {s.ollama_host}")
    
    print()
    print("Embedding Config:")
    if s.embedding_provider == "openai":
        print(f"  Model: {s.openai_embedding_model}")
        print(f"  API Key: {'✓ set' if s.openai_api_key else '✗ missing!'}")
    elif s.embedding_provider == "azure":
        print(f"  Deployment: {s.azure_openai_embedding_deployment or '✗ missing!'}")
        print(f"  Auth Type: {s.azure_auth_type}")
    elif s.embedding_provider == "ollama":
        print(f"  Model: {s.ollama_embedding_model}")
        print(f"  Host: {s.ollama_host}")
    
    print("=" * 50)


if __name__ == "__main__":
    print_config()
