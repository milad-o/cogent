"""
Central configuration for AgenticFlow examples.

Uses Pydantic Settings to automatically load from .env file and environment variables.
All examples should import settings from here instead of manually handling env vars.

Usage:
    from config import settings, get_model, get_embeddings

    model = get_model()       # Uses LLM_PROVIDER from .env
    embeddings = get_embeddings()  # Uses EMBEDDING_PROVIDER from .env

Required .env configuration:
    LLM_PROVIDER=openai|gemini|anthropic|groq|azure|ollama|mistral|github|cohere|cloudflare
    EMBEDDING_PROVIDER=openai|azure|ollama|github|cohere|cloudflare

    # Plus the appropriate API key for your chosen provider
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Find project root and examples directory
EXAMPLES_DIR = Path(__file__).parent
PROJECT_ROOT = EXAMPLES_DIR.parent

# Valid provider choices
LLMProvider = Literal[
    "gemini",
    "openai",
    "anthropic",
    "groq",
    "azure",
    "ollama",
    "mistral",
    "github",
    "cohere",
    "cloudflare",
]
EmbeddingProvider = Literal[
    "openai", "azure", "ollama", "github", "cohere", "cloudflare"
]


class Settings(BaseSettings):
    """Central settings for all examples.

    Automatically loads from:
    1. Environment variables
    2. .env file in project root

    You MUST set LLM_PROVIDER and EMBEDDING_PROVIDER explicitly in .env.
    """

    model_config = SettingsConfigDict(
        env_file=EXAMPLES_DIR / ".env",  # Load from examples/.env
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
    mistral_api_key: str | None = Field(default=None, alias="MISTRAL_API_KEY")
    github_token: str | None = Field(default=None, alias="GITHUB_TOKEN")
    cohere_api_key: str | None = Field(default=None, alias="COHERE_API_KEY")
    cloudflare_api_token: str | None = Field(default=None, alias="CLOUDFLARE_API_TOKEN")
    cloudflare_account_id: str | None = Field(
        default=None, alias="CLOUDFLARE_ACCOUNT_ID"
    )

    # ==========================================================================
    # Azure OpenAI Configuration
    # ==========================================================================

    azure_openai_endpoint: str | None = Field(
        default=None, alias="AZURE_OPENAI_ENDPOINT"
    )
    azure_openai_deployment: str | None = Field(
        default=None, alias="AZURE_OPENAI_DEPLOYMENT"
    )
    azure_openai_api_version: str = Field(
        default="2024-02-15-preview", alias="AZURE_OPENAI_API_VERSION"
    )

    # Azure auth type:
    # - api_key
    # - default (DefaultAzureCredential)
    # - managed_identity
    # - client_secret (service principal)
    azure_auth_type: Literal[
        "api_key", "managed_identity", "default", "client_secret"
    ] = Field(
        default="api_key",
        alias="AZURE_OPENAI_AUTH_TYPE",
    )

    # For api_key auth
    azure_openai_api_key: str | None = Field(default=None, alias="AZURE_OPENAI_API_KEY")

    # Optional Entra / identity client id:
    # - For managed_identity: user-assigned MI client id (system-assigned MI doesn't need it)
    # - For client_secret: service principal client id (required)
    azure_openai_client_id: str | None = Field(
        default=None,
        alias="AZURE_OPENAI_CLIENT_ID",
    )

    # For client_secret auth (service principal)
    azure_openai_tenant_id: str | None = Field(
        default=None, alias="AZURE_OPENAI_TENANT_ID"
    )
    azure_openai_client_secret: str | None = Field(
        default=None, alias="AZURE_OPENAI_CLIENT_SECRET"
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
    ollama_model: str = Field(default="qwen2.5:7b", alias="OLLAMA_CHAT_MODEL")
    ollama_embedding_model: str = Field(
        default="nomic-embed-text", alias="OLLAMA_EMBEDDING_MODEL"
    )

    # ==========================================================================
    # Model Names (can override defaults per provider)
    # ==========================================================================

    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_CHAT_MODEL")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_CHAT_MODEL")
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514", alias="ANTHROPIC_CHAT_MODEL"
    )
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_CHAT_MODEL")
    mistral_model: str = Field(
        default="mistral-small-latest", alias="MISTRAL_CHAT_MODEL"
    )
    github_model: str = Field(default="gpt-4o-mini", alias="GITHUB_CHAT_MODEL")
    cohere_model: str = Field(default="command-r-plus", alias="COHERE_CHAT_MODEL")
    cloudflare_model: str = Field(
        default="@cf/meta/llama-3.1-8b-instruct", alias="CLOUDFLARE_CHAT_MODEL"
    )

    # ==========================================================================
    # Embedding Model Names
    # ==========================================================================

    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )
    github_embedding_model: str = Field(
        default="text-embedding-3-large", alias="GITHUB_EMBEDDING_MODEL"
    )
    cohere_embedding_model: str = Field(
        default="embed-english-v3.0", alias="COHERE_EMBEDDING_MODEL"
    )
    cloudflare_embedding_model: str = Field(
        default="@cf/baai/bge-base-en-v1.5", alias="CLOUDFLARE_EMBEDDING_MODEL"
    )

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

    elif provider == "mistral":
        if not s.mistral_api_key:
            raise ValueError("LLM_PROVIDER=mistral requires MISTRAL_API_KEY")
        from agenticflow.models.mistral import MistralChat

        return MistralChat(model=s.mistral_model, api_key=s.mistral_api_key)

    elif provider == "github":
        if not s.github_token:
            raise ValueError("LLM_PROVIDER=github requires GITHUB_TOKEN")
        from agenticflow.models.azure import AzureAIFoundryChat

        return AzureAIFoundryChat.from_github(
            model=s.github_model,
            token=s.github_token,
        )

    elif provider == "cohere":
        if not s.cohere_api_key:
            raise ValueError("LLM_PROVIDER=cohere requires COHERE_API_KEY")
        from agenticflow.models.cohere import CohereChat

        return CohereChat(model=s.cohere_model, api_key=s.cohere_api_key)

    elif provider == "cloudflare":
        if not s.cloudflare_api_token:
            raise ValueError("LLM_PROVIDER=cloudflare requires CLOUDFLARE_API_TOKEN")
        if not s.cloudflare_account_id:
            raise ValueError("LLM_PROVIDER=cloudflare requires CLOUDFLARE_ACCOUNT_ID")
        from agenticflow.models.cloudflare import CloudflareChat

        return CloudflareChat(
            model=s.cloudflare_model,
            api_key=s.cloudflare_api_token,
            account_id=s.cloudflare_account_id,
        )

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def _create_azure_chat(s: Settings):
    """Create Azure OpenAI chat model with appropriate auth method."""
    if not s.azure_openai_endpoint:
        raise ValueError("LLM_PROVIDER=azure requires AZURE_OPENAI_ENDPOINT")
    if not s.azure_openai_deployment:
        raise ValueError("LLM_PROVIDER=azure requires AZURE_OPENAI_DEPLOYMENT")

    from agenticflow.models.azure import AzureEntraAuth, AzureOpenAIChat

    if s.azure_auth_type == "api_key":
        if not s.azure_openai_api_key:
            raise ValueError(
                "AZURE_OPENAI_AUTH_TYPE=api_key requires AZURE_OPENAI_API_KEY"
            )
        return AzureOpenAIChat(
            deployment=s.azure_openai_deployment,
            azure_endpoint=s.azure_openai_endpoint,
            api_version=s.azure_openai_api_version,
            api_key=s.azure_openai_api_key,
        )

    elif s.azure_auth_type == "managed_identity":
        return AzureOpenAIChat(
            deployment=s.azure_openai_deployment,
            azure_endpoint=s.azure_openai_endpoint,
            api_version=s.azure_openai_api_version,
            entra=AzureEntraAuth(
                method="managed_identity",
                managed_identity_client_id=s.azure_openai_client_id,
            ),
        )

    elif s.azure_auth_type == "default":
        return AzureOpenAIChat(
            deployment=s.azure_openai_deployment,
            azure_endpoint=s.azure_openai_endpoint,
            api_version=s.azure_openai_api_version,
            entra=AzureEntraAuth(method="default"),
        )

    elif s.azure_auth_type == "client_secret":
        missing = [
            name
            for name, value in {
                "AZURE_OPENAI_TENANT_ID": s.azure_openai_tenant_id,
                "AZURE_OPENAI_CLIENT_ID": s.azure_openai_client_id,
                "AZURE_OPENAI_CLIENT_SECRET": s.azure_openai_client_secret,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(
                "AZURE_OPENAI_AUTH_TYPE=client_secret requires: " + ", ".join(missing)
            )
        return AzureOpenAIChat(
            deployment=s.azure_openai_deployment,
            azure_endpoint=s.azure_openai_endpoint,
            api_version=s.azure_openai_api_version,
            entra=AzureEntraAuth(
                method="client_secret",
                tenant_id=s.azure_openai_tenant_id,
                client_id=s.azure_openai_client_id,
                client_secret=s.azure_openai_client_secret,
            ),
        )

    else:
        raise ValueError(f"Unknown AZURE_OPENAI_AUTH_TYPE: {s.azure_auth_type}")


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

    elif provider == "github":
        if not s.github_token:
            raise ValueError("EMBEDDING_PROVIDER=github requires GITHUB_TOKEN")
        # GitHub Models uses OpenAI-compatible embeddings via Foundry
        from agenticflow.vectorstore import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=s.github_embedding_model,
            api_key=s.github_token,
            base_url="https://models.github.ai/inference",
        )

    elif provider == "cohere":
        if not s.cohere_api_key:
            raise ValueError("EMBEDDING_PROVIDER=cohere requires COHERE_API_KEY")
        from agenticflow.models.cohere import CohereEmbedding

        return CohereEmbedding(model=s.cohere_embedding_model, api_key=s.cohere_api_key)

    elif provider == "cloudflare":
        if not s.cloudflare_api_token:
            raise ValueError(
                "EMBEDDING_PROVIDER=cloudflare requires CLOUDFLARE_API_TOKEN"
            )
        if not s.cloudflare_account_id:
            raise ValueError(
                "EMBEDDING_PROVIDER=cloudflare requires CLOUDFLARE_ACCOUNT_ID"
            )
        from agenticflow.models.cloudflare import CloudflareEmbedding

        return CloudflareEmbedding(
            model=s.cloudflare_embedding_model,
            api_key=s.cloudflare_api_token,
            account_id=s.cloudflare_account_id,
        )

    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}")


def _create_azure_embeddings(s: Settings):
    """Create Azure OpenAI embeddings with appropriate auth method."""
    if not s.azure_openai_endpoint:
        raise ValueError("EMBEDDING_PROVIDER=azure requires AZURE_OPENAI_ENDPOINT")
    if not s.azure_openai_embedding_deployment:
        raise ValueError(
            "EMBEDDING_PROVIDER=azure requires AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
        )

    from agenticflow.models.azure import AzureEntraAuth, AzureOpenAIEmbedding

    if s.azure_auth_type == "api_key":
        if not s.azure_openai_api_key:
            raise ValueError(
                "AZURE_OPENAI_AUTH_TYPE=api_key requires AZURE_OPENAI_API_KEY"
            )
        return AzureOpenAIEmbedding(
            deployment=s.azure_openai_embedding_deployment,
            azure_endpoint=s.azure_openai_endpoint,
            api_version=s.azure_openai_api_version,
            api_key=s.azure_openai_api_key,
        )

    elif s.azure_auth_type == "managed_identity":
        return AzureOpenAIEmbedding(
            deployment=s.azure_openai_embedding_deployment,
            azure_endpoint=s.azure_openai_endpoint,
            api_version=s.azure_openai_api_version,
            entra=AzureEntraAuth(
                method="managed_identity",
                managed_identity_client_id=s.azure_openai_client_id,
            ),
        )

    elif s.azure_auth_type == "default":
        return AzureOpenAIEmbedding(
            deployment=s.azure_openai_embedding_deployment,
            azure_endpoint=s.azure_openai_endpoint,
            api_version=s.azure_openai_api_version,
            entra=AzureEntraAuth(method="default"),
        )

    elif s.azure_auth_type == "client_secret":
        missing = [
            name
            for name, value in {
                "AZURE_OPENAI_TENANT_ID": s.azure_openai_tenant_id,
                "AZURE_OPENAI_CLIENT_ID": s.azure_openai_client_id,
                "AZURE_OPENAI_CLIENT_SECRET": s.azure_openai_client_secret,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(
                "AZURE_OPENAI_AUTH_TYPE=client_secret requires: " + ", ".join(missing)
            )
        return AzureOpenAIEmbedding(
            deployment=s.azure_openai_embedding_deployment,
            azure_endpoint=s.azure_openai_endpoint,
            api_version=s.azure_openai_api_version,
            entra=AzureEntraAuth(
                method="client_secret",
                tenant_id=s.azure_openai_tenant_id,
                client_id=s.azure_openai_client_id,
                client_secret=s.azure_openai_client_secret,
            ),
        )

    else:
        raise ValueError(f"Unknown AZURE_OPENAI_AUTH_TYPE: {s.azure_auth_type}")


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
    elif s.llm_provider == "cohere":
        print(f"  Model: {s.cohere_model}")
        print(f"  API Key: {'✓ set' if s.cohere_api_key else '✗ missing!'}")
    elif s.llm_provider == "cloudflare":
        print(f"  Model: {s.cloudflare_model}")
        print(f"  API Token: {'✓ set' if s.cloudflare_api_token else '✗ missing!'}")
        print(f"  Account ID: {s.cloudflare_account_id or '✗ missing!'}")
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
            print(f"  Client ID: {s.azure_openai_client_id or '(system-assigned)'}")
        elif s.azure_auth_type == "client_secret":
            print(
                f"  Tenant ID: {'✓ set' if s.azure_openai_tenant_id else '✗ missing!'}"
            )
            print(
                f"  Client ID: {'✓ set' if s.azure_openai_client_id else '✗ missing!'}"
            )
            print(
                f"  Client Secret: {'✓ set' if s.azure_openai_client_secret else '✗ missing!'}"
            )
    elif s.llm_provider == "ollama":
        print(f"  Model: {s.ollama_model}")
        print(f"  Host: {s.ollama_host}")
    elif s.llm_provider == "mistral":
        print(f"  Model: {s.mistral_model}")
        print(f"  API Key: {'✓ set' if s.mistral_api_key else '✗ missing!'}")
    elif s.llm_provider == "github":
        print(f"  Model: {s.github_model}")
        print(f"  Token: {'✓ set' if s.github_token else '✗ missing!'}")

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
    elif s.embedding_provider == "github":
        print(f"  Model: {s.github_embedding_model}")
        print(f"  Token: {'✓ set' if s.github_token else '✗ missing!'}")
    elif s.embedding_provider == "cohere":
        print(f"  Model: {s.cohere_embedding_model}")
        print(f"  API Key: {'✓ set' if s.cohere_api_key else '✗ missing!'}")
    elif s.embedding_provider == "cloudflare":
        print(f"  Model: {s.cloudflare_embedding_model}")
        print(f"  API Token: {'✓ set' if s.cloudflare_api_token else '✗ missing!'}")
        print(f"  Account ID: {s.cloudflare_account_id or '✗ missing!'}")

    print("=" * 50)


if __name__ == "__main__":
    print_config()
