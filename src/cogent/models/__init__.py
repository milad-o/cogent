"""
Native LLM and Embedding models for Cogent.

High-performance model interfaces using provider SDKs directly.
Supports OpenAI, Azure, Anthropic, Groq, Gemini, Cohere, Cloudflare, Ollama, and custom endpoints.

Provider-Specific Imports:
    from cogent.models.openai import ChatModel, OpenAIEmbedding
    from cogent.models.azure import AzureOpenAIChat, AzureOpenAIEmbedding, AzureAIFoundryChat
    from cogent.models.anthropic import AnthropicChat
    from cogent.models.groq import GroqChat
    from cogent.models.cohere import CohereChat, CohereEmbedding
    from cogent.models.cloudflare import CloudflareChat, CloudflareEmbedding
    from cogent.models.gemini import GeminiChat, GeminiEmbedding
    from cogent.models.ollama import OllamaChat, OllamaEmbedding
    from cogent.models.custom import CustomChat, CustomEmbedding

Convenience Imports:
    from cogent.models import ChatModel, create_chat, create_embedding

Example:
    # Simple usage
    from cogent.models import ChatModel

    llm = ChatModel(model="gpt-4o")
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])
    print(response.content)

    # With tools
    llm = ChatModel().bind_tools([search_tool])
    response = await llm.ainvoke(messages)

    # Azure with DefaultAzureCredential (Entra ID)
    from cogent.models.azure import AzureOpenAIChat, AzureEntraAuth

    llm = AzureOpenAIChat(
        azure_endpoint="https://your-resource.openai.azure.com",
        deployment="gpt-4o",
        entra=AzureEntraAuth(method="default"),
    )

    # GitHub Models (via Azure AI Foundry)
    from cogent.models.azure import AzureAIFoundryChat

    llm = AzureAIFoundryChat.from_github(
        model="meta/Meta-Llama-3.1-8B-Instruct",
        token=os.getenv("GITHUB_TOKEN"),
    )

    # Groq (fast inference)
    from cogent.models.groq import GroqChat

    llm = GroqChat(model="llama-3.3-70b-versatile")

    # Cohere
    from cogent.models.cohere import CohereChat, CohereEmbedding
    llm = CohereChat(model="command-r-plus")
    embedder = CohereEmbedding(model="embed-english-v3.0")

    # Cloudflare Workers AI
    from cogent.models.cloudflare import CloudflareChat, CloudflareEmbedding
    llm = CloudflareChat(model="@cf/meta/llama-3.3-70b-instruct")
    embedder = CloudflareEmbedding(model="@cf/baai/bge-base-en-v1.5")

    # Custom endpoint (vLLM, Together AI, etc.)
    from cogent.models.custom import CustomChat

    llm = CustomChat(
        base_url="http://localhost:8000/v1",
        model="meta-llama/Llama-3.2-3B-Instruct",
    )
"""

from __future__ import annotations

from typing import Any

# Message types and metadata
from cogent.core.messages import (
    BaseMessage,
    EmbeddingMetadata,
    EmbeddingResult,
    HumanMessage,
    MessageMetadata,
    SystemMessage,
    TokenUsage,
    ToolMessage,
)

# All provider models
from cogent.models.anthropic import AnthropicChat

# Base classes and utilities
from cogent.models.base import (
    AIMessage,
    BaseChatModel,
    BaseEmbedding,
    convert_messages,
    normalize_input,
)
from cogent.models.cloudflare import CloudflareChat, CloudflareEmbedding
from cogent.models.cohere import CohereChat, CohereEmbedding
from cogent.models.gemini import GeminiChat, GeminiEmbedding
from cogent.models.groq import GroqChat

# Mock models for testing
from cogent.models.mock import MockChatModel, MockEmbedding
from cogent.models.ollama import OllamaChat, OllamaEmbedding

# Default models (OpenAI)
from cogent.models.openai import OpenAIChat, OpenAIEmbedding

# xAI models
from cogent.models.xai import XAIChat

# DeepSeek models
from cogent.models.deepseek import DeepSeekChat

# Cerebras models
from cogent.models.cerebras import CerebrasChat

# Model registry for high-level API
from cogent.models.registry import (
    get_provider_for_model,
    list_model_aliases,
    resolve_and_create_model,
    resolve_model,
)

# Aliases for convenience
ChatModel = OpenAIChat
EmbeddingModel = OpenAIEmbedding


def create_chat(
    provider: str = "openai",
    model: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a chat model for any provider.

    Universal factory function supporting multiple invocation styles for maximum flexibility.
    Auto-loads API keys from environment (.env) and config files.

    Usage Patterns
    --------------

    Pattern 1: Model name only (simplest - auto-detects provider)
        >>> llm = create_chat("gpt-4o")              # OpenAI
        >>> llm = create_chat("gemini-2.5-pro")      # Google Gemini
        >>> llm = create_chat("claude-sonnet-4")     # Anthropic
        >>> llm = create_chat("llama-3.1-8b-instant")  # Groq
        >>> llm = create_chat("mistral-small-latest")  # Mistral

    Pattern 2: Provider:model syntax (explicit provider)
        >>> llm = create_chat("openai:gpt-4o")
        >>> llm = create_chat("gemini:gemini-2.5-flash")
        >>> llm = create_chat("anthropic:claude-sonnet-4-20250514")
        >>> llm = create_chat("groq:llama-3.3-70b-versatile")

    Pattern 3: Separate provider and model arguments
        >>> llm = create_chat("openai", "gpt-4o")
        >>> llm = create_chat("gemini", "gemini-2.5-pro")
        >>> llm = create_chat("anthropic", "claude-sonnet-4-20250514")

    Pattern 4: With additional configuration
        >>> llm = create_chat("openai", "gpt-4o", temperature=0.7, max_tokens=1000)
        >>> llm = create_chat("gemini:gemini-2.5-pro", temperature=0.9)
        >>> llm = create_chat("gpt-4o", api_key="sk-...", temperature=0)

    Supported Providers
    -------------------
    - openai: GPT-4o, GPT-4, GPT-3.5, o1, o3
    - gemini/google: Gemini 2.5 Pro, Flash, Flash-Lite
    - anthropic: Claude Sonnet, Opus, Haiku
    - groq: Llama, Mixtral, Qwen, DeepSeek (fast inference)
    - mistral: Mistral Small, Large, Codestral, Pixtral
    - cohere: Command R, Command R+
    - cloudflare: Workers AI models (@cf/...)
    - ollama: Local models
    - azure: Azure OpenAI with Entra ID auth
    - github: GitHub Models (via Azure AI Foundry)
    - custom: Any OpenAI-compatible endpoint

    Args:
        provider: Either provider name ("openai", "gemini") OR full model string
            when used as single argument. Supports "provider:model" syntax.
        model: Optional explicit model name. Use when specifying provider separately.
        **kwargs: Provider-specific configuration (api_key, temperature, max_tokens, etc.)

    Returns:
        BaseChatModel: Initialized chat model ready for use.

    Environment Variables
    ---------------------
    API keys automatically loaded from .env or environment:
    - OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, etc.

    Model overrides (optional):
    - OPENAI_CHAT_MODEL, GEMINI_CHAT_MODEL, etc.

    Examples:
        Basic usage with auto-detection:
            >>> llm = create_chat("gpt-4o")
            >>> response = await llm.ainvoke("Hello!")

        With provider prefix:
            >>> llm = create_chat("anthropic:claude-sonnet-4")
            >>> llm = create_chat("groq:llama-70b")

        Explicit provider and model:
            >>> llm = create_chat("openai", "gpt-4o")
            >>> llm = create_chat("gemini", "gemini-2.5-flash-lite")

        With configuration:
            >>> llm = create_chat("gpt-4o", temperature=0.7, max_tokens=500)
            >>> llm = create_chat("openai", "gpt-4o", api_key="sk-custom...")

        Azure OpenAI with Entra ID:
            >>> from cogent.models.azure import AzureEntraAuth
            >>> llm = create_chat(
            ...     "azure",
            ...     deployment="gpt-4o",
            ...     azure_endpoint="https://your-resource.openai.azure.com",
            ...     entra=AzureEntraAuth(method="default"),
            ... )

        GitHub Models:
            >>> import os
            >>> llm = create_chat(
            ...     "github",
            ...     model="meta/Meta-Llama-3.1-8B-Instruct",
            ...     token=os.getenv("GITHUB_TOKEN"),
            ... )

        Cloudflare Workers AI:
            >>> llm = create_chat("cloudflare", "@cf/meta/llama-3.1-8b-instruct")

        Custom endpoint (vLLM, Together AI, etc.):
            >>> llm = create_chat(
            ...     "custom",
            ...     base_url="http://localhost:8000/v1",
            ...     model="meta-llama/Llama-3.2-3B-Instruct",
            ... )
    """
    provider_lower = provider.lower()
    explicit_providers = {
        "openai",
        "azure",
        "azure-foundry",
        "github",
        "anthropic",
        "groq",
        "gemini",
        "google",
        "cohere",
        "cloudflare",
        "ollama",
        "mistral",
        "xai",
        "cerebras",
        "custom",
    }

    if model is None and provider_lower in explicit_providers:
        from cogent.config import get_model_override

        model = get_model_override(provider_lower, "chat")
    elif model is None:
        from cogent.models.registry import resolve_model

        provider_resolved, model = resolve_model(provider)
        provider = provider_resolved

    # Auto-load API key from config if not provided
    if "api_key" not in kwargs:
        from cogent.config import get_api_key

        api_key = get_api_key(provider, kwargs.get("api_key"))
        if api_key:
            kwargs["api_key"] = api_key

    provider_lower = provider.lower()
    if model is None:
        from cogent.config import get_model_override

        model = get_model_override(provider_lower, "embedding")

    provider = provider_lower

    if provider == "openai":
        from cogent.models.openai import OpenAIChat

        return OpenAIChat(model=model or "gpt-4o-mini", **kwargs)

    elif provider == "azure":
        from cogent.models.azure import AzureOpenAIChat

        return AzureOpenAIChat(
            deployment=model or kwargs.pop("deployment", None), **kwargs
        )

    elif provider == "azure-foundry":
        from cogent.models.azure import AzureAIFoundryChat

        if not model:
            raise ValueError("model required for azure-foundry provider")
        if "endpoint" not in kwargs:
            raise ValueError("endpoint required for azure-foundry provider")
        return AzureAIFoundryChat(model=model, **kwargs)

    elif provider == "github":
        from cogent.models.azure import AzureAIFoundryChat

        if not model:
            raise ValueError("model required for github provider")
        token = kwargs.pop("token", None)
        return AzureAIFoundryChat.from_github(model=model, token=token, **kwargs)

    elif provider == "anthropic":
        from cogent.models.anthropic import AnthropicChat

        return AnthropicChat(model=model or "claude-sonnet-4-20250514", **kwargs)

    elif provider == "groq":
        from cogent.models.groq import GroqChat

        return GroqChat(model=model or "llama-3.3-70b-versatile", **kwargs)

    elif provider == "gemini" or provider == "google":
        from cogent.models.gemini import GeminiChat

        return GeminiChat(model=model or "gemini-2.5-flash", **kwargs)

    elif provider == "cohere":
        from cogent.models.cohere import CohereChat

        return CohereChat(model=model or "command-r-plus", **kwargs)

    elif provider == "ollama":
        from cogent.models.ollama import OllamaChat

        return OllamaChat(model=model or "llama3.2", **kwargs)

    elif provider == "cloudflare":
        from cogent.models.cloudflare import CloudflareChat

        return CloudflareChat(
            model=model or "@cf/meta/llama-3.3-70b-instruct", **kwargs
        )

    elif provider == "custom":
        from cogent.models.custom import CustomChat

        return CustomChat(model=model or "gpt-3.5-turbo", **kwargs)

    elif provider == "mistral":
        from cogent.models.mistral import MistralChat

        return MistralChat(model=model or "mistral-small-latest", **kwargs)

    elif provider == "xai":
        from cogent.models.xai import XAIChat

        return XAIChat(model=model or "grok-4-1-fast", **kwargs)

    elif provider == "deepseek":
        from cogent.models.deepseek import DeepSeekChat

        return DeepSeekChat(model=model or "deepseek-chat", **kwargs)

    elif provider == "cerebras":
        from cogent.models.cerebras import CerebrasChat

        return CerebrasChat(model=model or "llama3.1-8b", **kwargs)

    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: openai, azure, azure-foundry, github, anthropic, groq, gemini, cohere, cloudflare, ollama, mistral, xai, deepseek, cerebras, custom"
        )


def create_embedding(
    provider: str = "openai",
    model: str | None = None,
    **kwargs: Any,
) -> BaseEmbedding:
    """Create an embedding model for any provider.

    Supports multiple invocation patterns like create_chat().

    Args:
        provider: Either provider name ("openai", "gemini") OR full model string
            when used as single argument. Supports "provider:model" syntax.
        model: Optional explicit model name. Use when specifying provider separately.
        **kwargs: Provider-specific configuration (api_key, etc.)

    Returns:
        BaseEmbedding: Initialized embedding model ready for use.

    Examples:
        # Pattern 1: Model name only (auto-detects provider)
        embedder = create_embedding("text-embedding-3-large")
        embedder = create_embedding("embed-english-v3.0")

        # Pattern 2: Provider:model syntax
        embedder = create_embedding("openai:text-embedding-3-large")
        embedder = create_embedding("cohere:embed-english-v3.0")

        # Pattern 3: Separate provider and model arguments
        embedder = create_embedding("openai", "text-embedding-3-large")
        embedder = create_embedding("gemini", "text-embedding-004")

        # Pattern 4: With configuration
        embedder = create_embedding("openai", "text-embedding-3-small", api_key="sk-...")

        # Azure
        embedder = create_embedding(
            "azure",
            deployment="text-embedding-3-small",
            azure_endpoint="https://your-resource.openai.azure.com",
        )
    """
    provider_lower = provider.lower()
    explicit_providers = {
        "openai",
        "azure",
        "gemini",
        "google",
        "cohere",
        "cloudflare",
        "ollama",
        "mistral",
        "custom",
    }

    if model is None and provider_lower in explicit_providers:
        from cogent.config import get_model_override

        model = get_model_override(provider_lower, "embedding")
    elif model is None:
        from cogent.models.registry import resolve_model

        provider_resolved, model = resolve_model(provider)
        provider = provider_resolved

    provider = provider.lower()

    if provider == "openai":
        from cogent.models.openai import OpenAIEmbedding

        return OpenAIEmbedding(model=model or "text-embedding-3-small", **kwargs)

    elif provider == "azure":
        from cogent.models.azure import AzureOpenAIEmbedding

        return AzureOpenAIEmbedding(
            deployment=model or kwargs.pop("deployment", None), **kwargs
        )

    elif provider == "gemini" or provider == "google":
        from cogent.models.gemini import GeminiEmbedding

        return GeminiEmbedding(model=model or "text-embedding-004", **kwargs)

    elif provider == "cohere":
        from cogent.models.cohere import CohereEmbedding

        return CohereEmbedding(model=model or "embed-english-v3.0", **kwargs)

    elif provider == "ollama":
        from cogent.models.ollama import OllamaEmbedding

        return OllamaEmbedding(model=model or "nomic-embed-text", **kwargs)

    elif provider == "cloudflare":
        from cogent.models.cloudflare import CloudflareEmbedding

        return CloudflareEmbedding(model=model or "@cf/baai/bge-base-en-v1.5", **kwargs)

    elif provider == "custom":
        from cogent.models.custom import CustomEmbedding

        return CustomEmbedding(model=model or "text-embedding-3-small", **kwargs)

    else:
        raise ValueError(
            f"Unknown embedding provider: {provider}. "
            f"Supported: openai, azure, gemini, cohere, cloudflare, ollama, custom"
        )


def is_native_model(model: Any) -> bool:
    """Check if a model is a native Cogent model.

    Args:
        model: Model instance to check.

    Returns:
        True if native model, False otherwise.
    """
    return isinstance(model, BaseChatModel)


__all__ = [
    # Base types
    "AIMessage",
    "BaseChatModel",
    "BaseEmbedding",
    # Message types and metadata
    "MessageMetadata",
    "TokenUsage",
    "EmbeddingMetadata",
    "EmbeddingResult",
    "BaseMessage",
    "SystemMessage",
    "HumanMessage",
    "ToolMessage",
    # Utilities
    "convert_messages",
    "normalize_input",
    "is_native_model",
    # Registry functions
    "resolve_model",
    "resolve_and_create_model",
    "list_model_aliases",
    "get_provider_for_model",
    # OpenAI models (also aliased as ChatModel/EmbeddingModel)
    "OpenAIChat",
    "OpenAIEmbedding",
    # Anthropic
    "AnthropicChat",
    # Gemini
    "GeminiChat",
    "GeminiEmbedding",
    # Groq
    "GroqChat",
    # Cohere
    "CohereChat",
    "CohereEmbedding",
    # Cloudflare
    "CloudflareChat",
    "CloudflareEmbedding",
    # Ollama
    "OllamaChat",
    "OllamaEmbedding",
    # xAI
    "XAIChat",
    # DeepSeek
    "DeepSeekChat",
    # Cerebras
    "CerebrasChat",
    # Aliases
    "ChatModel",  # Alias for OpenAIChat
    "EmbeddingModel",  # Alias for OpenAIEmbedding
    # Mock models for testing
    "MockChatModel",
    "MockEmbedding",
    # Factory functions
    "create_chat",
    "create_embedding",
]
