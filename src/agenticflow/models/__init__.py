"""
Native LLM and Embedding models for AgenticFlow.

High-performance model interfaces using provider SDKs directly.
Supports OpenAI, Azure, Anthropic, Groq, Gemini, Ollama, and custom endpoints.

Provider-Specific Imports:
    from agenticflow.models.openai import ChatModel, OpenAIEmbedding
    from agenticflow.models.azure import AzureChat, AzureEmbedding
    from agenticflow.models.anthropic import AnthropicChat
    from agenticflow.models.groq import GroqChat
    from agenticflow.models.gemini import GeminiChat, GeminiEmbedding
    from agenticflow.models.ollama import OllamaChat, OllamaEmbedding
    from agenticflow.models.custom import CustomChat, CustomEmbedding

Convenience Imports:
    from agenticflow.models import ChatModel, create_chat, create_embedding

Example:
    # Simple usage
    from agenticflow.models import ChatModel
    
    llm = ChatModel(model="gpt-4o")
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])
    print(response.content)
    
    # With tools
    llm = ChatModel().bind_tools([search_tool])
    response = await llm.ainvoke(messages)
    
    # Azure with DefaultAzureCredential
    from agenticflow.models.azure import AzureChat
    
    llm = AzureChat(
        azure_endpoint="https://your-resource.openai.azure.com",
        deployment="gpt-4o",
        use_azure_ad=True,
    )
    
    # Groq (fast inference)
    from agenticflow.models.groq import GroqChat
    
    llm = GroqChat(model="llama-3.3-70b-versatile")
    
    # Custom endpoint (vLLM, Together AI, etc.)
    from agenticflow.models.custom import CustomChat
    
    llm = CustomChat(
        base_url="http://localhost:8000/v1",
        model="meta-llama/Llama-3.2-3B-Instruct",
    )
"""

from __future__ import annotations

from typing import Any

# Base classes
from agenticflow.models.base import AIMessage, BaseChatModel, BaseEmbedding

# Default models (OpenAI)
from agenticflow.models.openai import OpenAIChat, OpenAIEmbedding

# Mock models for testing
from agenticflow.models.mock import MockEmbedding

# Aliases for convenience
ChatModel = OpenAIChat
EmbeddingModel = OpenAIEmbedding


def create_chat(
    provider: str = "openai",
    model: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a chat model for any provider.
    
    Args:
        provider: Provider name (openai, azure, anthropic, groq, gemini, ollama, custom)
        model: Model name (provider-specific). Uses provider default if not specified.
        **kwargs: Additional provider-specific arguments.
    
    Returns:
        Chat model instance.
    
    Example:
        # OpenAI
        llm = create_chat("openai", model="gpt-4o")
        
        # Azure with Azure AD
        llm = create_chat(
            "azure",
            deployment="gpt-4o",
            azure_endpoint="https://your-resource.openai.azure.com",
            use_azure_ad=True,
        )
        
        # Anthropic
        llm = create_chat("anthropic", model="claude-sonnet-4-20250514")
        
        # Groq
        llm = create_chat("groq", model="llama-3.3-70b-versatile")
        
        # Ollama
        llm = create_chat("ollama", model="llama3.2")
        
        # Gemini
        llm = create_chat("gemini", model="gemini-2.0-flash-exp")
        
        # Custom
        llm = create_chat(
            "custom",
            base_url="http://localhost:8000/v1",
            model="my-model",
        )
    """
    provider = provider.lower()
    
    if provider == "openai":
        from agenticflow.models.openai import OpenAIChat
        return OpenAIChat(model=model or "gpt-4o-mini", **kwargs)
    
    elif provider == "azure":
        from agenticflow.models.azure import AzureChat
        return AzureChat(deployment=model or kwargs.pop("deployment", None), **kwargs)
    
    elif provider == "anthropic":
        from agenticflow.models.anthropic import AnthropicChat
        return AnthropicChat(model=model or "claude-sonnet-4-20250514", **kwargs)
    
    elif provider == "groq":
        from agenticflow.models.groq import GroqChat
        return GroqChat(model=model or "llama-3.3-70b-versatile", **kwargs)
    
    elif provider == "gemini" or provider == "google":
        from agenticflow.models.gemini import GeminiChat
        return GeminiChat(model=model or "gemini-2.0-flash-exp", **kwargs)
    
    elif provider == "ollama":
        from agenticflow.models.ollama import OllamaChat
        return OllamaChat(model=model or "llama3.2", **kwargs)
    
    elif provider == "custom":
        from agenticflow.models.custom import CustomChat
        return CustomChat(model=model or "gpt-3.5-turbo", **kwargs)
    
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: openai, azure, anthropic, groq, gemini, ollama, custom"
        )


def create_embedding(
    provider: str = "openai",
    model: str | None = None,
    **kwargs: Any,
) -> BaseEmbedding:
    """Create an embedding model for any provider.
    
    Args:
        provider: Provider name (openai, azure, gemini, ollama, custom)
        model: Model name (provider-specific). Uses provider default if not specified.
        **kwargs: Additional provider-specific arguments.
    
    Returns:
        Embedding model instance.
    
    Example:
        # OpenAI
        embedder = create_embedding("openai")
        
        # Azure
        embedder = create_embedding(
            "azure",
            deployment="text-embedding-3-small",
            azure_endpoint="https://your-resource.openai.azure.com",
        )
        
        # Ollama
        embedder = create_embedding("ollama", model="nomic-embed-text")
        
        # Gemini
        embedder = create_embedding("gemini", model="text-embedding-004")
    """
    provider = provider.lower()
    
    if provider == "openai":
        from agenticflow.models.openai import OpenAIEmbedding
        return OpenAIEmbedding(model=model or "text-embedding-3-small", **kwargs)
    
    elif provider == "azure":
        from agenticflow.models.azure import AzureEmbedding
        return AzureEmbedding(deployment=model or kwargs.pop("deployment", None), **kwargs)
    
    elif provider == "gemini" or provider == "google":
        from agenticflow.models.gemini import GeminiEmbedding
        return GeminiEmbedding(model=model or "text-embedding-004", **kwargs)
    
    elif provider == "ollama":
        from agenticflow.models.ollama import OllamaEmbedding
        return OllamaEmbedding(model=model or "nomic-embed-text", **kwargs)
    
    elif provider == "custom":
        from agenticflow.models.custom import CustomEmbedding
        return CustomEmbedding(model=model or "text-embedding-3-small", **kwargs)
    
    else:
        raise ValueError(
            f"Unknown embedding provider: {provider}. "
            f"Supported: openai, azure, gemini, ollama, custom"
        )


def is_native_model(model: Any) -> bool:
    """Check if a model is a native AgenticFlow model.
    
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
    # OpenAI models (also aliased as ChatModel/EmbeddingModel)
    "OpenAIChat",
    "OpenAIEmbedding",
    "ChatModel",  # Alias for OpenAIChat
    "EmbeddingModel",  # Alias for OpenAIEmbedding
    # Mock models for testing
    "MockEmbedding",
    # Factory functions
    "create_chat",
    "create_embedding",
    # Utilities
    "is_native_model",
]
