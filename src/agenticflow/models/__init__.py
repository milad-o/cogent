"""
Native LLM and Embedding models for AgenticFlow.

High-performance model interfaces using provider SDKs directly.
Supports OpenAI, Azure, Anthropic, Groq, Gemini, Cohere, Ollama, and custom endpoints.

Provider-Specific Imports:
    from agenticflow.models.openai import ChatModel, OpenAIEmbedding
    from agenticflow.models.azure import AzureChat, AzureEmbedding, AzureAIFoundryChat
    from agenticflow.models.anthropic import AnthropicChat
    from agenticflow.models.groq import GroqChat
    from agenticflow.models.cohere import CohereChat, CohereEmbedding
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
    
    # GitHub Models (via Azure AI Foundry)
    from agenticflow.models.azure import AzureAIFoundryChat
    
    llm = AzureAIFoundryChat.from_github(
        model="meta/Meta-Llama-3.1-8B-Instruct",
        token=os.getenv("GITHUB_TOKEN"),
    )
    
    # Groq (fast inference)
    from agenticflow.models.groq import GroqChat
    
    llm = GroqChat(model="llama-3.3-70b-versatile")

    # Cohere
    from agenticflow.models.cohere import CohereChat, CohereEmbedding
    llm = CohereChat(model="command-r-plus")
    embedder = CohereEmbedding(model="embed-english-v3.0")
    
    # Custom endpoint (vLLM, Together AI, etc.)
    from agenticflow.models.custom import CustomChat
    
    llm = CustomChat(
        base_url="http://localhost:8000/v1",
        model="meta-llama/Llama-3.2-3B-Instruct",
    )
"""

from __future__ import annotations

from typing import Any

# Base classes and utilities
from agenticflow.models.base import AIMessage, BaseChatModel, BaseEmbedding, convert_messages

# Default models (OpenAI)
from agenticflow.models.openai import OpenAIChat, OpenAIEmbedding
from agenticflow.models.cohere import CohereChat, CohereEmbedding

# Mock models for testing
from agenticflow.models.mock import MockChatModel, MockEmbedding

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
        provider: Provider name (openai, azure, azure-foundry, github, anthropic, groq, gemini, cohere, ollama, custom)
        model: Model name (provider-specific). Uses provider default if not specified.
        **kwargs: Additional provider-specific arguments.
    
    Returns:
        Chat model instance.
    
    Example:
        # OpenAI
        llm = create_chat("openai", model="gpt-4o")
        
        # Azure OpenAI with Azure AD
        llm = create_chat(
            "azure",
            deployment="gpt-4o",
            azure_endpoint="https://your-resource.openai.azure.com",
            use_azure_ad=True,
        )
        
        # GitHub Models (via Azure AI Foundry)
        llm = create_chat(
            "github",
            model="meta/Meta-Llama-3.1-8B-Instruct",
            token=os.getenv("GITHUB_TOKEN"),
        )
        
        # Azure AI Foundry custom endpoint
        llm = create_chat(
            "azure-foundry",
            endpoint="https://your-foundry.azure.com/inference",
            model="your-model",
            api_key="your-key",
        )
        
        # Anthropic
        llm = create_chat("anthropic", model="claude-sonnet-4-20250514")
        
        # Groq
        llm = create_chat("groq", model="llama-3.3-70b-versatile")
        
        # Cohere
        llm = create_chat("cohere", model="command-r-plus")
        
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
    
    elif provider == "azure-foundry":
        from agenticflow.models.azure import AzureAIFoundryChat
        if not model:
            raise ValueError("model required for azure-foundry provider")
        if "endpoint" not in kwargs:
            raise ValueError("endpoint required for azure-foundry provider")
        return AzureAIFoundryChat(model=model, **kwargs)
    
    elif provider == "github":
        from agenticflow.models.azure import AzureAIFoundryChat
        if not model:
            raise ValueError("model required for github provider")
        token = kwargs.pop("token", None)
        return AzureAIFoundryChat.from_github(model=model, token=token, **kwargs)
    
    elif provider == "anthropic":
        from agenticflow.models.anthropic import AnthropicChat
        return AnthropicChat(model=model or "claude-sonnet-4-20250514", **kwargs)
    
    elif provider == "groq":
        from agenticflow.models.groq import GroqChat
        return GroqChat(model=model or "llama-3.3-70b-versatile", **kwargs)
    
    elif provider == "gemini" or provider == "google":
        from agenticflow.models.gemini import GeminiChat
        return GeminiChat(model=model or "gemini-2.0-flash-exp", **kwargs)

    elif provider == "cohere":
        from agenticflow.models.cohere import CohereChat
        return CohereChat(model=model or "command-r-plus", **kwargs)
    
    elif provider == "ollama":
        from agenticflow.models.ollama import OllamaChat
        return OllamaChat(model=model or "llama3.2", **kwargs)
    
    elif provider == "custom":
        from agenticflow.models.custom import CustomChat
        return CustomChat(model=model or "gpt-3.5-turbo", **kwargs)
    
    elif provider == "mistral":
        from agenticflow.models.mistral import MistralChat
        return MistralChat(model=model or "mistral-small-latest", **kwargs)
    
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: openai, azure, azure-foundry, github, anthropic, groq, gemini, cohere, ollama, mistral, custom"
        )


def create_embedding(
    provider: str = "openai",
    model: str | None = None,
    **kwargs: Any,
) -> BaseEmbedding:
    """Create an embedding model for any provider.
    
    Args:
        provider: Provider name (openai, azure, gemini, cohere, ollama, custom)
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

    elif provider == "cohere":
        from agenticflow.models.cohere import CohereEmbedding
        return CohereEmbedding(model=model or "embed-english-v3.0", **kwargs)
    
    elif provider == "ollama":
        from agenticflow.models.ollama import OllamaEmbedding
        return OllamaEmbedding(model=model or "nomic-embed-text", **kwargs)
    
    elif provider == "custom":
        from agenticflow.models.custom import CustomEmbedding
        return CustomEmbedding(model=model or "text-embedding-3-small", **kwargs)
    
    else:
        raise ValueError(
            f"Unknown embedding provider: {provider}. "
            f"Supported: openai, azure, gemini, cohere, ollama, custom"
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
    # Utilities
    "convert_messages",
    "is_native_model",
    # OpenAI models (also aliased as ChatModel/EmbeddingModel)
    "OpenAIChat",
    "OpenAIEmbedding",
    # Cohere
    "CohereChat",
    "CohereEmbedding",
    "ChatModel",  # Alias for OpenAIChat
    "EmbeddingModel",  # Alias for OpenAIEmbedding
    # Mock models for testing
    "MockChatModel",
    "MockEmbedding",
    # Factory functions
    "create_chat",
    "create_embedding",
]
