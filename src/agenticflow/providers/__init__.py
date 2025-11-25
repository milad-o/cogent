"""AgenticFlow Providers Module.

A modular, extensible provider system for LLM and embedding models.

Architecture:
- BaseProvider: Abstract base class defining the provider interface
- Provider implementations: OpenAI, Azure, Anthropic, Google, Ollama, etc.
- ModelSpec / EmbeddingSpec: Type-safe model specifications
- Factory functions: Create providers from string specs like 'openai/gpt-4o'

Usage:
    # Using string spec (provider/model format)
    from agenticflow.providers import create_model, create_embeddings
    
    model = create_model("openai/gpt-4o-mini")
    embeddings = create_embeddings("openai/text-embedding-3-small")
    
    # Using ModelSpec for more control
    from agenticflow.providers import ModelSpec
    
    spec = ModelSpec(
        provider="openai",
        model="gpt-4o",
        temperature=0.7,
        max_tokens=4096,
    )
    model = spec.create()
    
    # Using provider classes directly
    from agenticflow.providers import OpenAIProvider
    
    provider = OpenAIProvider()
    model = provider.create_chat_model("gpt-4o-mini")
    embeddings = provider.create_embeddings("text-embedding-3-small")
    
    # Azure with Managed Identity
    from agenticflow.providers import AzureOpenAIProvider, AzureAuthMethod
    
    provider = AzureOpenAIProvider(
        endpoint="https://my-resource.openai.azure.com",
        auth_method=AzureAuthMethod.MANAGED_IDENTITY,
    )
    model = provider.create_chat_model("gpt-4o-deployment")
"""

from agenticflow.providers.base import (
    BaseProvider,
    ModelSpec,
    EmbeddingSpec,
    ProviderRegistry,
)
from agenticflow.providers.factory import (
    acreate_embeddings,
    acreate_model,
    create_embeddings,
    create_model,
    get_provider,
    list_providers,
    parse_model_spec,
)
from agenticflow.providers.openai import OpenAIProvider
from agenticflow.providers.azure import (
    AzureOpenAIProvider,
    AzureAuthMethod,
    AzureConfig,
)
from agenticflow.providers.anthropic import AnthropicProvider
from agenticflow.providers.google import GoogleProvider
from agenticflow.providers.ollama import OllamaProvider
from agenticflow.providers.enums import Provider

__all__ = [
    # Base classes
    "BaseProvider",
    "ModelSpec",
    "EmbeddingSpec",
    "ProviderRegistry",
    # Enums
    "Provider",
    "AzureAuthMethod",
    "AzureConfig",
    # Provider implementations
    "OpenAIProvider",
    "AzureOpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "OllamaProvider",
    # Factory functions
    "create_model",
    "acreate_model",
    "create_embeddings",
    "acreate_embeddings",
    "get_provider",
    "parse_model_spec",
    "list_providers",
]
