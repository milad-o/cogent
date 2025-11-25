"""Model provider utilities for LLM and embedding model creation.

Provides flexible model initialization supporting:
- Direct LangChain model objects
- Model name strings (e.g., "gpt-4o", "claude-3-opus")
- Provider-prefixed strings (e.g., "openai:gpt-4o", "azure:gpt-4")
- Configuration dictionaries

Supported LLM Providers:
- OpenAI (gpt-4o, gpt-4o-mini, gpt-4-turbo, etc.)
- Azure OpenAI
- Anthropic (claude-3-opus, claude-3-sonnet, etc.)
- Google (gemini-pro, gemini-1.5-pro, etc.)
- Ollama (local models)
- AWS Bedrock

Supported Embedding Providers:
- OpenAI (text-embedding-3-small, text-embedding-3-large)
- Azure OpenAI
- Cohere
- HuggingFace
- Ollama (local embeddings)
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.embeddings import Embeddings


# Provider name mappings
LLM_PROVIDERS = {
    "openai": "langchain_openai.ChatOpenAI",
    "azure": "langchain_openai.AzureChatOpenAI",
    "azure_openai": "langchain_openai.AzureChatOpenAI",
    "anthropic": "langchain_anthropic.ChatAnthropic",
    "google": "langchain_google_genai.ChatGoogleGenerativeAI",
    "gemini": "langchain_google_genai.ChatGoogleGenerativeAI",
    "ollama": "langchain_ollama.ChatOllama",
    "bedrock": "langchain_aws.ChatBedrock",
    "cohere": "langchain_cohere.ChatCohere",
    "mistral": "langchain_mistralai.ChatMistralAI",
    "groq": "langchain_groq.ChatGroq",
    "fireworks": "langchain_fireworks.ChatFireworks",
    "together": "langchain_together.ChatTogether",
}

EMBEDDING_PROVIDERS = {
    "openai": "langchain_openai.OpenAIEmbeddings",
    "azure": "langchain_openai.AzureOpenAIEmbeddings",
    "azure_openai": "langchain_openai.AzureOpenAIEmbeddings",
    "cohere": "langchain_cohere.CohereEmbeddings",
    "huggingface": "langchain_huggingface.HuggingFaceEmbeddings",
    "ollama": "langchain_ollama.OllamaEmbeddings",
    "bedrock": "langchain_aws.BedrockEmbeddings",
    "google": "langchain_google_genai.GoogleGenerativeAIEmbeddings",
    "gemini": "langchain_google_genai.GoogleGenerativeAIEmbeddings",
    "mistral": "langchain_mistralai.MistralAIEmbeddings",
    "voyage": "langchain_voyageai.VoyageAIEmbeddings",
}

# Default model names per provider
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "azure": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "google": "gemini-1.5-flash",
    "ollama": "llama3.2",
    "mistral": "mistral-large-latest",
    "groq": "llama-3.3-70b-versatile",
}

DEFAULT_EMBEDDINGS = {
    "openai": "text-embedding-3-small",
    "azure": "text-embedding-3-small",
    "cohere": "embed-english-v3.0",
    "ollama": "nomic-embed-text",
    "google": "models/embedding-001",
}


def _import_class(class_path: str) -> type:
    """Dynamically import a class from a module path."""
    module_path, class_name = class_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _parse_model_string(model_string: str) -> tuple[str | None, str]:
    """Parse a model string into provider and model name.
    
    Args:
        model_string: Model string like "gpt-4o" or "openai:gpt-4o"
        
    Returns:
        Tuple of (provider, model_name). Provider may be None.
    """
    if ":" in model_string:
        provider, model = model_string.split(":", 1)
        return provider.lower(), model
    return None, model_string


def _infer_provider(model_name: str) -> str:
    """Infer the provider from a model name.
    
    Args:
        model_name: Model name like "gpt-4o" or "claude-3-opus"
        
    Returns:
        Inferred provider name.
    """
    model_lower = model_name.lower()
    
    # OpenAI models
    if model_lower.startswith(("gpt-", "o1-", "o3-", "chatgpt-")):
        return "openai"
    if "davinci" in model_lower or "curie" in model_lower:
        return "openai"
    
    # Anthropic models
    if model_lower.startswith("claude"):
        return "anthropic"
    
    # Google models
    if model_lower.startswith(("gemini", "palm")):
        return "google"
    
    # Mistral models
    if model_lower.startswith(("mistral", "mixtral", "codestral")):
        return "mistral"
    
    # Cohere models
    if model_lower.startswith(("command", "embed-")):
        return "cohere"
    
    # Ollama (local) - common local model names
    if model_lower.startswith(("llama", "phi", "qwen", "deepseek", "codellama")):
        return "ollama"
    
    # Default to OpenAI
    return "openai"


def create_chat_model(
    model: str | BaseChatModel | dict[str, Any] | None = None,
    *,
    provider: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a chat model from various input formats.
    
    Supports flexible model specification:
    - Direct LangChain model object (returned as-is)
    - Model name string (e.g., "gpt-4o")
    - Provider-prefixed string (e.g., "openai:gpt-4o", "azure:gpt-4")
    - Configuration dictionary
    
    Args:
        model: Model specification (string, object, or dict)
        provider: Explicit provider override
        temperature: Model temperature (0.0-2.0)
        max_tokens: Maximum response tokens
        **kwargs: Additional provider-specific parameters
        
    Returns:
        Configured LangChain chat model.
        
    Examples:
        # Simple model name (auto-detects OpenAI)
        >>> model = create_chat_model("gpt-4o-mini")
        
        # Provider-prefixed
        >>> model = create_chat_model("anthropic:claude-3-5-sonnet-latest")
        
        # Azure OpenAI
        >>> model = create_chat_model(
        ...     "azure:gpt-4o",
        ...     azure_endpoint="https://my-resource.openai.azure.com",
        ...     api_version="2024-02-15-preview",
        ...     azure_deployment="my-gpt4-deployment",
        ... )
        
        # Direct LangChain object
        >>> from langchain_openai import ChatOpenAI
        >>> model = create_chat_model(ChatOpenAI(model="gpt-4o"))
        
        # Configuration dict
        >>> model = create_chat_model({
        ...     "provider": "openai",
        ...     "model": "gpt-4o",
        ...     "temperature": 0.5,
        ... })
        
        # Use default model for a provider
        >>> model = create_chat_model(provider="anthropic")
    """
    from langchain_core.language_models import BaseChatModel
    
    # If already a model object, return as-is
    if isinstance(model, BaseChatModel):
        return model
    
    # Handle dict configuration
    if isinstance(model, dict):
        config = model.copy()
        provider = config.pop("provider", provider)
        model = config.pop("model", config.pop("model_name", None))
        temperature = config.pop("temperature", temperature)
        max_tokens = config.pop("max_tokens", max_tokens)
        kwargs.update(config)
    
    # Parse model string
    if isinstance(model, str):
        parsed_provider, model_name = _parse_model_string(model)
        if parsed_provider:
            provider = parsed_provider
        model = model_name
    
    # Infer provider if not specified
    if provider is None:
        if model:
            provider = _infer_provider(model)
        else:
            provider = "openai"
    
    provider = provider.lower()
    
    # Use default model if none specified
    if not model:
        model = DEFAULT_MODELS.get(provider, "gpt-4o-mini")
    
    # Get provider class path
    if provider not in LLM_PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: {', '.join(LLM_PROVIDERS.keys())}"
        )
    
    class_path = LLM_PROVIDERS[provider]
    
    try:
        model_class = _import_class(class_path)
    except ImportError as e:
        package = class_path.split(".")[0]
        raise ImportError(
            f"Provider '{provider}' requires {package}. "
            f"Install with: pip install {package}"
        ) from e
    
    # Build kwargs based on provider
    model_kwargs: dict[str, Any] = {
        "temperature": temperature,
    }
    
    if max_tokens:
        model_kwargs["max_tokens"] = max_tokens
    
    # Provider-specific parameter mapping
    if provider in ("openai", "mistral", "groq", "fireworks", "together"):
        model_kwargs["model"] = model
    elif provider in ("azure", "azure_openai"):
        model_kwargs["azure_deployment"] = kwargs.pop("azure_deployment", model)
        model_kwargs["api_version"] = kwargs.pop("api_version", "2024-02-15-preview")
        if "azure_endpoint" in kwargs:
            model_kwargs["azure_endpoint"] = kwargs.pop("azure_endpoint")
    elif provider == "anthropic":
        model_kwargs["model"] = model
    elif provider in ("google", "gemini"):
        model_kwargs["model"] = model
    elif provider == "ollama":
        model_kwargs["model"] = model
    elif provider == "bedrock":
        model_kwargs["model_id"] = model
    elif provider == "cohere":
        model_kwargs["model"] = model
    
    model_kwargs.update(kwargs)
    
    return model_class(**model_kwargs)


def create_embeddings(
    model: str | Embeddings | dict[str, Any] | None = None,
    *,
    provider: str | None = None,
    **kwargs: Any,
) -> Embeddings:
    """Create an embeddings model from various input formats.
    
    Supports flexible specification:
    - Direct LangChain Embeddings object (returned as-is)
    - Model name string (e.g., "text-embedding-3-small")
    - Provider-prefixed string (e.g., "openai:text-embedding-3-large")
    - Configuration dictionary
    
    Args:
        model: Model specification (string, object, or dict)
        provider: Explicit provider override
        **kwargs: Additional provider-specific parameters
        
    Returns:
        Configured LangChain embeddings model.
        
    Examples:
        # Simple model name (auto-detects OpenAI)
        >>> embeddings = create_embeddings("text-embedding-3-small")
        
        # Provider-prefixed
        >>> embeddings = create_embeddings("cohere:embed-english-v3.0")
        
        # Azure OpenAI
        >>> embeddings = create_embeddings(
        ...     "azure:text-embedding-3-small",
        ...     azure_endpoint="https://my-resource.openai.azure.com",
        ...     azure_deployment="my-embedding-deployment",
        ... )
        
        # Direct LangChain object
        >>> from langchain_openai import OpenAIEmbeddings
        >>> embeddings = create_embeddings(OpenAIEmbeddings())
        
        # Use default for provider
        >>> embeddings = create_embeddings(provider="openai")
        
        # Local Ollama embeddings
        >>> embeddings = create_embeddings("ollama:nomic-embed-text")
    """
    from langchain_core.embeddings import Embeddings
    
    # If already an embeddings object, return as-is
    if isinstance(model, Embeddings):
        return model
    
    # Handle dict configuration
    if isinstance(model, dict):
        config = model.copy()
        provider = config.pop("provider", provider)
        model = config.pop("model", config.pop("model_name", None))
        kwargs.update(config)
    
    # Parse model string
    if isinstance(model, str):
        parsed_provider, model_name = _parse_model_string(model)
        if parsed_provider:
            provider = parsed_provider
        model = model_name
    
    # Infer provider from model name
    if provider is None:
        if model:
            provider = _infer_embedding_provider(model)
        else:
            provider = "openai"
    
    provider = provider.lower()
    
    # Use default model if none specified
    if not model:
        model = DEFAULT_EMBEDDINGS.get(provider, "text-embedding-3-small")
    
    # Get provider class path
    if provider not in EMBEDDING_PROVIDERS:
        raise ValueError(
            f"Unknown embedding provider: {provider}. "
            f"Supported: {', '.join(EMBEDDING_PROVIDERS.keys())}"
        )
    
    class_path = EMBEDDING_PROVIDERS[provider]
    
    try:
        embeddings_class = _import_class(class_path)
    except ImportError as e:
        package = class_path.split(".")[0]
        raise ImportError(
            f"Embedding provider '{provider}' requires {package}. "
            f"Install with: pip install {package}"
        ) from e
    
    # Build kwargs based on provider
    model_kwargs: dict[str, Any] = {}
    
    # Provider-specific parameter mapping
    if provider == "openai":
        model_kwargs["model"] = model
    elif provider in ("azure", "azure_openai"):
        model_kwargs["azure_deployment"] = kwargs.pop("azure_deployment", model)
        if "azure_endpoint" in kwargs:
            model_kwargs["azure_endpoint"] = kwargs.pop("azure_endpoint")
        if "api_version" in kwargs:
            model_kwargs["api_version"] = kwargs.pop("api_version")
    elif provider == "cohere":
        model_kwargs["model"] = model
    elif provider == "huggingface":
        model_kwargs["model_name"] = model
    elif provider == "ollama":
        model_kwargs["model"] = model
    elif provider == "bedrock":
        model_kwargs["model_id"] = model
    elif provider in ("google", "gemini"):
        model_kwargs["model"] = model
    elif provider == "mistral":
        model_kwargs["model"] = model
    elif provider == "voyage":
        model_kwargs["model"] = model
    
    model_kwargs.update(kwargs)
    
    return embeddings_class(**model_kwargs)


def _infer_embedding_provider(model_name: str) -> str:
    """Infer the embedding provider from a model name."""
    model_lower = model_name.lower()
    
    # OpenAI embeddings
    if "embedding" in model_lower and ("ada" in model_lower or model_lower.startswith("text-embedding")):
        return "openai"
    
    # Cohere embeddings
    if model_lower.startswith("embed-"):
        return "cohere"
    
    # Ollama local embeddings
    if model_lower.startswith(("nomic", "mxbai", "snowflake")):
        return "ollama"
    
    # Google embeddings
    if "gecko" in model_lower or model_lower.startswith("models/embedding"):
        return "google"
    
    # HuggingFace (sentence transformers)
    if "/" in model_name:  # HF format: org/model
        return "huggingface"
    
    # Default to OpenAI
    return "openai"


# Convenience aliases
def openai_chat(
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    **kwargs: Any,
) -> BaseChatModel:
    """Create an OpenAI chat model.
    
    Args:
        model: Model name (default: gpt-4o-mini)
        temperature: Temperature setting
        **kwargs: Additional parameters
        
    Returns:
        ChatOpenAI instance.
    """
    return create_chat_model(model, provider="openai", temperature=temperature, **kwargs)


def azure_chat(
    deployment: str,
    *,
    azure_endpoint: str,
    api_version: str = "2024-02-15-preview",
    temperature: float = 0.7,
    **kwargs: Any,
) -> BaseChatModel:
    """Create an Azure OpenAI chat model.
    
    Args:
        deployment: Azure deployment name
        azure_endpoint: Azure OpenAI endpoint URL
        api_version: API version
        temperature: Temperature setting
        **kwargs: Additional parameters
        
    Returns:
        AzureChatOpenAI instance.
    """
    return create_chat_model(
        deployment,
        provider="azure",
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        temperature=temperature,
        **kwargs,
    )


def anthropic_chat(
    model: str = "claude-3-5-sonnet-latest",
    temperature: float = 0.7,
    **kwargs: Any,
) -> BaseChatModel:
    """Create an Anthropic chat model.
    
    Args:
        model: Model name (default: claude-3-5-sonnet-latest)
        temperature: Temperature setting
        **kwargs: Additional parameters
        
    Returns:
        ChatAnthropic instance.
    """
    return create_chat_model(model, provider="anthropic", temperature=temperature, **kwargs)


def ollama_chat(
    model: str = "llama3.2",
    temperature: float = 0.7,
    **kwargs: Any,
) -> BaseChatModel:
    """Create an Ollama (local) chat model.
    
    Args:
        model: Model name (default: llama3.2)
        temperature: Temperature setting
        **kwargs: Additional parameters
        
    Returns:
        ChatOllama instance.
    """
    return create_chat_model(model, provider="ollama", temperature=temperature, **kwargs)


def openai_embeddings(
    model: str = "text-embedding-3-small",
    **kwargs: Any,
) -> Embeddings:
    """Create OpenAI embeddings.
    
    Args:
        model: Model name (default: text-embedding-3-small)
        **kwargs: Additional parameters
        
    Returns:
        OpenAIEmbeddings instance.
    """
    return create_embeddings(model, provider="openai", **kwargs)


def azure_embeddings(
    deployment: str,
    *,
    azure_endpoint: str,
    api_version: str = "2024-02-15-preview",
    **kwargs: Any,
) -> Embeddings:
    """Create Azure OpenAI embeddings.
    
    Args:
        deployment: Azure deployment name
        azure_endpoint: Azure OpenAI endpoint URL
        api_version: API version
        **kwargs: Additional parameters
        
    Returns:
        AzureOpenAIEmbeddings instance.
    """
    return create_embeddings(
        deployment,
        provider="azure",
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        **kwargs,
    )


__all__ = [
    # Main factory functions
    "create_chat_model",
    "create_embeddings",
    # Convenience aliases
    "openai_chat",
    "azure_chat",
    "anthropic_chat",
    "ollama_chat",
    "openai_embeddings",
    "azure_embeddings",
    # Provider mappings (for reference)
    "LLM_PROVIDERS",
    "EMBEDDING_PROVIDERS",
]
