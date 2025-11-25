"""Base provider classes and specifications.

This module defines the abstract base class for all providers and
the data structures for model/embedding specifications.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel


@dataclass
class ModelSpec:
    """Specification for creating a chat model.

    A type-safe way to configure LLM models with full IDE support.

    Attributes:
        provider: Provider name (e.g., 'openai', 'azure', 'anthropic').
        model: Model name or deployment name.
        temperature: Sampling temperature (0.0-2.0).
        max_tokens: Maximum tokens in response.
        top_p: Nucleus sampling parameter.
        stop: Stop sequences.
        extra: Additional provider-specific parameters.

    Examples:
        >>> spec = ModelSpec(provider="openai", model="gpt-4o-mini")
        >>> model = spec.create()

        >>> spec = ModelSpec(
        ...     provider="azure",
        ...     model="gpt-4o-deployment",
        ...     temperature=0.5,
        ...     extra={"azure_endpoint": "https://my.openai.azure.com"},
        ... )
    """

    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_string(cls, spec_string: str, **kwargs: Any) -> ModelSpec:
        """Create a ModelSpec from a string like 'openai/gpt-4o'.

        Supports formats:
        - 'provider/model' (preferred): 'openai/gpt-4o-mini'
        - 'provider:model' (legacy): 'openai:gpt-4o-mini'
        - 'model' (auto-detect): 'gpt-4o-mini' -> infers OpenAI

        Args:
            spec_string: Model specification string.
            **kwargs: Additional parameters to pass to ModelSpec.

        Returns:
            Configured ModelSpec instance.

        Examples:
            >>> spec = ModelSpec.from_string("openai/gpt-4o")
            >>> spec = ModelSpec.from_string("anthropic/claude-3-5-sonnet-latest")
            >>> spec = ModelSpec.from_string("gpt-4o")  # auto-detects openai
        """
        provider, model = _parse_spec_string(spec_string)
        return cls(provider=provider, model=model, **kwargs)

    def create(self) -> BaseChatModel:
        """Create a LangChain chat model from this spec.

        Returns:
            Configured BaseChatModel instance.

        Raises:
            ValueError: If provider is not registered.
            ImportError: If provider package is not installed.
        """
        provider_instance = ProviderRegistry.get(self.provider)
        return provider_instance.create_chat_model(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            stop=self.stop,
            **self.extra,
        )

    async def acreate(self) -> BaseChatModel:
        """Async version of create().

        Some providers may need async initialization (e.g., fetching tokens).

        Returns:
            Configured BaseChatModel instance.
        """
        provider_instance = ProviderRegistry.get(self.provider)
        return await provider_instance.acreate_chat_model(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            stop=self.stop,
            **self.extra,
        )


@dataclass
class EmbeddingSpec:
    """Specification for creating an embedding model.

    Attributes:
        provider: Provider name (e.g., 'openai', 'azure', 'cohere').
        model: Model name or deployment name.
        dimensions: Output embedding dimensions (if supported).
        extra: Additional provider-specific parameters.

    Examples:
        >>> spec = EmbeddingSpec(provider="openai", model="text-embedding-3-small")
        >>> embeddings = spec.create()

        >>> spec = EmbeddingSpec(
        ...     provider="openai",
        ...     model="text-embedding-3-large",
        ...     dimensions=1024,  # Reduce from 3072 for efficiency
        ... )
    """

    provider: str
    model: str
    dimensions: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_string(cls, spec_string: str, **kwargs: Any) -> EmbeddingSpec:
        """Create an EmbeddingSpec from a string like 'openai/text-embedding-3-small'.

        Args:
            spec_string: Embedding specification string.
            **kwargs: Additional parameters.

        Returns:
            Configured EmbeddingSpec instance.
        """
        provider, model = _parse_spec_string(spec_string)
        return cls(provider=provider, model=model, **kwargs)

    def create(self) -> Embeddings:
        """Create a LangChain embeddings model from this spec.

        Returns:
            Configured Embeddings instance.
        """
        provider_instance = ProviderRegistry.get(self.provider)
        return provider_instance.create_embeddings(
            model=self.model,
            dimensions=self.dimensions,
            **self.extra,
        )

    async def acreate(self) -> Embeddings:
        """Async version of create()."""
        provider_instance = ProviderRegistry.get(self.provider)
        return await provider_instance.acreate_embeddings(
            model=self.model,
            dimensions=self.dimensions,
            **self.extra,
        )


class BaseProvider(ABC):
    """Abstract base class for LLM/embedding providers.

    Subclass this to implement a new provider. Each provider must implement
    the sync versions of create methods. Async versions have default
    implementations that call the sync versions.

    Class Attributes:
        name: Unique provider identifier (e.g., 'openai', 'azure').
        supports_chat: Whether this provider supports chat models.
        supports_embeddings: Whether this provider supports embeddings.
        default_chat_model: Default model name for chat.
        default_embedding_model: Default model name for embeddings.

    Example:
        >>> class MyProvider(BaseProvider):
        ...     name = "myprovider"
        ...     supports_chat = True
        ...     supports_embeddings = False
        ...     default_chat_model = "my-model-v1"
        ...
        ...     def create_chat_model(self, model, **kwargs):
        ...         # Implementation
        ...         pass
    """

    name: ClassVar[str]
    supports_chat: ClassVar[bool] = True
    supports_embeddings: ClassVar[bool] = True
    default_chat_model: ClassVar[str | None] = None
    default_embedding_model: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register provider subclasses."""
        super().__init_subclass__(**kwargs)
        # Only register if name is defined (not abstract)
        if hasattr(cls, "name") and cls.name:
            ProviderRegistry.register(cls)

    @abstractmethod
    def create_chat_model(
        self,
        model: str | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Create a chat model instance.

        Args:
            model: Model name. If None, uses default_chat_model.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.
            top_p: Nucleus sampling parameter.
            stop: Stop sequences.
            **kwargs: Provider-specific parameters.

        Returns:
            Configured LangChain chat model.

        Raises:
            NotImplementedError: If provider doesn't support chat.
            ImportError: If required package not installed.
        """
        ...

    async def acreate_chat_model(
        self,
        model: str | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Async version of create_chat_model.

        Default implementation calls sync version. Override for providers
        that need async initialization (e.g., token fetching).
        """
        return self.create_chat_model(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            **kwargs,
        )

    @abstractmethod
    def create_embeddings(
        self,
        model: str | None = None,
        *,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> Embeddings:
        """Create an embeddings model instance.

        Args:
            model: Model name. If None, uses default_embedding_model.
            dimensions: Output dimensions (if supported by model).
            **kwargs: Provider-specific parameters.

        Returns:
            Configured LangChain embeddings model.

        Raises:
            NotImplementedError: If provider doesn't support embeddings.
            ImportError: If required package not installed.
        """
        ...

    async def acreate_embeddings(
        self,
        model: str | None = None,
        *,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> Embeddings:
        """Async version of create_embeddings.

        Default implementation calls sync version.
        """
        return self.create_embeddings(
            model=model,
            dimensions=dimensions,
            **kwargs,
        )

    def _import_class(self, class_path: str) -> type:
        """Dynamically import a class from a module path.

        Args:
            class_path: Full path like 'langchain_openai.ChatOpenAI'.

        Returns:
            The imported class.

        Raises:
            ImportError: If module or class not found.
        """
        import importlib

        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)


class ProviderRegistry:
    """Registry for provider classes.

    Providers auto-register when subclassing BaseProvider with a `name` attribute.
    You can also manually register providers.

    Example:
        >>> ProviderRegistry.register(MyProvider)
        >>> provider = ProviderRegistry.get("myprovider")
    """

    _providers: ClassVar[dict[str, type[BaseProvider]]] = {}
    _instances: ClassVar[dict[str, BaseProvider]] = {}

    @classmethod
    def register(cls, provider_class: type[BaseProvider]) -> None:
        """Register a provider class.

        Args:
            provider_class: Provider class to register.
        """
        cls._providers[provider_class.name] = provider_class

    @classmethod
    def get(cls, name: str) -> BaseProvider:
        """Get a provider instance by name.

        Creates singleton instances for each provider.

        Args:
            name: Provider name (e.g., 'openai', 'azure').

        Returns:
            Provider instance.

        Raises:
            ValueError: If provider not found.
        """
        name = name.lower()

        # Return cached instance
        if name in cls._instances:
            return cls._instances[name]

        # Create new instance
        if name not in cls._providers:
            available = ", ".join(sorted(cls._providers.keys()))
            raise ValueError(
                f"Unknown provider: '{name}'. Available providers: {available}"
            )

        instance = cls._providers[name]()
        cls._instances[name] = instance
        return instance

    @classmethod
    def get_with_config(cls, name: str, **config: Any) -> BaseProvider:
        """Get a provider instance with custom configuration.

        Unlike get(), this always creates a new instance with the given config.
        Useful for Azure with custom endpoints or auth methods.

        Args:
            name: Provider name.
            **config: Configuration to pass to provider constructor.

        Returns:
            New provider instance with given configuration.
        """
        name = name.lower()
        if name not in cls._providers:
            available = ", ".join(sorted(cls._providers.keys()))
            raise ValueError(
                f"Unknown provider: '{name}'. Available providers: {available}"
            )
        return cls._providers[name](**config)

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names.

        Returns:
            Sorted list of provider names.
        """
        return sorted(cls._providers.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a provider is registered.

        Args:
            name: Provider name.

        Returns:
            True if provider is registered.
        """
        return name.lower() in cls._providers


def _parse_spec_string(spec_string: str) -> tuple[str, str]:
    """Parse a model spec string into (provider, model).

    Supports:
    - 'provider/model' (preferred): 'openai/gpt-4o'
    - 'provider:model' (legacy): 'openai:gpt-4o'
    - 'model' (auto-detect): 'gpt-4o' -> ('openai', 'gpt-4o')

    Args:
        spec_string: Model specification string.

    Returns:
        Tuple of (provider, model).
    """
    # Preferred format: provider/model
    if "/" in spec_string:
        provider, model = spec_string.split("/", 1)
        return provider.lower(), model

    # Legacy format: provider:model
    if ":" in spec_string:
        provider, model = spec_string.split(":", 1)
        return provider.lower(), model

    # Auto-detect provider from model name
    provider = _infer_provider(spec_string)
    return provider, spec_string


def _infer_provider(model_name: str) -> str:
    """Infer provider from model name.

    Args:
        model_name: Model name like 'gpt-4o' or 'claude-3-opus'.

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

    # Local models (Ollama)
    if model_lower.startswith(("llama", "phi", "qwen", "deepseek", "codellama")):
        return "ollama"

    # Embedding models
    if "embedding" in model_lower or model_lower.startswith("text-embedding"):
        return "openai"

    # Default to OpenAI
    return "openai"


__all__ = [
    "BaseProvider",
    "ModelSpec",
    "EmbeddingSpec",
    "ProviderRegistry",
]
