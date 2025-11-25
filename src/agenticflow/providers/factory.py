"""Factory functions for creating models and embeddings.

High-level convenience functions that parse spec strings and create
model/embedding instances.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agenticflow.providers.base import (
    BaseProvider,
    EmbeddingSpec,
    ModelSpec,
    ProviderRegistry,
    _parse_spec_string,
)

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel


def create_model(
    spec: str | ModelSpec | BaseChatModel | dict[str, Any],
    *,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a chat model from various input formats.

    This is the primary factory function for creating LLM instances.

    Args:
        spec: Model specification. Can be:
            - String: 'openai/gpt-4o', 'anthropic/claude-3-5-sonnet-latest'
            - ModelSpec: Explicit specification object
            - BaseChatModel: Returned as-is
            - Dict: Configuration dictionary
        temperature: Sampling temperature (0.0-2.0).
        max_tokens: Maximum response tokens.
        **kwargs: Additional provider-specific parameters.

    Returns:
        Configured LangChain chat model.

    Examples:
        # String spec (provider/model format)
        >>> model = create_model("openai/gpt-4o-mini")
        >>> model = create_model("anthropic/claude-3-5-sonnet-latest")

        # Auto-detect provider from model name
        >>> model = create_model("gpt-4o")  # Infers OpenAI
        >>> model = create_model("claude-3-opus")  # Infers Anthropic

        # Legacy colon format (still supported)
        >>> model = create_model("openai:gpt-4o")

        # With parameters
        >>> model = create_model("openai/gpt-4o", temperature=0.5, max_tokens=1000)

        # Using ModelSpec
        >>> spec = ModelSpec(provider="openai", model="gpt-4o", temperature=0.3)
        >>> model = create_model(spec)

        # Pass through existing model
        >>> from langchain_openai import ChatOpenAI
        >>> existing = ChatOpenAI(model="gpt-4o")
        >>> model = create_model(existing)  # Returns existing

        # Dict configuration
        >>> model = create_model({
        ...     "provider": "openai",
        ...     "model": "gpt-4o",
        ...     "temperature": 0.5,
        ... })
    """
    from langchain_core.language_models import BaseChatModel as LCBaseChatModel

    # Pass through existing model
    if isinstance(spec, LCBaseChatModel):
        return spec

    # Handle ModelSpec
    if isinstance(spec, ModelSpec):
        return spec.create()

    # Handle dict
    if isinstance(spec, dict):
        config = spec.copy()
        provider = config.pop("provider", None)
        model = config.pop("model", config.pop("model_name", None))
        temp = config.pop("temperature", temperature)
        max_tok = config.pop("max_tokens", max_tokens)
        config.update(kwargs)

        if provider and model:
            model_spec = ModelSpec(
                provider=provider,
                model=model,
                temperature=temp,
                max_tokens=max_tok,
                extra=config,
            )
            return model_spec.create()
        elif model:
            # Just model name, parse it
            return create_model(
                model, temperature=temp, max_tokens=max_tok, **config
            )
        else:
            raise ValueError("Dict spec must include 'model' key")

    # Handle string
    if isinstance(spec, str):
        provider_name, model_name = _parse_spec_string(spec)
        provider = ProviderRegistry.get(provider_name)
        return provider.create_chat_model(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    raise TypeError(
        f"spec must be str, ModelSpec, BaseChatModel, or dict, got {type(spec)}"
    )


async def acreate_model(
    spec: str | ModelSpec | BaseChatModel | dict[str, Any],
    *,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Async version of create_model.

    Useful for providers that need async initialization (e.g., token fetching).

    Args:
        spec: Model specification (same as create_model).
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.
        **kwargs: Additional parameters.

    Returns:
        Configured chat model.
    """
    from langchain_core.language_models import BaseChatModel as LCBaseChatModel

    if isinstance(spec, LCBaseChatModel):
        return spec

    if isinstance(spec, ModelSpec):
        return await spec.acreate()

    if isinstance(spec, dict):
        config = spec.copy()
        provider = config.pop("provider", None)
        model = config.pop("model", config.pop("model_name", None))
        temp = config.pop("temperature", temperature)
        max_tok = config.pop("max_tokens", max_tokens)
        config.update(kwargs)

        if provider and model:
            model_spec = ModelSpec(
                provider=provider,
                model=model,
                temperature=temp,
                max_tokens=max_tok,
                extra=config,
            )
            return await model_spec.acreate()
        elif model:
            return await acreate_model(
                model, temperature=temp, max_tokens=max_tok, **config
            )
        else:
            raise ValueError("Dict spec must include 'model' key")

    if isinstance(spec, str):
        provider_name, model_name = _parse_spec_string(spec)
        provider = ProviderRegistry.get(provider_name)
        return await provider.acreate_chat_model(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    raise TypeError(
        f"spec must be str, ModelSpec, BaseChatModel, or dict, got {type(spec)}"
    )


def create_embeddings(
    spec: str | EmbeddingSpec | Embeddings | dict[str, Any],
    *,
    dimensions: int | None = None,
    **kwargs: Any,
) -> Embeddings:
    """Create an embeddings model from various input formats.

    Args:
        spec: Embedding specification. Can be:
            - String: 'openai/text-embedding-3-small'
            - EmbeddingSpec: Explicit specification object
            - Embeddings: Returned as-is
            - Dict: Configuration dictionary
        dimensions: Output dimensions (if supported by model).
        **kwargs: Additional provider-specific parameters.

    Returns:
        Configured LangChain embeddings model.

    Examples:
        >>> embeddings = create_embeddings("openai/text-embedding-3-small")
        >>> embeddings = create_embeddings("cohere/embed-english-v3.0")

        # With dimensions (for text-embedding-3-* models)
        >>> embeddings = create_embeddings(
        ...     "openai/text-embedding-3-large",
        ...     dimensions=1024,
        ... )
    """
    from langchain_core.embeddings import Embeddings as LCEmbeddings

    # Pass through existing embeddings
    if isinstance(spec, LCEmbeddings):
        return spec

    # Handle EmbeddingSpec
    if isinstance(spec, EmbeddingSpec):
        return spec.create()

    # Handle dict
    if isinstance(spec, dict):
        config = spec.copy()
        provider = config.pop("provider", None)
        model = config.pop("model", config.pop("model_name", None))
        dims = config.pop("dimensions", dimensions)
        config.update(kwargs)

        if provider and model:
            emb_spec = EmbeddingSpec(
                provider=provider,
                model=model,
                dimensions=dims,
                extra=config,
            )
            return emb_spec.create()
        elif model:
            return create_embeddings(model, dimensions=dims, **config)
        else:
            raise ValueError("Dict spec must include 'model' key")

    # Handle string
    if isinstance(spec, str):
        provider_name, model_name = _parse_spec_string(spec)
        provider = ProviderRegistry.get(provider_name)
        return provider.create_embeddings(
            model=model_name,
            dimensions=dimensions,
            **kwargs,
        )

    raise TypeError(
        f"spec must be str, EmbeddingSpec, Embeddings, or dict, got {type(spec)}"
    )


async def acreate_embeddings(
    spec: str | EmbeddingSpec | Embeddings | dict[str, Any],
    *,
    dimensions: int | None = None,
    **kwargs: Any,
) -> Embeddings:
    """Async version of create_embeddings.

    Args:
        spec: Embedding specification.
        dimensions: Output dimensions.
        **kwargs: Additional parameters.

    Returns:
        Configured embeddings model.
    """
    from langchain_core.embeddings import Embeddings as LCEmbeddings

    if isinstance(spec, LCEmbeddings):
        return spec

    if isinstance(spec, EmbeddingSpec):
        return await spec.acreate()

    if isinstance(spec, dict):
        config = spec.copy()
        provider = config.pop("provider", None)
        model = config.pop("model", config.pop("model_name", None))
        dims = config.pop("dimensions", dimensions)
        config.update(kwargs)

        if provider and model:
            emb_spec = EmbeddingSpec(
                provider=provider,
                model=model,
                dimensions=dims,
                extra=config,
            )
            return await emb_spec.acreate()
        elif model:
            return await acreate_embeddings(model, dimensions=dims, **config)
        else:
            raise ValueError("Dict spec must include 'model' key")

    if isinstance(spec, str):
        provider_name, model_name = _parse_spec_string(spec)
        provider = ProviderRegistry.get(provider_name)
        return await provider.acreate_embeddings(
            model=model_name,
            dimensions=dimensions,
            **kwargs,
        )

    raise TypeError(
        f"spec must be str, EmbeddingSpec, Embeddings, or dict, got {type(spec)}"
    )


def get_provider(name: str, **config: Any) -> BaseProvider:
    """Get a provider instance by name.

    Args:
        name: Provider name (e.g., 'openai', 'azure', 'anthropic').
        **config: Optional configuration for the provider.

    Returns:
        Provider instance.

    Examples:
        >>> provider = get_provider("openai")
        >>> provider = get_provider("azure", endpoint="https://...")
    """
    if config:
        return ProviderRegistry.get_with_config(name, **config)
    return ProviderRegistry.get(name)


def parse_model_spec(spec_string: str) -> tuple[str, str]:
    """Parse a model spec string into (provider, model).

    Args:
        spec_string: Model spec like 'openai/gpt-4o' or 'gpt-4o'.

    Returns:
        Tuple of (provider, model).

    Examples:
        >>> parse_model_spec("openai/gpt-4o")
        ('openai', 'gpt-4o')
        >>> parse_model_spec("gpt-4o")
        ('openai', 'gpt-4o')  # Auto-detected
    """
    return _parse_spec_string(spec_string)


def list_providers() -> list[str]:
    """List all registered provider names.

    Returns:
        Sorted list of provider names.
    """
    return ProviderRegistry.list_providers()


__all__ = [
    "create_model",
    "acreate_model",
    "create_embeddings",
    "acreate_embeddings",
    "get_provider",
    "parse_model_spec",
    "list_providers",
]
