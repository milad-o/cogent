"""OpenAI provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agenticflow.providers.base import BaseProvider
from agenticflow.providers.enums import DEFAULT_CHAT_MODELS, DEFAULT_EMBEDDING_MODELS

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel


class OpenAIProvider(BaseProvider):
    """OpenAI provider for GPT models and embeddings.

    Supports:
    - Chat models: gpt-4o, gpt-4o-mini, gpt-4-turbo, o1-preview, o1-mini, etc.
    - Embeddings: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002

    Authentication:
        Set OPENAI_API_KEY environment variable or pass api_key parameter.

    Example:
        >>> from agenticflow.providers import OpenAIProvider
        >>>
        >>> provider = OpenAIProvider()
        >>> model = provider.create_chat_model("gpt-4o-mini", temperature=0.7)
        >>> embeddings = provider.create_embeddings("text-embedding-3-small")
        >>>
        >>> # With custom API key
        >>> provider = OpenAIProvider(api_key="sk-...")
        >>> model = provider.create_chat_model("gpt-4o")
    """

    name = "openai"
    supports_chat = True
    supports_embeddings = True
    default_chat_model = DEFAULT_CHAT_MODELS["openai"]
    default_embedding_model = DEFAULT_EMBEDDING_MODELS["openai"]

    def __init__(
        self,
        *,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
    ) -> None:
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            organization: OpenAI organization ID.
            base_url: Custom base URL (for proxies or compatible APIs).
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
        """
        self.api_key = api_key
        self.organization = organization
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

    def _get_common_kwargs(self) -> dict[str, Any]:
        """Get common kwargs for OpenAI clients."""
        kwargs: dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.organization:
            kwargs["organization"] = self.organization
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout:
            kwargs["timeout"] = self.timeout
        kwargs["max_retries"] = self.max_retries
        return kwargs

    def create_chat_model(
        self,
        model: str | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        streaming: bool = False,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Create an OpenAI chat model.

        Args:
            model: Model name (e.g., 'gpt-4o-mini'). Defaults to gpt-4o-mini.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum tokens in response.
            top_p: Nucleus sampling parameter.
            stop: Stop sequences.
            streaming: Enable streaming responses.
            **kwargs: Additional ChatOpenAI parameters.

        Returns:
            Configured ChatOpenAI instance.

        Raises:
            ImportError: If langchain-openai is not installed.
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI provider requires langchain-openai. "
                "Install with: uv add langchain-openai"
            ) from e

        model = model or self.default_chat_model

        model_kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "streaming": streaming,
            **self._get_common_kwargs(),
        }

        if max_tokens is not None:
            model_kwargs["max_tokens"] = max_tokens
        if top_p is not None:
            model_kwargs["top_p"] = top_p
        if stop is not None:
            model_kwargs["stop"] = stop

        model_kwargs.update(kwargs)

        return ChatOpenAI(**model_kwargs)

    def create_embeddings(
        self,
        model: str | None = None,
        *,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> Embeddings:
        """Create OpenAI embeddings.

        Args:
            model: Model name (e.g., 'text-embedding-3-small').
            dimensions: Output dimensions (only for text-embedding-3-* models).
            **kwargs: Additional OpenAIEmbeddings parameters.

        Returns:
            Configured OpenAIEmbeddings instance.

        Raises:
            ImportError: If langchain-openai is not installed.
        """
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as e:
            raise ImportError(
                "OpenAI provider requires langchain-openai. "
                "Install with: uv add langchain-openai"
            ) from e

        model = model or self.default_embedding_model

        embedding_kwargs: dict[str, Any] = {
            "model": model,
            **self._get_common_kwargs(),
        }

        # Only text-embedding-3-* models support dimensions
        if dimensions is not None and model.startswith("text-embedding-3"):
            embedding_kwargs["dimensions"] = dimensions

        embedding_kwargs.update(kwargs)

        return OpenAIEmbeddings(**embedding_kwargs)


__all__ = ["OpenAIProvider"]
