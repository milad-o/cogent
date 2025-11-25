"""Ollama provider implementation for local models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agenticflow.providers.base import BaseProvider
from agenticflow.providers.enums import DEFAULT_CHAT_MODELS, DEFAULT_EMBEDDING_MODELS

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel


class OllamaProvider(BaseProvider):
    """Ollama provider for local models.

    Supports:
    - Chat models: llama3.2, llama3.1, mistral, phi3, qwen2, etc.
    - Embeddings: nomic-embed-text, mxbai-embed-large, all-minilm, etc.

    Requirements:
        Ollama must be installed and running locally.
        Download models with: ollama pull <model-name>

    Example:
        >>> from agenticflow.providers import OllamaProvider
        >>>
        >>> provider = OllamaProvider()
        >>> model = provider.create_chat_model("llama3.2")
        >>>
        >>> # Custom Ollama server
        >>> provider = OllamaProvider(base_url="http://localhost:11434")
    """

    name = "ollama"
    supports_chat = True
    supports_embeddings = True
    default_chat_model = DEFAULT_CHAT_MODELS["ollama"]
    default_embedding_model = DEFAULT_EMBEDDING_MODELS["ollama"]

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        timeout: float | None = None,
    ) -> None:
        """Initialize Ollama provider.

        Args:
            base_url: Ollama server URL.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url
        self.timeout = timeout

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
        """Create an Ollama chat model.

        Args:
            model: Model name (e.g., 'llama3.2').
            temperature: Sampling temperature.
            max_tokens: Maximum tokens (num_predict in Ollama).
            top_p: Nucleus sampling parameter.
            stop: Stop sequences.
            streaming: Enable streaming responses.
            **kwargs: Additional ChatOllama parameters.

        Returns:
            Configured ChatOllama instance.

        Raises:
            ImportError: If langchain-ollama is not installed.
        """
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise ImportError(
                "Ollama provider requires langchain-ollama. "
                "Install with: uv add langchain-ollama"
            ) from e

        model = model or self.default_chat_model

        model_kwargs: dict[str, Any] = {
            "model": model,
            "base_url": self.base_url,
            "temperature": temperature,
        }

        if self.timeout:
            model_kwargs["timeout"] = self.timeout

        if max_tokens is not None:
            model_kwargs["num_predict"] = max_tokens
        if top_p is not None:
            model_kwargs["top_p"] = top_p
        if stop is not None:
            model_kwargs["stop"] = stop

        model_kwargs.update(kwargs)

        return ChatOllama(**model_kwargs)

    def create_embeddings(
        self,
        model: str | None = None,
        *,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> Embeddings:
        """Create Ollama embeddings.

        Args:
            model: Model name (e.g., 'nomic-embed-text').
            dimensions: Not supported by Ollama embeddings.
            **kwargs: Additional OllamaEmbeddings parameters.

        Returns:
            Configured OllamaEmbeddings instance.

        Raises:
            ImportError: If langchain-ollama is not installed.
        """
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as e:
            raise ImportError(
                "Ollama provider requires langchain-ollama. "
                "Install with: uv add langchain-ollama"
            ) from e

        model = model or self.default_embedding_model

        embedding_kwargs: dict[str, Any] = {
            "model": model,
            "base_url": self.base_url,
        }

        embedding_kwargs.update(kwargs)

        return OllamaEmbeddings(**embedding_kwargs)


__all__ = ["OllamaProvider"]
