"""Anthropic provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agenticflow.providers.base import BaseProvider
from agenticflow.providers.enums import DEFAULT_CHAT_MODELS

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel


class AnthropicProvider(BaseProvider):
    """Anthropic provider for Claude models.

    Supports:
    - Chat models: claude-3-5-sonnet-latest, claude-3-opus-latest,
                   claude-3-haiku-20240307, etc.

    Note: Anthropic does not provide embedding models.

    Authentication:
        Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.

    Example:
        >>> from agenticflow.providers import AnthropicProvider
        >>>
        >>> provider = AnthropicProvider()
        >>> model = provider.create_chat_model("claude-3-5-sonnet-latest")
        >>>
        >>> # With custom API key
        >>> provider = AnthropicProvider(api_key="sk-ant-...")
    """

    name = "anthropic"
    supports_chat = True
    supports_embeddings = False
    default_chat_model = DEFAULT_CHAT_MODELS["anthropic"]
    default_embedding_model = None

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
    ) -> None:
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            base_url: Custom base URL.
            timeout: Request timeout in seconds.
            max_retries: Maximum retries for failed requests.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

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
        """Create an Anthropic chat model.

        Args:
            model: Model name (e.g., 'claude-3-5-sonnet-latest').
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens in response. Required for Anthropic.
            top_p: Nucleus sampling parameter.
            stop: Stop sequences.
            streaming: Enable streaming responses.
            **kwargs: Additional ChatAnthropic parameters.

        Returns:
            Configured ChatAnthropic instance.

        Raises:
            ImportError: If langchain-anthropic is not installed.
        """
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ImportError(
                "Anthropic provider requires langchain-anthropic. "
                "Install with: uv add langchain-anthropic"
            ) from e

        model = model or self.default_chat_model

        model_kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "streaming": streaming,
        }

        if self.api_key:
            model_kwargs["api_key"] = self.api_key
        if self.base_url:
            model_kwargs["base_url"] = self.base_url
        if self.timeout:
            model_kwargs["timeout"] = self.timeout
        model_kwargs["max_retries"] = self.max_retries

        # Anthropic requires max_tokens
        if max_tokens is not None:
            model_kwargs["max_tokens"] = max_tokens
        else:
            model_kwargs["max_tokens"] = 4096  # Default for Anthropic

        if top_p is not None:
            model_kwargs["top_p"] = top_p
        if stop is not None:
            model_kwargs["stop_sequences"] = stop

        model_kwargs.update(kwargs)

        return ChatAnthropic(**model_kwargs)

    def create_embeddings(
        self,
        model: str | None = None,
        *,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> Embeddings:
        """Anthropic does not support embeddings.

        Raises:
            NotImplementedError: Always, as Anthropic has no embedding models.
        """
        raise NotImplementedError(
            "Anthropic does not provide embedding models. "
            "Consider using OpenAI, Cohere, or Voyage for embeddings."
        )


__all__ = ["AnthropicProvider"]
