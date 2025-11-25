"""Google (Gemini) provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agenticflow.providers.base import BaseProvider
from agenticflow.providers.enums import DEFAULT_CHAT_MODELS, DEFAULT_EMBEDDING_MODELS

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel


class GoogleProvider(BaseProvider):
    """Google provider for Gemini models.

    Supports:
    - Chat models: gemini-1.5-pro, gemini-1.5-flash, gemini-pro, etc.
    - Embeddings: models/embedding-001, models/text-embedding-004

    Authentication:
        Set GOOGLE_API_KEY environment variable or pass api_key parameter.

    Example:
        >>> from agenticflow.providers import GoogleProvider
        >>>
        >>> provider = GoogleProvider()
        >>> model = provider.create_chat_model("gemini-1.5-flash")
        >>> embeddings = provider.create_embeddings("models/embedding-001")
    """

    name = "google"
    supports_chat = True
    supports_embeddings = True
    default_chat_model = DEFAULT_CHAT_MODELS["google"]
    default_embedding_model = DEFAULT_EMBEDDING_MODELS["google"]

    def __init__(
        self,
        *,
        api_key: str | None = None,
        transport: str = "rest",
    ) -> None:
        """Initialize Google provider.

        Args:
            api_key: Google API key. If None, uses GOOGLE_API_KEY env var.
            transport: Transport method ('rest' or 'grpc').
        """
        self.api_key = api_key
        self.transport = transport

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
        """Create a Google Gemini chat model.

        Args:
            model: Model name (e.g., 'gemini-1.5-flash').
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            top_p: Nucleus sampling parameter.
            stop: Stop sequences.
            streaming: Enable streaming responses.
            **kwargs: Additional ChatGoogleGenerativeAI parameters.

        Returns:
            Configured ChatGoogleGenerativeAI instance.

        Raises:
            ImportError: If langchain-google-genai is not installed.
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError(
                "Google provider requires langchain-google-genai. "
                "Install with: uv add langchain-google-genai"
            ) from e

        model = model or self.default_chat_model

        model_kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "transport": self.transport,
        }

        if self.api_key:
            model_kwargs["google_api_key"] = self.api_key

        if max_tokens is not None:
            model_kwargs["max_output_tokens"] = max_tokens
        if top_p is not None:
            model_kwargs["top_p"] = top_p
        if stop is not None:
            model_kwargs["stop"] = stop

        model_kwargs.update(kwargs)

        return ChatGoogleGenerativeAI(**model_kwargs)

    def create_embeddings(
        self,
        model: str | None = None,
        *,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> Embeddings:
        """Create Google embeddings.

        Args:
            model: Model name (e.g., 'models/embedding-001').
            dimensions: Not supported by Google embeddings.
            **kwargs: Additional GoogleGenerativeAIEmbeddings parameters.

        Returns:
            Configured GoogleGenerativeAIEmbeddings instance.

        Raises:
            ImportError: If langchain-google-genai is not installed.
        """
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as e:
            raise ImportError(
                "Google provider requires langchain-google-genai. "
                "Install with: uv add langchain-google-genai"
            ) from e

        model = model or self.default_embedding_model

        embedding_kwargs: dict[str, Any] = {
            "model": model,
        }

        if self.api_key:
            embedding_kwargs["google_api_key"] = self.api_key

        embedding_kwargs.update(kwargs)

        return GoogleGenerativeAIEmbeddings(**embedding_kwargs)


# Alias for 'gemini' provider name
class GeminiProvider(GoogleProvider):
    """Alias for GoogleProvider registered as 'gemini'."""

    name = "gemini"


__all__ = ["GoogleProvider", "GeminiProvider"]
