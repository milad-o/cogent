"""Azure OpenAI provider implementation with Managed Identity support."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from agenticflow.providers.base import BaseProvider
from agenticflow.providers.enums import DEFAULT_CHAT_MODELS, DEFAULT_EMBEDDING_MODELS

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel


class AzureAuthMethod(str, Enum):
    """Authentication methods for Azure OpenAI.

    Attributes:
        API_KEY: Traditional API key authentication (AZURE_OPENAI_API_KEY).
        MANAGED_IDENTITY: Azure Managed Identity (system or user-assigned).
        DEFAULT_CREDENTIAL: DefaultAzureCredential (tries multiple methods).
        TOKEN_PROVIDER: Custom token provider callable.
        AD_TOKEN: Direct Azure AD token string.
    """

    API_KEY = "api_key"
    MANAGED_IDENTITY = "managed_identity"
    DEFAULT_CREDENTIAL = "default_credential"
    TOKEN_PROVIDER = "token_provider"
    AD_TOKEN = "ad_token"


# Azure Cognitive Services scope for token requests
AZURE_COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"


@dataclass
class AzureConfig:
    """Configuration for Azure OpenAI authentication.

    Supports multiple authentication methods:
    - API Key (traditional)
    - Managed Identity (for Azure-hosted apps)
    - DefaultAzureCredential (automatic credential chain)
    - Custom token provider
    - Direct AD token

    Attributes:
        endpoint: Azure OpenAI endpoint URL.
        auth_method: Authentication method to use.
        api_key: API key (for API_KEY auth method).
        api_version: Azure OpenAI API version.
        managed_identity_client_id: Client ID for user-assigned managed identity.
        ad_token: Direct Azure AD token (for AD_TOKEN auth method).
        token_provider: Custom sync token provider callable.
        async_token_provider: Custom async token provider callable.

    Example:
        # API Key authentication
        >>> config = AzureConfig(
        ...     endpoint="https://my-resource.openai.azure.com",
        ...     auth_method=AzureAuthMethod.API_KEY,
        ...     api_key="your-api-key",
        ... )

        # Managed Identity (system-assigned)
        >>> config = AzureConfig(
        ...     endpoint="https://my-resource.openai.azure.com",
        ...     auth_method=AzureAuthMethod.MANAGED_IDENTITY,
        ... )

        # Managed Identity (user-assigned)
        >>> config = AzureConfig(
        ...     endpoint="https://my-resource.openai.azure.com",
        ...     auth_method=AzureAuthMethod.MANAGED_IDENTITY,
        ...     managed_identity_client_id="your-client-id",
        ... )

        # DefaultAzureCredential (automatic)
        >>> config = AzureConfig(
        ...     endpoint="https://my-resource.openai.azure.com",
        ...     auth_method=AzureAuthMethod.DEFAULT_CREDENTIAL,
        ... )
    """

    endpoint: str
    auth_method: AzureAuthMethod = AzureAuthMethod.API_KEY
    api_key: str | None = None
    api_version: str = "2024-08-01-preview"
    managed_identity_client_id: str | None = None
    ad_token: str | None = None
    token_provider: Callable[[], str] | None = None
    async_token_provider: Callable[[], Awaitable[str]] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def get_token_provider(self) -> Callable[[], str] | None:
        """Get or create a sync token provider based on auth method.

        Returns:
            Token provider callable or None for API key auth.

        Raises:
            ImportError: If azure-identity is not installed.
            ValueError: If auth method requires missing configuration.
        """
        if self.auth_method == AzureAuthMethod.API_KEY:
            return None

        if self.auth_method == AzureAuthMethod.TOKEN_PROVIDER:
            if self.token_provider is None:
                raise ValueError(
                    "TOKEN_PROVIDER auth method requires 'token_provider' callable"
                )
            return self.token_provider

        if self.auth_method == AzureAuthMethod.AD_TOKEN:
            if self.ad_token is None:
                raise ValueError("AD_TOKEN auth method requires 'ad_token' string")
            # Return a callable that returns the static token
            return lambda: self.ad_token  # type: ignore[return-value]

        # Managed Identity or DefaultCredential
        return self._create_azure_identity_provider()

    def get_async_token_provider(self) -> Callable[[], Awaitable[str]] | None:
        """Get or create an async token provider based on auth method.

        Returns:
            Async token provider callable or None for API key auth.
        """
        if self.auth_method == AzureAuthMethod.API_KEY:
            return None

        if self.auth_method == AzureAuthMethod.TOKEN_PROVIDER:
            return self.async_token_provider

        if self.auth_method == AzureAuthMethod.AD_TOKEN:
            if self.ad_token is None:
                raise ValueError("AD_TOKEN auth method requires 'ad_token' string")

            async def _static_token() -> str:
                return self.ad_token  # type: ignore[return-value]

            return _static_token

        # Managed Identity or DefaultCredential - use async credential
        return self._create_async_azure_identity_provider()

    def _create_azure_identity_provider(self) -> Callable[[], str]:
        """Create a sync token provider using azure-identity.

        Returns:
            Token provider callable.

        Raises:
            ImportError: If azure-identity is not installed.
        """
        try:
            from azure.identity import (
                DefaultAzureCredential,
                ManagedIdentityCredential,
            )
        except ImportError as e:
            raise ImportError(
                "Azure Managed Identity requires azure-identity. "
                "Install with: uv add azure-identity"
            ) from e

        if self.auth_method == AzureAuthMethod.MANAGED_IDENTITY:
            if self.managed_identity_client_id:
                credential = ManagedIdentityCredential(
                    client_id=self.managed_identity_client_id
                )
            else:
                credential = ManagedIdentityCredential()
        else:
            # DEFAULT_CREDENTIAL
            credential = DefaultAzureCredential()

        def _get_token() -> str:
            token = credential.get_token(AZURE_COGNITIVE_SERVICES_SCOPE)
            return token.token

        return _get_token

    def _create_async_azure_identity_provider(self) -> Callable[[], Awaitable[str]]:
        """Create an async token provider using azure-identity.

        Returns:
            Async token provider callable.

        Raises:
            ImportError: If azure-identity is not installed.
        """
        try:
            from azure.identity.aio import (
                DefaultAzureCredential as AsyncDefaultCredential,
            )
            from azure.identity.aio import (
                ManagedIdentityCredential as AsyncManagedIdentityCredential,
            )
        except ImportError as e:
            raise ImportError(
                "Azure async authentication requires azure-identity. "
                "Install with: uv add azure-identity"
            ) from e

        if self.auth_method == AzureAuthMethod.MANAGED_IDENTITY:
            if self.managed_identity_client_id:
                credential = AsyncManagedIdentityCredential(
                    client_id=self.managed_identity_client_id
                )
            else:
                credential = AsyncManagedIdentityCredential()
        else:
            # DEFAULT_CREDENTIAL
            credential = AsyncDefaultCredential()

        async def _get_token_async() -> str:
            token = await credential.get_token(AZURE_COGNITIVE_SERVICES_SCOPE)
            return token.token

        return _get_token_async


class AzureOpenAIProvider(BaseProvider):
    """Azure OpenAI provider with full authentication support.

    Supports multiple authentication methods:
    - API Key: Traditional key-based auth
    - Managed Identity: For Azure-hosted applications
    - DefaultAzureCredential: Automatic credential chain
    - Custom token provider: For advanced scenarios

    Example:
        # API Key authentication
        >>> provider = AzureOpenAIProvider(
        ...     endpoint="https://my-resource.openai.azure.com",
        ...     api_key="your-api-key",
        ... )
        >>> model = provider.create_chat_model("gpt-4o-deployment")

        # Managed Identity (system-assigned)
        >>> provider = AzureOpenAIProvider(
        ...     endpoint="https://my-resource.openai.azure.com",
        ...     auth_method=AzureAuthMethod.MANAGED_IDENTITY,
        ... )

        # Managed Identity (user-assigned)
        >>> provider = AzureOpenAIProvider(
        ...     endpoint="https://my-resource.openai.azure.com",
        ...     auth_method=AzureAuthMethod.MANAGED_IDENTITY,
        ...     managed_identity_client_id="your-client-id",
        ... )

        # DefaultAzureCredential (tries multiple auth methods)
        >>> provider = AzureOpenAIProvider(
        ...     endpoint="https://my-resource.openai.azure.com",
        ...     auth_method=AzureAuthMethod.DEFAULT_CREDENTIAL,
        ... )

        # Using AzureConfig
        >>> config = AzureConfig(
        ...     endpoint="https://my-resource.openai.azure.com",
        ...     auth_method=AzureAuthMethod.MANAGED_IDENTITY,
        ... )
        >>> provider = AzureOpenAIProvider.from_config(config)
    """

    name = "azure"
    supports_chat = True
    supports_embeddings = True
    default_chat_model = DEFAULT_CHAT_MODELS["azure"]
    default_embedding_model = DEFAULT_EMBEDDING_MODELS["azure"]

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str = "2024-08-01-preview",
        auth_method: AzureAuthMethod = AzureAuthMethod.API_KEY,
        managed_identity_client_id: str | None = None,
        ad_token: str | None = None,
        token_provider: Callable[[], str] | None = None,
        async_token_provider: Callable[[], Awaitable[str]] | None = None,
    ) -> None:
        """Initialize Azure OpenAI provider.

        Args:
            endpoint: Azure OpenAI endpoint URL. If None, uses AZURE_OPENAI_ENDPOINT env var.
            api_key: API key. If None, uses AZURE_OPENAI_API_KEY env var.
            api_version: Azure OpenAI API version.
            auth_method: Authentication method to use.
            managed_identity_client_id: Client ID for user-assigned managed identity.
            ad_token: Direct Azure AD token.
            token_provider: Custom sync token provider callable.
            async_token_provider: Custom async token provider callable.
        """
        self.config = AzureConfig(
            endpoint=endpoint or "",
            api_key=api_key,
            api_version=api_version,
            auth_method=auth_method,
            managed_identity_client_id=managed_identity_client_id,
            ad_token=ad_token,
            token_provider=token_provider,
            async_token_provider=async_token_provider,
        )

    @classmethod
    def from_config(cls, config: AzureConfig) -> AzureOpenAIProvider:
        """Create provider from AzureConfig.

        Args:
            config: Azure configuration object.

        Returns:
            Configured AzureOpenAIProvider instance.
        """
        provider = cls(
            endpoint=config.endpoint,
            api_key=config.api_key,
            api_version=config.api_version,
            auth_method=config.auth_method,
            managed_identity_client_id=config.managed_identity_client_id,
            ad_token=config.ad_token,
            token_provider=config.token_provider,
            async_token_provider=config.async_token_provider,
        )
        provider.config.extra = config.extra
        return provider

    def _get_auth_kwargs(self, for_async: bool = False) -> dict[str, Any]:
        """Get authentication kwargs for Azure clients.

        Args:
            for_async: Whether to get async token provider.

        Returns:
            Dict with appropriate auth parameters.
        """
        kwargs: dict[str, Any] = {}

        if self.config.endpoint:
            kwargs["azure_endpoint"] = self.config.endpoint
        kwargs["api_version"] = self.config.api_version

        if self.config.auth_method == AzureAuthMethod.API_KEY:
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            # If no api_key, let LangChain use env var
        elif self.config.auth_method == AzureAuthMethod.AD_TOKEN:
            kwargs["azure_ad_token"] = self.config.ad_token
        else:
            # Token provider methods
            token_provider = self.config.get_token_provider()
            if token_provider:
                kwargs["azure_ad_token_provider"] = token_provider

            if for_async:
                async_provider = self.config.get_async_token_provider()
                if async_provider:
                    kwargs["azure_ad_async_token_provider"] = async_provider

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
        """Create an Azure OpenAI chat model.

        Args:
            model: Azure deployment name. Defaults to gpt-4o-mini.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum tokens in response.
            top_p: Nucleus sampling parameter.
            stop: Stop sequences.
            streaming: Enable streaming responses.
            **kwargs: Additional AzureChatOpenAI parameters.

        Returns:
            Configured AzureChatOpenAI instance.

        Raises:
            ImportError: If langchain-openai is not installed.
        """
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError as e:
            raise ImportError(
                "Azure provider requires langchain-openai. "
                "Install with: uv add langchain-openai"
            ) from e

        deployment = model or self.default_chat_model

        model_kwargs: dict[str, Any] = {
            "azure_deployment": deployment,
            "temperature": temperature,
            "streaming": streaming,
            **self._get_auth_kwargs(for_async=False),
        }

        if max_tokens is not None:
            model_kwargs["max_tokens"] = max_tokens
        if top_p is not None:
            model_kwargs["top_p"] = top_p
        if stop is not None:
            model_kwargs["stop"] = stop

        model_kwargs.update(self.config.extra)
        model_kwargs.update(kwargs)

        return AzureChatOpenAI(**model_kwargs)

    async def acreate_chat_model(
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
        """Create an Azure OpenAI chat model with async token provider.

        This method sets up the async token provider for better performance
        in async contexts.

        Args:
            model: Azure deployment name.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.
            top_p: Nucleus sampling.
            stop: Stop sequences.
            streaming: Enable streaming.
            **kwargs: Additional parameters.

        Returns:
            Configured AzureChatOpenAI instance.
        """
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError as e:
            raise ImportError(
                "Azure provider requires langchain-openai. "
                "Install with: uv add langchain-openai"
            ) from e

        deployment = model or self.default_chat_model

        model_kwargs: dict[str, Any] = {
            "azure_deployment": deployment,
            "temperature": temperature,
            "streaming": streaming,
            **self._get_auth_kwargs(for_async=True),
        }

        if max_tokens is not None:
            model_kwargs["max_tokens"] = max_tokens
        if top_p is not None:
            model_kwargs["top_p"] = top_p
        if stop is not None:
            model_kwargs["stop"] = stop

        model_kwargs.update(self.config.extra)
        model_kwargs.update(kwargs)

        return AzureChatOpenAI(**model_kwargs)

    def create_embeddings(
        self,
        model: str | None = None,
        *,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> Embeddings:
        """Create Azure OpenAI embeddings.

        Args:
            model: Azure deployment name for embeddings.
            dimensions: Output dimensions (if supported).
            **kwargs: Additional AzureOpenAIEmbeddings parameters.

        Returns:
            Configured AzureOpenAIEmbeddings instance.

        Raises:
            ImportError: If langchain-openai is not installed.
        """
        try:
            from langchain_openai import AzureOpenAIEmbeddings
        except ImportError as e:
            raise ImportError(
                "Azure provider requires langchain-openai. "
                "Install with: uv add langchain-openai"
            ) from e

        deployment = model or self.default_embedding_model

        embedding_kwargs: dict[str, Any] = {
            "azure_deployment": deployment,
            **self._get_auth_kwargs(for_async=False),
        }

        if dimensions is not None:
            embedding_kwargs["dimensions"] = dimensions

        embedding_kwargs.update(self.config.extra)
        embedding_kwargs.update(kwargs)

        return AzureOpenAIEmbeddings(**embedding_kwargs)

    async def acreate_embeddings(
        self,
        model: str | None = None,
        *,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> Embeddings:
        """Create Azure OpenAI embeddings with async token provider.

        Args:
            model: Azure deployment name.
            dimensions: Output dimensions.
            **kwargs: Additional parameters.

        Returns:
            Configured AzureOpenAIEmbeddings instance.
        """
        try:
            from langchain_openai import AzureOpenAIEmbeddings
        except ImportError as e:
            raise ImportError(
                "Azure provider requires langchain-openai. "
                "Install with: uv add langchain-openai"
            ) from e

        deployment = model or self.default_embedding_model

        embedding_kwargs: dict[str, Any] = {
            "azure_deployment": deployment,
            **self._get_auth_kwargs(for_async=True),
        }

        if dimensions is not None:
            embedding_kwargs["dimensions"] = dimensions

        embedding_kwargs.update(self.config.extra)
        embedding_kwargs.update(kwargs)

        return AzureOpenAIEmbeddings(**embedding_kwargs)


# Register alias
class AzureOpenAIProviderAlias(AzureOpenAIProvider):
    """Alias for AzureOpenAIProvider registered as 'azure_openai'."""

    name = "azure_openai"


__all__ = [
    "AzureOpenAIProvider",
    "AzureAuthMethod",
    "AzureConfig",
    "AZURE_COGNITIVE_SERVICES_SCOPE",
]
