"""
Azure OpenAI models for AgenticFlow.

Supports multiple authentication methods:
- API Key authentication
- DefaultAzureCredential (for local development and production)
- ManagedIdentityCredential (for Azure-hosted services)

Usage:
    from agenticflow.models.azure import AzureChat, AzureEmbedding
    
    # API Key auth
    llm = AzureChat(
        azure_endpoint="https://your-resource.openai.azure.com",
        api_key="your-key",
        deployment="gpt-4o",
    )
    
    # DefaultAzureCredential (recommended for Azure)
    llm = AzureChat(
        azure_endpoint="https://your-resource.openai.azure.com",
        deployment="gpt-4o",
        use_azure_ad=True,
    )
    
    # ManagedIdentityCredential (for Azure VMs/Functions/etc)
    llm = AzureChat(
        azure_endpoint="https://your-resource.openai.azure.com",
        deployment="gpt-4o",
        use_managed_identity=True,
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from agenticflow.models.base import AIMessage, BaseChatModel, BaseEmbedding


def _parse_response(response: Any) -> AIMessage:
    """Parse Azure OpenAI response into AIMessage."""
    choice = response.choices[0]
    message = choice.message
    
    tool_calls = []
    if message.tool_calls:
        for tc in message.tool_calls:
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "args": __import__("json").loads(tc.function.arguments),
            })
    
    return AIMessage(
        content=message.content or "",
        tool_calls=tool_calls,
    )


def _format_tools(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert tools to API format."""
    formatted = []
    for tool in tools:
        if hasattr(tool, "to_dict"):
            formatted.append(tool.to_dict())
        elif hasattr(tool, "to_openai"):  # backward compat
            formatted.append(tool.to_openai())
        elif hasattr(tool, "name") and hasattr(tool, "description"):
            schema = getattr(tool, "args_schema", {}) or {}
            formatted.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": schema,
                },
            })
        elif isinstance(tool, dict):
            formatted.append(tool)
    return formatted


def _get_token_provider(
    use_azure_ad: bool = False,
    use_managed_identity: bool = False,
    managed_identity_client_id: str | None = None,
) -> Any | None:
    """Get Azure AD token provider."""
    if not (use_azure_ad or use_managed_identity):
        return None
    
    try:
        from azure.identity import DefaultAzureCredential, ManagedIdentityCredential, get_bearer_token_provider
    except ImportError:
        raise ImportError(
            "azure-identity required for Azure AD authentication. "
            "Install with: uv add azure-identity"
        )
    
    if use_managed_identity:
        if managed_identity_client_id:
            credential = ManagedIdentityCredential(client_id=managed_identity_client_id)
        else:
            credential = ManagedIdentityCredential()
    else:
        credential = DefaultAzureCredential()
    
    return get_bearer_token_provider(
        credential,
        "https://cognitiveservices.azure.com/.default",
    )


@dataclass
class AzureChat(BaseChatModel):
    """Azure OpenAI chat model.
    
    Supports API Key, DefaultAzureCredential, and ManagedIdentityCredential.
    
    Example:
        from agenticflow.models.azure import AzureChat
        
        # API Key auth (simple)
        llm = AzureChat(
            azure_endpoint="https://your-resource.openai.azure.com",
            api_key="your-api-key",
            deployment="gpt-4o",
        )
        
        # DefaultAzureCredential (recommended for Azure)
        llm = AzureChat(
            azure_endpoint="https://your-resource.openai.azure.com",
            deployment="gpt-4o",
            use_azure_ad=True,
        )
        
        # ManagedIdentityCredential (for Azure VMs/Functions)
        llm = AzureChat(
            azure_endpoint="https://your-resource.openai.azure.com",
            deployment="gpt-4o",
            use_managed_identity=True,
            managed_identity_client_id="optional-client-id",
        )
        
        # Use environment variables
        # AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
        llm = AzureChat()
        
        response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])
    """
    
    model: str = ""  # Azure uses deployment name, not model
    azure_endpoint: str | None = None
    deployment: str | None = None
    api_version: str = "2024-08-01-preview"
    use_azure_ad: bool = False
    use_managed_identity: bool = False
    managed_identity_client_id: str | None = None
    
    def _init_client(self) -> None:
        """Initialize Azure OpenAI clients."""
        try:
            from openai import AsyncAzureOpenAI, AzureOpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: uv add openai")
        
        endpoint = self.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError("azure_endpoint or AZURE_OPENAI_ENDPOINT environment variable required")
        
        self.deployment = self.deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        if not self.deployment:
            raise ValueError("deployment or AZURE_OPENAI_DEPLOYMENT environment variable required")
        
        kwargs: dict[str, Any] = {
            "azure_endpoint": endpoint,
            "api_version": self.api_version,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        
        # Determine authentication method
        token_provider = _get_token_provider(
            self.use_azure_ad,
            self.use_managed_identity,
            self.managed_identity_client_id,
        )
        
        if token_provider:
            kwargs["azure_ad_token_provider"] = token_provider
        else:
            api_key = self.api_key or os.environ.get("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key or AZURE_OPENAI_API_KEY required when not using Azure AD. "
                    "Or set use_azure_ad=True or use_managed_identity=True"
                )
            kwargs["api_key"] = api_key
        
        self._client = AzureOpenAI(**kwargs)
        self._async_client = AsyncAzureOpenAI(**kwargs)
    
    def bind_tools(
        self,
        tools: list[Any],
        *,
        parallel_tool_calls: bool = True,
    ) -> "AzureChat":
        """Bind tools to the model."""
        self._ensure_initialized()
        
        new_model = AzureChat(
            azure_endpoint=self.azure_endpoint,
            deployment=self.deployment,
            api_version=self.api_version,
            api_key=self.api_key,
            use_azure_ad=self.use_azure_ad,
            use_managed_identity=self.use_managed_identity,
            managed_identity_client_id=self.managed_identity_client_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        new_model._tools = tools
        new_model._parallel_tool_calls = parallel_tool_calls
        new_model._client = self._client
        new_model._async_client = self._async_client
        new_model._initialized = True
        return new_model
    
    def invoke(self, messages: list[dict[str, Any]]) -> AIMessage:
        """Invoke synchronously."""
        self._ensure_initialized()
        response = self._client.chat.completions.create(**self._build_request(messages))
        return _parse_response(response)
    
    async def ainvoke(self, messages: list[dict[str, Any]]) -> AIMessage:
        """Invoke asynchronously."""
        self._ensure_initialized()
        response = await self._async_client.chat.completions.create(**self._build_request(messages))
        return _parse_response(response)
    
    async def astream(self, messages: list[dict[str, Any]]) -> AsyncIterator[AIMessage]:
        """Stream response asynchronously."""
        self._ensure_initialized()
        kwargs = self._build_request(messages)
        kwargs["stream"] = True
        
        async for chunk in await self._async_client.chat.completions.create(**kwargs):
            if chunk.choices and chunk.choices[0].delta.content:
                yield AIMessage(content=chunk.choices[0].delta.content)
    
    def _build_request(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Build API request."""
        kwargs: dict[str, Any] = {
            "model": self.deployment,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if self._tools:
            kwargs["tools"] = _format_tools(self._tools)
            kwargs["parallel_tool_calls"] = self._parallel_tool_calls
        return kwargs


@dataclass
class AzureEmbedding(BaseEmbedding):
    """Azure OpenAI embedding model.
    
    Supports API Key and Azure AD authentication.
    
    Example:
        from agenticflow.models.azure import AzureEmbedding
        
        # API Key auth
        embedder = AzureEmbedding(
            azure_endpoint="https://your-resource.openai.azure.com",
            api_key="your-api-key",
            deployment="text-embedding-ada-002",
        )
        
        # Azure AD auth
        embedder = AzureEmbedding(
            azure_endpoint="https://your-resource.openai.azure.com",
            deployment="text-embedding-3-small",
            use_azure_ad=True,
        )
        
        vectors = await embedder.aembed(["Hello", "World"])
    """
    
    model: str = ""  # Azure uses deployment name
    azure_endpoint: str | None = None
    deployment: str | None = None
    api_version: str = "2024-08-01-preview"
    use_azure_ad: bool = False
    use_managed_identity: bool = False
    managed_identity_client_id: str | None = None
    
    def _init_client(self) -> None:
        """Initialize Azure OpenAI clients."""
        try:
            from openai import AsyncAzureOpenAI, AzureOpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: uv add openai")
        
        endpoint = self.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError("azure_endpoint or AZURE_OPENAI_ENDPOINT environment variable required")
        
        self.deployment = self.deployment or os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        if not self.deployment:
            raise ValueError("deployment or AZURE_OPENAI_EMBEDDING_DEPLOYMENT environment variable required")
        
        kwargs: dict[str, Any] = {
            "azure_endpoint": endpoint,
            "api_version": self.api_version,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        
        token_provider = _get_token_provider(
            self.use_azure_ad,
            self.use_managed_identity,
            self.managed_identity_client_id,
        )
        
        if token_provider:
            kwargs["azure_ad_token_provider"] = token_provider
        else:
            api_key = self.api_key or os.environ.get("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key or AZURE_OPENAI_API_KEY required when not using Azure AD"
                )
            kwargs["api_key"] = api_key
        
        self._client = AzureOpenAI(**kwargs)
        self._async_client = AsyncAzureOpenAI(**kwargs)
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts synchronously."""
        self._ensure_initialized()
        
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self._client.embeddings.create(**self._build_request(batch))
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([d.embedding for d in sorted_data])
        return all_embeddings
    
    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts asynchronously."""
        self._ensure_initialized()
        import asyncio
        
        async def embed_batch(batch: list[str]) -> list[list[float]]:
            response = await self._async_client.embeddings.create(**self._build_request(batch))
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [d.embedding for d in sorted_data]
        
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        results = await asyncio.gather(*[embed_batch(b) for b in batches])
        
        all_embeddings: list[list[float]] = []
        for batch_result in results:
            all_embeddings.extend(batch_result)
        return all_embeddings
    
    def _build_request(self, texts: list[str]) -> dict[str, Any]:
        """Build API request."""
        kwargs: dict[str, Any] = {
            "model": self.deployment,
            "input": texts,
        }
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        return kwargs
