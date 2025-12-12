"""
Azure OpenAI and Azure AI Foundry models for AgenticFlow.

Azure OpenAI:
- API Key authentication
- DefaultAzureCredential (for local development and production)
- ManagedIdentityCredential (for Azure-hosted services)

Azure AI Foundry:
- Unified inference endpoint for GitHub Models and custom deployments
- Uses azure-ai-inference SDK

Usage:
    from agenticflow.models.azure import AzureChat, AzureEmbedding, AzureAIFoundryChat
    
    # Azure OpenAI with API Key
    llm = AzureChat(
        azure_endpoint="https://your-resource.openai.azure.com",
        api_key="your-key",
        deployment="gpt-4o",
    )
    
    # Azure OpenAI with DefaultAzureCredential
    llm = AzureChat(
        azure_endpoint="https://your-resource.openai.azure.com",
        deployment="gpt-4o",
        use_azure_ad=True,
    )
    
    # GitHub Models (free tier via Azure AI Foundry)
    llm = AzureAIFoundryChat.from_github(
        model="meta/Meta-Llama-3.1-8B-Instruct",
        token=os.getenv("GITHUB_TOKEN"),
    )
    
    # Custom Azure AI Foundry endpoint
    llm = AzureAIFoundryChat(
        endpoint="https://your-foundry.azure.com/inference",
        model="your-model",
        api_key="your-key",
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from agenticflow.models.base import AIMessage, BaseChatModel, BaseEmbedding, convert_messages


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
    
    def _build_request(self, messages: list[dict[str, Any]] | list[Any]) -> dict[str, Any]:
        """Build API request.
        
        Args:
            messages: List of messages - can be dicts or BaseMessage objects.
            
        Returns:
            Dict of API request parameters.
        """
        # Convert message objects to dicts if needed
        formatted_messages = convert_messages(messages)
        
        kwargs: dict[str, Any] = {
            "model": self.deployment,
            "messages": formatted_messages,
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


# ==============================================================================
# Azure AI Foundry (Microsoft Foundry Inference SDK)
# ==============================================================================


def _parse_foundry_response(response: Any) -> AIMessage:
    """Parse Azure AI Foundry response into AIMessage."""
    choice = response.choices[0]
    message = choice.message
    
    tool_calls = []
    if hasattr(message, 'tool_calls') and message.tool_calls:
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


def _format_foundry_messages(messages: list[dict[str, Any]] | list[Any]) -> list[Any]:
    """Format messages for Azure AI Foundry SDK."""
    try:
        from azure.ai.inference.models import (
            SystemMessage,
            UserMessage,
            AssistantMessage,
            ToolMessage,
            ChatCompletionsToolCall,
            FunctionCall,
        )
    except ImportError:
        raise ImportError(
            "azure-ai-inference required for Azure AI Foundry. "
            "Install with: uv add azure-ai-inference"
        )
    
    formatted_messages = convert_messages(messages)
    foundry_messages = []
    
    for msg in formatted_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            foundry_messages.append(SystemMessage(content))
        elif role == "user":
            foundry_messages.append(UserMessage(content))
        elif role == "assistant":
            # Handle tool calls if present
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                # Convert tool_calls to SDK format
                sdk_tool_calls = []
                for tc in tool_calls:
                    func_info = tc.get("function", {})
                    func_name = func_info.get("name", tc.get("name", ""))
                    func_args = func_info.get("arguments", "")
                    if not isinstance(func_args, str):
                        import json
                        func_args = json.dumps(func_args)
                    sdk_tool_calls.append(
                        ChatCompletionsToolCall(
                            id=tc.get("id", ""),
                            function=FunctionCall(name=func_name, arguments=func_args),
                        )
                    )
                foundry_messages.append(AssistantMessage(content=content or "", tool_calls=sdk_tool_calls))
            else:
                foundry_messages.append(AssistantMessage(content))
        elif role == "tool":
            # Tool response
            tool_call_id = msg.get("tool_call_id", "")
            foundry_messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))
    
    return foundry_messages


@dataclass
class AzureAIFoundryChat(BaseChatModel):
    """Azure AI Foundry chat model.
    
    Unified inference client for Azure AI Foundry, supporting:
    - GitHub Models (free tier with rate limits)
    - Custom Azure AI Foundry deployments
    - Bring-your-own-key (BYOK) scenarios
    
    Uses the azure-ai-inference SDK (Microsoft Foundry Inference SDK).
    
    Example:
        from agenticflow.models.azure import AzureAIFoundryChat
        import os
        
        # GitHub Models (free tier)
        llm = AzureAIFoundryChat.from_github(
            model="meta/Meta-Llama-3.1-8B-Instruct",
            token=os.getenv("GITHUB_TOKEN"),
        )
        
        # Custom Azure AI Foundry endpoint
        llm = AzureAIFoundryChat(
            endpoint="https://your-foundry-endpoint.azure.com/inference",
            model="your-model",
            api_key="your-api-key",
        )
        
        # Use with streaming
        async for chunk in llm.astream([{"role": "user", "content": "Hello!"}]):
            print(chunk.content, end="")
        
        # With tools
        llm = llm.bind_tools([search_tool])
        response = await llm.ainvoke(messages)
    """
    
    endpoint: str = ""
    model: str = ""
    
    _foundry_client: Any = field(default=None, init=False, repr=False)
    
    @classmethod
    def from_github(
        cls,
        model: str,
        token: str | None = None,
        temperature: float | None = 1.0,
        max_tokens: int | None = 1000,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> "AzureAIFoundryChat":
        """Create a GitHub Models client.
        
        Args:
            model: Model name (e.g., "meta/Meta-Llama-3.1-8B-Instruct")
            token: GitHub token (or set GITHUB_TOKEN env var)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            
        Returns:
            AzureAIFoundryChat configured for GitHub Models.
            
        Example:
            >>> llm = AzureAIFoundryChat.from_github(
            ...     model="meta/Meta-Llama-3.1-8B-Instruct",
            ...     token=os.getenv("GITHUB_TOKEN")
            ... )
            >>> response = await llm.ainvoke([{"role": "user", "content": "Hi!"}])
        """
        token = token or os.environ.get("GITHUB_TOKEN")
        if not token:
            raise ValueError(
                "GitHub token required. Provide token parameter or set GITHUB_TOKEN environment variable"
            )
        
        return cls(
            endpoint="https://models.github.ai/inference",
            model=model,
            api_key=token,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
    
    def _init_client(self) -> None:
        """Initialize Azure AI Foundry client."""
        try:
            from azure.ai.inference import ChatCompletionsClient
            from azure.core.credentials import AzureKeyCredential
        except ImportError:
            raise ImportError(
                "azure-ai-inference required for Azure AI Foundry. "
                "Install with: uv add azure-ai-inference"
            )
        
        if not self.api_key:
            raise ValueError("api_key required for Azure AI Foundry")
        
        self._foundry_client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key),
        )
    
    def bind_tools(
        self,
        tools: list[Any],
        *,
        parallel_tool_calls: bool = True,
    ) -> "AzureAIFoundryChat":
        """Bind tools to the model."""
        self._ensure_initialized()
        
        new_model = AzureAIFoundryChat(
            endpoint=self.endpoint,
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        new_model._tools = tools
        new_model._parallel_tool_calls = parallel_tool_calls
        new_model._foundry_client = self._foundry_client
        new_model._initialized = True
        return new_model
    
    def invoke(self, messages: list[dict[str, Any]]) -> AIMessage:
        """Invoke synchronously."""
        self._ensure_initialized()
        response = self._foundry_client.complete(**self._build_request(messages))
        return _parse_foundry_response(response)
    
    async def ainvoke(self, messages: list[dict[str, Any]]) -> AIMessage:
        """Invoke asynchronously."""
        self._ensure_initialized()
        # Note: azure-ai-inference SDK doesn't have async support yet
        # Fallback to sync call in thread pool
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke, messages)
    
    async def astream(self, messages: list[dict[str, Any]]) -> AsyncIterator[AIMessage]:
        """Stream response asynchronously."""
        self._ensure_initialized()
        
        kwargs = self._build_request(messages)
        
        # Enable streaming and usage tracking
        kwargs["stream"] = True
        if "model_extras" not in kwargs:
            kwargs["model_extras"] = {}
        kwargs["model_extras"]["stream_options"] = {"include_usage": True}
        
        # Run streaming in thread pool since SDK is sync
        import asyncio
        loop = asyncio.get_event_loop()
        
        # Get iterator in executor
        def get_stream():
            return self._foundry_client.complete(**kwargs)
        
        stream = await loop.run_in_executor(None, get_stream)
        
        # Yield chunks
        for update in stream:
            if update.choices and update.choices[0].delta:
                content = update.choices[0].delta.content
                if content:
                    yield AIMessage(content=content)
    
    def _build_request(self, messages: list[dict[str, Any]] | list[Any]) -> dict[str, Any]:
        """Build API request for Azure AI Foundry.
        
        Args:
            messages: List of messages (dicts or BaseMessage objects).
            
        Returns:
            Dict of API request parameters.
        """
        foundry_messages = _format_foundry_messages(messages)
        
        kwargs: dict[str, Any] = {
            "messages": foundry_messages,
            "model": self.model,
        }
        
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if self._tools:
            kwargs["tools"] = _format_tools(self._tools)
        
        return kwargs
