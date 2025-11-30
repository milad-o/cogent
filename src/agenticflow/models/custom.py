"""
Custom OpenAI-compatible models for AgenticFlow.

Supports any OpenAI-compatible API endpoint like:
- vLLM
- LiteLLM
- LocalAI
- Together AI
- Anyscale
- Custom deployments

Usage:
    from agenticflow.models.custom import CustomChat, CustomEmbedding
    
    # vLLM endpoint
    llm = CustomChat(
        base_url="http://localhost:8000/v1",
        model="meta-llama/Llama-3.2-3B-Instruct",
    )
    
    # Together AI
    llm = CustomChat(
        base_url="https://api.together.xyz/v1",
        api_key="your-key",
        model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from agenticflow.models.base import AIMessage, BaseChatModel, BaseEmbedding


def _tools_to_openai(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert tools to OpenAI format."""
    openai_tools = []
    for tool in tools:
        if hasattr(tool, "to_openai"):
            openai_tools.append(tool.to_openai())
        elif hasattr(tool, "name") and hasattr(tool, "description"):
            schema = getattr(tool, "args_schema", {}) or {}
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": schema,
                },
            })
        elif isinstance(tool, dict):
            openai_tools.append(tool)
    return openai_tools


def _parse_response(response: Any) -> AIMessage:
    """Parse OpenAI-compatible response into AIMessage."""
    choice = response.choices[0]
    message = choice.message
    
    tool_calls = []
    if message.tool_calls:
        for tc in message.tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                args = __import__("json").loads(args)
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "args": args,
            })
    
    return AIMessage(
        content=message.content or "",
        tool_calls=tool_calls,
    )


@dataclass
class CustomChat(BaseChatModel):
    """Custom OpenAI-compatible chat model.
    
    Use with any API that implements the OpenAI chat completions format.
    
    Example:
        from agenticflow.models.custom import CustomChat
        
        # vLLM locally
        llm = CustomChat(
            base_url="http://localhost:8000/v1",
            model="meta-llama/Llama-3.2-3B-Instruct",
        )
        
        # Together AI
        llm = CustomChat(
            base_url="https://api.together.xyz/v1",
            api_key=os.environ["TOGETHER_API_KEY"],
            model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
        )
        
        # LiteLLM proxy
        llm = CustomChat(
            base_url="http://localhost:4000/v1",
            model="gpt-4o",  # Route to configured backend
        )
        
        # Anyscale
        llm = CustomChat(
            base_url="https://api.endpoints.anyscale.com/v1",
            api_key="your-key",
            model="meta-llama/Llama-3-8b-chat-hf",
        )
        
        response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])
    """
    
    model: str = "gpt-3.5-turbo"  # Default, should be overridden
    base_url: str = "http://localhost:8000/v1"
    
    def _init_client(self) -> None:
        """Initialize OpenAI-compatible client."""
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: uv add openai")
        
        # Support various env var patterns
        api_key = self.api_key
        if not api_key:
            for env_var in ["CUSTOM_API_KEY", "OPENAI_API_KEY", "API_KEY"]:
                api_key = os.environ.get(env_var)
                if api_key:
                    break
        
        # Some endpoints don't need API key
        if not api_key:
            api_key = "not-needed"
        
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        self._async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
    
    def bind_tools(
        self,
        tools: list[Any],
        *,
        parallel_tool_calls: bool = True,
    ) -> "CustomChat":
        """Bind tools to the model."""
        self._ensure_initialized()
        
        new_model = CustomChat(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
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
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if self._tools:
            kwargs["tools"] = _tools_to_openai(self._tools)
            kwargs["parallel_tool_calls"] = self._parallel_tool_calls
        return kwargs


@dataclass
class CustomEmbedding(BaseEmbedding):
    """Custom OpenAI-compatible embedding model.
    
    Use with any API that implements the OpenAI embeddings format.
    
    Example:
        from agenticflow.models.custom import CustomEmbedding
        
        # vLLM embedding
        embedder = CustomEmbedding(
            base_url="http://localhost:8000/v1",
            model="BAAI/bge-small-en-v1.5",
        )
        
        # Together AI
        embedder = CustomEmbedding(
            base_url="https://api.together.xyz/v1",
            api_key="your-key",
            model="togethercomputer/m2-bert-80M-8k-retrieval",
        )
        
        vectors = await embedder.aembed(["Hello", "World"])
    """
    
    model: str = "text-embedding-3-small"
    base_url: str = "http://localhost:8000/v1"
    
    def _init_client(self) -> None:
        """Initialize OpenAI-compatible client."""
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: uv add openai")
        
        api_key = self.api_key
        if not api_key:
            for env_var in ["CUSTOM_API_KEY", "OPENAI_API_KEY", "API_KEY"]:
                api_key = os.environ.get(env_var)
                if api_key:
                    break
        
        if not api_key:
            api_key = "not-needed"
        
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        self._async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts synchronously."""
        self._ensure_initialized()
        
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self._client.embeddings.create(
                model=self.model,
                input=batch,
            )
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([d.embedding for d in sorted_data])
        return all_embeddings
    
    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts asynchronously."""
        self._ensure_initialized()
        import asyncio
        
        async def embed_batch(batch: list[str]) -> list[list[float]]:
            response = await self._async_client.embeddings.create(
                model=self.model,
                input=batch,
            )
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [d.embedding for d in sorted_data]
        
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        results = await asyncio.gather(*[embed_batch(b) for b in batches])
        
        all_embeddings: list[list[float]] = []
        for batch_result in results:
            all_embeddings.extend(batch_result)
        return all_embeddings
