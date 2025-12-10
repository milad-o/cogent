"""
OpenAI models for AgenticFlow.

Usage:
    from agenticflow.models.openai import OpenAIChat, OpenAIEmbedding
    
    # Chat
    llm = OpenAIChat(model="gpt-4o")
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])
    
    # With tools
    llm = OpenAIChat().bind_tools([my_tool])
    
    # Embeddings
    embedder = OpenAIEmbedding()
    vectors = await embedder.aembed(["Hello", "World"])
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from agenticflow.models.base import AIMessage, BaseChatModel, BaseEmbedding, convert_messages


def _parse_response(response: Any) -> AIMessage:
    """Parse OpenAI response into AIMessage."""
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
            # Our native Tool format
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


@dataclass
class OpenAIChat(BaseChatModel):
    """OpenAI chat model.
    
    High-performance chat model using OpenAI SDK directly.
    Supports GPT-4o, GPT-4, GPT-3.5, and other OpenAI models.
    
    Example:
        from agenticflow.models.openai import OpenAIChat
        
        # Simple usage
        llm = OpenAIChat()  # Uses gpt-4o-mini by default
        response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])
        print(response.content)
        
        # With custom model
        llm = OpenAIChat(model="gpt-4o", temperature=0.7)
        
        # With tools
        llm = OpenAIChat().bind_tools([search_tool, calc_tool])
        response = await llm.ainvoke(messages)
        if response.tool_calls:
            # Handle tool calls
            pass
        
        # Streaming
        async for chunk in llm.astream(messages):
            print(chunk.content, end="")
    """
    
    model: str = "gpt-4o-mini"
    base_url: str = ""
    
    def _init_client(self) -> None:
        """Initialize OpenAI clients."""
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: uv add openai")
        
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        
        kwargs: dict[str, Any] = {
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        
        self._client = OpenAI(**kwargs)
        self._async_client = AsyncOpenAI(**kwargs)
    
    def bind_tools(
        self,
        tools: list[Any],
        *,
        parallel_tool_calls: bool = True,
    ) -> "OpenAIChat":
        """Bind tools to the model."""
        self._ensure_initialized()
        
        new_model = OpenAIChat(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            base_url=self.base_url,
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
            "model": self.model,
            "messages": formatted_messages,
        }
        
        # Only include temperature for models that support it
        # o1, o3, gpt-5 series don't support temperature
        model_lower = self.model.lower()
        supports_temperature = not any(
            prefix in model_lower 
            for prefix in ("o1", "o3", "gpt-5")
        )
        if supports_temperature and self.temperature is not None:
            kwargs["temperature"] = self.temperature
        
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if self._tools:
            kwargs["tools"] = _format_tools(self._tools)
            kwargs["parallel_tool_calls"] = self._parallel_tool_calls
        return kwargs


@dataclass
class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embedding model.
    
    Example:
        from agenticflow.models.openai import OpenAIEmbedding
        
        embedder = OpenAIEmbedding()  # Uses text-embedding-3-small by default
        
        # Single text
        vector = await embedder.aembed_query("Hello world")
        
        # Batch embedding
        vectors = await embedder.aembed(["Hello", "World"])
        
        # With dimension reduction
        embedder = OpenAIEmbedding(dimensions=256)
    """
    
    model: str = "text-embedding-3-small"
    base_url: str | None = None
    
    def _init_client(self) -> None:
        """Initialize OpenAI clients."""
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: uv add openai")
        
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        
        kwargs: dict[str, Any] = {
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        
        self._client = OpenAI(**kwargs)
        self._async_client = AsyncOpenAI(**kwargs)
    
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
            "model": self.model,
            "input": texts,
        }
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        return kwargs
