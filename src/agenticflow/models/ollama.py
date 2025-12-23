"""
Ollama models for AgenticFlow.

Ollama runs LLMs locally. Supports Llama, Mistral, Qwen, and other models.

Usage:
    from agenticflow.models.ollama import OllamaChat, OllamaEmbedding
    
    # Chat
    llm = OllamaChat(model="llama3.2")
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])
    
    # Embeddings
    embedder = OllamaEmbedding(model="nomic-embed-text")
    vectors = await embedder.aembed(["Hello", "World"])
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from agenticflow.models.base import AIMessage, BaseChatModel, BaseEmbedding, convert_messages, normalize_input, convert_messages


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


def _parse_response(response: Any) -> AIMessage:
    """Parse Ollama response into AIMessage."""
    choice = response.choices[0]
    message = choice.message
    
    tool_calls = []
    if message.tool_calls:
        for tc in message.tool_calls:
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "args": __import__("json").loads(tc.function.arguments)
                    if isinstance(tc.function.arguments, str) else tc.function.arguments,
            })
    
    return AIMessage(
        content=message.content or "",
        tool_calls=tool_calls,
    )


@dataclass
class OllamaChat(BaseChatModel):
    """Ollama chat model.
    
    Runs LLMs locally using Ollama. Supports Llama, Mistral, Qwen, and many others.
    
    Example:
        from agenticflow.models.ollama import OllamaChat
        
        # Default model
        llm = OllamaChat()  # Uses llama3.2 by default
        
        # Custom model
        llm = OllamaChat(model="mistral")
        
        # Custom host
        llm = OllamaChat(
            model="codellama",
            host="http://192.168.1.100:11434",
        )
        
        # With tools (not all models support this)
        llm = OllamaChat(model="llama3.2").bind_tools([my_tool])
        
        response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])
    """
    
    model: str = "llama3.2"
    host: str = "http://localhost:11434"
    
    def _init_client(self) -> None:
        """Initialize Ollama client using OpenAI-compatible API."""
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: uv add openai")
        
        host = self.host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        base_url = f"{host.rstrip('/')}/v1"
        
        self._client = OpenAI(
            base_url=base_url,
            api_key="ollama",  # Ollama doesn't require real API key
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        self._async_client = AsyncOpenAI(
            base_url=base_url,
            api_key="ollama",
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
    
    def bind_tools(
        self,
        tools: list[Any],
        *,
        parallel_tool_calls: bool = True,
    ) -> "OllamaChat":
        """Bind tools to the model."""
        self._ensure_initialized()
        
        new_model = OllamaChat(
            model=self.model,
            host=self.host,
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
    
    def _build_request(self, messages: list[Any]) -> dict[str, Any]:
        """Build API request, converting messages to dict format."""
        # Convert message objects to dicts
        converted_messages = convert_messages(messages)
        
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": converted_messages,
            "temperature": self.temperature,
        }
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if self._tools:
            kwargs["tools"] = _format_tools(self._tools)
        return kwargs


@dataclass
class OllamaEmbedding(BaseEmbedding):
    """Ollama embedding model.
    
    Generate embeddings locally using Ollama.
    
    Example:
        from agenticflow.models.ollama import OllamaEmbedding
        
        embedder = OllamaEmbedding()  # Uses nomic-embed-text by default
        
        # Custom model
        embedder = OllamaEmbedding(model="mxbai-embed-large")
        
        vectors = await embedder.aembed(["Hello", "World"])
    """
    
    model: str = "nomic-embed-text"
    host: str = "http://localhost:11434"
    
    def _init_client(self) -> None:
        """Initialize Ollama client using OpenAI-compatible API."""
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: uv add openai")
        
        host = self.host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        base_url = f"{host.rstrip('/')}/v1"
        
        self._client = OpenAI(
            base_url=base_url,
            api_key="ollama",
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        self._async_client = AsyncOpenAI(
            base_url=base_url,
            api_key="ollama",
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
