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
    result = await embedder.embed(["Hello", "World"])
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from agenticflow.core.messages import (
    EmbeddingMetadata,
    EmbeddingResult,
    MessageMetadata,
    TokenUsage,
)
from agenticflow.models.base import (
    AIMessage,
    BaseChatModel,
    BaseEmbedding,
    convert_messages,
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
            formatted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": schema,
                    },
                }
            )
        elif isinstance(tool, dict):
            formatted.append(tool)
    return formatted


def _parse_response(response: Any) -> AIMessage:
    """Parse Ollama response into AIMessage with metadata."""
    from agenticflow.core.messages import MessageMetadata, TokenUsage

    choice = response.choices[0]
    message = choice.message

    tool_calls = []
    if message.tool_calls:
        for tc in message.tool_calls:
            tool_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": __import__("json").loads(tc.function.arguments)
                    if isinstance(tc.function.arguments, str)
                    else tc.function.arguments,
                }
            )

    metadata = MessageMetadata(
        model=response.model if hasattr(response, "model") else None,
        tokens=TokenUsage(
            prompt_tokens=response.usage.prompt_tokens
            if hasattr(response, "usage") and response.usage
            else 0,
            completion_tokens=response.usage.completion_tokens
            if hasattr(response, "usage") and response.usage
            else 0,
            total_tokens=response.usage.total_tokens
            if hasattr(response, "usage") and response.usage
            else 0,
        )
        if hasattr(response, "usage") and response.usage
        else None,
        finish_reason=choice.finish_reason
        if hasattr(choice, "finish_reason")
        else None,
        response_id=response.id if hasattr(response, "id") else None,
    )

    return AIMessage(
        content=message.content or "",
        tool_calls=tool_calls,
        metadata=metadata,
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

        from agenticflow.config import get_config_value

        host = (
            get_config_value("ollama", "host", self.host, ["OLLAMA_HOST"])
            or "http://localhost:11434"
        )
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
    ) -> OllamaChat:
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
        response = await self._async_client.chat.completions.create(
            **self._build_request(messages)
        )
        return _parse_response(response)

    async def astream(self, messages: list[dict[str, Any]]) -> AsyncIterator[AIMessage]:
        """Stream response asynchronously with metadata.

        Yields:
            AIMessage objects with incremental content and metadata.
        """
        self._ensure_initialized()
        kwargs = self._build_request(messages)
        kwargs["stream"] = True
        kwargs["stream_options"] = {
            "include_usage": True
        }  # Request token usage in final chunk

        start_time = time.time()
        chunk_metadata = {
            "id": None,
            "model": None,
            "finish_reason": None,
            "usage": None,
        }

        async for chunk in await self._async_client.chat.completions.create(**kwargs):
            # Accumulate metadata
            if hasattr(chunk, "id") and chunk.id:
                chunk_metadata["id"] = chunk.id
            if hasattr(chunk, "model") and chunk.model:
                chunk_metadata["model"] = chunk.model
            if chunk.choices and chunk.choices[0].finish_reason:
                chunk_metadata["finish_reason"] = chunk.choices[0].finish_reason
            if hasattr(chunk, "usage") and chunk.usage:
                chunk_metadata["usage"] = chunk.usage
                # Yield final metadata chunk
                metadata = MessageMetadata(
                    id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    model=chunk_metadata["model"],
                    tokens=TokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    ),
                    finish_reason=chunk_metadata["finish_reason"],
                    response_id=chunk_metadata["id"],
                    duration=time.time() - start_time,
                )
                yield AIMessage(content="", metadata=metadata)

            # Yield content chunks
            if chunk.choices and chunk.choices[0].delta.content:
                metadata = MessageMetadata(
                    id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    model=chunk_metadata["model"],
                    tokens=None,
                    finish_reason=chunk_metadata.get("finish_reason"),
                    response_id=chunk_metadata["id"],
                    duration=time.time() - start_time,
                )
                yield AIMessage(
                    content=chunk.choices[0].delta.content, metadata=metadata
                )

    def _build_request(self, messages: list[Any]) -> dict[str, Any]:
        """Build API request, converting messages to dict format."""
        # Convert message objects to dicts
        converted_messages = convert_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": converted_messages,
        }

        model_lower = (self.model or "").lower()
        supports_temperature = not any(
            prefix in model_lower for prefix in ("o1", "o3", "gpt-5")
        )
        if supports_temperature and self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens:
            if not supports_temperature:
                kwargs["max_completion_tokens"] = self.max_tokens
            else:
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

        result = await embedder.embed(["Hello", "World"])
    """

    model: str = "nomic-embed-text"
    host: str = "http://localhost:11434"

    def _init_client(self) -> None:
        """Initialize Ollama client using OpenAI-compatible API."""
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: uv add openai")

        from agenticflow.config import get_config_value

        host = (
            get_config_value("ollama", "host", self.host, ["OLLAMA_HOST"])
            or "http://localhost:11434"
        )
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

    async def embed(self, texts: str | list[str]) -> EmbeddingResult:
        """Embed one or more texts asynchronously with metadata.

        Args:
            texts: Single text or list of texts to embed.

        Returns:
            EmbeddingResult with vectors and metadata.
        """
        self._ensure_initialized()
        import asyncio
        import time

        from agenticflow.core.messages import (
            EmbeddingResult,
        )

        # Normalize input
        texts_list = [texts] if isinstance(texts, str) else texts
        start_time = time.time()

        async def embed_batch(batch: list[str]) -> list[list[float]]:
            response = await self._async_client.embeddings.create(
                model=self.model,
                input=batch,
            )
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [d.embedding for d in sorted_data]

        batches = [
            texts_list[i : i + self.batch_size]
            for i in range(0, len(texts_list), self.batch_size)
        ]
        results = await asyncio.gather(*[embed_batch(b) for b in batches])

        all_embeddings: list[list[float]] = []
        for batch_result in results:
            all_embeddings.extend(batch_result)

        metadata = EmbeddingMetadata(
            model=self.model,
            tokens=None,  # Ollama typically doesn't return token counts for embeddings
            duration=time.time() - start_time,
            dimensions=len(all_embeddings[0]) if all_embeddings else None,
            num_texts=len(texts_list),
        )

        return EmbeddingResult(embeddings=all_embeddings, metadata=metadata)
