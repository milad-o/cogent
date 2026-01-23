"""
OpenAI models for AgenticFlow.

Usage:
    from agenticflow.models.openai import OpenAIChat, OpenAIEmbedding

    # Chat Completions API
    llm = OpenAIChat(model="gpt-4o")
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])

    # Responses API (beta) - optimized for tool use
    llm = OpenAIChat(model="gpt-4o", use_responses_api=True)
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])

    # With tools
    llm = OpenAIChat().bind_tools([my_tool])

    # Embeddings
    embedder = OpenAIEmbedding()
    vectors = await embedder.aembed(["Hello", "World"])
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from agenticflow.core.messages import EmbeddingResult
from agenticflow.models.base import (
    AIMessage,
    BaseChatModel,
    BaseEmbedding,
    convert_messages,
    normalize_input,
)


def _parse_response(response: Any) -> AIMessage:
    """Parse OpenAI response into AIMessage with metadata."""
    from agenticflow.core.messages import (
        MessageMetadata,
        TokenUsage,
    )

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

    metadata = MessageMetadata(
        model=response.model,
        tokens=TokenUsage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        ) if response.usage else None,
        finish_reason=choice.finish_reason,
        response_id=response.id,
    )

    return AIMessage(
        content=message.content or "",
        tool_calls=tool_calls,
        metadata=metadata,
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

        # Simple usage (Chat Completions API)
        llm = OpenAIChat()  # Uses gpt-4o-mini by default
        response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])
        print(response.content)

        # Responses API (beta) - optimized for tool use
        llm = OpenAIChat(use_responses_api=True)
        response = await llm.ainvoke(messages)

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
    use_responses_api: bool = False  # Use beta Responses API instead of Chat Completions

    def _init_client(self) -> None:
        """Initialize OpenAI clients."""
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: uv add openai")

        from agenticflow.config import get_api_key

        api_key = get_api_key("openai", self.api_key)

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
    ) -> OpenAIChat:
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
            use_responses_api=self.use_responses_api,
        )
        new_model._tools = tools
        new_model._parallel_tool_calls = parallel_tool_calls
        new_model._client = self._client
        new_model._async_client = self._async_client
        new_model._initialized = True
        return new_model

    def invoke(self, messages: str | list[dict[str, Any]] | list[Any]) -> AIMessage:
        """Invoke synchronously.

        Args:
            messages: Can be a string, list of dicts, or list of message objects.
        """
        self._ensure_initialized()
        kwargs = self._build_request(normalize_input(messages))
        if self.use_responses_api:
            response = self._client.beta.responses.create(**kwargs)
        else:
            response = self._client.chat.completions.create(**kwargs)
        return _parse_response(response)

    async def ainvoke(self, messages: str | list[dict[str, Any]] | list[Any]) -> AIMessage:
        """Invoke asynchronously.

        Args:
            messages: Can be a string, list of dicts, or list of message objects.
        """
        self._ensure_initialized()
        kwargs = self._build_request(normalize_input(messages))
        if self.use_responses_api:
            response = await self._async_client.beta.responses.create(**kwargs)
        else:
            response = await self._async_client.chat.completions.create(**kwargs)
        return _parse_response(response)

    async def astream(self, messages: str | list[dict[str, Any]] | list[Any]) -> AsyncIterator[AIMessage]:
        """Stream response asynchronously with metadata.

        Args:
            messages: Can be a string, list of dicts, or list of message objects.
            
        Yields:
            AIMessage objects with incremental content and metadata. The last chunk
            will have empty content but complete metadata (finish_reason and token usage).
        """
        import time

        from ..core.messages import MessageMetadata, TokenUsage

        self._ensure_initialized()
        kwargs = self._build_request(normalize_input(messages))
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}  # Request usage in final chunk

        start_time = time.time()
        chunk_metadata = {
            "id": None,
            "model": None,
            "finish_reason": None,
            "usage": None,
        }
        has_yielded_final = False

        if self.use_responses_api:
            stream = await self._async_client.beta.responses.create(**kwargs)
        else:
            stream = await self._async_client.chat.completions.create(**kwargs)

        async for chunk in stream:
            # Accumulate metadata from chunks
            if chunk.id:
                chunk_metadata["id"] = chunk.id
            if chunk.model:
                chunk_metadata["model"] = chunk.model

            # Check for finish_reason (comes in second-to-last chunk)
            if chunk.choices and chunk.choices[0].finish_reason:
                chunk_metadata["finish_reason"] = chunk.choices[0].finish_reason

            # Check for usage (comes in final chunk with stream_options)
            if hasattr(chunk, 'usage') and chunk.usage:
                chunk_metadata["usage"] = chunk.usage
                # This is the final chunk with usage - yield it with complete metadata
                metadata = MessageMetadata(
                    id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    model=chunk_metadata["model"],
                    tokens=TokenUsage(
                        prompt_tokens=chunk_metadata["usage"].prompt_tokens,
                        completion_tokens=chunk_metadata["usage"].completion_tokens,
                        total_tokens=chunk_metadata["usage"].total_tokens,
                    ),
                    finish_reason=chunk_metadata["finish_reason"],
                    response_id=chunk_metadata["id"],
                    duration=time.time() - start_time,
                )
                # Yield final metadata-only chunk
                yield AIMessage(content="", metadata=metadata)
                has_yielded_final = True

            # Yield content chunks with partial metadata
            if chunk.choices and chunk.choices[0].delta.content:
                metadata = MessageMetadata(
                    id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    model=chunk_metadata["model"],
                    tokens=None,  # Will be in final chunk
                    finish_reason=chunk_metadata.get("finish_reason"),  # May be None until near end
                    response_id=chunk_metadata["id"],
                    duration=time.time() - start_time,
                )

                yield AIMessage(
                    content=chunk.choices[0].delta.content,
                    metadata=metadata,
                )

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
            if not supports_temperature:
                kwargs["max_completion_tokens"] = self.max_tokens
            else:
                kwargs["max_tokens"] = self.max_tokens
        if self._tools:
            kwargs["tools"] = _format_tools(self._tools)
            kwargs["parallel_tool_calls"] = self._parallel_tool_calls

        # Structured output support
        if hasattr(self, "_response_format") and self._response_format:
            kwargs["response_format"] = self._response_format

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

        from agenticflow.config import get_api_key

        api_key = get_api_key("openai", self.api_key)

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

    def embed(self, texts: list[str]) -> EmbeddingResult:
        """Embed texts synchronously with metadata."""
        self._ensure_initialized()
        import time

        from agenticflow.core.messages import (
            EmbeddingMetadata,
            EmbeddingResult,
            TokenUsage,
        )

        start_time = time.time()
        all_embeddings: list[list[float]] = []
        total_prompt_tokens = 0
        model_name = None

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self._client.embeddings.create(**self._build_request(batch))
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([d.embedding for d in sorted_data])

            # Accumulate metadata
            if response.usage:
                total_prompt_tokens += response.usage.prompt_tokens
            if response.model:
                model_name = response.model

        # Build metadata
        metadata = EmbeddingMetadata(
            model=model_name or self.model,
            tokens=TokenUsage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=0,
                total_tokens=total_prompt_tokens,
            ) if total_prompt_tokens > 0 else None,
            duration=time.time() - start_time,
            dimensions=len(all_embeddings[0]) if all_embeddings else self.dimensions,
            num_texts=len(texts),
        )

        return EmbeddingResult(embeddings=all_embeddings, metadata=metadata)

    async def aembed(self, texts: list[str]) -> EmbeddingResult:
        """Embed texts asynchronously with metadata."""
        self._ensure_initialized()
        import asyncio
        import time

        from agenticflow.core.messages import (
            EmbeddingMetadata,
            EmbeddingResult,
            TokenUsage,
        )

        start_time = time.time()
        total_prompt_tokens = 0
        model_name = None

        async def embed_batch(batch: list[str]) -> tuple[list[list[float]], int, str | None]:
            response = await self._async_client.embeddings.create(**self._build_request(batch))
            sorted_data = sorted(response.data, key=lambda x: x.index)
            embeddings = [d.embedding for d in sorted_data]
            tokens = response.usage.prompt_tokens if response.usage else 0
            model = response.model if hasattr(response, 'model') else None
            return embeddings, tokens, model

        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        results = await asyncio.gather(*[embed_batch(b) for b in batches])

        all_embeddings: list[list[float]] = []
        for batch_embeddings, tokens, model in results:
            all_embeddings.extend(batch_embeddings)
            total_prompt_tokens += tokens
            if model:
                model_name = model

        # Build metadata
        metadata = EmbeddingMetadata(
            model=model_name or self.model,
            tokens=TokenUsage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=0,
                total_tokens=total_prompt_tokens,
            ) if total_prompt_tokens > 0 else None,
            duration=time.time() - start_time,
            dimensions=len(all_embeddings[0]) if all_embeddings else self.dimensions,
            num_texts=len(texts),
        )

        return EmbeddingResult(embeddings=all_embeddings, metadata=metadata)

    def _build_request(self, texts: list[str]) -> dict[str, Any]:
        """Build API request."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        return kwargs
