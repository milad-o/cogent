"""
Mistral AI models for Cogent.

Supports all Mistral models via the official API.

Usage:
    from cogent.models.mistral import MistralChat, MistralEmbedding

    # Chat
    llm = MistralChat(model="mistral-large-latest")
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])

    # With tools
    llm = MistralChat(model="mistral-large-latest")
    bound = llm.bind_tools([search_tool])
    response = await bound.ainvoke(messages)

    # Embeddings
    embedder = MistralEmbedding(model="mistral-embed")
    result = await embedder.embed(["Hello", "World"])

Available chat models:
    - mistral-large-latest: State-of-the-art, best for complex tasks
    - mistral-small-latest: Budget-friendly, good for most tasks
    - codestral-latest: Optimized for coding tasks
    - ministral-8b-latest: Efficient smaller model
    - open-mistral-nemo: Open source 12B model

Available embedding models:
    - mistral-embed: 1024-dimensional embeddings
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from cogent.core.messages import (
    EmbeddingMetadata,
    EmbeddingResult,
    MessageMetadata,
    TokenUsage,
)
from cogent.models.base import (
    AIMessage,
    BaseChatModel,
    BaseEmbedding,
    convert_messages,
    normalize_input,
)


def _format_tools(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert tools to Mistral API format."""
    formatted = []
    for tool in tools:
        if hasattr(tool, "to_dict"):
            formatted.append(tool.to_dict())
        elif hasattr(tool, "to_openai"):
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
    """Parse Mistral response into AIMessage with metadata."""
    from cogent.core.messages import MessageMetadata, TokenUsage

    choice = response.choices[0]
    message = choice.message

    tool_calls = []
    if message.tool_calls:
        for tc in message.tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                args = __import__("json").loads(args)
            tool_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": args,
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
class MistralChat(BaseChatModel):
    """Mistral AI chat model.

    Uses the Mistral API (OpenAI-compatible) for high-performance inference.

    Example:
        from cogent.models.mistral import MistralChat

        # State-of-the-art model
        llm = MistralChat(model="mistral-large-latest")

        # Budget-friendly
        llm = MistralChat(model="mistral-small-latest")

        # For coding
        llm = MistralChat(model="codestral-latest")

        response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])

    Available models:
        - mistral-large-latest: Best for complex reasoning
        - mistral-small-latest: Cost-effective, good quality
        - codestral-latest: Optimized for code
        - ministral-8b-latest: Efficient 8B model
        - open-mistral-nemo: Open 12B model
    """

    model: str = "mistral-small-latest"
    base_url: str = "https://api.mistral.ai/v1"

    _tool_choice: str | dict[str, Any] | None = field(default=None, repr=False)
    _parallel_tool_calls: bool = field(default=True, repr=False)

    def _init_client(self) -> None:
        """Initialize Mistral client (OpenAI-compatible)."""
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: uv add openai"
            ) from None

        from cogent.config import get_api_key

        api_key = get_api_key("mistral", self.api_key)
        if not api_key:
            raise ValueError(
                "Mistral API key required. Set MISTRAL_API_KEY environment variable "
                "or pass api_key parameter."
            )

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

    async def ainvoke(
        self,
        messages: str | list[dict[str, Any]] | list[Any],
        **kwargs: Any,
    ) -> AIMessage:
        """Async invoke the model.

        Args:
            messages: Can be a string, list of dicts, or list of message objects.
            **kwargs: Additional arguments (temperature, max_tokens, etc.)

        Returns:
            AIMessage with response content and any tool calls.
        """
        await self._ensure_initialized_async()

        # Convert messages to dict format
        converted_messages = convert_messages(normalize_input(messages))

        params: dict[str, Any] = {
            "model": self.model,
            "messages": converted_messages,
        }

        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        # Add tools if bound
        if self._tools:
            params["tools"] = _format_tools(self._tools)
            params["parallel_tool_calls"] = self._parallel_tool_calls
            if self._tool_choice:
                params["tool_choice"] = self._tool_choice

        # Apply overrides
        params.update(kwargs)

        response = await self._async_client.chat.completions.create(**params)
        return _parse_response(response)

    def invoke(
        self,
        messages: str | list[dict[str, Any]] | list[Any],
        **kwargs: Any,
    ) -> AIMessage:
        """Sync invoke the model.

        Args:
            messages: Can be a string, list of dicts, or list of message objects.
            **kwargs: Additional arguments.

        Returns:
            AIMessage with response content and any tool calls.
        """
        self._ensure_initialized()

        # Convert messages to dict format
        converted_messages = convert_messages(normalize_input(messages))

        params: dict[str, Any] = {
            "model": self.model,
            "messages": converted_messages,
        }

        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        if self._tools:
            params["tools"] = _format_tools(self._tools)
            params["parallel_tool_calls"] = self._parallel_tool_calls
            if self._tool_choice:
                params["tool_choice"] = self._tool_choice

        if self._tools:
            params["tools"] = _format_tools(self._tools)
            params["parallel_tool_calls"] = self._parallel_tool_calls
            if self._tool_choice:
                params["tool_choice"] = self._tool_choice

        params.update(kwargs)

        response = self._client.chat.completions.create(**params)
        return _parse_response(response)

    async def astream(
        self,
        messages: str | list[dict[str, Any]] | list[Any],
        **kwargs: Any,
    ) -> AsyncIterator[AIMessage]:
        """Stream responses from the model with metadata.

        Args:
            messages: Can be a string, list of dicts, or list of message objects.
            **kwargs: Additional arguments.

        Yields:
            AIMessage chunks with incremental content and metadata.
        """
        self._ensure_initialized()

        # Convert messages to dict format
        converted_messages = convert_messages(normalize_input(messages))

        params: dict[str, Any] = {
            "model": self.model,
            "messages": converted_messages,
            "stream": True,
        }

        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        params.update(kwargs)

        start_time = time.time()
        chunk_metadata = {
            "id": None,
            "model": None,
            "finish_reason": None,
            "usage": None,
        }

        stream = await self._async_client.chat.completions.create(**params)

        async for chunk in stream:
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

    def bind_tools(
        self,
        tools: list[Any],
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool = True,
        **kwargs: Any,
    ) -> MistralChat:
        """Bind tools to the model.

        Args:
            tools: List of tools (BaseTool instances or dicts).
            tool_choice: How to choose tools ("auto", "none", "any", or specific).
            **kwargs: Additional arguments (ignored for compatibility).

        Returns:
            New model instance with tools bound.
        """
        new_model = MistralChat(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        new_model._tools = tools
        new_model._tool_choice = tool_choice
        new_model._parallel_tool_calls = parallel_tool_calls
        # Only copy initialization state if original was actually initialized
        if self._initialized and (
            self._client is not None or self._async_client is not None
        ):
            new_model._client = self._client
            new_model._async_client = self._async_client
            new_model._initialized = True
        return new_model


@dataclass
class MistralEmbedding(BaseEmbedding):
    """Mistral AI embedding model.

    Uses the Mistral API (OpenAI-compatible) for text embeddings.

    Example:
        from cogent.models.mistral import MistralEmbedding

        embedder = MistralEmbedding(model="mistral-embed")
        result = await embedder.embed(["Hello", "World"])
        print(f"Generated {len(result.embeddings[0])} dimensional embeddings")

    Available models:
        - mistral-embed: 1024-dimensional embeddings
    """

    model: str = "mistral-embed"
    base_url: str = "https://api.mistral.ai/v1"

    def _init_client(self) -> None:
        """Initialize Mistral client (OpenAI-compatible)."""
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: uv add openai"
            ) from None

        from cogent.config import get_api_key

        api_key = get_api_key("mistral", self.api_key)
        if not api_key:
            raise ValueError(
                "Mistral API key required. Set MISTRAL_API_KEY environment variable "
                "or pass api_key parameter."
            )

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

    async def embed(self, texts: str | list[str]) -> EmbeddingResult:
        """Embed texts asynchronously with metadata.

        Args:
            texts: A single string or list of strings to embed.

        Returns:
            EmbeddingResult with embeddings and metadata.
        """
        self._ensure_initialized()
        import asyncio
        import time

        from cogent.core.messages import (
            EmbeddingResult,
            TokenUsage,
        )

        # Normalize input
        texts_list = [texts] if isinstance(texts, str) else texts

        start_time = time.time()
        total_prompt_tokens = 0
        model_name = None

        async def embed_batch(
            batch: list[str],
        ) -> tuple[list[list[float]], int, str | None]:
            response = await self._async_client.embeddings.create(
                **self._build_request(batch)
            )
            sorted_data = sorted(response.data, key=lambda x: x.index)
            embeddings = [d.embedding for d in sorted_data]
            tokens = response.usage.prompt_tokens if response.usage else 0
            model = response.model if hasattr(response, "model") else None
            return embeddings, tokens, model

        batches = [
            texts_list[i : i + self.batch_size]
            for i in range(0, len(texts_list), self.batch_size)
        ]
        results = await asyncio.gather(*[embed_batch(b) for b in batches])

        all_embeddings: list[list[float]] = []
        for batch_embeddings, tokens, model in results:
            all_embeddings.extend(batch_embeddings)
            total_prompt_tokens += tokens
            if model:
                model_name = model

        metadata = EmbeddingMetadata(
            model=model_name or self.model,
            tokens=TokenUsage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=0,
                total_tokens=total_prompt_tokens,
            )
            if total_prompt_tokens > 0
            else None,
            duration=time.time() - start_time,
            dimensions=len(all_embeddings[0]) if all_embeddings else None,
            num_texts=len(texts_list),
        )

        return EmbeddingResult(embeddings=all_embeddings, metadata=metadata)

    def _build_request(self, texts: list[str]) -> dict[str, Any]:
        """Build API request."""
        return {
            "model": self.model,
            "input": texts,
        }
