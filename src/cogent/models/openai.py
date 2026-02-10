"""
OpenAI models for Cogent.

Supports Chat Completions API and Reasoning Config for o-series models.

Usage:
    from cogent.models.openai import OpenAIChat, OpenAIEmbedding

    # Chat Completions API
    llm = OpenAIChat(model="gpt-4o")
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])

    # With Reasoning (o-series models)
    llm = OpenAIChat(
        model="o3-mini",
        reasoning_effort="high",  # low, medium, or high
    )
    response = await llm.ainvoke("Solve this complex problem...")
    # Reasoning tokens tracked in response.metadata.tokens.reasoning_tokens

    # Responses API (beta) - optimized for tool use
    llm = OpenAIChat(model="gpt-4o", use_responses_api=True)
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])

    # With tools
    llm = OpenAIChat().bind_tools([my_tool])

    # Embeddings
    embedder = OpenAIEmbedding()
    result = await embedder.embed(["Hello", "World"])

Reasoning-enabled models:
    - o3: Most capable reasoning model
    - o3-mini: Fast reasoning (supports low/medium/high)
    - o1: Advanced reasoning
    - o1-mini: Fast reasoning
    - o1-preview: Preview version
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from cogent.core.messages import EmbeddingResult
from cogent.models.base import (
    AIMessage,
    BaseChatModel,
    BaseEmbedding,
    convert_messages,
    normalize_input,
)

# Models that support reasoning config (o-series)
REASONING_MODELS = {
    "o3",
    "o3-mini",
    "o3-mini-2025-01-31",
    "o1",
    "o1-2024-12-17",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o1-preview",
    "o1-preview-2024-09-12",
}

# Models that support reasoning_effort parameter
REASONING_EFFORT_MODELS = {
    "o3-mini",
    "o3-mini-2025-01-31",
}


def _parse_response(response: Any) -> AIMessage:
    """Parse OpenAI response into AIMessage with metadata including reasoning tokens."""
    from cogent.core.messages import (
        MessageMetadata,
        TokenUsage,
    )

    choice = response.choices[0]
    message = choice.message

    tool_calls = []
    if message.tool_calls:
        for tc in message.tool_calls:
            tool_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": __import__("json").loads(tc.function.arguments),
                }
            )

    # Extract reasoning tokens if available (o-series models)
    reasoning_tokens = None
    if response.usage:
        if hasattr(response.usage, "completion_tokens_details"):
            details = response.usage.completion_tokens_details
            if hasattr(details, "reasoning_tokens") and details.reasoning_tokens:
                reasoning_tokens = details.reasoning_tokens

    metadata = MessageMetadata(
        model=response.model,
        tokens=TokenUsage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
            reasoning_tokens=reasoning_tokens,
        )
        if response.usage
        else None,
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


@dataclass
class OpenAIChat(BaseChatModel):
    """OpenAI chat model with Reasoning support.

    High-performance chat model using OpenAI SDK directly.
    Supports GPT-4o, GPT-4, o-series (reasoning), and other OpenAI models.

    Example:
        from cogent.models.openai import OpenAIChat

        # Simple usage (Chat Completions API)
        llm = OpenAIChat()  # Uses gpt-4o-mini by default
        response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])
        print(response.content)

        # With Reasoning (o-series models)
        llm = OpenAIChat(
            model="o3-mini",
            reasoning_effort="high",  # low, medium, or high
        )
        response = await llm.ainvoke("Solve this step by step...")

        # Access reasoning token usage
        if response.metadata.tokens:
            print(f"Reasoning tokens: {response.metadata.tokens.reasoning_tokens}")

        # Responses API (beta) - optimized for tool use
        llm = OpenAIChat(use_responses_api=True)
        response = await llm.ainvoke(messages)

        # With tools
        llm = OpenAIChat().bind_tools([search_tool, calc_tool])
        response = await llm.ainvoke(messages)

        # Streaming
        async for chunk in llm.astream(messages):
            print(chunk.content, end="")

    Attributes:
        model: Model name (default: gpt-4o-mini).
        reasoning_effort: Effort level for o-series models.
            - "low": Faster, less thorough reasoning
            - "medium": Balanced (default for o3-mini)
            - "high": Most thorough reasoning
        use_responses_api: Use beta Responses API for optimized tool use.
    """

    model: str = "gpt-4o-mini"
    base_url: str = ""
    use_responses_api: bool = False
    reasoning_effort: Literal["low", "medium", "high"] | None = field(default=None)

    def _init_client(self) -> None:
        """Initialize OpenAI clients."""
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: uv add openai"
            ) from None

        from cogent.config import get_api_key

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
            reasoning_effort=self.reasoning_effort,
        )
        new_model._tools = tools
        new_model._parallel_tool_calls = parallel_tool_calls
        new_model._client = self._client
        new_model._async_client = self._async_client
        new_model._initialized = True
        return new_model

    def with_reasoning(
        self,
        effort: Literal["low", "medium", "high"] = "medium",
    ) -> OpenAIChat:
        """Enable reasoning with specified effort level.

        Reasoning allows o-series models to think through complex problems
        before providing a response.

        Args:
            effort: Reasoning effort level.
                - "low": Faster, less thorough
                - "medium": Balanced (default)
                - "high": Most thorough reasoning

        Returns:
            New OpenAIChat instance with reasoning enabled.

        Example:
            llm = OpenAIChat(model="o3-mini").with_reasoning("high")
            response = await llm.ainvoke("What is 127 * 893?")
            print(response.metadata.tokens.reasoning_tokens)

        Note:
            - Only supported on o-series models (o3-mini, o1, etc.)
            - Reasoning tokens are tracked in response metadata
        """
        self._ensure_initialized()

        # Validate model supports reasoning
        if self.model not in REASONING_MODELS:
            raise ValueError(
                f"Model {self.model} does not support reasoning. "
                f"Use o3, o3-mini, o1, o1-mini, or o1-preview."
            )

        new_model = OpenAIChat(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            use_responses_api=self.use_responses_api,
            reasoning_effort=effort,
        )
        new_model._tools = getattr(self, "_tools", [])
        new_model._parallel_tool_calls = getattr(self, "_parallel_tool_calls", True)
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

    async def ainvoke(
        self, messages: str | list[dict[str, Any]] | list[Any]
    ) -> AIMessage:
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

    async def astream(
        self, messages: str | list[dict[str, Any]] | list[Any]
    ) -> AsyncIterator[AIMessage]:
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
        kwargs["stream_options"] = {
            "include_usage": True
        }  # Request usage in final chunk

        start_time = time.time()
        chunk_metadata = {
            "id": None,
            "model": None,
            "finish_reason": None,
            "usage": None,
        }

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
            if hasattr(chunk, "usage") and chunk.usage:
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

            # Yield content chunks with partial metadata
            if chunk.choices and chunk.choices[0].delta.content:
                metadata = MessageMetadata(
                    id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    model=chunk_metadata["model"],
                    tokens=None,  # Will be in final chunk
                    finish_reason=chunk_metadata.get(
                        "finish_reason"
                    ),  # May be None until near end
                    response_id=chunk_metadata["id"],
                    duration=time.time() - start_time,
                )

                yield AIMessage(
                    content=chunk.choices[0].delta.content,
                    metadata=metadata,
                )

    def _build_request(
        self, messages: list[dict[str, Any]] | list[Any]
    ) -> dict[str, Any]:
        """Build API request with reasoning support.

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

        # Check if model is a reasoning model (o-series)
        model_lower = self.model.lower()
        is_reasoning_model = any(
            prefix in model_lower for prefix in ("o1", "o3", "gpt-5")
        )

        # Only include temperature for models that support it
        # o1, o3, gpt-5 series don't support temperature
        if not is_reasoning_model and self.temperature is not None:
            kwargs["temperature"] = self.temperature

        # Max tokens handling
        if self.max_tokens:
            if is_reasoning_model:
                kwargs["max_completion_tokens"] = self.max_tokens
            else:
                kwargs["max_tokens"] = self.max_tokens

        # Reasoning effort for o3-mini and supported models
        if self.reasoning_effort and self.model in REASONING_EFFORT_MODELS:
            kwargs["reasoning_effort"] = self.reasoning_effort

        # Tools
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
        from cogent.models.openai import OpenAIEmbedding

        embedder = OpenAIEmbedding()  # Uses text-embedding-3-small by default

        # Single text
        result = await embedder.embed("Hello world")

        # Batch embedding
        result = await embedder.embed(["Hello", "World"])

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
            raise ImportError(
                "openai package required. Install with: uv add openai"
            ) from None

        from cogent.config import get_api_key

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

    async def embed(self, texts: str | list[str]) -> EmbeddingResult:
        """Embed one or more texts asynchronously with metadata."""
        self._ensure_initialized()
        import asyncio
        import time

        from cogent.core.messages import (
            EmbeddingMetadata,
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

        # Build metadata
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
            dimensions=len(all_embeddings[0]) if all_embeddings else self.dimensions,
            num_texts=len(texts_list),
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
