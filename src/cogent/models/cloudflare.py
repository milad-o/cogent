"""
Cloudflare Workers AI models for AgenticFlow.

Cloudflare exposes OpenAI-compatible chat and embedding endpoints:
- Chat: https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1/chat/completions
- Embeddings: https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1/embeddings

Authentication: Bearer token via CLOUDFLARE_API_TOKEN.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
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
)


def _parse_response(response: Any) -> AIMessage:
    """Parse Cloudflare response into AIMessage with metadata."""
    from cogent.core.messages import MessageMetadata, TokenUsage

    choice = response.choices[0]
    message = choice.message
    tool_calls = []
    if getattr(message, "tool_calls", None):
        for tc in message.tool_calls:
            tool_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": __import__("json").loads(tc.function.arguments),
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
        content=message.content or "", tool_calls=tool_calls, metadata=metadata
    )


@dataclass
class CloudflareChat(BaseChatModel):
    """Cloudflare Workers AI chat via OpenAI-compatible API."""

    model: str = "@cf/meta/llama-3.3-70b-instruct"
    account_id: str | None = None
    base_url: str = ""

    def _init_client(self) -> None:
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package required. Install with: uv add openai"
            ) from exc

        from cogent.config import get_api_key, get_config_value

        api_key = get_api_key("cloudflare", self.api_key)
        account_id = get_config_value(
            "cloudflare", "account_id", self.account_id, ["CLOUDFLARE_ACCOUNT_ID"]
        )
        if not account_id:
            raise ValueError("Cloudflare account_id is required")

        base_url = (
            self.base_url
            or f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
        )

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        self._client = OpenAI(**kwargs)
        self._async_client = AsyncOpenAI(**kwargs)

    def bind_tools(
        self,
        tools: list[Any],
        *,
        parallel_tool_calls: bool = True,
    ) -> CloudflareChat:
        self._ensure_initialized()
        new_model = CloudflareChat(
            model=self.model,
            account_id=self.account_id,
            api_key=self.api_key,
            base_url=self.base_url,
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
        self._ensure_initialized()
        response = self._client.chat.completions.create(**self._build_request(messages))
        return _parse_response(response)

    async def ainvoke(self, messages: list[dict[str, Any]]) -> AIMessage:
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
        kwargs["stream_options"] = {"include_usage": True}

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

            # Yield content chunks with partial metadata
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

        # Yield final metadata chunk if we have usage or finish_reason
        if chunk_metadata.get("usage") or chunk_metadata.get("finish_reason"):
            metadata = MessageMetadata(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                model=chunk_metadata["model"],
                tokens=TokenUsage(
                    prompt_tokens=chunk_metadata["usage"].prompt_tokens
                    if chunk_metadata.get("usage")
                    else 0,
                    completion_tokens=chunk_metadata["usage"].completion_tokens
                    if chunk_metadata.get("usage")
                    else 0,
                    total_tokens=chunk_metadata["usage"].total_tokens
                    if chunk_metadata.get("usage")
                    else 0,
                )
                if chunk_metadata.get("usage")
                else None,
                finish_reason=chunk_metadata["finish_reason"],
                response_id=chunk_metadata["id"],
                duration=time.time() - start_time,
            )
            yield AIMessage(content="", metadata=metadata)

    def _build_request(
        self, messages: list[dict[str, Any]] | list[Any]
    ) -> dict[str, Any]:
        formatted_messages = convert_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
        }

        model_lower = (self.model or "").lower()
        supports_temperature = not any(
            prefix in model_lower for prefix in ("o1", "o3", "gpt-5")
        )
        if supports_temperature and self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens:
            if not supports_temperature:
                payload["max_completion_tokens"] = self.max_tokens
            else:
                payload["max_tokens"] = self.max_tokens
        if self._tools:
            payload["tools"] = [
                t.to_dict() if hasattr(t, "to_dict") else t for t in self._tools
            ]
            payload["parallel_tool_calls"] = self._parallel_tool_calls

        # Structured output support
        if hasattr(self, "_response_format") and self._response_format:
            payload["response_format"] = self._response_format

        return payload


@dataclass
class CloudflareEmbedding(BaseEmbedding):
    """Cloudflare Workers AI embeddings via OpenAI-compatible API."""

    model: str = "@cf/baai/bge-base-en-v1.5"
    account_id: str | None = None
    base_url: str = ""

    def _init_client(self) -> None:
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package required. Install with: uv add openai"
            ) from exc

        from cogent.config import get_api_key, get_config_value

        api_key = get_api_key("cloudflare", self.api_key)
        account_id = get_config_value(
            "cloudflare", "account_id", self.account_id, ["CLOUDFLARE_ACCOUNT_ID"]
        )
        if not account_id:
            raise ValueError("Cloudflare account_id is required")

        base_url = (
            self.base_url
            or f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
        )

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        self._client = OpenAI(**kwargs)
        self._async_client = AsyncOpenAI(**kwargs)

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

        from cogent.core.messages import (
            EmbeddingResult,
        )

        # Normalize input
        texts_list = [texts] if isinstance(texts, str) else texts
        start_time = time.time()

        async def embed_batch(batch: list[str]) -> list[list[float]]:
            response = await self._async_client.embeddings.create(
                **self._build_request(batch)
            )
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [d.embedding for d in sorted_data]

        batches = [
            texts_list[i : i + self.batch_size]
            for i in range(0, len(texts_list), self.batch_size)
        ]
        results = await asyncio.gather(*[embed_batch(b) for b in batches])
        all_embeddings: list[list[float]] = []
        for res in results:
            all_embeddings.extend(res)

        metadata = EmbeddingMetadata(
            model=self.model,
            tokens=None,  # Cloudflare typically doesn't track token usage
            duration=time.time() - start_time,
            dimensions=len(all_embeddings[0]) if all_embeddings else self.dimension,
            num_texts=len(texts_list),
        )

        return EmbeddingResult(embeddings=all_embeddings, metadata=metadata)

    def _build_request(self, texts: list[str]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        return payload

    @property
    def dimension(self) -> int:
        if self.dimensions:
            return self.dimensions
        return 768
