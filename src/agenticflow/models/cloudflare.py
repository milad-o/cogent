"""
Cloudflare Workers AI models for AgenticFlow.

Cloudflare exposes OpenAI-compatible chat and embedding endpoints:
- Chat: https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1/chat/completions
- Embeddings: https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1/embeddings

Authentication: Bearer token via CLOUDFLARE_API_TOKEN.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, AsyncIterator

from agenticflow.models.base import AIMessage, BaseChatModel, BaseEmbedding, convert_messages


def _parse_response(response: Any) -> AIMessage:
    choice = response.choices[0]
    message = choice.message
    tool_calls = []
    if getattr(message, "tool_calls", None):
        for tc in message.tool_calls:
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "args": __import__("json").loads(tc.function.arguments),
            })
    return AIMessage(content=message.content or "", tool_calls=tool_calls)


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
            raise ImportError("openai package required. Install with: uv add openai") from exc

        api_key = self.api_key or os.environ.get("CLOUDFLARE_API_TOKEN")
        account_id = self.account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        if not account_id:
            raise ValueError("Cloudflare account_id is required")

        base_url = self.base_url or f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"

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
    ) -> "CloudflareChat":
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
        response = await self._async_client.chat.completions.create(**self._build_request(messages))
        return _parse_response(response)

    async def astream(self, messages: list[dict[str, Any]]) -> AsyncIterator[AIMessage]:
        self._ensure_initialized()
        kwargs = self._build_request(messages)
        kwargs["stream"] = True
        async for chunk in await self._async_client.chat.completions.create(**kwargs):
            if chunk.choices and chunk.choices[0].delta.content:
                yield AIMessage(content=chunk.choices[0].delta.content)

    def _build_request(self, messages: list[dict[str, Any]] | list[Any]) -> dict[str, Any]:
        formatted_messages = convert_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if self._tools:
            payload["tools"] = [t.to_dict() if hasattr(t, "to_dict") else t for t in self._tools]
            payload["parallel_tool_calls"] = self._parallel_tool_calls
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
            raise ImportError("openai package required. Install with: uv add openai") from exc

        api_key = self.api_key or os.environ.get("CLOUDFLARE_API_TOKEN")
        account_id = self.account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        if not account_id:
            raise ValueError("Cloudflare account_id is required")

        base_url = self.base_url or f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        self._client = OpenAI(**kwargs)
        self._async_client = AsyncOpenAI(**kwargs)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_initialized()
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self._client.embeddings.create(**self._build_request(batch))
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([d.embedding for d in sorted_data])
        return all_embeddings

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_initialized()
        import asyncio

        async def embed_batch(batch: list[str]) -> list[list[float]]:
            response = await self._async_client.embeddings.create(**self._build_request(batch))
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [d.embedding for d in sorted_data]

        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        results = await asyncio.gather(*[embed_batch(b) for b in batches])
        all_embeddings: list[list[float]] = []
        for res in results:
            all_embeddings.extend(res)
        return all_embeddings

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
