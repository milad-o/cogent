"""
Cohere models for AgenticFlow.

Supports Cohere chat (Command models) and embeddings.

Usage:
    from agenticflow.models.cohere import CohereChat, CohereEmbedding

    llm = CohereChat(model="command-r-plus")
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])

    embedder = CohereEmbedding(model="embed-english-v3.0")
    vectors = await embedder.embed(["Hello", "World"])
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
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
    normalize_input,
)


def _schema_to_parameter_definitions(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert JSON schema to Cohere parameter definitions."""
    if not schema:
        return {}
    properties = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    params: dict[str, Any] = {}
    for name, spec in properties.items():
        if not isinstance(spec, dict):
            continue
        params[name] = {
            "type": spec.get("type", "string"),
            "description": spec.get("description", ""),
            "required": name in required,
        }
    return params


def _format_tools(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert bound tools into Cohere format."""
    formatted: list[dict[str, Any]] = []
    for tool in tools:
        if hasattr(tool, "name"):
            schema = getattr(tool, "args_schema", {}) or {}
            formatted.append(
                {
                    "name": tool.name,
                    "description": getattr(tool, "description", "") or "",
                    "parameter_definitions": _schema_to_parameter_definitions(schema),
                }
            )
            continue
        if isinstance(tool, dict):
            if "parameter_definitions" in tool:
                formatted.append(tool)
                continue
            if "function" in tool:
                func = tool.get("function", {}) or {}
                formatted.append(
                    {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "parameter_definitions": _schema_to_parameter_definitions(
                            func.get("parameters", {}) or {}
                        ),
                    }
                )
                continue
            formatted.append(tool)
    return formatted


def _parse_response(response: Any) -> AIMessage:
    """Parse Cohere chat response into AIMessage with metadata."""
    from agenticflow.core.messages import MessageMetadata, TokenUsage

    tool_calls: list[dict[str, Any]] = []
    content_parts: list[str] = []

    message = getattr(response, "message", None)
    if message:
        parts = getattr(message, "content", None) or []
        for part in parts:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    content_parts.append(part.get("text", ""))
            elif hasattr(part, "text"):
                content_parts.append(getattr(part, "text", ""))

        calls = getattr(message, "tool_calls", None) or []
        for idx, tc in enumerate(calls):
            name = getattr(tc, "name", None) or (
                tc.get("name") if isinstance(tc, dict) else ""
            )
            args = getattr(tc, "parameters", None)
            if args is None and isinstance(tc, dict):
                args = tc.get("parameters") or tc.get("args") or tc.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(
                {
                    "id": getattr(tc, "id", None)
                    or (tc.get("id") if isinstance(tc, dict) else f"call_{idx}"),
                    "name": name,
                    "args": args or {},
                }
            )

    content = "".join(content_parts) if content_parts else ""

    metadata = MessageMetadata(
        model=response.response_id if hasattr(response, "response_id") else None,
        tokens=TokenUsage(
            prompt_tokens=response.usage.input_tokens
            if hasattr(response, "usage") and hasattr(response.usage, "input_tokens")
            else 0,
            completion_tokens=response.usage.output_tokens
            if hasattr(response, "usage") and hasattr(response.usage, "output_tokens")
            else 0,
            total_tokens=(response.usage.input_tokens + response.usage.output_tokens)
            if hasattr(response, "usage") and hasattr(response.usage, "input_tokens")
            else 0,
        )
        if hasattr(response, "usage")
        else None,
        finish_reason=response.finish_reason
        if hasattr(response, "finish_reason")
        else None,
        response_id=response.response_id if hasattr(response, "response_id") else None,
    )

    return AIMessage(content=content, tool_calls=tool_calls, metadata=metadata)


@dataclass
class CohereChat(BaseChatModel):
    """Cohere Command family chat models."""

    model: str = "command-r-plus"
    _use_responses_api: bool = field(default=True, repr=False)

    def _init_client(self) -> None:
        """Initialize Cohere clients."""
        try:
            from cohere import AsyncClientV2, ClientV2
        except ImportError as exc:
            raise ImportError(
                "cohere package required. Install with: uv add cohere"
            ) from exc

        from agenticflow.config import get_api_key

        api_key = get_api_key("cohere", self.api_key)
        client_kwargs = {
            "api_key": api_key,
            "timeout": self.timeout,
        }
        self._client = ClientV2(**client_kwargs)
        self._async_client = AsyncClientV2(**client_kwargs)

    def bind_tools(
        self,
        tools: list[Any],
        *,
        parallel_tool_calls: bool = True,
    ) -> CohereChat:
        """Bind tools to the model."""
        self._ensure_initialized()

        new_model = CohereChat(
            model=self.model,
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

    def invoke(self, messages: str | list[dict[str, Any]] | list[Any]) -> AIMessage:
        """Invoke synchronously.

        Args:
            messages: Can be a string, list of dicts, or list of message objects.
        """
        self._ensure_initialized()
        kwargs = self._build_request(normalize_input(messages))
        response = self._client.chat(**kwargs)
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
        response = await self._async_client.chat(**kwargs)
        return _parse_response(response)

    async def astream(
        self, messages: str | list[dict[str, Any]] | list[Any]
    ) -> AsyncIterator[AIMessage]:
        """Stream response asynchronously with metadata.

        Args:
            messages: Can be a string, list of dicts, or list of message objects.

        Yields:
            AIMessage objects with incremental content and metadata.
        """
        self._ensure_initialized()
        kwargs = self._build_request(normalize_input(messages))

        start_time = time.time()
        chunk_metadata = {
            "id": None,
            "model": None,
            "finish_reason": None,
            "usage": None,
        }

        stream = self._async_client.chat_stream(**kwargs)

        async for event in stream:
            # Handle different event types
            if event.type == "content-delta":
                if hasattr(event, "delta") and hasattr(event.delta, "message"):
                    if hasattr(event.delta.message, "content") and hasattr(
                        event.delta.message.content, "text"
                    ):
                        text = event.delta.message.content.text
                        if text:
                            metadata = MessageMetadata(
                                id=str(uuid.uuid4()),
                                timestamp=time.time(),
                                model=chunk_metadata.get("model"),
                                tokens=None,
                                finish_reason=chunk_metadata.get("finish_reason"),
                                response_id=chunk_metadata.get("id"),
                                duration=time.time() - start_time,
                            )
                            yield AIMessage(content=text, metadata=metadata)

            elif event.type == "message-end" and hasattr(event, "delta"):
                if hasattr(event.delta, "finish_reason"):
                    chunk_metadata["finish_reason"] = event.delta.finish_reason
                if hasattr(event.delta, "usage") and hasattr(
                    event.delta.usage, "tokens"
                ):
                    usage = event.delta.usage.tokens
                    chunk_metadata["usage"] = usage
                    # Yield final metadata chunk
                    final_metadata = MessageMetadata(
                        id=str(uuid.uuid4()),
                        timestamp=time.time(),
                        model=chunk_metadata.get("model"),
                        tokens=TokenUsage(
                            prompt_tokens=int(usage.input_tokens)
                            if hasattr(usage, "input_tokens")
                            else 0,
                            completion_tokens=int(usage.output_tokens)
                            if hasattr(usage, "output_tokens")
                            else 0,
                            total_tokens=int(
                                usage.input_tokens + usage.output_tokens
                            )
                            if hasattr(usage, "input_tokens")
                            and hasattr(usage, "output_tokens")
                            else 0,
                        ),
                        finish_reason=chunk_metadata.get("finish_reason"),
                        response_id=chunk_metadata.get("id"),
                        duration=time.time() - start_time,
                    )
                    yield AIMessage(content="", metadata=final_metadata)

    def _build_request(
        self, messages: list[dict[str, Any]] | list[Any]
    ) -> dict[str, Any]:
        """Build request for Cohere Chat API."""
        formatted_messages = convert_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if self._tools:
            kwargs["tools"] = _format_tools(self._tools)

        # Structured output support (if configured)
        if hasattr(self, "_response_format") and self._response_format:
            kwargs["response_format"] = self._response_format

        return kwargs


@dataclass
class CohereEmbedding(BaseEmbedding):
    """Cohere embedding model."""

    model: str = "embed-english-v3.0"

    def _init_client(self) -> None:
        """Initialize Cohere clients."""
        try:
            from cohere import AsyncClientV2, ClientV2
        except ImportError as exc:
            raise ImportError(
                "cohere package required. Install with: uv add cohere"
            ) from exc

        from agenticflow.config import get_api_key

        api_key = get_api_key("cohere", self.api_key)
        client_kwargs = {
            "api_key": api_key,
            "timeout": self.timeout,
        }
        self._client = ClientV2(**client_kwargs)
        self._async_client = AsyncClientV2(**client_kwargs)

    async def embed(self, texts: str | list[str]) -> EmbeddingResult:
        """Embed one or more texts asynchronously with metadata.

        Args:
            texts: Single text or list of texts to embed.

        Returns:
            EmbeddingResult with vectors and metadata.
        """
        self._ensure_initialized()
        import time

        from agenticflow.core.messages import (
            EmbeddingResult,
            TokenUsage,
        )

        # Normalize input
        texts_list = [texts] if isinstance(texts, str) else texts
        start_time = time.time()

        response = await self._async_client.embed(
            model=self.model, texts=texts_list, input_type="search_document"
        )
        embeddings = getattr(response, "embeddings", None) or getattr(
            response, "data", None
        )
        vectors = [list(vec) for vec in embeddings] if embeddings else []

        # Cohere response may have meta with billed_units
        tokens = None
        if hasattr(response, "meta") and hasattr(response.meta, "billed_units"):
            if hasattr(response.meta.billed_units, "input_tokens"):
                tokens = TokenUsage(
                    prompt_tokens=response.meta.billed_units.input_tokens,
                    completion_tokens=0,
                    total_tokens=response.meta.billed_units.input_tokens,
                )

        metadata = EmbeddingMetadata(
            model=self.model,
            tokens=tokens,
            duration=time.time() - start_time,
            dimensions=len(vectors[0]) if vectors else self.dimension,
            num_texts=len(texts_list),
        )

        return EmbeddingResult(embeddings=vectors, metadata=metadata)

    async def embed_query(self, query: str) -> EmbeddingResult:
        """Embed a search query with Cohere-specific query input type.

        Cohere uses different embeddings for queries vs documents.
        Override the protocol method to use input_type="search_query".

        Args:
            query: Query text to embed.

        Returns:
            EmbeddingResult with vector and metadata.
        """
        self._ensure_initialized()
        import time

        from agenticflow.core.messages import (
            EmbeddingResult,
            TokenUsage,
        )

        start_time = time.time()
        response = await self._async_client.embed(
            model=self.model,
            texts=[query],
            input_type="search_query",  # Cohere-specific for queries
        )
        embeddings = getattr(response, "embeddings", None) or getattr(
            response, "data", None
        )
        vectors = [list(vec) for vec in embeddings] if embeddings else []

        # Cohere response may have meta with billed_units
        tokens = None
        if hasattr(response, "meta") and hasattr(response.meta, "billed_units"):
            if hasattr(response.meta.billed_units, "input_tokens"):
                tokens = TokenUsage(
                    prompt_tokens=response.meta.billed_units.input_tokens,
                    completion_tokens=0,
                    total_tokens=response.meta.billed_units.input_tokens,
                )

        metadata = EmbeddingMetadata(
            model=self.model,
            tokens=tokens,
            duration=time.time() - start_time,
            dimensions=len(vectors[0]) if vectors else self.dimension,
            num_texts=1,
        )

        return EmbeddingResult(embeddings=vectors, metadata=metadata)

    @property
    def dimension(self) -> int:
        """Return embedding dimension when known."""
        if self.dimensions:
            return self.dimensions
        return 1024
