"""
Cohere models for AgenticFlow.

Supports Cohere chat (Command models) and embeddings.

Usage:
    from agenticflow.models.cohere import CohereChat, CohereEmbedding

    llm = CohereChat(model="command-r-plus")
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])

    embedder = CohereEmbedding(model="embed-english-v3.0")
    vectors = await embedder.aembed(["Hello", "World"])
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from agenticflow.models.base import AIMessage, BaseChatModel, BaseEmbedding, convert_messages, normalize_input, convert_messages


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
            formatted.append({
                "name": tool.name,
                "description": getattr(tool, "description", "") or "",
                "parameter_definitions": _schema_to_parameter_definitions(schema),
            })
            continue
        if isinstance(tool, dict):
            if "parameter_definitions" in tool:
                formatted.append(tool)
                continue
            if "function" in tool:
                func = tool.get("function", {}) or {}
                formatted.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameter_definitions": _schema_to_parameter_definitions(func.get("parameters", {}) or {}),
                })
                continue
            formatted.append(tool)
    return formatted


def _parse_response(response: Any) -> AIMessage:
    """Parse Cohere chat response into AIMessage."""
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
            name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else "")
            args = getattr(tc, "parameters", None)
            if args is None and isinstance(tc, dict):
                args = tc.get("parameters") or tc.get("args") or tc.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append({
                "id": getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else f"call_{idx}"),
                "name": name,
                "args": args or {},
            })

    content = "".join(content_parts) if content_parts else ""
    return AIMessage(content=content, tool_calls=tool_calls)


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
            raise ImportError("cohere package required. Install with: uv add cohere") from exc

        api_key = self.api_key or os.environ.get("COHERE_API_KEY")
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
    ) -> "CohereChat":
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

    async def ainvoke(self, messages: str | list[dict[str, Any]] | list[Any]) -> AIMessage:
        """Invoke asynchronously.
        
        Args:
            messages: Can be a string, list of dicts, or list of message objects.
        """
        self._ensure_initialized()
        kwargs = self._build_request(normalize_input(messages))
        response = await self._async_client.chat(**kwargs)
        return _parse_response(response)

    def _build_request(self, messages: list[dict[str, Any]] | list[Any]) -> dict[str, Any]:
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
            raise ImportError("cohere package required. Install with: uv add cohere") from exc

        api_key = self.api_key or os.environ.get("COHERE_API_KEY")
        client_kwargs = {
            "api_key": api_key,
            "timeout": self.timeout,
        }
        self._client = ClientV2(**client_kwargs)
        self._async_client = AsyncClientV2(**client_kwargs)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts synchronously."""
        self._ensure_initialized()
        response = self._client.embed(model=self.model, texts=texts, input_type="search_document")
        embeddings = getattr(response, "embeddings", None) or getattr(response, "data", None)
        return [list(vec) for vec in embeddings] if embeddings else []

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts asynchronously."""
        self._ensure_initialized()
        response = await self._async_client.embed(model=self.model, texts=texts, input_type="search_document")
        embeddings = getattr(response, "embeddings", None) or getattr(response, "data", None)
        return [list(vec) for vec in embeddings] if embeddings else []

    @property
    def dimension(self) -> int:
        """Return embedding dimension when known."""
        if self.dimensions:
            return self.dimensions
        return 1024
