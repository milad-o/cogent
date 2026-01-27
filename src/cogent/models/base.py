"""
Base classes for AgenticFlow models.

All chat and embedding models inherit from these base classes.
"""

from __future__ import annotations

import contextlib
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

# Import AIMessage from core messages - single source of truth
from cogent.core.messages import AIMessage, EmbeddingResult


@runtime_checkable
class MessageLike(Protocol):
    """Protocol for message-like objects.

    Any object with role and content attributes can be treated as a message.
    This includes dict messages and message objects (SystemMessage, HumanMessage, etc.).
    """

    @property
    def role(self) -> str:
        """Message role (system, user, assistant, tool)."""
        ...

    @property
    def content(self) -> str | list[dict[str, object]] | None:
        """Message content (text or multimodal parts)."""
        ...


MessageInput = str | dict[str, object] | MessageLike
"""Accepted message input types: string, dict, or message-like object."""


def normalize_input(messages: str | list[MessageInput]) -> list[MessageInput]:
    """Normalize various input types to a list of messages.

    Args:
        messages: Can be a string or list of messages.

    Returns:
        List of messages (dicts or message-like objects).
    """
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return messages


def convert_messages(messages: list[MessageInput]) -> list[dict[str, object]]:
    """Convert messages to dict format for API calls.

    Handles both dict messages and message objects (SystemMessage, HumanMessage, etc.).
    This is a shared utility used by all model providers.
    """

    def _is_multimodal_content(value: object) -> bool:
        """Return True if `content` is already in provider multimodal format.

        We intentionally preserve OpenAI-style content parts, e.g.:
        - [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {...}}]

        This enables vision-capable chat models to receive images without
        AgenticFlow flattening them into a single string.
        """
        if isinstance(value, list) and value:
            return all(isinstance(item, dict) and "type" in item for item in value)
        # Some providers may represent a single part as a dict.
        return bool(isinstance(value, dict) and "type" in value)

    def _normalize_content(
        value: object,
    ) -> str | list[dict[str, object]] | dict[str, object]:
        """Coerce arbitrary content to a provider-safe representation.

        Default behavior is to convert to string for broad provider compatibility.
        If content is already in a recognized multimodal format, preserve it.
        """
        if _is_multimodal_content(value):
            return value
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            parts: list[str] = []
            for item in value:
                if isinstance(item, dict):
                    # Prefer human-readable fields if present
                    parts.append(
                        item.get("text", "")
                        or item.get("content", "")
                        or json.dumps(item, default=str)
                    )
                else:
                    parts.append(str(item))
            return " ".join(p for p in parts if p)
        if isinstance(value, dict):
            return json.dumps(value, default=str)
        try:
            return str(value)
        except Exception:
            return ""

    def _normalize_message_dict(raw: dict[str, object]) -> dict[str, object]:
        msg = dict(raw)  # shallow copy
        if "content" in msg:
            msg["content"] = _normalize_content(msg.get("content"))
        return msg

    result: list[dict[str, object]] = []
    for msg in messages:
        # Already a dict
        if isinstance(msg, dict):
            result.append(_normalize_message_dict(msg))
            continue

        # Message object with to_dict method
        if hasattr(msg, "to_dict"):
            result.append(_normalize_message_dict(msg.to_dict()))
            continue

        # Message object with to_openai method (backward compat)
        if hasattr(msg, "to_openai"):
            result.append(_normalize_message_dict(msg.to_openai()))
            continue

        # Message object with role and content attributes
        if hasattr(msg, "role") and hasattr(msg, "content"):
            msg_dict: dict[str, object] = {
                "role": getattr(msg, "role", "user"),
                "content": _normalize_content(getattr(msg, "content", "")),
            }
            # Handle tool calls on assistant messages
            if hasattr(msg, "tool_calls") and getattr(msg, "tool_calls", None):
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.get("id", f"call_{i}")
                        if isinstance(tc, dict)
                        else getattr(tc, "id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", "")
                            if isinstance(tc, dict)
                            else getattr(tc, "name", ""),
                            "arguments": json.dumps(
                                tc.get("args", {})
                                if isinstance(tc, dict)
                                else getattr(tc, "args", {}),
                                default=str,
                            ),
                        },
                    }
                    for i, tc in enumerate(msg.tool_calls)
                ]
            # Handle tool result messages
            if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            if hasattr(msg, "name") and getattr(msg, "name", None):
                msg_dict["name"] = msg.name
            result.append(msg_dict)
            continue

        # Fallback: try to convert to string
        result.append({"role": "user", "content": _normalize_content(msg)})

    # Provider-facing sanitization:
    # Some APIs (OpenAI/Azure) validate that every `role="tool"` message must be
    # a response to a *preceding* assistant message that contains `tool_calls`,
    # and that tool_call_id values match one of those tool_calls ids.
    #
    # To be defensive across all internal execution paths, we:
    # - Ensure assistant tool_calls all have stable, non-empty string ids
    # - Ensure tool messages have a tool_call_id that matches a prior tool_call
    # - If a tool message cannot be paired, we drop it to avoid a hard 400
    def _sanitize_tool_messages(
        formatted: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        # Tracks ids for the *most recent* assistant tool_calls message.
        # OpenAI/Azure require tool messages to follow that assistant message
        # (possibly with other tool messages in between, but no other role).
        last_tool_call_ids: list[str] = []
        consumed_last_ids: set[str] = set()
        in_tool_response_zone: bool = False  # True after assistant with tool_calls

        out: list[dict[str, object]] = []
        for idx, m in enumerate(formatted):
            role = m.get("role")

            if role == "assistant" and m.get("tool_calls"):
                # Normalize ids on the assistant tool_calls message.
                last_tool_call_ids = []
                consumed_last_ids = set()
                tool_calls = m.get("tool_calls")
                if isinstance(tool_calls, list):
                    for i, tc in enumerate(tool_calls):
                        if not isinstance(tc, dict):
                            continue
                        tc_id = tc.get("id")
                        if not tc_id:
                            tc_id = f"call_{idx}_{i}"
                        tc_id_str = str(tc_id)
                        tc["id"] = tc_id_str
                        last_tool_call_ids.append(tc_id_str)
                out.append(m)
                in_tool_response_zone = True  # Tool messages can now follow
                continue

            if role == "tool":
                # Tool messages are only valid in the tool response zone
                # (after an assistant with tool_calls, possibly after other tool messages).
                if not in_tool_response_zone or not last_tool_call_ids:
                    continue  # Drop orphan tool message

                tool_call_id = m.get("tool_call_id")
                if tool_call_id:
                    tool_call_id = str(tool_call_id)
                    m["tool_call_id"] = tool_call_id

                # Ensure the tool_call_id matches one of the preceding tool_calls ids.
                if tool_call_id and tool_call_id in last_tool_call_ids:
                    out.append(m)
                    consumed_last_ids.add(tool_call_id)
                    continue

                # Try to infer in-order from the preceding tool_calls.
                inferred: str | None = None
                for candidate in last_tool_call_ids:
                    if candidate not in consumed_last_ids:
                        inferred = candidate
                        break
                if inferred is not None:
                    m["tool_call_id"] = inferred
                    consumed_last_ids.add(inferred)
                    out.append(m)
                    continue

                # Can't pair to preceding assistant tool_calls -> drop.
                continue

            # Any non-tool, non-assistant-with-tool_calls message ends the zone.
            in_tool_response_zone = False
            last_tool_call_ids = []
            consumed_last_ids = set()
            out.append(m)

        return out

    return _sanitize_tool_messages(result)


class ModelProvider(str, Enum):
    """Supported model providers."""

    OPENAI = "openai"
    COHERE = "cohere"
    CLOUDFLARE = "cloudflare"
    AZURE = "azure"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    OLLAMA = "ollama"
    CUSTOM = "custom"


@dataclass
class BaseChatModel(ABC):
    """Abstract base class for all chat models.

    All chat model implementations must inherit from this class
    and implement the required methods.

    Features:
    - Lazy initialization (no API key needed at construction)
    - Sync and async invocation
    - Tool binding with parallel execution
    - Streaming support

    Example:
        class MyChatModel(BaseChatModel):
            model: str = "my-model"

            def _init_client(self) -> None:
                self._client = MyClient(api_key=self.api_key)

            def invoke(self, messages):
                self._ensure_initialized()
                return self._client.chat(messages)
    """

    model: str = ""
    temperature: float = 0.0
    max_tokens: int | None = None
    api_key: str | None = None
    timeout: float = 60.0
    max_retries: int = 2

    # Tool binding state
    _tools: list[Callable[..., object]] = field(default_factory=list, repr=False)
    _parallel_tool_calls: bool = field(default=True, repr=False)

    # Client state (lazy initialized)
    _client: object = field(default=None, repr=False)
    _async_client: object = field(default=None, repr=False)
    _initialized: bool = field(default=False, repr=False)

    def _ensure_initialized(self) -> None:
        """Lazily initialize clients on first use."""
        if not self._initialized:
            with contextlib.suppress(Exception):
                __import__("agenticflow.config")
            self._init_client()
            self._initialized = True

    @abstractmethod
    def _init_client(self) -> None:
        """Initialize the API client. Must be implemented by subclasses."""
        ...

    @abstractmethod
    def invoke(self, messages: str | list[MessageInput]) -> AIMessage:
        """Invoke the model synchronously.

        Args:
            messages: Can be:
                - A string (converted to user message)
                - List of message dicts with 'role' and 'content'
                - List of message objects (SystemMessage, HumanMessage, etc.)
                - Mixed list of dicts and objects

        Returns:
            AIMessage with response content and optional tool calls.
        """
        ...

    @abstractmethod
    async def ainvoke(self, messages: str | list[MessageInput]) -> AIMessage:
        """Invoke the model asynchronously.

        Args:
            messages: Can be:
                - A string (converted to user message)
                - List of message dicts with 'role' and 'content'
                - List of message objects (SystemMessage, HumanMessage, etc.)
                - Mixed list of dicts and objects

        Returns:
            AIMessage with response content and optional tool calls.
        """
        ...

    @abstractmethod
    def bind_tools(
        self,
        tools: list[Callable[..., object]],
        *,
        parallel_tool_calls: bool = True,
    ) -> BaseChatModel:
        """Bind tools to the model.

        Args:
            tools: List of tools to bind.
            parallel_tool_calls: Allow parallel tool execution.

        Returns:
            New model instance with tools bound.
        """
        ...

    async def astream(
        self,
        messages: str | list[MessageInput],
    ) -> AsyncIterator[AIMessage]:
        """Stream response asynchronously.

        Default implementation yields single response.
        Override for true streaming support.

        Args:
            messages: Can be:
                - A string (converted to user message)
                - List of message dicts with 'role' and 'content'
                - List of message objects (SystemMessage, HumanMessage, etc.)
                - Mixed list of dicts and objects

        Yields:
            AIMessage chunks with partial content.
        """
        response = await self.ainvoke(messages)
        yield response


@dataclass
class BaseEmbedding(ABC):
    """Abstract base class for all embedding models.

    All embedding implementations must inherit from this class
    and implement the required methods.

    Features:
    - Lazy initialization
    - Single and batch embedding
    - Sync and async support
    - Configurable dimensions
    """

    model: str = ""
    dimensions: int | None = None
    api_key: str | None = None
    timeout: float = 60.0
    max_retries: int = 2
    batch_size: int = 100

    # Client state (lazy initialized)
    _client: object = field(default=None, repr=False)
    _async_client: object = field(default=None, repr=False)
    _initialized: bool = field(default=False, repr=False)

    def _ensure_initialized(self) -> None:
        """Lazily initialize clients on first use."""
        if not self._initialized:
            self._init_client()
            self._initialized = True

    @abstractmethod
    def _init_client(self) -> None:
        """Initialize the API client. Must be implemented by subclasses."""
        ...

    @abstractmethod
    async def embed(self, texts: str | list[str]) -> EmbeddingResult:
        """Embed one or more texts asynchronously with metadata.

        Primary method for embedding. Accepts single text or batch.
        Returns full metadata including token usage, duration, and model information.

        Args:
            texts: Single text or list of texts to embed.

        Returns:
            EmbeddingResult with vectors and metadata.

        Example:
            # Single text
            result = await embedder.embed("Hello world")
            vector = result.embeddings[0]

            # Multiple texts
            result = await embedder.embed(["Hello", "World"])
            vectors = result.embeddings
            print(f"Tokens used: {result.metadata.tokens}")
        """
        ...

    # VectorStore protocol methods (semantic distinction for specialized use)
    async def embed_documents(self, texts: str | list[str]) -> EmbeddingResult:
        """Embed documents for VectorStore.

        Protocol method with semantic meaning for document embedding.
        Some models may optimize differently for documents vs queries.

        Args:
            texts: Single document or list of documents to embed.

        Returns:
            EmbeddingResult with vectors and metadata.

        Example:
            result = await embedder.embed_documents(["Doc 1", "Doc 2"])
            vectors = result.embeddings
        """
        return await self.embed(texts)

    async def embed_query(self, query: str) -> EmbeddingResult:
        """Embed a search query for VectorStore.

        Protocol method with semantic meaning for query embedding.
        Some models (e.g., Cohere) embed queries differently than documents.

        Args:
            query: Query text to embed.

        Returns:
            EmbeddingResult with vector and metadata.

        Example:
            result = await embedder.embed_query("search query")
            vector = result.embeddings[0]
        """
        return await self.embed(query)


# Common default embedding models per provider
DEFAULT_EMBEDDING_MODELS: dict[ModelProvider, str] = {
    ModelProvider.OPENAI: "text-embedding-3-small",
    ModelProvider.COHERE: "embed-english-v3.0",
    ModelProvider.CLOUDFLARE: "@cf/baai/bge-base-en-v1.5",
    ModelProvider.AZURE: "text-embedding-3-small",
    ModelProvider.ANTHROPIC: "voyage-3",
    ModelProvider.GOOGLE: "text-embedding-004",
    ModelProvider.GROQ: "nomic-embed-text",  # Via Groq
    ModelProvider.OLLAMA: "nomic-embed-text",
    ModelProvider.CUSTOM: "text-embedding-3-small",
}
