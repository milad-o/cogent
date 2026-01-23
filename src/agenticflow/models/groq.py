"""
Groq models for AgenticFlow.

Groq provides extremely fast inference for open-source LLMs.
Supports Llama, Mixtral, and other models.

Usage:
    from agenticflow.models.groq import GroqChat

    # Standard completions
    llm = GroqChat(model="llama-3.3-70b-versatile")
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])

    # Responses API (optimized for tool use)
    llm = GroqChat(model="llama-3.3-70b-versatile", use_responses_api=True)
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from agenticflow.core.messages import MessageMetadata, TokenUsage
from agenticflow.models.base import AIMessage, BaseChatModel, normalize_input


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


def _convert_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert messages to dict format.

    Handles both dict messages and message objects (SystemMessage, HumanMessage, etc.).
    """
    result = []
    for msg in messages:
        # Already a dict
        if isinstance(msg, dict):
            result.append(msg)
            continue

        # Message object with to_dict method
        if hasattr(msg, "to_dict"):
            result.append(msg.to_dict())
            continue

        # Message object with to_openai method (backward compat)
        if hasattr(msg, "to_openai"):
            result.append(msg.to_openai())
            continue

        # Message object with role and content attributes
        if hasattr(msg, "role") and hasattr(msg, "content"):
            msg_dict: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content or "",
            }
            # Handle tool calls on assistant messages
            if hasattr(msg, "tool_calls") and msg.tool_calls:
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
                            "arguments": __import__("json").dumps(
                                tc.get("args", {})
                                if isinstance(tc, dict)
                                else getattr(tc, "args", {})
                            ),
                        },
                    }
                    for i, tc in enumerate(msg.tool_calls)
                ]
            # Handle tool result messages
            if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            if hasattr(msg, "name") and msg.name:
                msg_dict["name"] = msg.name
            result.append(msg_dict)
            continue

        # Fallback: try to convert to string
        result.append({"role": "user", "content": str(msg)})

    return result


def _parse_response(response: Any) -> AIMessage:
    """Parse Groq response into AIMessage with metadata."""
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
class GroqChat(BaseChatModel):
    """Groq chat model.

    Lightning-fast inference for open-source LLMs.

    Available models (as of 2024):
    - llama-3.3-70b-versatile (recommended)
    - llama-3.1-70b-versatile
    - llama-3.1-8b-instant
    - mixtral-8x7b-32768
    - gemma2-9b-it

    Example:
        from agenticflow.models.groq import GroqChat

        # Default model
        llm = GroqChat()  # Uses llama-3.3-70b-versatile

        # Custom model
        llm = GroqChat(model="mixtral-8x7b-32768")

        # With tools
        llm = GroqChat().bind_tools([search_tool])

        # Responses API (optimized for tool use)
        llm = GroqChat(model="llama-3.3-70b-versatile", use_responses_api=True)

        response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])

        # Streaming
        async for chunk in llm.astream(messages):
            print(chunk.content, end="")
    """

    model: str = "llama-3.3-70b-versatile"
    use_responses_api: bool = False

    def _init_client(self) -> None:
        """Initialize Groq client using OpenAI-compatible API."""
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: uv add openai"
            ) from None

        from agenticflow.config import get_api_key

        api_key = get_api_key("groq", self.api_key)

        self._client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        self._async_client = AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def bind_tools(
        self,
        tools: list[Any],
        *,
        parallel_tool_calls: bool = True,
    ) -> GroqChat:
        """Bind tools to the model."""
        self._ensure_initialized()

        new_model = GroqChat(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
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
        messages = normalize_input(messages)
        kwargs = self._build_request(messages)
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
        messages = normalize_input(messages)
        kwargs = self._build_request(messages)
        if self.use_responses_api:
            response = await self._async_client.beta.responses.create(**kwargs)
        else:
            response = await self._async_client.chat.completions.create(**kwargs)
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

        if self.use_responses_api:
            stream = await self._async_client.beta.responses.create(**kwargs)
        else:
            stream = await self._async_client.chat.completions.create(**kwargs)

        async for chunk in stream:
            # Accumulate metadata
            if chunk.id:
                chunk_metadata["id"] = chunk.id
            if chunk.model:
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
        converted_messages = _convert_messages(messages)

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
            kwargs["parallel_tool_calls"] = self._parallel_tool_calls

        # Structured output support
        if hasattr(self, "_response_format") and self._response_format:
            kwargs["response_format"] = self._response_format

        return kwargs
