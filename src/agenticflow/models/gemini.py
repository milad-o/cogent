"""
Google Gemini models for AgenticFlow.

Supports Gemini 2.0, 1.5 Pro, 1.5 Flash, and other Google AI models.

Usage:
    from agenticflow.models.gemini import GeminiChat, GeminiEmbedding

    llm = GeminiChat(model="gemini-2.0-flash")
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])
"""

from __future__ import annotations

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


def _messages_to_gemini(
    messages: list[dict[str, Any]], types: Any
) -> tuple[str | None, list[Any]]:
    """Convert messages to Gemini format using the new google.genai types.

    Handles both dict messages and message objects (SystemMessage, HumanMessage, etc.).
    Returns proper types.Content objects for the SDK.
    """
    system_instruction = None
    gemini_messages = []

    for msg in messages:
        # Handle message objects (SystemMessage, HumanMessage, AIMessage, ToolMessage)
        if hasattr(msg, "role"):
            role = msg.role
            content = getattr(msg, "content", "")
            tool_calls = getattr(msg, "tool_calls", [])
            name = getattr(msg, "name", "")
        else:
            # Handle dict messages
            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            name = msg.get("name", "")

        if role == "system":
            system_instruction = content
        elif role == "user":
            gemini_messages.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text=content)],
                )
            )
        elif role == "assistant":
            if tool_calls:
                parts = []
                if content:
                    parts.append(types.Part(text=content))
                for tc in tool_calls:
                    # Handle tool call dicts or objects
                    if hasattr(tc, "name"):
                        tc_name = tc.name
                        tc_args = getattr(tc, "args", {})
                    else:
                        tc_name = tc.get("name", tc.get("function", {}).get("name", ""))
                        tc_args = tc.get(
                            "args", tc.get("function", {}).get("arguments", {})
                        )

                    # Parse args if it's a JSON string (OpenAI format)
                    if isinstance(tc_args, str):
                        import json

                        try:
                            tc_args = json.loads(tc_args)
                        except json.JSONDecodeError:
                            tc_args = {}

                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                name=tc_name,
                                args=tc_args,
                            )
                        )
                    )
                gemini_messages.append(types.Content(role="model", parts=parts))
            else:
                gemini_messages.append(
                    types.Content(
                        role="model",
                        parts=[types.Part(text=content)],
                    )
                )
        elif role == "tool":
            # Get the function name - try 'name' attribute first, then extract from tool_call_id
            tool_name = name
            if not tool_name:
                # Try to get tool_call_id and extract name from it
                tool_call_id = (
                    getattr(msg, "tool_call_id", "")
                    if hasattr(msg, "tool_call_id")
                    else msg.get("tool_call_id", "")
                )
                if tool_call_id and tool_call_id.startswith("call_"):
                    tool_name = tool_call_id[5:]  # Remove "call_" prefix
                else:
                    tool_name = tool_call_id or "unknown_function"

            gemini_messages.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=tool_name,
                                response={"result": content},
                            )
                        )
                    ],
                )
            )

    return system_instruction, gemini_messages


def _convert_schema_types(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert JSON Schema to Gemini Schema format.

    The new SDK uses string type values like "OBJECT", "STRING", etc.
    """
    if not schema:
        return schema

    result = {}
    type_mapping = {
        "object": "OBJECT",
        "string": "STRING",
        "integer": "INTEGER",
        "number": "NUMBER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
    }

    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            result["type"] = type_mapping.get(value.lower(), "STRING")
        elif key == "properties" and isinstance(value, dict):
            # Recursively convert properties
            result["properties"] = {
                k: _convert_schema_types(v) for k, v in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            # Recursively convert array items
            result["items"] = _convert_schema_types(value)
        else:
            result[key] = value

    return result


def _tools_to_gemini(tools: list[Any], types: Any) -> list[Any]:
    """Convert tools to Gemini FunctionDeclaration format.

    Gemini expects function declarations with:
    - name: function name
    - description: what the function does
    - parameters: Schema object
    """
    function_declarations = []
    for tool in tools:
        if hasattr(tool, "name") and hasattr(tool, "description"):
            schema = getattr(tool, "args_schema", {}) or {}
            # Build parameters with proper Gemini Schema format
            parameters = {
                "type": "OBJECT",
                "properties": {},
            }

            # Handle both formats: {"properties": {...}} and flat {param: schema}
            if isinstance(schema, dict):
                if "properties" in schema:
                    # Standard JSON schema format
                    parameters["properties"] = {
                        k: _convert_schema_types(v)
                        for k, v in schema["properties"].items()
                    }
                    if schema.get("required"):
                        parameters["required"] = schema["required"]
                else:
                    # Flat format: {param_name: param_schema, ...}
                    parameters["properties"] = {
                        k: _convert_schema_types(v) for k, v in schema.items()
                    }

            function_declarations.append(
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=parameters,
                )
            )
        elif isinstance(tool, dict):
            if "function" in tool:
                func = tool["function"]
                params = func.get("parameters", {})
                if params:
                    params = _convert_schema_types(params)
                function_declarations.append(
                    types.FunctionDeclaration(
                        name=func["name"],
                        description=func.get("description", ""),
                        parameters=params or None,
                    )
                )
            else:
                # Already in correct format
                function_declarations.append(types.FunctionDeclaration(**tool))

    return function_declarations if function_declarations else []


def _parse_response(response: Any) -> AIMessage:
    """Parse Gemini response into AIMessage with metadata."""
    from agenticflow.core.messages import MessageMetadata, TokenUsage

    content = ""
    tool_calls = []

    # Handle the response structure
    candidate = response.candidates[0] if response.candidates else None
    if not candidate:
        return AIMessage(content="")

    # Gemini can return None for content when blocked or failed
    if not candidate.content or not hasattr(candidate.content, "parts"):
        return AIMessage(content="")

    for part in candidate.content.parts:
        if hasattr(part, "text") and part.text:
            content += part.text
        if hasattr(part, "function_call") and part.function_call:
            fc = part.function_call
            # Extract args properly from Gemini's format
            args_dict = {}
            if fc.args:
                # fc.args is a google.protobuf.Struct, convert to dict
                for key, value in fc.args.items():
                    args_dict[key] = value

            tool_calls.append(
                {
                    "id": f"call_{fc.name}",  # Gemini doesn't provide IDs
                    "name": fc.name,
                    "args": args_dict,
                }
            )

    metadata = MessageMetadata(
        model=response.model_version if hasattr(response, "model_version") else None,
        tokens=TokenUsage(
            prompt_tokens=response.usage_metadata.prompt_token_count
            if hasattr(response, "usage_metadata")
            else 0,
            completion_tokens=response.usage_metadata.candidates_token_count
            if hasattr(response, "usage_metadata")
            else 0,
            total_tokens=response.usage_metadata.total_token_count
            if hasattr(response, "usage_metadata")
            else 0,
        )
        if hasattr(response, "usage_metadata")
        else None,
        finish_reason=candidate.finish_reason.name
        if hasattr(candidate, "finish_reason")
        else None,
    )

    return AIMessage(
        content=content,
        tool_calls=tool_calls,
        metadata=metadata,
    )


@dataclass
class GeminiChat(BaseChatModel):
    """Google Gemini chat model.

    High-performance chat model using the Google GenAI SDK.

    Available models:
    - gemini-2.0-flash (latest)
    - gemini-2.0-flash-exp (experimental)
    - gemini-1.5-pro
    - gemini-1.5-flash
    - gemini-1.5-flash-8b

    Example:
        from agenticflow.models.gemini import GeminiChat

        # Default model
        llm = GeminiChat()  # Uses gemini-2.0-flash

        # Custom model
        llm = GeminiChat(model="gemini-1.5-pro")

        # With tools
        llm = GeminiChat().bind_tools([search_tool])

        response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])

        # Streaming
        async for chunk in llm.astream(messages):
            print(chunk.content, end="")
    """

    model: str = "gemini-2.0-flash"
    _tools: list[Any] = field(default_factory=list, repr=False)
    _parallel_tool_calls: bool = field(default=True, repr=False)

    def _init_client(self) -> None:
        """Initialize Gemini client using the new google.genai SDK."""
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "google-genai package required. Install with: uv add google-genai"
            )

        from agenticflow.config import get_api_key

        api_key = get_api_key("gemini", self.api_key)

        # Create the centralized client
        self._client = genai.Client(api_key=api_key)
        # Store types module for later use
        self._types = types

    def bind_tools(
        self,
        tools: list[Any],
        *,
        parallel_tool_calls: bool = True,
    ) -> GeminiChat:
        """Bind tools to the model."""
        self._ensure_initialized()

        new_model = GeminiChat(
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
        new_model._types = self._types
        new_model._initialized = True
        return new_model

    def _build_config(self, system_instruction: str | None = None) -> Any:
        """Build GenerateContentConfig with current settings."""
        config_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
        }

        if self.max_tokens:
            config_kwargs["max_output_tokens"] = self.max_tokens

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if self._tools:
            function_declarations = _tools_to_gemini(self._tools, self._types)
            if function_declarations:
                # Wrap function declarations in a Tool object
                tool = self._types.Tool(function_declarations=function_declarations)
                config_kwargs["tools"] = [tool]
                # Use AUTO mode for balanced tool usage
                from google.genai.types import FunctionCallingConfigMode

                config_kwargs["tool_config"] = self._types.ToolConfig(
                    function_calling_config=self._types.FunctionCallingConfig(
                        mode=FunctionCallingConfigMode.AUTO
                    )
                )

        # Structured output support
        if hasattr(self, "_response_schema") and self._response_schema:
            config_kwargs["response_schema"] = self._response_schema

        return self._types.GenerateContentConfig(**config_kwargs)

    def invoke(self, messages: str | list[dict[str, Any]] | list[Any]) -> AIMessage:
        """Invoke synchronously.

        Args:
            messages: Can be a string, list of dicts, or list of message objects.
        """
        self._ensure_initialized()

        normalized = convert_messages(normalize_input(messages))
        system, gemini_messages = _messages_to_gemini(normalized, self._types)

        config = self._build_config(system_instruction=system)

        response = self._client.models.generate_content(
            model=self.model,
            contents=gemini_messages,
            config=config,
        )
        return _parse_response(response)

    async def ainvoke(
        self, messages: str | list[dict[str, Any]] | list[Any]
    ) -> AIMessage:
        """Invoke asynchronously.

        Args:
            messages: Can be a string, list of dicts, or list of message objects.
        """
        self._ensure_initialized()

        normalized = convert_messages(normalize_input(messages))
        system, gemini_messages = _messages_to_gemini(normalized, self._types)

        config = self._build_config(system_instruction=system)

        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=gemini_messages,
            config=config,
        )
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

        normalized = convert_messages(normalize_input(messages))
        system, gemini_messages = _messages_to_gemini(normalized, self._types)

        config = self._build_config(system_instruction=system)

        start_time = time.time()
        chunk_metadata = {
            "model": None,
            "finish_reason": None,
            "usage": None,
        }

        stream = await self._client.aio.models.generate_content_stream(
            model=self.model,
            contents=gemini_messages,
            config=config,
        )
        async for chunk in stream:
            # Accumulate metadata
            if hasattr(chunk, "model_version") and chunk.model_version:
                chunk_metadata["model"] = chunk.model_version
            if hasattr(chunk, "candidates") and chunk.candidates:
                if hasattr(chunk.candidates[0], "finish_reason"):
                    chunk_metadata["finish_reason"] = str(
                        chunk.candidates[0].finish_reason
                    )
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                chunk_metadata["usage"] = chunk.usage_metadata

            # Yield content chunks
            if chunk.text:
                metadata = MessageMetadata(
                    id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    model=chunk_metadata.get("model"),
                    tokens=TokenUsage(
                        prompt_tokens=chunk_metadata["usage"].prompt_token_count
                        if chunk_metadata["usage"]
                        else None,
                        completion_tokens=chunk_metadata["usage"].candidates_token_count
                        if chunk_metadata["usage"]
                        else None,
                        total_tokens=chunk_metadata["usage"].total_token_count
                        if chunk_metadata["usage"]
                        else None,
                    )
                    if chunk_metadata["usage"]
                    else None,
                    finish_reason=chunk_metadata.get("finish_reason"),
                    duration=time.time() - start_time,
                )
                yield AIMessage(content=chunk.text, metadata=metadata)

        # Yield final metadata chunk if we have complete metadata
        if chunk_metadata["usage"] or chunk_metadata["finish_reason"]:
            final_metadata = MessageMetadata(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                model=chunk_metadata.get("model"),
                tokens=TokenUsage(
                    prompt_tokens=chunk_metadata["usage"].prompt_token_count
                    if chunk_metadata["usage"]
                    else None,
                    completion_tokens=chunk_metadata["usage"].candidates_token_count
                    if chunk_metadata["usage"]
                    else None,
                    total_tokens=chunk_metadata["usage"].total_token_count
                    if chunk_metadata["usage"]
                    else None,
                )
                if chunk_metadata["usage"]
                else None,
                finish_reason=chunk_metadata.get("finish_reason"),
                duration=time.time() - start_time,
            )
            yield AIMessage(content="", metadata=final_metadata)


@dataclass
class GeminiEmbedding(BaseEmbedding):
    """Google Gemini embedding model.

    Example:
        from agenticflow.models.gemini import GeminiEmbedding

        embedder = GeminiEmbedding()  # Uses gemini-embedding-001

        vectors = await embedder.aembed(["Hello", "World"])
    """

    model: str = "gemini-embedding-001"

    def _init_client(self) -> None:
        """Initialize Gemini embedding client using the new google.genai SDK."""
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai package required. Install with: uv add google-genai"
            )

        from agenticflow.config import get_api_key

        api_key = get_api_key("gemini", self.api_key)

        # Create the centralized client
        self._client = genai.Client(api_key=api_key)

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
        )

        # Normalize input
        texts_list = [texts] if isinstance(texts, str) else texts
        start_time = time.time()

        result = await self._client.aio.models.embed_content(
            model=self.model,
            contents=texts_list,
        )
        # Handle both single and multiple embeddings
        embeddings = result.embeddings
        vectors = [emb.values for emb in embeddings] if embeddings else []

        metadata = EmbeddingMetadata(
            model=self.model,
            tokens=None,  # Gemini doesn't provide token counts for embeddings
            duration=time.time() - start_time,
            dimensions=len(vectors[0]) if vectors else None,
            num_texts=len(texts_list),
        )

        return EmbeddingResult(embeddings=vectors, metadata=metadata)
