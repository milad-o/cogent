"""
Google Gemini models for Cogent.

Supports Gemini 2.5, 2.0, 1.5 Pro, 1.5 Flash, and other Google AI models.
Includes Thinking Config for enhanced reasoning on complex tasks.

Usage:
    from cogent.models.gemini import GeminiChat, GeminiEmbedding

    # Standard usage
    llm = GeminiChat(model="gemini-2.0-flash")
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])

    # With Thinking (Gemini 2.5 models)
    llm = GeminiChat(
        model="gemini-2.5-flash-preview-05-20",
        thinking_budget=8192,  # Token budget for thinking
    )
    response = await llm.ainvoke("Solve this complex problem...")
    print(response.thoughts)  # Access thought summary
    print(response.content)   # Access final response

Thinking-enabled models:
    - gemini-2.5-pro-preview-05-06: Most capable, uses thinking_budget
    - gemini-2.5-flash-preview-05-20: Fast, uses thinking_budget
    - gemini-3-flash-preview: Gemini 3 Flash (Preview), uses thinking_budget
    - gemini-3-pro-preview: Gemini 3 Pro (Preview), uses thinking_budget
    - gemini-2.0-flash-thinking-exp-01-21: Uses thinking_level
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

# Models that support thinking_budget (Gemini 2.5 and 3.0)
THINKING_BUDGET_MODELS = {
    # Gemini 2.5
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-pro",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash",
    # Gemini 3.0 (Preview)
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3-pro-image-preview",
}

# Models that use thinking_level instead (experimental)
THINKING_LEVEL_MODELS = {
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.0-flash-thinking-exp",
}


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

            # Handle Pydantic models (convert to JSON schema first)
            if hasattr(schema, "model_json_schema"):
                # Pydantic v2
                schema = schema.model_json_schema()
            elif hasattr(schema, "schema"):
                # Pydantic v1
                schema = schema.schema()

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


def _parse_response(response: Any, include_thoughts: bool = False) -> AIMessage:
    """Parse Gemini response into AIMessage with metadata.

    Args:
        response: Gemini API response.
        include_thoughts: Whether to include thought summaries in response.

    Returns:
        AIMessage with content, tool_calls, thoughts (if enabled), and metadata.
    """
    from cogent.core.messages import MessageMetadata, TokenUsage

    content = ""
    thoughts = ""
    thought_signature = None
    tool_calls = []

    # Handle the response structure
    if not hasattr(response, "candidates") or not response.candidates:
        return AIMessage(content="")

    candidate = response.candidates[0]
    if not candidate:
        return AIMessage(content="")

    # Gemini can return None for content when blocked or failed
    if (
        not candidate.content
        or not hasattr(candidate.content, "parts")
        or candidate.content.parts is None
    ):
        return AIMessage(content="")

    for part in candidate.content.parts:
        # Handle thought parts (Gemini 2.5 thinking)
        if hasattr(part, "thought") and part.thought:
            thoughts += part.text if hasattr(part, "text") else ""
            if hasattr(part, "thought_signature"):
                thought_signature = part.thought_signature
        elif hasattr(part, "text") and part.text:
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

    # Extract thinking tokens if available
    thinking_tokens = None
    if hasattr(response, "usage_metadata") and hasattr(response.usage_metadata, "thoughts_token_count"):
        thinking_tokens = response.usage_metadata.thoughts_token_count

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
            reasoning_tokens=thinking_tokens,
        )
        if hasattr(response, "usage_metadata")
        else None,
        finish_reason=candidate.finish_reason.name
        if hasattr(candidate, "finish_reason")
        else None,
    )

    msg = AIMessage(
        content=content,
        tool_calls=tool_calls,
        metadata=metadata,
    )

    # Add thoughts if present
    if thoughts and include_thoughts:
        msg.thoughts = thoughts
        if thought_signature:
            msg.thought_signature = thought_signature

    return msg


@dataclass
class GeminiChat(BaseChatModel):
    """Google Gemini chat model with Thinking support.

    High-performance chat model using the Google GenAI SDK.
    Supports Thinking Config for enhanced reasoning on complex tasks.

    Available models:
    - gemini-2.5-pro-preview-05-06 (thinking with budget)
    - gemini-2.5-flash-preview-05-20 (fast thinking with budget)
    - gemini-2.0-flash (latest stable non-thinking)
    - gemini-2.0-flash-exp (experimental)
    - gemini-1.5-pro
    - gemini-1.5-flash

    Gemini 3 (Preview - Not Production Ready):
    - gemini-3-pro-preview (most capable, 1M context, thinking support)
    - gemini-3-flash-preview (fast, 1M context, thinking support)
    - gemini-3-pro-image-preview (Nano Banana Pro, 131K context, thinking support)

    Example:
        from cogent.models.gemini import GeminiChat

        # Default model
        llm = GeminiChat()  # Uses gemini-2.0-flash

        # With Thinking (Gemini 2.5)
        llm = GeminiChat(
            model="gemini-2.5-flash-preview-05-20",
            thinking_budget=8192,  # Token budget for thinking
        )
        response = await llm.ainvoke("Solve this step by step...")
        print(response.thoughts)  # Thought summary
        print(response.content)   # Final answer

        # With tools
        llm = GeminiChat().bind_tools([search_tool])

        # Streaming
        async for chunk in llm.astream(messages):
            print(chunk.content, end="")

    Attributes:
        model: Model name (default: gemini-2.0-flash).
        thinking_budget: Token budget for thinking (Gemini 2.5+ models).
            - 0: Disable thinking (default for cost efficiency)
            - 1-24576: Fixed budget
            - None: Use model's default
        thinking_level: Thinking intensity for experimental models.
            - "minimal", "low", "medium", "high"
        include_thoughts: Whether to include thought summaries in response.
    """

    model: str = "gemini-2.5-flash"
    thinking_budget: int | None = field(
        default=0
    )  # Default: disabled for cost efficiency
    thinking_level: str | None = field(default=None)  # For experimental models
    include_thoughts: bool = field(default=True)

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
            ) from None

        from cogent.config import get_api_key

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
            thinking_budget=self.thinking_budget,
            thinking_level=self.thinking_level,
            include_thoughts=self.include_thoughts,
        )
        new_model._tools = tools
        new_model._parallel_tool_calls = parallel_tool_calls
        new_model._client = self._client
        new_model._types = self._types
        new_model._initialized = True
        return new_model

    def with_thinking(
        self,
        budget: int | None = 8192,
        *,
        level: str | None = None,
        include_thoughts: bool = True,
    ) -> GeminiChat:
        """Enable Thinking with specified budget or level.

        Thinking allows the model to reason through complex problems
        before providing a response.

        Args:
            budget: Token budget for thinking (Gemini 2.5 models).
                - 0: Dynamic thinking (model decides)
                - 1-24576: Fixed budget
            level: Thinking level for experimental models.
                - "minimal", "low", "medium", "high"
            include_thoughts: Whether to include thought summaries (default True).

        Returns:
            New GeminiChat instance with thinking enabled.

        Example:
            # Gemini 2.5 with budget
            llm = GeminiChat(model="gemini-2.5-flash-preview-05-20").with_thinking(8192)

            # Experimental model with level
            llm = GeminiChat(model="gemini-2.0-flash-thinking-exp").with_thinking(level="high")
        """
        self._ensure_initialized()

        new_model = GeminiChat(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
            thinking_budget=budget,
            thinking_level=level,
            include_thoughts=include_thoughts,
        )
        new_model._tools = getattr(self, "_tools", [])
        new_model._parallel_tool_calls = getattr(self, "_parallel_tool_calls", True)
        new_model._client = self._client
        new_model._types = self._types
        new_model._initialized = True
        return new_model

    def _build_config(self, system_instruction: str | None = None) -> Any:
        """Build GenerateContentConfig with current settings including thinking."""
        config_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
        }

        if self.max_tokens:
            config_kwargs["max_output_tokens"] = self.max_tokens

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        # Thinking configuration (Gemini 2.5 and experimental models)
        # Only enable thinking if explicitly configured (budget > 0 or level set)
        if (
            self.thinking_budget is not None and self.thinking_budget > 0
        ) or self.thinking_level is not None:
            thinking_config: dict[str, Any] = {}

            # Gemini 2.5 models use thinking_budget
            if self.thinking_budget is not None and self.thinking_budget > 0:
                thinking_config["thinking_budget"] = self.thinking_budget

            # Experimental models use thinking_level
            if self.thinking_level is not None:
                # Map string levels to SDK enum values
                level_mapping = {
                    "minimal": "THINKING_LEVEL_MINIMAL",
                    "low": "THINKING_LEVEL_LOW",
                    "medium": "THINKING_LEVEL_MEDIUM",
                    "high": "THINKING_LEVEL_HIGH",
                }
                thinking_config["thinking_level"] = level_mapping.get(
                    self.thinking_level.lower(), self.thinking_level
                )

            # Include thought summaries in response
            if self.include_thoughts:
                thinking_config["include_thoughts"] = True

            config_kwargs["thinking_config"] = self._types.ThinkingConfig(
                **thinking_config
            )

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
        return _parse_response(
            response,
            include_thoughts=self.include_thoughts
            and (
                (self.thinking_budget is not None and self.thinking_budget > 0)
                or self.thinking_level is not None
            ),
        )

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
        return _parse_response(
            response,
            include_thoughts=self.include_thoughts
            and (
                (self.thinking_budget is not None and self.thinking_budget > 0)
                or self.thinking_level is not None
            ),
        )

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
            if hasattr(chunk, "candidates") and chunk.candidates and hasattr(chunk.candidates[0], "finish_reason"):
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
        from cogent.models.gemini import GeminiEmbedding

        embedder = GeminiEmbedding()  # Uses gemini-embedding-001

        vectors = await embedder.embed(["Hello", "World"])
    """

    model: str = "gemini-embedding-001"

    def _init_client(self) -> None:
        """Initialize Gemini embedding client using the new google.genai SDK."""
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai package required. Install with: uv add google-genai"
            ) from None

        from cogent.config import get_api_key

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

        from cogent.core.messages import (
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
