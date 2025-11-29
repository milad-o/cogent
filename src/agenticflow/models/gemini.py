"""
Google Gemini models for AgenticFlow.

Supports Gemini 2.0, 1.5 Pro, 1.5 Flash, and other Google AI models.

Usage:
    from agenticflow.models.gemini import Chat, Embedding
    
    llm = Chat(model="gemini-2.0-flash")
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from agenticflow.models.base import AIMessage, BaseChatModel, BaseEmbedding


def _messages_to_gemini(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert messages to Gemini format.
    
    Handles both dict messages and message objects (SystemMessage, HumanMessage, etc.).
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
            gemini_messages.append({
                "role": "user",
                "parts": [{"text": content}],
            })
        elif role == "assistant":
            if tool_calls:
                parts = []
                if content:
                    parts.append({"text": content})
                for tc in tool_calls:
                    # Handle tool call dicts or objects
                    if hasattr(tc, "name"):
                        tc_name = tc.name
                        tc_args = getattr(tc, "args", {})
                    else:
                        tc_name = tc.get("name", tc.get("function", {}).get("name", ""))
                        tc_args = tc.get("args", tc.get("function", {}).get("arguments", {}))
                    parts.append({
                        "functionCall": {
                            "name": tc_name,
                            "args": tc_args,
                        }
                    })
                gemini_messages.append({"role": "model", "parts": parts})
            else:
                gemini_messages.append({
                    "role": "model",
                    "parts": [{"text": content}],
                })
        elif role == "tool":
            gemini_messages.append({
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": name,
                        "response": {"result": content},
                    }
                }],
            })
    
    return system_instruction, gemini_messages


def _tools_to_gemini(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert tools to Gemini format."""
    function_declarations = []
    for tool in tools:
        if hasattr(tool, "name") and hasattr(tool, "description"):
            schema = getattr(tool, "args_schema", {}) or {}
            # Gemini uses a slightly different schema format
            parameters = {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            }
            function_declarations.append({
                "name": tool.name,
                "description": tool.description or "",
                "parameters": parameters,
            })
        elif isinstance(tool, dict):
            if "function" in tool:
                func = tool["function"]
                function_declarations.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                })
            else:
                function_declarations.append(tool)
    
    return [{"functionDeclarations": function_declarations}] if function_declarations else []


def _parse_response(response: Any) -> AIMessage:
    """Parse Gemini response into AIMessage."""
    content = ""
    tool_calls = []
    
    # Handle the response structure
    candidate = response.candidates[0] if response.candidates else None
    if not candidate:
        return AIMessage(content="")
    
    for part in candidate.content.parts:
        if hasattr(part, "text") and part.text:
            content += part.text
        if hasattr(part, "function_call") and part.function_call:
            fc = part.function_call
            tool_calls.append({
                "id": f"call_{fc.name}",  # Gemini doesn't provide IDs
                "name": fc.name,
                "args": dict(fc.args) if fc.args else {},
            })
    
    return AIMessage(
        content=content,
        tool_calls=tool_calls,
    )


@dataclass
class GeminiChat(BaseChatModel):
    """Google Gemini chat model.
    
    High-performance chat model using Google's Generative AI SDK.
    
    Available models:
    - gemini-2.0-flash-exp (latest, experimental)
    - gemini-1.5-pro
    - gemini-1.5-flash
    - gemini-1.5-flash-8b
    
    Example:
        from agenticflow.models.gemini import Chat
        
        # Default model
        llm = Chat()  # Uses gemini-2.0-flash-exp
        
        # Custom model
        llm = Chat(model="gemini-1.5-pro")
        
        # With tools
        llm = Chat().bind_tools([search_tool])
        
        response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])
        
        # Streaming
        async for chunk in llm.astream(messages):
            print(chunk.content, end="")
    """
    
    model: str = "gemini-2.0-flash-exp"
    
    def _init_client(self) -> None:
        """Initialize Gemini client."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package required. "
                "Install with: uv add google-generativeai"
            )
        
        api_key = self.api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        
        generation_config = {
            "temperature": self.temperature,
        }
        if self.max_tokens:
            generation_config["max_output_tokens"] = self.max_tokens
        
        self._client = genai.GenerativeModel(
            model_name=self.model,
            generation_config=generation_config,
        )
        # Store genai for later use
        self._genai = genai
    
    def bind_tools(
        self,
        tools: list[Any],
        *,
        parallel_tool_calls: bool = True,
    ) -> "GeminiChat":
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
        new_model._genai = self._genai
        
        # Recreate client with tools
        generation_config = {
            "temperature": self.temperature,
        }
        if self.max_tokens:
            generation_config["max_output_tokens"] = self.max_tokens
        
        new_model._client = self._genai.GenerativeModel(
            model_name=self.model,
            generation_config=generation_config,
            tools=_tools_to_gemini(tools),
        )
        new_model._initialized = True
        return new_model
    
    def invoke(self, messages: list[dict[str, Any]]) -> AIMessage:
        """Invoke synchronously."""
        self._ensure_initialized()
        
        system, gemini_messages = _messages_to_gemini(messages)
        
        # Gemini needs a chat session for multi-turn
        chat = self._client.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])
        
        # Get the last message
        last_msg = gemini_messages[-1] if gemini_messages else {"parts": [{"text": ""}]}
        last_content = last_msg.get("parts", [{}])[0].get("text", "")
        
        response = chat.send_message(last_content)
        return _parse_response(response)
    
    async def ainvoke(self, messages: list[dict[str, Any]]) -> AIMessage:
        """Invoke asynchronously."""
        self._ensure_initialized()
        
        system, gemini_messages = _messages_to_gemini(messages)
        
        chat = self._client.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])
        
        last_msg = gemini_messages[-1] if gemini_messages else {"parts": [{"text": ""}]}
        last_content = last_msg.get("parts", [{}])[0].get("text", "")
        
        response = await chat.send_message_async(last_content)
        return _parse_response(response)
    
    async def astream(self, messages: list[dict[str, Any]]) -> AsyncIterator[AIMessage]:
        """Stream response asynchronously."""
        self._ensure_initialized()
        
        system, gemini_messages = _messages_to_gemini(messages)
        
        chat = self._client.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])
        
        last_msg = gemini_messages[-1] if gemini_messages else {"parts": [{"text": ""}]}
        last_content = last_msg.get("parts", [{}])[0].get("text", "")
        
        response = await chat.send_message_async(last_content, stream=True)
        async for chunk in response:
            if chunk.text:
                yield AIMessage(content=chunk.text)


@dataclass
class GeminiEmbedding(BaseEmbedding):
    """Google Gemini embedding model.
    
    Example:
        from agenticflow.models.gemini import GeminiEmbedding
        
        embedder = Embedding()  # Uses text-embedding-004
        
        vectors = await embedder.aembed(["Hello", "World"])
    """
    
    model: str = "text-embedding-004"
    
    def _init_client(self) -> None:
        """Initialize Gemini embedding client."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package required. "
                "Install with: uv add google-generativeai"
            )
        
        api_key = self.api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        
        self._genai = genai
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts synchronously."""
        self._ensure_initialized()
        
        result = self._genai.embed_content(
            model=f"models/{self.model}",
            content=texts,
        )
        return result["embedding"] if isinstance(result["embedding"][0], list) else [result["embedding"]]
    
    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts asynchronously.
        
        Note: google-generativeai doesn't have native async embed,
        so we use sync in executor.
        """
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, self.embed, texts
        )
