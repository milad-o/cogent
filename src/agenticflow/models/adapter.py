"""
LangChain adapter for native AgenticFlow models.

This module provides compatibility with LangChain-based code.
"""

from __future__ import annotations

from typing import Any, Iterator

from agenticflow.models.base import BaseChatModel


class NativeModelAdapter:
    """Adapter that wraps native OpenAIChat for LangChain compatibility.
    
    This allows using native AgenticFlow models with LangChain-based
    code that expects a LangChain chat model interface.
    
    Example:
        from agenticflow.models.openai import OpenAIChat
        from agenticflow.models.adapter import NativeModelAdapter
        
        # Wrap native model
        native = OpenAIChat(model="gpt-4o")
        langchain_compatible = NativeModelAdapter(native)
        
        # Use with LangChain code
        result = langchain_compatible.invoke([...])
    """
    
    def __init__(self, native_model: BaseChatModel):
        """Initialize adapter with native model.
        
        Args:
            native_model: Native AgenticFlow chat model to wrap.
        """
        self._native = native_model
        self._tools: list[Any] = []
    
    @property
    def native_model(self) -> BaseChatModel:
        """Get the underlying native model."""
        return self._native
    
    def bind_tools(
        self,
        tools: list[Any],
        *,
        parallel_tool_calls: bool = True,
    ) -> "NativeModelAdapter":
        """Bind tools to the model.
        
        Returns a new adapter with tools bound.
        """
        bound_native = self._native.bind_tools(
            tools,
            parallel_tool_calls=parallel_tool_calls,
        )
        return NativeModelAdapter(bound_native)
    
    def invoke(self, messages: list[Any], **kwargs: Any) -> Any:
        """Invoke model synchronously.
        
        Accepts LangChain message format and returns LangChain-compatible output.
        """
        from langchain_core.messages import AIMessage as LCAIMessage
        
        # Convert LangChain messages to dict format
        dict_messages = self._convert_messages(messages)
        
        # Call native model
        result = self._native.invoke(dict_messages)
        
        # Convert to LangChain format
        return self._to_langchain(result)
    
    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> Any:
        """Invoke model asynchronously.
        
        Accepts LangChain message format and returns LangChain-compatible output.
        """
        from langchain_core.messages import AIMessage as LCAIMessage
        
        # Convert LangChain messages to dict format
        dict_messages = self._convert_messages(messages)
        
        # Call native model
        result = await self._native.ainvoke(dict_messages)
        
        # Convert to LangChain format
        return self._to_langchain(result)
    
    def stream(self, messages: list[Any], **kwargs: Any) -> Iterator[Any]:
        """Stream response synchronously."""
        # For sync streaming, we fall back to invoke
        yield self.invoke(messages, **kwargs)
    
    async def astream(self, messages: list[Any], **kwargs: Any) -> Any:
        """Stream response asynchronously."""
        from langchain_core.messages import AIMessageChunk
        
        dict_messages = self._convert_messages(messages)
        
        async for chunk in self._native.astream(dict_messages):
            yield AIMessageChunk(content=chunk.content)
    
    def _convert_messages(self, messages: list[Any]) -> list[dict[str, Any]]:
        """Convert LangChain messages to dict format."""
        result = []
        for msg in messages:
            if hasattr(msg, "type"):
                # LangChain message object
                if msg.type == "system":
                    result.append({"role": "system", "content": msg.content})
                elif msg.type == "human":
                    result.append({"role": "user", "content": msg.content})
                elif msg.type == "ai":
                    entry = {"role": "assistant", "content": msg.content}
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        entry["tool_calls"] = [
                            {
                                "id": tc.get("id", ""),
                                "function": {
                                    "name": tc.get("name", ""),
                                    "arguments": tc.get("args", {}),
                                },
                            }
                            for tc in msg.tool_calls
                        ]
                    result.append(entry)
                elif msg.type == "tool":
                    result.append({
                        "role": "tool",
                        "content": msg.content,
                        "tool_call_id": getattr(msg, "tool_call_id", ""),
                    })
            elif isinstance(msg, dict):
                result.append(msg)
        return result
    
    def _to_langchain(self, result: Any) -> Any:
        """Convert native result to LangChain format."""
        from langchain_core.messages import AIMessage as LCAIMessage
        
        # Convert tool calls to LangChain format
        tool_calls = []
        if result.tool_calls:
            for tc in result.tool_calls:
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                })
        
        return LCAIMessage(
            content=result.content,
            tool_calls=tool_calls,
        )
