"""
Native tool abstraction for AgenticFlow.

Replaces langchain_core.tools with lightweight native implementations.
Compatible with OpenAI function calling format.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints


@dataclass
class BaseTool:
    """Base class for tools that agents can use.
    
    A tool wraps a function and provides:
    - Name and description for the LLM
    - JSON schema for parameters
    - Sync and async invocation
    
    Example:
        @tool
        def search(query: str) -> str:
            '''Search the web for information.'''
            return f"Results for: {query}"
        
        # Or create directly:
        tool = BaseTool(
            name="search",
            description="Search the web",
            func=search_fn,
            args_schema={"query": {"type": "string"}},
        )
    """
    
    name: str
    description: str
    func: Callable[..., Any]
    args_schema: dict[str, Any] = field(default_factory=dict)
    
    def invoke(self, args: dict[str, Any]) -> Any:
        """Invoke the tool synchronously.
        
        Args:
            args: Dictionary of arguments matching the schema.
            
        Returns:
            Tool result.
        """
        return self.func(**args)
    
    async def ainvoke(self, args: dict[str, Any]) -> Any:
        """Invoke the tool asynchronously.
        
        Args:
            args: Dictionary of arguments matching the schema.
            
        Returns:
            Tool result.
        """
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(**args)
        else:
            return await asyncio.to_thread(self.func, **args)
    
    def to_openai(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format.
        
        Returns:
            OpenAI tool definition dict.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.args_schema,
                    "required": list(self.args_schema.keys()),
                },
            },
        }


def _python_type_to_json_schema(py_type: type) -> dict[str, Any]:
    """Convert Python type hint to JSON schema type."""
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }
    
    # Handle Optional, Union, etc.
    origin = getattr(py_type, "__origin__", None)
    if origin is list:
        args = getattr(py_type, "__args__", (Any,))
        return {"type": "array", "items": _python_type_to_json_schema(args[0]) if args else {}}
    
    return type_map.get(py_type, {"type": "string"})


def _extract_schema_from_function(func: Callable[..., Any]) -> dict[str, Any]:
    """Extract JSON schema from function signature and docstring."""
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
    
    schema: dict[str, Any] = {}
    
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        
        param_schema: dict[str, Any] = {}
        
        # Get type from hints
        if param_name in hints:
            param_schema = _python_type_to_json_schema(hints[param_name])
        else:
            param_schema = {"type": "string"}
        
        # Try to extract description from docstring
        doc = func.__doc__ or ""
        # Look for :param name: or Args: name: patterns
        for line in doc.split("\n"):
            line = line.strip()
            if f"{param_name}:" in line or f":param {param_name}:" in line:
                # Extract description after the colon
                parts = line.split(":", 2)
                if len(parts) >= 2:
                    param_schema["description"] = parts[-1].strip()
                break
        
        schema[param_name] = param_schema
    
    return schema


def tool(func: Callable[..., Any] | None = None, *, name: str | None = None, description: str | None = None) -> BaseTool | Callable[[Callable[..., Any]], BaseTool]:
    """Decorator to create a tool from a function.
    
    The function's docstring becomes the tool description.
    Parameter types and descriptions are extracted from type hints and docstring.
    
    Example:
        @tool
        def search(query: str) -> str:
            '''Search the web for information.
            
            Args:
                query: The search query string.
            '''
            return f"Results for: {query}"
        
        # With custom name:
        @tool(name="web_search")
        def search(query: str) -> str:
            '''Search the web.'''
            return f"Results for: {query}"
    """
    def decorator(fn: Callable[..., Any]) -> BaseTool:
        tool_name = name or fn.__name__
        tool_desc = description or (fn.__doc__ or "").split("\n")[0].strip() or f"Tool: {tool_name}"
        args_schema = _extract_schema_from_function(fn)
        
        return BaseTool(
            name=tool_name,
            description=tool_desc,
            func=fn,
            args_schema=args_schema,
        )
    
    if func is not None:
        return decorator(func)
    return decorator


def tools_to_openai(tools: list[BaseTool]) -> list[dict[str, Any]]:
    """Convert a list of tools to OpenAI format."""
    return [t.to_openai() for t in tools]
