"""
Native tool abstraction for AgenticFlow.

Lightweight tool implementations compatible with OpenAI function calling format.
Supports context injection via `ctx: RunContext` parameter.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, get_type_hints

if TYPE_CHECKING:
    from agenticflow.core.context import RunContext


@dataclass
class BaseTool:
    """Base class for tools that agents can use.

    A tool wraps a function and provides:
    - Name and description for the LLM
    - JSON schema for parameters
    - Sync and async invocation
    - Context injection (if function has `ctx: RunContext` param)

    Example:
        @tool
        def search(query: str) -> str:
            '''Search the web for information.'''
            return f"Results for: {query}"

        # With context access:
        @tool
        def get_user_data(ctx: RunContext) -> str:
            '''Get data for the current user.'''
            return f"User: {ctx.user_id}"

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
    return_info: str = field(default="", repr=False)
    _needs_context: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Check if function needs context injection."""
        self._needs_context = _function_needs_context(self.func)

    def invoke(self, args: dict[str, Any], ctx: RunContext | None = None) -> Any:
        """Invoke the tool synchronously.

        Args:
            args: Dictionary of arguments matching the schema.
            ctx: Optional RunContext to inject.

        Returns:
            Tool result.
        """
        if self._needs_context and ctx is not None:
            return self.func(**args, ctx=ctx)
        return self.func(**args)

    async def ainvoke(self, args: dict[str, Any], ctx: RunContext | None = None) -> Any:
        """Invoke the tool asynchronously.

        Args:
            args: Dictionary of arguments matching the schema.
            ctx: Optional RunContext to inject.

        Returns:
            Tool result.
        """
        if self._needs_context and ctx is not None:
            if asyncio.iscoroutinefunction(self.func):
                return await self.func(**args, ctx=ctx)
            else:
                return await asyncio.to_thread(self.func, **args, ctx=ctx)
        else:
            if asyncio.iscoroutinefunction(self.func):
                return await self.func(**args)
            else:
                return await asyncio.to_thread(self.func, **args)

    def to_dict(self) -> dict[str, Any]:
        """Convert to function calling format for chat APIs.

        Returns:
            Tool definition dict.
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

    # Alias for backward compatibility
    to_openai = to_dict


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


def _type_to_readable_string(py_type: type) -> str:
    """Convert Python type hint to a readable string for descriptions."""
    # Handle None type
    if py_type is type(None):
        return "None"

    # Handle basic types
    if py_type in (str, int, float, bool, list, dict):
        return py_type.__name__

    # Handle generic types like list[str], dict[str, int], Optional[str]
    origin = getattr(py_type, "__origin__", None)
    args = getattr(py_type, "__args__", ())

    if origin is list:
        if args:
            return f"list[{_type_to_readable_string(args[0])}]"
        return "list"

    if origin is dict:
        if len(args) >= 2:
            return f"dict[{_type_to_readable_string(args[0])}, {_type_to_readable_string(args[1])}]"
        return "dict"

    # Handle Union (including Optional which is Union[X, None])
    import types
    if origin is types.UnionType or (hasattr(origin, "__name__") and origin.__name__ == "Union"):
        type_strs = [_type_to_readable_string(arg) for arg in args]
        # Check for Optional pattern (X | None)
        if len(type_strs) == 2 and "None" in type_strs:
            non_none = [t for t in type_strs if t != "None"][0]
            return f"{non_none} | None"
        return " | ".join(type_strs)

    # Handle | union syntax (Python 3.10+)
    if hasattr(py_type, "__class__") and py_type.__class__.__name__ == "UnionType":
        type_strs = [_type_to_readable_string(arg) for arg in args]
        return " | ".join(type_strs)

    # Fallback: try to get name or str representation
    if hasattr(py_type, "__name__"):
        return py_type.__name__

    return str(py_type).replace("typing.", "")


def _extract_return_info(func: Callable[..., Any]) -> str:
    """Extract return type and description from function.

    Combines the return type annotation with the Returns section
    from the docstring to create a descriptive string.

    Returns:
        A string describing what the function returns, or empty string.
    """
    parts: list[str] = []

    # Get return type from annotations
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
    return_type = hints.get("return")

    if return_type is not None and return_type is not type(None):
        type_str = _type_to_readable_string(return_type)
        parts.append(type_str)

    # Extract Returns section from docstring
    doc = func.__doc__ or ""
    returns_desc = ""

    # Look for Returns: or :returns: section
    lines = doc.split("\n")
    in_returns = False
    returns_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Check for start of Returns section
        if stripped.lower().startswith("returns:") or stripped.lower().startswith(":returns:"):
            in_returns = True
            # Get content after "Returns:"
            after_colon = stripped.split(":", 1)[-1].strip()
            if after_colon:
                returns_lines.append(after_colon)
            continue

        # Check for end of Returns section (new section starts)
        if in_returns:
            if stripped and (stripped.endswith(":") or stripped.startswith("Args:") or
                            stripped.startswith("Raises:") or stripped.startswith("Example:") or
                            stripped.startswith("Note:")):
                break
            if stripped:
                returns_lines.append(stripped)

    if returns_lines:
        returns_desc = " ".join(returns_lines)

    # Combine type and description
    if parts and returns_desc:
        return f"{parts[0]} - {returns_desc}"
    elif parts:
        return parts[0]
    elif returns_desc:
        return returns_desc

    return ""


def _function_needs_context(func: Callable[..., Any]) -> bool:
    """Check if function has a `ctx` parameter for context injection."""
    sig = inspect.signature(func)
    return "ctx" in sig.parameters


def _extract_schema_from_function(func: Callable[..., Any]) -> dict[str, Any]:
    """Extract JSON schema from function signature and docstring.

    Note: The `ctx` parameter is excluded from the schema as it's
    automatically injected and not passed by the LLM.
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

    schema: dict[str, Any] = {}

    for param_name, _param in sig.parameters.items():
        # Skip self/cls and the special ctx parameter
        if param_name in ("self", "cls", "ctx"):
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
        base_desc = description or (fn.__doc__ or "").split("\n")[0].strip() or f"Tool: {tool_name}"
        args_schema = _extract_schema_from_function(fn)
        return_info = _extract_return_info(fn)

        # Append return info to description for LLM visibility
        tool_desc = f"{base_desc} Returns: {return_info}" if return_info else base_desc

        return BaseTool(
            name=tool_name,
            description=tool_desc,
            func=fn,
            args_schema=args_schema,
            return_info=return_info,
        )

    if func is not None:
        return decorator(func)
    return decorator


def tools_to_dict(tools: list[BaseTool]) -> list[dict[str, Any]]:
    """Convert a list of tools to dict format for chat APIs."""
    return [t.to_dict() for t in tools]


# Alias for backward compatibility
tools_to_openai = tools_to_dict
