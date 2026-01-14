"""
ToolRegistry - extensible registry for agent tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from agenticflow.tools.base import BaseTool

if TYPE_CHECKING:
    from agenticflow.observability.bus import TraceBus


class ToolRegistry:
    """
    Extensible registry for agent tools.
    
    The ToolRegistry manages available tools that agents can use.
    It supports:
    - Registering native AgenticFlow tools
    - Getting tool by name
    - Generating tool descriptions for LLM prompts
    - Event emission for tool registration
    
    Attributes:
        event_bus: Optional EventBus for registration events
        
    Example:
        ```python
        from agenticflow.tools import tool
        
        @tool
        def search_web(query: str) -> str:
            '''Search the web for information.'''
            return f"Results for: {query}"
        
        registry = ToolRegistry()
        registry.register(search_web)
        
        # Use in agent
        agent = Agent(
            config=config,
            event_bus=event_bus,
            tool_registry=registry,
        )
        ```
    """

    def __init__(self, event_bus: TraceBus | None = None) -> None:
        """
        Initialize the ToolRegistry.
        
        Args:
            event_bus: Optional EventBus for registration events
        """
        self._tools: dict[str, BaseTool] = {}
        self.event_bus = event_bus

    def register(self, tool_instance: BaseTool) -> ToolRegistry:
        """
        Register a tool.
        
        Args:
            tool_instance: The tool to register
            
        Returns:
            Self for method chaining
        """
        self._tools[tool_instance.name] = tool_instance
        return self

    def register_many(self, tools: list[BaseTool]) -> ToolRegistry:
        """
        Register multiple tools.
        
        Args:
            tools: List of tools to register
            
        Returns:
            Self for method chaining
        """
        for t in tools:
            self.register(t)
        return self

    def unregister(self, name: str) -> bool:
        """
        Unregister a tool by name.
        
        Args:
            name: Name of the tool to unregister
            
        Returns:
            True if tool was removed, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> BaseTool | None:
        """
        Get a tool by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            The tool or None if not found
        """
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            name: Name of the tool
            
        Returns:
            True if tool exists
        """
        return name in self._tools

    def get_tool_descriptions(self) -> str:
        """
        Get formatted descriptions of all tools for LLM prompts.
        
        Returns:
            Formatted string with tool descriptions
        """
        descriptions = []
        for name, tool_obj in self._tools.items():
            # Get schema if available
            if tool_obj.args_schema:
                schema = tool_obj.args_schema.model_json_schema()
            else:
                schema = {}

            props = schema.get("properties", {})
            args_desc = ", ".join(
                f"{k}: {v.get('type', 'any')}" for k, v in props.items()
            )
            descriptions.append(f"- {name}({args_desc}): {tool_obj.description}")

        return "\n".join(descriptions)

    def get_tool_schema(self, name: str) -> dict | None:
        """
        Get the JSON schema for a tool's arguments.
        
        Args:
            name: Name of the tool
            
        Returns:
            JSON schema dict or None if not found
        """
        tool = self._tools.get(name)
        if not tool:
            return None

        if tool.args_schema:
            return tool.args_schema.model_json_schema()
        return {}

    @property
    def tool_names(self) -> list[str]:
        """List of all registered tool names."""
        return list(self._tools.keys())

    @property
    def tools(self) -> list[BaseTool]:
        """List of all registered tools."""
        return list(self._tools.values())

    def __len__(self) -> int:
        """Number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools

    def __iter__(self):
        """Iterate over tool names."""
        return iter(self._tools)

    def to_dict(self) -> dict:
        """
        Convert registry info to dictionary.
        
        Returns:
            Dictionary with tool information
        """
        return {
            name: {
                "description": tool.description,
                "schema": self.get_tool_schema(name),
            }
            for name, tool in self._tools.items()
        }

    def clear(self) -> None:
        """Remove all registered tools."""
        self._tools.clear()


def create_tool_from_function(
    func: Callable,
    name: str | None = None,
    description: str | None = None,
) -> BaseTool:
    """
    Create a native tool from a function.
    
    This is a convenience wrapper around @tool decorator.
    
    Args:
        func: The function to wrap
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        
    Returns:
        A native BaseTool
    """
    from agenticflow.tools.base import tool as tool_decorator
    
    # The tool decorator creates a tool from the function
    # We can override the name by setting __name__ and description via docstring
    if name:
        func.__name__ = name
    if description:
        func.__doc__ = description
    
    return tool_decorator(func)
