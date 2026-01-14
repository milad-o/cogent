"""
ToolGate - Dynamic tool filtering interceptors.

ToolGate interceptors filter available tools based on context like
permissions, conversation stage, or custom logic.

Example:
    from agenticflow import Agent
    from agenticflow.interceptors import PermissionGate
    
    class MyPermissionGate(PermissionGate):
        def allowed_tools(self, ctx: InterceptContext) -> list[str]:
            if ctx.run_context and ctx.run_context.user_role == "admin":
                return ["*"]  # All tools
            return ["read_data", "search"]  # Limited tools
    
    agent = Agent(
        name="assistant",
        model=model,
        tools=[read_data, write_data, admin_delete],
        intercept=[MyPermissionGate()],
    )
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from agenticflow.interceptors.base import (
    Interceptor,
    InterceptContext,
    InterceptResult,
)

if TYPE_CHECKING:
    pass


class ToolGate(Interceptor):
    """Base class for tool filtering interceptors.
    
    ToolGate filters the available tools before each model call (PRE_THINK).
    This allows dynamic tool selection based on:
    - User permissions
    - Conversation stage
    - Custom business logic
    
    Override the `filter` method to implement your filtering logic.
    
    Example:
        class AuthGate(ToolGate):
            async def filter(
                self,
                tools: list[BaseTool],
                ctx: InterceptContext,
            ) -> list[BaseTool]:
                if ctx.run_context and ctx.run_context.is_authenticated:
                    return tools
                return [t for t in tools if t.name.startswith("public_")]
    """
    
    @abstractmethod
    async def filter(
        self,
        tools: list[Any],
        ctx: InterceptContext,
    ) -> list[Any]:
        """Filter the available tools.
        
        Args:
            tools: Current list of available tools.
            ctx: Intercept context with run_context, state, etc.
            
        Returns:
            Filtered list of tools.
        """
        ...
    
    async def pre_think(self, ctx: InterceptContext) -> InterceptResult:
        """Filter tools before each model call."""
        if ctx.tools is None:
            return InterceptResult.ok()
        
        filtered = await self.filter(ctx.tools, ctx)
        return InterceptResult.modify_tools(filtered)


class PermissionGate(ToolGate):
    """Filter tools based on a permission allowlist.
    
    Override `allowed_tools` to return the list of allowed tool names
    based on the current context.
    
    Args:
        default_tools: Tools allowed when no context is available.
        
    Example:
        class RoleBasedGate(PermissionGate):
            def allowed_tools(self, ctx: InterceptContext) -> list[str]:
                role = ctx.run_context.role if ctx.run_context else "guest"
                return {
                    "admin": ["*"],
                    "user": ["read", "search", "write"],
                    "guest": ["read", "search"],
                }.get(role, ["read"])
    """
    
    def __init__(self, default_tools: list[str] | None = None) -> None:
        """Initialize PermissionGate.
        
        Args:
            default_tools: Tool names allowed by default (when no context).
                          Use ["*"] to allow all tools.
        """
        self.default_tools = default_tools or []
    
    def allowed_tools(self, ctx: InterceptContext) -> list[str]:
        """Return list of allowed tool names.
        
        Override this method to implement permission logic.
        Return ["*"] to allow all tools.
        
        Args:
            ctx: Intercept context.
            
        Returns:
            List of allowed tool names, or ["*"] for all.
        """
        return self.default_tools
    
    async def filter(
        self,
        tools: list[Any],
        ctx: InterceptContext,
    ) -> list[Any]:
        """Filter tools based on allowed_tools list."""
        allowed = self.allowed_tools(ctx)
        
        if "*" in allowed:
            return tools
        
        return [t for t in tools if getattr(t, "name", None) in allowed]


class ConversationGate(ToolGate):
    """Filter tools based on conversation stage.
    
    Enables more tools as the conversation progresses, reducing
    initial complexity and improving model accuracy.
    
    Args:
        stages: Dict mapping message count thresholds to tool names.
                Tools accumulate as thresholds are passed.
                
    Example:
        gate = ConversationGate({
            0: ["greeting", "help"],      # Start with basic tools
            5: ["search", "lookup"],      # Add more after 5 messages
            10: ["write", "execute"],     # Add powerful tools after 10
        })
    """
    
    def __init__(self, stages: dict[int, list[str]]) -> None:
        """Initialize ConversationGate.
        
        Args:
            stages: Dict mapping message count to tool names to enable.
        """
        self.stages = stages
    
    async def filter(
        self,
        tools: list[Any],
        ctx: InterceptContext,
    ) -> list[Any]:
        """Filter tools based on message count."""
        message_count = len(ctx.messages) if ctx.messages else 0
        
        # Accumulate allowed tools from all passed thresholds
        allowed: set[str] = set()
        for threshold, tool_names in sorted(self.stages.items()):
            if message_count >= threshold:
                allowed.update(tool_names)
        
        if "*" in allowed:
            return tools
        
        return [t for t in tools if getattr(t, "name", None) in allowed]


__all__ = [
    "ToolGate",
    "PermissionGate",
    "ConversationGate",
]
