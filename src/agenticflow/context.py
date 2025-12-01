"""
RunContext - Invocation-scoped context for agents.

RunContext provides dependency injection for tools and interceptors,
allowing typed data to be passed at invocation time without global state.

Example:
    from dataclasses import dataclass
    from agenticflow import Agent, RunContext, tool
    
    @dataclass
    class AppContext(RunContext):
        user_id: str
        db: Database
        api_key: str
    
    @tool
    def get_user_data(ctx: RunContext) -> str:
        '''Get data for the current user.'''
        user = ctx.db.get_user(ctx.user_id)
        return f"User: {user.name}"
    
    agent = Agent(name="assistant", model=model, tools=[get_user_data])
    
    result = await agent.run(
        "Get my profile data",
        context=AppContext(user_id="123", db=db, api_key=key),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunContext:
    """Base class for invocation-scoped context.
    
    Subclass this to define typed context data that will be available
    to tools and interceptors during agent execution.
    
    The context is:
    - Passed to agent.run() at invocation time
    - Available to interceptors via InterceptContext.run_context
    - Available to tools that declare a `ctx: RunContext` parameter
    - Immutable during execution (create new instance to change)
    
    Attributes:
        metadata: Optional dict for untyped extension data.
    
    Example:
        @dataclass
        class MyContext(RunContext):
            user_id: str
            session_id: str
            permissions: list[str] = field(default_factory=list)
            
            def has_permission(self, perm: str) -> bool:
                return perm in self.permissions
        
        # Use in tool
        @tool
        def admin_action(action: str, ctx: RunContext) -> str:
            '''Perform admin action. Requires admin permission.'''
            if not ctx.has_permission("admin"):
                return "Permission denied"
            return f"Performed: {action}"
    """
    
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a metadata value by key.
        
        Args:
            key: The metadata key.
            default: Default value if key not found.
            
        Returns:
            The value or default.
        """
        return self.metadata.get(key, default)
    
    def with_metadata(self, **kwargs: Any) -> RunContext:
        """Create a new context with additional metadata.
        
        This does not modify the current context.
        
        Args:
            **kwargs: Metadata to add.
            
        Returns:
            New RunContext with merged metadata.
        """
        new_metadata = {**self.metadata, **kwargs}
        # Create a copy with updated metadata
        # This works for subclasses too via __class__
        import copy
        new_ctx = copy.copy(self)
        new_ctx.metadata = new_metadata
        return new_ctx


# Default empty context for when none is provided
EMPTY_CONTEXT = RunContext()


__all__ = [
    "RunContext",
    "EMPTY_CONTEXT",
]
