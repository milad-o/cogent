"""Context propagation between agents.

Demonstrates automatic context propagation when using agent.as_tool().
Shows delegation depth tracking pattern to prevent infinite loops.
"""

import asyncio
from dataclasses import dataclass, field

from cogent import Agent, Observer, RunContext, tool
from cogent.core import generate_id


@dataclass
class DelegationContext(RunContext):
    """Context with delegation depth tracking."""

    user_id: str = ""
    user_name: str = ""
    permissions: list[str] = field(default_factory=list)
    
    # Delegation tracking
    depth: int = 0
    max_depth: int = 3
    task_id: str = field(default_factory=generate_id)
    parent_task_id: str | None = None
    
    def has_permission(self, perm: str) -> bool:
        return perm in self.permissions
    
    @property
    def can_delegate(self) -> bool:
        return self.depth < self.max_depth
    
    def create_child_context(self) -> "DelegationContext":
        """Create context for delegated task."""
        return DelegationContext(
            query=self.query,
            user_id=self.user_id,
            user_name=self.user_name,
            permissions=self.permissions,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            task_id=generate_id(),
            parent_task_id=self.task_id,
            metadata=self.metadata,
        )


@tool
def check_permissions(action: str, ctx: RunContext) -> str:
    """Check if user has permission for an action."""
    if isinstance(ctx, DelegationContext):
        if not ctx.can_delegate:
            return f"Max delegation depth ({ctx.max_depth}) reached"
        
        if ctx.has_permission(action):
            result = f"User {ctx.user_name} has '{action}' permission"
        else:
            result = f"User {ctx.user_name} lacks '{action}' permission"
        
        result += f"\nDepth: {ctx.depth}, Task: {ctx.task_id[:8]}"
        
        if ctx.query:
            result += f"\nOriginal query: {ctx.query}"
        
        return result
    return "No context available"


async def main() -> None:
    """Demonstrate context propagation with delegation tracking."""
    observer = Observer(level="progress")

    specialist = Agent(
        name="PermissionChecker",
        model="gemini-2.5-flash",
        tools=[check_permissions],
        instructions="Check user permissions.",
        observer=observer,
    )

    orchestrator = Agent(
        name="Orchestrator",
        model="gemini-2.5-flash",
        tools=[specialist.as_tool()],
        instructions="Use PermissionChecker to answer permission questions.",
        observer=observer,
        memory=True,  # ADD MEMORY
    )

    ctx = DelegationContext(
        user_id="user_123",
        user_name="Alice",
        permissions=["read", "write"],
        depth=0,
        max_depth=3,
    )

    # First question
    result1 = await orchestrator.run("Can I delete files?", context=ctx, thread_id="session1")
    print(f"\nResult 1: {result1.content}\n")
    
    # Second question - orchestrator remembers first question, specialist doesn't
    result2 = await orchestrator.run("What about the previous action?", context=ctx, thread_id="session1")
    print(f"\nResult 2: {result2.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
