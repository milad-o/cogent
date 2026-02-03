"""Context propagation between agents.

Demonstrates automatic context propagation when using agent.as_tool().
Context flows from orchestrator → specialist without explicit passing.
"""

import asyncio
from dataclasses import dataclass, field

from cogent import Agent, Observer, RunContext, tool


@dataclass
class UserContext(RunContext):
    """Context with user information."""

    user_id: str = ""
    user_name: str = ""
    permissions: list[str] = field(default_factory=list)
    
    def has_permission(self, perm: str) -> bool:
        return perm in self.permissions


@tool
def check_permissions(action: str, ctx: RunContext) -> str:
    """Check if user has permission for an action."""
    if isinstance(ctx, UserContext):
        if ctx.has_permission(action):
            result = f"✅ User '{ctx.user_name}' has '{action}' permission"
        else:
            result = f"❌ User '{ctx.user_name}' lacks '{action}' permission"
        
        if ctx.query:
            result += f"\n📝 Original query: {ctx.query}"
        
        return result
    return "No context available"


async def main() -> None:
    """Demonstrate context propagation through agent delegation."""
    observer = Observer(level="progress")

    # Specialist agent - uses tools
    specialist = Agent(
        name="PermissionChecker",
        model="gemini-2.5-flash",
        tools=[check_permissions],
        instructions="Check user permissions using the check_permissions tool.",
        observer=observer,
    )

    # Orchestrator agent - delegates to specialist
    orchestrator = Agent(
        name="Orchestrator",
        model="gemini-2.5-flash",
        tools=[specialist.as_tool()],  # Context flows automatically!
        instructions="Use PermissionChecker to answer permission questions.",
        observer=observer,
    )

    # Create context with user info
    ctx = UserContext(
        user_id="user_123",
        user_name="Alice",
        permissions=["read", "write"],
    )

    # Context flows: orchestrator → specialist.as_tool() → check_permissions
    result = await orchestrator.run("Can I delete files?", context=ctx)
    print(f"\n{result.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
