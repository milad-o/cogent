"""Context propagation between agents.

Demonstrates automatic context propagation when using agent.as_tool().
Context flows automatically (just like with regular tools), enabling
sub-agents to access user identity, permissions, and the original query.
"""

import asyncio
from dataclasses import dataclass, field

from cogent import Agent, Observer, RunContext, tool


# Context definition
@dataclass
class UserContext(RunContext):
    """Application-specific context with user info."""

    user_id: str = ""
    user_name: str = ""
    permissions: list[str] = field(default_factory=list)

    def has_permission(self, perm: str) -> bool:
        """Check if user has a specific permission."""
        return perm in self.permissions


# Context-aware tools
@tool
def get_user_profile(ctx: RunContext) -> str:
    """Get the current user's profile information."""
    if isinstance(ctx, UserContext):
        return f"User: {ctx.user_name} (ID: {ctx.user_id})"
    return "No user context available"


@tool
def check_permissions(action: str, ctx: RunContext) -> str:
    """Check if user has permission for an action."""
    if isinstance(ctx, UserContext):
        result = ""
        if ctx.has_permission(action):
            result = f"[OK] User {ctx.user_name} has '{action}' permission"
        else:
            result = f"[NO] User {ctx.user_name} lacks '{action}' permission"
        
        # Show original user query for context
        if ctx.query:
            result += f"\n(Original query: {ctx.query})"
        
        return result
    return "No user context available"


async def demo_context_propagation():
    """Demonstrate automatic context propagation."""
    print("\n=== Context Propagation Demo ===")

    observer = Observer(level="trace")

    specialist = Agent(
        name="PermissionChecker",
        model="gemini-2.5-flash",
        tools=[get_user_profile, check_permissions],
        instructions="Check user permissions and report what they can do.",
        observer=observer,
    )

    orchestrator = Agent(
        name="Orchestrator",
        model="gemini-2.5-flash",
        tools=[specialist.as_tool()],  # Context propagates automatically
        instructions="Use the PermissionChecker tool to answer questions about user permissions.",
        observer=observer,
    )

    ctx = UserContext(
        user_id="user_123",
        user_name="Alice",
        permissions=["read", "write"],
    )

    result = await orchestrator.run(
        "Can I delete files?",
        context=ctx,
        max_iterations=3,
    )

    print(f"Result: {result.content}")


async def main() -> None:
    """Demonstrate context propagation with propagate_context."""
    print("\nContext Propagation Demo")
    print("=" * 40)

    await demo_context_propagation()

    print()


if __name__ == "__main__":
    asyncio.run(main())
