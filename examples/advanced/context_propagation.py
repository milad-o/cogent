"""Context propagation between agents.

Demonstrates automatic context propagation when using agent.as_tool().
Context flows automatically (just like with regular tools), enabling
sub-agents to access user identity, permissions, and the original query.

Also shows common patterns:
- Delegation depth tracking to prevent infinite loops
- Task lineage tracking for debugging
- Using query field for original user intent
"""

import asyncio
from dataclasses import dataclass, field

from cogent import Agent, Observer, RunContext, tool
from cogent.core import generate_id


# Enhanced context with common patterns
@dataclass
class UserContext(RunContext):
    """Application-specific context with user info and execution tracking."""

    # User information
    user_id: str = ""
    user_name: str = ""
    permissions: list[str] = field(default_factory=list)
    
    # Delegation tracking (common pattern)
    depth: int = 0
    max_depth: int = 3
    task_id: str = field(default_factory=generate_id)
    parent_task_id: str | None = None
    
    def has_permission(self, perm: str) -> bool:
        """Check if user has a specific permission."""
        return perm in self.permissions
    
    @property
    def can_delegate(self) -> bool:
        """Check if we can delegate further."""
        return self.depth < self.max_depth
    
    def create_child_context(self) -> "UserContext":
        """Create context for delegated task."""
        return UserContext(
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


# Context-aware tools
@tool
def get_user_profile(ctx: RunContext) -> str:
    """Get the current user's profile information."""
    if isinstance(ctx, UserContext):
        info = f"User: {ctx.user_name} (ID: {ctx.user_id})"
        info += f"\nTask: {ctx.task_id}"
        if ctx.parent_task_id:
            info += f"\nParent Task: {ctx.parent_task_id}"
        info += f"\nDepth: {ctx.depth}/{ctx.max_depth}"
        return info
    return "No user context available"


@tool
def check_permissions(action: str, ctx: RunContext) -> str:
    """Check if user has permission for an action."""
    if isinstance(ctx, UserContext):
        result = ""
        
        # Check delegation depth
        if not ctx.can_delegate:
            result += f"[DEPTH] Maximum delegation depth ({ctx.max_depth}) reached\n"
        
        # Check permissions
        if ctx.has_permission(action):
            result += f"✅ User {ctx.user_name} has '{action}' permission"
        else:
            result += f"❌ User {ctx.user_name} lacks '{action}' permission"
        
        # Show execution context
        result += f"\n📊 Execution depth: {ctx.depth}"
        result += f"\n🆔 Task ID: {ctx.task_id[:8]}..."
        
        # Show original user query for context
        if ctx.query:
            result += f"\n💭 Original query: '{ctx.query}'"
        
        return result
    return "No user context available"


@tool
def show_task_lineage(ctx: RunContext) -> str:
    """Show the task execution lineage."""
    if isinstance(ctx, UserContext):
        info = f"Task Lineage:\n"
        info += f"  Current Task: {ctx.task_id}\n"
        info += f"  Parent Task: {ctx.parent_task_id or 'None (root task)'}\n"
        info += f"  Depth: {ctx.depth}\n"
        info += f"  Can Delegate: {ctx.can_delegate}\n"
        info += f"  Original Query: {ctx.query or 'Not set'}\n"
        return info
    return "No context available"


async def demo_context_propagation():
    """Demonstrate automatic context propagation with execution tracking."""
    print("\n=== Context Propagation with Execution Tracking ===\n")

    observer = Observer(level="progress")

    # Specialist agent with context-aware tools
    specialist = Agent(
        name="PermissionChecker",
        model="gemini-2.5-flash",
        tools=[get_user_profile, check_permissions, show_task_lineage],
        instructions="""Check user permissions and provide detailed information.
        You can use show_task_lineage to see execution depth and task hierarchy.""",
        observer=observer,
    )

    # Orchestrator that delegates to specialist
    orchestrator = Agent(
        name="Orchestrator",
        model="gemini-2.5-flash",
        tools=[specialist.as_tool()],  # Context propagates automatically!
        instructions="""Use the PermissionChecker tool to answer questions about user permissions.
        The context (user info, task lineage, original query) flows automatically.""",
        observer=observer,
    )

    # Create rich context with tracking
    ctx = UserContext(
        user_id="user_123",
        user_name="Alice",
        permissions=["read", "write"],
        depth=0,  # Top-level task
        max_depth=3,
    )

    print(f"👤 User: {ctx.user_name}")
    print(f"🔑 Permissions: {ctx.permissions}")
    print(f"📝 Task ID: {ctx.task_id}")
    print(f"🔢 Starting depth: {ctx.depth}/{ctx.max_depth}")
    print()

    # Run orchestrator - context flows through entire delegation chain
    result = await orchestrator.run(
        "Can I delete files?",  # Original query - accessible via ctx.query
        context=ctx,
        max_iterations=5,
    )

    print(f"\n{'='*60}")
    print(f"📤 Final Result:\n{result.content}")
    print(f"{'='*60}\n")


async def main() -> None:
    """Demonstrate context propagation with common patterns."""
    print("\n" + "="*60)
    print("Context Propagation with Execution Tracking Demo")
    print("="*60)
    print("\nDemonstrates:")
    print("  ✓ Automatic context propagation via agent.as_tool()")
    print("  ✓ Delegation depth tracking (prevents infinite loops)")
    print("  ✓ Task lineage (parent-child relationships)")
    print("  ✓ Original query tracking via ctx.query")
    print("  ✓ User permissions flowing through delegation chain")

    await demo_context_propagation()

    print("\n" + "="*60)
    print("💡 Key Takeaways:")
    print("  • Context flows automatically (like regular tools)")
    print("  • Sub-agents access original query via ctx.query")
    print("  • Track delegation depth to prevent loops")
    print("  • Use task IDs for lineage tracking")
    print("  • Extend RunContext for your specific needs")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
