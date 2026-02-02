"""Context Hand-off Demo: Sharing context between agents in multi-agent workflows.

This demonstrates THREE approaches for context hand-off:
1. AUTO-PROPAGATION: Using propagate_context=True in as_tool() (NEW!)
2. MANUAL WRAPPER: Creating custom context-aware tool wrappers
3. STRUCTURED OUTPUT: Using typed data for explicit hand-off

Usage:
    uv run python examples/advanced/context_handoff_demo.py
"""

import asyncio
from dataclasses import dataclass, field

from cogent import Agent, Observer, RunContext, tool
from cogent.tools.base import BaseTool


# =============================================================================
# Typed RunContext
# =============================================================================


@dataclass
class UserContext(RunContext):
    """Application-specific context with user info."""

    user_id: str = ""
    user_name: str = ""
    permissions: list[str] = field(default_factory=list)

    def has_permission(self, perm: str) -> bool:
        """Check if user has a specific permission."""
        return perm in self.permissions


# =============================================================================
# Tools that use context
# =============================================================================


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
        if ctx.has_permission(action):
            return f"[OK] User {ctx.user_name} has '{action}' permission"
        return f"[NO] User {ctx.user_name} lacks '{action}' permission"
    return "No user context available"


# =============================================================================
# Option 1: Auto-Propagation with propagate_context=True
# =============================================================================


async def demo_option1_auto_propagation():
    """Option 1: Using propagate_context=True (EASIEST!)."""
    print("=" * 70)
    print("OPTION 1: Auto-Propagation (propagate_context=True)")
    print("=" * 70)
    print("NEW FEATURE: Context automatically flows to child agent\n")

    observer = Observer(level="trace")

    # Specialist agent that uses context-aware tools
    specialist = Agent(
        name="PermissionChecker",
        model="gemini-2.5-flash",
        tools=[get_user_profile, check_permissions],
        instructions="Check user permissions and report what they can do.",
        observer=observer,
    )

    # Orchestrator delegates to specialist WITH context propagation
    orchestrator = Agent(
        name="Orchestrator",
        model="gemini-2.5-flash",
        tools=[specialist.as_tool(propagate_context=True)],  # 🔑 Simple! No manual config needed
        instructions="Use the PermissionChecker tool to answer questions about user permissions.",
        observer=observer,
    )

    ctx = UserContext(
        user_id="user_123",
        user_name="Alice",
        permissions=["read", "write"],
    )

    print(f"Context: {ctx.user_name} with permissions: {ctx.permissions}")
    print()

    result = await orchestrator.run(
        "Can I delete files?",
        context=ctx,
        max_iterations=3,
    )

    print(f"\nResult:\n{result.content}")
    print("\n[SUCCESS] Context automatically propagated to specialist!\n")


# =============================================================================
# Option 2: Manual Context-Aware Wrapper
# =============================================================================


async def demo_option2_manual_wrapper():
    """Option 2: Manual wrapper with closure over context."""
    print("=" * 70)
    print("OPTION 2: Manual Context-Aware Wrapper")
    print("=" * 70)
    print("Full control: Create custom wrapper with captured context\n")

    # Specialist agent
    specialist = Agent(
        name="PermissionChecker",
        model="gemini-2.0-flash",
        tools=[get_user_profile, check_permissions],
        instructions="Check user permissions and report what they can do.",
    )

    ctx = UserContext(
        user_id="user_456",
        user_name="Bob",
        permissions=["read"],
    )

    # Create a custom tool that closes over the context
    def create_context_aware_tool(agent: Agent, context: RunContext) -> BaseTool:
        """Create an agent tool that captures context in closure."""
        from pydantic import BaseModel, Field

        class AgentToolInput(BaseModel):
            task: str = Field(description="Task for the agent")

        async def execute_with_context(task: str) -> str:
            """Execute agent with the captured context."""
            response = await agent.run(task, context=context)
            return str(response.content) if response.content else ""

        return BaseTool(
            name=f"{agent.name}_with_context",
            description=f"Execute task using {agent.name} (with user context)",
            func=execute_with_context,
            args_schema=AgentToolInput,
        )

    # Orchestrator uses custom wrapper
    orchestrator = Agent(
        name="Orchestrator",
        model="gemini-2.0-flash",
        tools=[create_context_aware_tool(specialist, ctx)],
        instructions="Use the permission checker to analyze user capabilities.",
    )

    print(f"Context: {ctx.user_name} with permissions: {ctx.permissions}")
    print()

    result = await orchestrator.run(
        "Check what permissions I have and whether I can write files",
    )

    print(f"\nResult:\n{result.content}")
    print("\nContext manually passed via closure!\n")


# =============================================================================
# Option 3: Structured Output as Explicit Hand-off
# =============================================================================


async def demo_option3_structured_output():
    """Option 3: Using structured output for explicit data passing."""
    print("=" * 70)
    print("OPTION 3: Structured Output as Explicit Hand-off")
    print("=" * 70)
    print("Type-safe: Pass data explicitly via structured outputs\n")

    from pydantic import BaseModel, Field

    class UserInfo(BaseModel):
        """User information to pass between agents."""

        user_id: str
        user_name: str
        permissions: list[str]

    class PermissionAnalysis(BaseModel):
        """Analysis of user permissions."""

        can_read: bool
        can_write: bool
        can_delete: bool
        summary: str

    # Agent 1: Extract user info from context
    info_extractor = Agent(
        name="InfoExtractor",
        model="gemini-2.0-flash",
        output=UserInfo,
        tools=[get_user_profile, check_permissions],
        instructions="""Extract user information using the tools.
        Return structured UserInfo with id, name, and permissions list.""",
    )

    # Agent 2: Analyze the structured data
    analyzer = Agent(
        name="PermissionAnalyzer",
        model="gemini-2.0-flash",
        output=PermissionAnalysis,
        instructions="""Analyze user permissions from the provided UserInfo.
        Check for read, write, and delete permissions.
        Provide a clear summary of what the user can and cannot do.""",
    )

    # Orchestrator coordinates with explicit data passing
    orchestrator = Agent(
        name="Orchestrator",
        model="gemini-2.0-flash",
        tools=[
            info_extractor.as_tool(
                name="extract_user_info",
                description="Extract user information from context",
                return_full_response=True,
            ),
            analyzer.as_tool(
                name="analyze_permissions",
                description="Analyze user permissions from structured UserInfo",
            ),
        ],
        instructions="""
        1. Use extract_user_info to get user data from context
        2. Extract the UserInfo from the response content
        3. Pass it to analyze_permissions: "Analyze permissions for user: {json of UserInfo}"
        4. Return the analysis
        """,
    )

    ctx = UserContext(
        user_id="user_789",
        user_name="Charlie",
        permissions=["read", "write", "delete"],
    )

    print(f"Context: {ctx.user_name} with permissions: {ctx.permissions}")
    print()

    result = await orchestrator.run(
        "Extract my user info and analyze what I can do",
        context=ctx,
    )

    print(f"\nResult:\n{result.content}")
    print("\nData explicitly passed via structured output!\n")


# =============================================================================
# Comparison Demo
# =============================================================================


async def demo_comparison():
    """Side-by-side comparison of all three approaches."""
    print("=" * 70)
    print("COMPARISON: Which Approach to Use?")
    print("=" * 70)
    print()
    print("+---------------------+-------------+------------+--------------+")
    print("| Approach            | Ease of Use | Type Safety| Use Case     |")
    print("+---------------------+-------------+------------+--------------+")
    print("| 1. Auto-Propagation | *****       | ***        | Same user    |")
    print("|                     | (Easiest)   |            | context      |")
    print("+---------------------+-------------+------------+--------------+")
    print("| 2. Manual Wrapper   | ***         | ****       | Custom logic |")
    print("|                     | (Medium)    |            | per agent    |")
    print("+---------------------+-------------+------------+--------------+")
    print("| 3. Structured Output| **          | *****      | Explicit     |")
    print("|                     | (Complex)   | (Safest)   | data passing |")
    print("+---------------------+-------------+------------+--------------+")
    print()
    print("Recommendations:")
    print("  - Use Option 1 when: Child agent needs same user context (DB, auth)")
    print("  - Use Option 2 when: Need custom context transformation per agent")
    print("  - Use Option 3 when: Want explicit, type-safe data contracts")
    print()


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Run all demos."""
    print("\n" + "=" * 70)
    print("CONTEXT HAND-OFF: Three Approaches")
    print("=" * 70)
    print()

    await demo_option1_auto_propagation()
    await demo_option2_manual_wrapper()
    await demo_option3_structured_output()
    await demo_comparison()

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("NEW: propagate_context=True makes context sharing easy!")
    print("Manual wrappers give full control over context flow")
    print("Structured output ensures type-safe, explicit hand-offs")
    print()
    print("Choose the approach that best fits your use case!")
    print()


if __name__ == "__main__":
    asyncio.run(main())
