"""
Context Layer

Cogent's Context Layer for runtime injection and dynamic behavior control:
- RunContext: Pass invocation-scoped data to tools/interceptors
- ToolGate: Dynamic tool filtering based on permissions/stage
- ToolGuard/CircuitBreaker: Tool retry and failure protection
- PromptAdapter: Dynamic system prompt modification

Usage: uv run python examples/advanced/context_layer.py
"""

import asyncio
from dataclasses import dataclass, field

from cogent import Agent, RunContext
from cogent.interceptors import (
    ContextPrompt,
    ConversationGate,
    InterceptContext,
    LambdaPrompt,
    PermissionGate,
    ToolGuard,
)
from cogent.tools.base import tool


@dataclass
class AppContext(RunContext):
    """Application-specific context passed to agent.run()."""
    user_id: str = ""
    user_role: str = "guest"
    department: str = ""
    metadata: dict = field(default_factory=dict)


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for '{query}': Found 3 articles."


@tool
def read_data(table: str) -> str:
    """Read data from a database table."""
    return f"Data from {table}: [row1, row2, row3]"


@tool
def write_data(table: str, data: str) -> str:
    """Write data to a database table."""
    return f"Wrote '{data}' to {table}"


@tool
def admin_delete(table: str) -> str:
    """Delete all data from a table (admin only!)."""
    return f"Deleted all data from {table}"


@tool
def get_user_profile(ctx: RunContext) -> str:
    """Get the current user's profile. Uses RunContext for user info."""
    if isinstance(ctx, AppContext):
        return f"Profile for {ctx.user_id}: role={ctx.user_role}, dept={ctx.department}"
    return "No user context available"


async def demo_run_context():
    """Pass data to tools via RunContext."""
    print("\n" + "=" * 60)
    print("1. RunContext - Pass Data to Tools")
    print("=" * 60)

    agent = Agent(
        name="ProfileBot",
        model="gpt4",
        tools=[get_user_profile],
        instructions="Help users with their profiles.",
    )

    ctx = AppContext(user_id="alice123", user_role="admin", department="Engineering")

    print(f"\nContext: {ctx.user_id} ({ctx.user_role})")
    result = await agent.run("What's my profile?", context=ctx)
    print(f"Result: {result.unwrap()[:200]}...")


class RoleBasedGate(PermissionGate):
    """Filter tools based on user role."""

    ROLE_PERMISSIONS = {
        "admin": ["*"],  # All tools
        "user": ["search", "read_data"],
        "guest": ["search"],
    }

    async def check(self, tool_name: str, ctx: InterceptContext) -> bool:
        role = "guest"
        if ctx.run_context and isinstance(ctx.run_context, AppContext):
            role = ctx.run_context.user_role

        allowed = self.ROLE_PERMISSIONS.get(role, [])
        is_allowed = "*" in allowed or tool_name in allowed
        print(f"  🔒 {tool_name}: {'✓ allowed' if is_allowed else '✗ denied'} for {role}")
        return is_allowed


async def demo_permission_gate():
    """Filter tools by user role."""
    print("\n" + "=" * 60)
    print("2. PermissionGate - Role-Based Tool Access")
    print("=" * 60)

    agent = Agent(
        name="SecureBot",
        model="gpt4",
        tools=[search, read_data, write_data, admin_delete],
        instructions="Help with data tasks using available tools.",
        intercept=[RoleBasedGate()],
    )

    print("\n--- Guest user (limited access) ---")
    guest_ctx = AppContext(user_id="guest1", user_role="guest")
    result = await agent.run("Search for database info", context=guest_ctx)
    print(f"Result: {result.unwrap()[:150]}...")

    print("\n--- Admin user (full access) ---")
    admin_ctx = AppContext(user_id="admin1", user_role="admin")
    result = await agent.run("Delete test_table", context=admin_ctx)
    print(f"Result: {result.unwrap()[:150]}...")


async def demo_conversation_gate():
    """Unlock more tools as conversation progresses."""
    print("\n" + "=" * 60)
    print("3. ConversationGate - Progressive Tool Unlock")
    print("=" * 60)

    gate = ConversationGate(stages={
        0: ["search"],
        3: ["search", "read_data"],
        6: ["search", "read_data", "write_data"],
    })

    agent = Agent(
        name="ProgressiveBot",
        model="gpt4",
        tools=[search, read_data, write_data],
        instructions="Help with data tasks using available tools.",
        intercept=[gate],
    )

    print("\nFirst message (only search available)...")
    result = await agent.run("Search for database best practices")
    print(f"Result: {result.unwrap()[:150]}...")
    print("\nNote: More tools unlock as message history grows!")


_call_count = 0


@tool
def flaky_api(query: str) -> str:
    """A flaky API that sometimes fails."""
    global _call_count
    _call_count += 1
    if _call_count % 3 != 0:
        raise TimeoutError(f"API timeout on attempt {_call_count}")
    return f"API success for: {query}"


async def demo_tool_guard():
    """Automatic retry for failed tool calls."""
    print("\n" + "=" * 60)
    print("4. ToolGuard - Retry Failed Tools")
    print("=" * 60)

    global _call_count
    _call_count = 0

    guard = ToolGuard(
        max_retries=3,
        backoff=2.0,
        initial_delay=0.1,
        retry_on=[TimeoutError],
        on_retry=lambda name, attempt, err, delay: print(
            f"  ⚡ Retry {name} (attempt {attempt}): {err}"
        ),
    )

    agent = Agent(
        name="ResilientBot",
        model="gpt4",
        tools=[flaky_api],
        instructions="Use the flaky_api tool to answer queries.",
        intercept=[guard],
    )

    print("\nCalling flaky API (will retry on failures)...")
    result = await agent.run("Query the flaky API for weather data")
    print(f"\nResult: {result.unwrap()[:200]}...")


async def demo_lambda_prompt():
    """Quick prompt customization with lambda."""
    print("\n" + "=" * 60)
    print("5. LambdaPrompt - Simple Prompt Customization")
    print("=" * 60)

    adapter = LambdaPrompt(
        adapter_fn=lambda prompt, ctx: (
            f"{prompt}\n\n"
            f"Current user: {ctx.run_context.user_id if ctx.run_context else 'anonymous'}\n"
            f"Role: {getattr(ctx.run_context, 'user_role', 'guest')}"
        )
    )

    agent = Agent(
        name="PersonalizedBot",
        model="gpt4",
        instructions="You are a helpful assistant.",
        intercept=[adapter],
    )

    ctx = AppContext(user_id="alice", user_role="admin")

    print("\nPrompt will include user context...")
    result = await agent.run("Hello, who am I?", context=ctx)
    print(f"\nResult: {result.unwrap()[:300]}...")


async def demo_context_prompt():
    """Inject RunContext data via template."""
    print("\n" + "=" * 60)
    print("6. ContextPrompt - Template-Based Injection")
    print("=" * 60)

    adapter = ContextPrompt(
        template="User Details:\n- ID: {user_id}\n- Department: {department}\n- Role: {user_role}",
    )

    agent = Agent(
        name="TemplateBot",
        model="gpt4",
        instructions="Help users with their department-specific needs.",
        intercept=[adapter],
    )

    ctx = AppContext(user_id="bob456", user_role="user", department="Sales")

    print(f"\nInjecting context: {ctx.user_id} ({ctx.user_role})")
    result = await agent.run("What resources are available for my team?", context=ctx)
    print(f"\nResult: {result.unwrap()[:300]}...")


async def demo_combined():
    """Combine multiple context layer features."""
    print("\n" + "=" * 60)
    print("7. Combined Context Layer")
    print("=" * 60)

    agent = Agent(
        name="EnterpriseBot",
        model="gpt4",
        tools=[search, read_data, write_data],
        instructions="Help enterprise users with their tasks.",
        intercept=[
            RoleBasedGate(),
            LambdaPrompt(
                adapter_fn=lambda p, ctx: (
                    f"{p}\n\nUser: {getattr(ctx.run_context, 'user_id', 'unknown')}"
                )
            ),
        ],
    )

    ctx = AppContext(user_id="enterprise_user", user_role="user", department="Operations")

    print(f"\nEnterprise context: {ctx.user_id} ({ctx.user_role})")
    result = await agent.run("Search for operational best practices", context=ctx)
    print(f"\nResult: {result.unwrap()[:300]}...")


async def main():
    """Run all context layer examples."""
    print("\n" + "=" * 60)
    print("AGENTICFLOW CONTEXT LAYER EXAMPLES")
    print("=" * 60)

    await demo_run_context()
    await demo_permission_gate()
    await demo_conversation_gate()
    await demo_tool_guard()
    await demo_lambda_prompt()
    await demo_context_prompt()
    await demo_combined()

    print("\n" + "=" * 60)
    print("✓ All examples completed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
