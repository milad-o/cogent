"""
Example 26: Context Layer

Demonstrates AgenticFlow's Context Layer for runtime injection and
dynamic behavior control.

Key features:
- RunContext: Pass invocation-scoped data to tools/interceptors
- ToolGate: Dynamic tool filtering based on permissions/stage
- Failover: Automatic model fallback on errors
- ToolGuard/CircuitBreaker: Tool retry and failure protection
- PromptAdapter: Dynamic system prompt modification

Run: uv run python examples/26_context_layer.py
"""

import asyncio
from dataclasses import dataclass, field

from config import get_model

from agenticflow import Agent, RunContext
from agenticflow.tools.base import tool
from agenticflow.interceptors import (
    InterceptContext,
    InterceptResult,
    ToolGate,
    PermissionGate,
    ConversationGate,
    ToolGuard,
    CircuitBreaker,
    PromptAdapter,
    ContextPrompt,
    ConversationPrompt,
    LambdaPrompt,
)


# =============================================================================
# Custom RunContext for our app
# =============================================================================

@dataclass
class AppContext(RunContext):
    """Application-specific context passed to agent.run()."""
    user_id: str = ""
    user_role: str = "guest"  # guest, user, admin
    session_id: str = ""
    department: str = ""
    
    # For accumulating metadata
    metadata: dict = field(default_factory=dict)


# =============================================================================
# Tools for demonstration
# =============================================================================

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for '{query}': Found 3 relevant articles."


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
def send_email(to: str, subject: str) -> str:
    """Send an email."""
    return f"Sent email to {to}: {subject}"


@tool
def generate_report(report_type: str) -> str:
    """Generate a detailed report."""
    return f"Generated {report_type} report with 50 pages of analysis."


# =============================================================================
# Example 1: RunContext - Pass Data to Tools
# =============================================================================

@tool
def get_user_profile(ctx: RunContext) -> str:
    """Get the current user's profile. Uses RunContext for user info."""
    # The tool receives the RunContext automatically!
    if isinstance(ctx, AppContext):
        return f"Profile for {ctx.user_id}: role={ctx.user_role}, dept={ctx.department}"
    return "No user context available"


@tool
def log_action(action: str, ctx: RunContext) -> str:
    """Log an action with user context."""
    if isinstance(ctx, AppContext):
        return f"Logged: {action} by user {ctx.user_id} (session: {ctx.session_id})"
    return f"Logged: {action} (no user context)"


async def example_run_context():
    """Pass context data to tools via RunContext."""
    print("=" * 60)
    print("Example 1: RunContext - Pass Data to Tools")
    print("=" * 60)
    
    model = get_model()
    
    agent = Agent(
        name="ContextAwareBot",
        model=model,
        tools=[get_user_profile, log_action, search],
        instructions="Help users with their requests. Use get_user_profile to personalize.",
    )
    
    # Create context for this invocation
    ctx = AppContext(
        user_id="alice123",
        user_role="admin",
        session_id="sess-abc-123",
        department="Engineering",
    )
    
    print(f"\nRunning with context: user={ctx.user_id}, role={ctx.user_role}")
    result = await agent.run(
        "What's my profile? Then log that I viewed it.",
        context=ctx,
    )
    
    print(f"\nResult: {result[:300]}...")


# =============================================================================
# Example 2: PermissionGate - Role-Based Tool Access
# =============================================================================

class RoleBasedGate(PermissionGate):
    """Filter tools based on user role from RunContext."""
    
    ROLE_PERMISSIONS = {
        "guest": ["search"],
        "user": ["search", "read_data", "send_email"],
        "admin": ["*"],  # All tools
    }
    
    def allowed_tools(self, ctx: InterceptContext) -> list[str]:
        if ctx.run_context and isinstance(ctx.run_context, AppContext):
            role = ctx.run_context.user_role
            return self.ROLE_PERMISSIONS.get(role, ["search"])
        return ["search"]  # Default for no context


async def example_permission_gate():
    """Filter tools based on user role."""
    print("\n" + "=" * 60)
    print("Example 2: PermissionGate - Role-Based Tool Access")
    print("=" * 60)
    
    model = get_model()
    
    agent = Agent(
        name="SecureBot",
        model=model,
        tools=[search, read_data, write_data, admin_delete],
        instructions="Help with data tasks. Use available tools.",
        intercept=[RoleBasedGate()],
    )
    
    # Guest user - limited access
    print("\n--- Guest user (limited tools) ---")
    guest_ctx = AppContext(user_id="guest1", user_role="guest")
    result = await agent.run(
        "Search for Python tutorials",
        context=guest_ctx,
    )
    print(f"Result: {result[:200]}...")
    
    # Admin user - full access
    print("\n--- Admin user (all tools) ---")
    admin_ctx = AppContext(user_id="admin1", user_role="admin")
    result = await agent.run(
        "Read data from the users table",
        context=admin_ctx,
    )
    print(f"Result: {result[:200]}...")


# =============================================================================
# Example 3: ConversationGate - Progressive Tool Unlock
# =============================================================================

async def example_conversation_gate():
    """Unlock more tools as conversation progresses."""
    print("\n" + "=" * 60)
    print("Example 3: ConversationGate - Progressive Tool Unlock")
    print("=" * 60)
    
    model = get_model()
    
    # Start with basic tools, unlock more as conversation grows
    gate = ConversationGate(stages={
        0: ["search"],                    # Start: only search
        3: ["search", "read_data"],       # After 3 messages: add read
        6: ["search", "read_data", "write_data"],  # After 6: add write
    })
    
    agent = Agent(
        name="ProgressiveBot",
        model=model,
        tools=[search, read_data, write_data],
        instructions="Help with data tasks using available tools.",
        intercept=[gate],
    )
    
    print("\n--- First message (only search available) ---")
    result = await agent.run("Search for database best practices")
    print(f"Result: {result[:150]}...")
    
    print("\nNote: More tools unlock as message history grows!")


# =============================================================================
# Example 4: Custom ToolGate - Department-Based Access
# =============================================================================

class DepartmentGate(ToolGate):
    """Filter tools based on department."""
    
    DEPT_TOOLS = {
        "Engineering": ["search", "read_data", "write_data"],
        "Marketing": ["search", "send_email", "generate_report"],
        "HR": ["search", "read_data"],
    }
    
    async def filter(self, tools, ctx: InterceptContext):
        dept = "Engineering"  # Default
        if ctx.run_context and isinstance(ctx.run_context, AppContext):
            dept = ctx.run_context.department or "Engineering"
        
        allowed = self.DEPT_TOOLS.get(dept, ["search"])
        print(f"  🏢 Department '{dept}' has access to: {allowed}")
        
        return [t for t in tools if t.name in allowed or "*" in allowed]


async def example_department_gate():
    """Filter tools by department."""
    print("\n" + "=" * 60)
    print("Example 4: Custom ToolGate - Department-Based Access")
    print("=" * 60)
    
    model = get_model()
    
    agent = Agent(
        name="DeptBot",
        model=model,
        tools=[search, read_data, write_data, send_email, generate_report],
        instructions="Help with department tasks.",
        intercept=[DepartmentGate()],
    )
    
    # Marketing department
    print("\n--- Marketing department ---")
    mkt_ctx = AppContext(user_id="mark1", department="Marketing")
    result = await agent.run(
        "Generate a sales report",
        context=mkt_ctx,
    )
    print(f"Result: {result[:150]}...")
    
    # Engineering department  
    print("\n--- Engineering department ---")
    eng_ctx = AppContext(user_id="eng1", department="Engineering")
    result = await agent.run(
        "Read data from the metrics table",
        context=eng_ctx,
    )
    print(f"Result: {result[:150]}...")


# =============================================================================
# Example 5: ToolGuard - Retry Failed Tools
# =============================================================================

# Simulated flaky tool
_call_count = 0

@tool
def flaky_api(query: str) -> str:
    """A flaky API that sometimes fails."""
    global _call_count
    _call_count += 1
    if _call_count % 3 != 0:  # Fail 2 out of 3 times
        raise TimeoutError(f"API timeout on attempt {_call_count}")
    return f"API success for: {query}"


async def example_tool_guard():
    """Automatic retry for failed tool calls."""
    print("\n" + "=" * 60)
    print("Example 5: ToolGuard - Retry Failed Tools")
    print("=" * 60)
    
    model = get_model()
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
        model=model,
        tools=[flaky_api, search],
        instructions="Use the flaky_api tool to answer queries.",
        intercept=[guard],
    )
    
    print("\nCalling flaky API (will retry on failures)...")
    result = await agent.run("Query the flaky API for weather data")
    print(f"\nResult: {result[:200]}...")


# =============================================================================
# Example 6: CircuitBreaker - Prevent Cascading Failures
# =============================================================================

async def example_circuit_breaker():
    """Circuit breaker to protect against failing tools."""
    print("\n" + "=" * 60)
    print("Example 6: CircuitBreaker - Prevent Cascading Failures")
    print("=" * 60)
    
    model = get_model()
    
    breaker = CircuitBreaker(
        failure_threshold=3,   # Open after 3 failures
        reset_timeout=10.0,    # Try again after 10 seconds
        tools=["flaky_api"],   # Only protect this tool
    )
    
    agent = Agent(
        name="ProtectedBot",
        model=model,
        tools=[search],  # Using safe tools for demo
        instructions="Help with searches.",
        intercept=[breaker],
    )
    
    print("\nCircuit breaker configured:")
    print(f"  - Failure threshold: {breaker.failure_threshold}")
    print(f"  - Reset timeout: {breaker.reset_timeout}s")
    
    result = await agent.run("Search for circuit breaker patterns")
    print(f"\nResult: {result[:200]}...")


# =============================================================================
# Example 7: LambdaPrompt - Simple Prompt Customization
# =============================================================================

async def example_lambda_prompt():
    """Quick prompt customization with lambda."""
    print("\n" + "=" * 60)
    print("Example 7: LambdaPrompt - Simple Prompt Customization")
    print("=" * 60)
    
    model = get_model()
    
    # Add user name and role to every prompt
    adapter = LambdaPrompt(
        adapter_fn=lambda prompt, ctx: (
            f"{prompt}\n\n"
            f"Current user: {ctx.run_context.user_id if ctx.run_context else 'anonymous'}\n"
            f"Role: {getattr(ctx.run_context, 'user_role', 'guest')}"
        )
    )
    
    agent = Agent(
        name="PersonalizedBot",
        model=model,
        tools=[search],
        instructions="You are a helpful assistant.",
        intercept=[adapter],
    )
    
    ctx = AppContext(user_id="alice", user_role="admin")
    
    print("\nPrompt will include user context...")
    result = await agent.run("Hello, who am I?", context=ctx)
    print(f"\nResult: {result[:300]}...")


# =============================================================================
# Example 8: ContextPrompt - Template-Based Injection
# =============================================================================

async def example_context_prompt():
    """Inject RunContext data via template."""
    print("\n" + "=" * 60)
    print("Example 8: ContextPrompt - Template-Based Injection")
    print("=" * 60)
    
    model = get_model()
    
    adapter = ContextPrompt(
        template="User Details:\n- ID: {user_id}\n- Department: {department}\n- Role: {user_role}",
    )
    
    agent = Agent(
        name="TemplateBot",
        model=model,
        tools=[search],
        instructions="Help users with their department-specific needs.",
        intercept=[adapter],
    )
    
    ctx = AppContext(
        user_id="bob456",
        user_role="user",
        department="Sales",
    )
    
    print(f"\nInjecting context: {ctx}")
    result = await agent.run("What resources are available for my team?", context=ctx)
    print(f"\nResult: {result[:300]}...")


# =============================================================================
# Example 9: ConversationPrompt - Stage-Based Instructions
# =============================================================================

async def example_conversation_prompt():
    """Change instructions based on conversation length."""
    print("\n" + "=" * 60)
    print("Example 9: ConversationPrompt - Stage-Based Instructions")
    print("=" * 60)
    
    model = get_model()
    
    adapter = ConversationPrompt(
        stages={
            0: "This is a new conversation. Be welcoming and introduce yourself.",
            5: "The conversation is progressing. Be more detailed in responses.",
            10: "This is a long conversation. Be concise to avoid repetition.",
        }
    )
    
    agent = Agent(
        name="AdaptiveBot",
        model=model,
        instructions="You are a helpful assistant.",
        intercept=[adapter],
    )
    
    print("\nFirst message (short conversation)...")
    result = await agent.run("Hi there!")
    print(f"\nResult: {result[:200]}...")
    
    print("\nNote: Instructions change as conversation grows!")


# =============================================================================
# Example 10: Combined Context Layer
# =============================================================================

async def example_combined():
    """Combine multiple context layer features."""
    print("\n" + "=" * 60)
    print("Example 10: Combined Context Layer")
    print("=" * 60)
    
    model = get_model()
    
    # Create a fully-featured agent
    agent = Agent(
        name="EnterpriseBot",
        model=model,
        tools=[search, read_data, write_data, send_email, generate_report],
        instructions="Help enterprise users with their tasks.",
        intercept=[
            # 1. Permission-based tool filtering
            RoleBasedGate(),
            
            # 2. Personalized prompts
            LambdaPrompt(
                adapter_fn=lambda p, ctx: (
                    f"{p}\n\nUser: {getattr(ctx.run_context, 'user_id', 'unknown')}"
                )
            ),
            
            # 3. Conversation-aware instructions
            ConversationPrompt(stages={
                0: "Start with a greeting.",
                5: "Be more detailed.",
            }),
        ],
    )
    
    # Simulate enterprise user
    ctx = AppContext(
        user_id="enterprise_user",
        user_role="user",
        department="Operations",
        session_id="sess-xyz-789",
    )
    
    print(f"\nEnterprise context: {ctx.user_id} ({ctx.user_role})")
    result = await agent.run(
        "Search for operational best practices",
        context=ctx,
    )
    print(f"\nResult: {result[:300]}...")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all context layer examples."""
    print("\n🔌 AgenticFlow Context Layer Examples\n")
    
    await example_run_context()
    await example_permission_gate()
    await example_conversation_gate()
    await example_department_gate()
    await example_tool_guard()
    await example_circuit_breaker()
    await example_lambda_prompt()
    await example_context_prompt()
    await example_conversation_prompt()
    await example_combined()
    
    print("\n" + "=" * 60)
    print("✅ All context layer examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
