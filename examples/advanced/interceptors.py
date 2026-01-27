"""
Interceptors

Cogent's interceptor system for cross-cutting concerns:
- BudgetGuard for limiting model/tool calls
- Custom interceptors for logging, validation, security
- PIIShield for PII detection and redaction
- RateLimiter for controlling execution rate
- TokenLimiter for token budget control

Usage: uv run python examples/advanced/interceptors.py
"""

import asyncio

from cogent import Agent
from cogent.interceptors import (
    Auditor,
    BudgetGuard,
    ContentFilter,
    InterceptContext,
    Interceptor,
    InterceptResult,
    PIIAction,
    PIIShield,
    RateLimiter,
    TokenLimiter,
)
from cogent.tools.base import tool


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for '{query}': Found 3 articles."


@tool
def calculate(expression: str) -> str:
    """Calculate a math expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"


async def demo_budget_guard():
    """Limit model and tool calls with BudgetGuard."""
    print("\n" + "=" * 60)
    print("1. BudgetGuard - Cost Control")
    print("=" * 60)

    guard = BudgetGuard(
        model_calls=3,
        tool_calls=5,
    )

    agent = Agent(
        name="ResearchBot",
        model="gpt4",
        tools=[search, calculate, get_weather],
        instructions="Help with research tasks. Use tools when helpful.",
        intercept=[guard],
    )

    print("\nRunning task with budget limits...")
    result = await agent.run("Search for Python tips, calculate 2+2, get weather in NYC")

    text = result.unwrap()
    print(f"\nResult: {str(text)[:200]}...")
    print("\nBudget usage:")
    print(f"  Model calls: {guard.current_model_calls}/{guard.model_calls}")
    print(f"  Tool calls: {guard.current_tool_calls}/{guard.tool_calls}")


class LoggingInterceptor(Interceptor):
    """Log all agent interactions."""

    async def before_turn(self, ctx: InterceptContext) -> InterceptResult:
        print(f"  📝 Starting turn with message: {ctx.messages[-1].content[:50]}...")
        return InterceptResult()

    async def after_turn(self, ctx: InterceptContext) -> InterceptResult:
        if ctx.response:
            content = str(ctx.response.content)[:50]
            print(f"  ✅ Turn complete: {content}...")
        return InterceptResult()


async def demo_logging_interceptor():
    """Custom logging interceptor."""
    print("\n" + "=" * 60)
    print("2. Custom Logging Interceptor")
    print("=" * 60)

    agent = Agent(
        name="LoggedBot",
        model="gpt4",
        instructions="Be helpful.",
        intercept=[LoggingInterceptor()],
    )

    print("\nRunning with logging...")
    await agent.run("What is 2+2?")
    await agent.run("Tell me a joke")


class ValidationInterceptor(Interceptor):
    """Validate input before processing."""

    async def before_turn(self, ctx: InterceptContext) -> InterceptResult:
        last_msg = ctx.messages[-1].content if ctx.messages else ""

        if "forbidden" in last_msg.lower():
            print("  ❌ Blocked: Message contains forbidden content")
            return InterceptResult(
                stop=True,
                response="I cannot process messages containing forbidden content."
            )

        return InterceptResult()


async def demo_validation_interceptor():
    """Validate and block certain inputs."""
    print("\n" + "=" * 60)
    print("3. Validation Interceptor")
    print("=" * 60)

    agent = Agent(
        name="ValidatedBot",
        model="gpt4",
        instructions="Be helpful.",
        intercept=[ValidationInterceptor()],
    )

    print("\nValid request...")
    result = await agent.run("What is Python?")
    print(f"Result: {result.unwrap()[:100]}...")

    print("\nBlocked request...")
    result = await agent.run("This contains forbidden content")
    print(f"Result: {result.unwrap()[:100]}...")


async def demo_pii_shield():
    """Detect and redact PII."""
    print("\n" + "=" * 60)
    print("4. PIIShield - PII Detection")
    print("=" * 60)

    shield = PIIShield(
        patterns=["email", "phone", "ssn"],
        action=PIIAction.MASK,
    )

    agent = Agent(
        name="SecureBot",
        model="gpt4",
        instructions="Be helpful.",
        intercept=[shield],
    )

    print("\nMessage with PII...")
    result = await agent.run("Contact me at john@example.com or 555-123-4567")
    print(f"\nResult: {result.unwrap()[:200]}...")


async def demo_rate_limiter():
    """Control execution rate."""
    print("\n" + "=" * 60)
    print("5. RateLimiter - Execution Control")
    print("=" * 60)

    limiter = RateLimiter(
        calls_per_window=3,
        window_seconds=10.0,
    )

    agent = Agent(
        name="RateLimitedBot",
        model="gpt4",
        instructions="Be concise.",
        intercept=[limiter],
    )

    print("\nExecuting multiple requests...")
    for i in range(3):
        result = await agent.run(f"Request {i+1}: What is {i}+{i}?")
        print(f"  Request {i+1}: {result.unwrap()[:50]}...")


async def demo_token_limiter():
    """Limit token usage per turn."""
    print("\n" + "=" * 60)
    print("6. TokenLimiter - Token Budget")
    print("=" * 60)

    limiter = TokenLimiter(
        max_tokens=100,
    )

    agent = Agent(
        name="TokenLimitedBot",
        model="gpt4",
        instructions="Be extremely concise.",
        intercept=[limiter],
    )

    print("\nRunning with token limits...")
    result = await agent.run("Explain quantum computing")
    print(f"\nResult: {result.unwrap()[:200]}...")


async def demo_content_filter():
    """Filter inappropriate content."""
    print("\n" + "=" * 60)
    print("7. ContentFilter - Content Moderation")
    print("=" * 60)

    content_filter = ContentFilter(
        blocked_patterns=["badword", "offensive"],
    )

    agent = Agent(
        name="FilteredBot",
        model="gpt4",
        instructions="Be helpful.",
        intercept=[content_filter],
    )

    print("\nClean message...")
    result = await agent.run("Hello, how are you?")
    print(f"Result: {result.unwrap()[:100]}...")

    print("\nBlocked message...")
    result = await agent.run("This contains badword")
    print(f"Result: {result.unwrap()[:100]}...")


async def demo_auditor():
    """Audit all agent interactions."""
    print("\n" + "=" * 60)
    print("8. Auditor - Interaction Logging")
    print("=" * 60)

    audit_log = []

    auditor = Auditor(
        callback=lambda event: audit_log.append(event),
    )

    agent = Agent(
        name="AuditedBot",
        model="gpt4",
        tools=[calculate],
        instructions="Use tools to answer.",
        intercept=[auditor],
    )

    print("\nRunning with audit logging...")
    await agent.run("Calculate 5 * 8")

    print(f"\nAudit log entries: {len(audit_log)}")
    for i, entry in enumerate(audit_log[:3], 1):
        print(f"  [{i}] {entry.event_type}: {entry.task[:50]}...")


async def demo_combined():
    """Combine multiple interceptors."""
    print("\n" + "=" * 60)
    print("9. Combined Interceptors")
    print("=" * 60)

    agent = Agent(
        name="SecureBot",
        model="gpt4",
        tools=[search, calculate],
        instructions="Be helpful and concise.",
        intercept=[
            BudgetGuard(model_calls=5, tool_calls=10),
            LoggingInterceptor(),
            ValidationInterceptor(),
        ],
    )

    print("\nRunning with multiple interceptors...")
    result = await agent.run("Search for AI news and calculate 7+3")
    print(f"\nResult: {result.unwrap()[:200]}...")


async def main():
    """Run all interceptor examples."""
    print("\n" + "=" * 60)
    print("AGENTICFLOW INTERCEPTORS EXAMPLES")
    print("=" * 60)

    await demo_budget_guard()
    await demo_logging_interceptor()
    await demo_validation_interceptor()
    await demo_pii_shield()
    await demo_rate_limiter()
    await demo_token_limiter()
    await demo_content_filter()
    await demo_auditor()
    await demo_combined()

    print("\n" + "=" * 60)
    print("✓ All demos completed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
