"""
Example 29: Interceptors

Demonstrates AgenticFlow's interceptor system for cross-cutting concerns
like cost control, security, and observability.

Key features:
- Composable interceptors that hook into agent execution
- BudgetGuard for limiting model/tool calls
- Custom interceptors for logging, validation, etc.

Run: uv run python examples/advanced/interceptors.py
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataclasses import dataclass

from config import get_model

from agenticflow import Agent
from agenticflow.tools.base import tool
from agenticflow.interceptors import (
    Interceptor,
    InterceptContext,
    InterceptResult,
    Phase,
    BudgetGuard,
    PIIShield,
    PIIAction,
    TokenLimiter,
    ContentFilter,
    RateLimiter,
    Auditor,
)


# =============================================================================
# Tools for demonstration
# =============================================================================

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for '{query}': Found 3 relevant articles about the topic."


@tool
def calculate(expression: str) -> str:
    """Calculate a math expression."""
    try:
        result = eval(expression)  # Simple demo only!
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"


# =============================================================================
# Example 1: BudgetGuard - Cost Control
# =============================================================================

async def example_budget_guard():
    """Limit model and tool calls with BudgetGuard."""
    print("=" * 60)
    print("Example 1: BudgetGuard - Cost Control")
    print("=" * 60)
    
    model = get_model()
    
    # Create a guard that limits calls
    guard = BudgetGuard(
        model_calls=3,   # Max 3 LLM calls
        tool_calls=5,    # Max 5 tool calls
    )
    
    agent = Agent(
        name="ResearchBot",
        model=model,
        tools=[search, calculate, get_weather],
        instructions="Help with research tasks. Use tools when helpful.",
        intercept=[guard],  # Add the guard
    )
    
    # Run a task
    print("\nRunning task with budget limits...")
    result = await agent.run(
        "Search for Python tips, calculate 2+2, and get weather in NYC"
    )
    
    print(f"\nResult: {result[:200]}..." if len(result) > 200 else f"\nResult: {result}")
    print(f"\nBudget usage:")
    print(f"  Model calls: {guard.current_model_calls}/{guard.model_calls}")
    print(f"  Tool calls: {guard.current_tool_calls}/{guard.tool_calls}")


# =============================================================================
# Example 2: Custom Logging Interceptor
# =============================================================================

class LoggingInterceptor(Interceptor):
    """Log all agent execution phases."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.events: list[str] = []
    
    async def pre_run(self, ctx: InterceptContext) -> InterceptResult:
        self._log(f"🚀 Starting task: {ctx.task[:50]}...")
        return InterceptResult.ok()
    
    async def pre_think(self, ctx: InterceptContext) -> InterceptResult:
        self._log(f"🧠 Model call #{ctx.model_calls + 1}")
        return InterceptResult.ok()
    
    async def post_think(self, ctx: InterceptContext) -> InterceptResult:
        if self.verbose and ctx.model_response:
            content = getattr(ctx.model_response, 'content', '')
            if content:
                self._log(f"   Response: {content[:100]}...")
        return InterceptResult.ok()
    
    async def pre_act(self, ctx: InterceptContext) -> InterceptResult:
        self._log(f"🔧 Calling tool: {ctx.tool_name}({ctx.tool_args})")
        return InterceptResult.ok()
    
    async def post_act(self, ctx: InterceptContext) -> InterceptResult:
        if self.verbose:
            result = str(ctx.tool_result)[:100]
            self._log(f"   Result: {result}...")
        return InterceptResult.ok()
    
    async def post_run(self, ctx: InterceptContext) -> InterceptResult:
        self._log(f"✅ Completed! Model calls: {ctx.model_calls}, Tool calls: {ctx.tool_calls}")
        return InterceptResult.ok()
    
    def _log(self, msg: str) -> None:
        self.events.append(msg)
        print(f"  {msg}")


async def example_logging_interceptor():
    """Custom interceptor for logging."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Logging Interceptor")
    print("=" * 60)
    
    model = get_model()
    logger = LoggingInterceptor(verbose=True)
    
    agent = Agent(
        name="LoggedBot",
        model=model,
        tools=[search, calculate],
        instructions="Help with tasks concisely.",
        intercept=[logger],
    )
    
    print("\nExecution log:")
    await agent.run("What is 15 * 7?")
    
    print(f"\nTotal logged events: {len(logger.events)}")


# =============================================================================
# Example 3: Tool Argument Modifier
# =============================================================================

class QueryEnhancer(Interceptor):
    """Enhance search queries with context."""
    
    def __init__(self, context: str = "in 2024"):
        self.context = context
    
    async def pre_act(self, ctx: InterceptContext) -> InterceptResult:
        if ctx.tool_name == "search":
            original = ctx.tool_args.get("query", "")
            enhanced = f"{original} {self.context}"
            print(f"  📝 Enhanced query: '{original}' → '{enhanced}'")
            return InterceptResult.modify_args({"query": enhanced})
        return InterceptResult.ok()


async def example_query_enhancer():
    """Modify tool arguments on the fly."""
    print("\n" + "=" * 60)
    print("Example 3: Tool Argument Modifier")
    print("=" * 60)
    
    model = get_model()
    
    agent = Agent(
        name="EnhancedBot",
        model=model,
        tools=[search],
        instructions="Search for requested information.",
        intercept=[QueryEnhancer(context="latest news 2024")],
    )
    
    result = await agent.run("Search for Python updates")
    print(f"\nResult: {result[:200]}...")


# =============================================================================
# Example 4: Chaining Multiple Interceptors
# =============================================================================

async def example_chained_interceptors():
    """Chain multiple interceptors together."""
    print("\n" + "=" * 60)
    print("Example 4: Chaining Multiple Interceptors")
    print("=" * 60)
    
    model = get_model()
    
    # Create multiple interceptors
    budget = BudgetGuard(model_calls=5, tool_calls=10)
    logger = LoggingInterceptor(verbose=False)
    enhancer = QueryEnhancer(context="2024")
    
    agent = Agent(
        name="FullyInterceptedBot",
        model=model,
        tools=[search, calculate, get_weather],
        instructions="Help with research. Be thorough but concise.",
        intercept=[
            budget,    # First: check budget
            logger,    # Second: log activity
            enhancer,  # Third: enhance queries
        ],
    )
    
    print("\nRunning with 3 chained interceptors...")
    result = await agent.run("Search for AI trends and calculate 100/4")
    
    print(f"\n--- Summary ---")
    print(f"Budget: {budget.current_model_calls} model, {budget.current_tool_calls} tool calls")
    print(f"Events logged: {len(logger.events)}")


# =============================================================================
# Example 5: Early Exit Interceptor
# =============================================================================

class SafetyGuard(Interceptor):
    """Block dangerous operations."""
    
    def __init__(self, blocked_tools: list[str]):
        self.blocked_tools = blocked_tools
    
    async def pre_act(self, ctx: InterceptContext) -> InterceptResult:
        if ctx.tool_name in self.blocked_tools:
            print(f"  ⛔ Blocked tool: {ctx.tool_name}")
            return InterceptResult.stop(
                f"Operation blocked: {ctx.tool_name} is not allowed for safety reasons."
            )
        return InterceptResult.ok()


async def example_safety_guard():
    """Block specific tools for safety."""
    print("\n" + "=" * 60)
    print("Example 5: Safety Guard - Block Dangerous Tools")
    print("=" * 60)
    
    @tool
    def delete_file(path: str) -> str:
        """Delete a file (dangerous!)."""
        return f"Deleted {path}"
    
    model = get_model()
    
    agent = Agent(
        name="SafeBot",
        model=model,
        tools=[search, delete_file],
        instructions="Help with tasks. You can search and delete files.",
        intercept=[SafetyGuard(blocked_tools=["delete_file"])],
    )
    
    result = await agent.run("Delete the file /important/data.txt")
    print(f"\nResult: {result}")


# =============================================================================
# Example 6: PIIShield - Protect Sensitive Data
# =============================================================================

async def example_pii_shield():
    """Mask or block PII in messages."""
    print("\n" + "=" * 60)
    print("Example 6: PIIShield - Protect Sensitive Data")
    print("=" * 60)
    
    model = get_model()
    
    # Create a PII shield that masks emails and SSNs
    shield = PIIShield(
        patterns=["email", "ssn", "credit_card"],
        action=PIIAction.MASK,
    )
    
    agent = Agent(
        name="SecureBot",
        model=model,
        instructions="Help with user requests.",
        intercept=[shield],
    )
    
    print("\n--- Masking PII in messages ---")
    result = await agent.run(
        "My email is john.doe@company.com and SSN is 123-45-6789. "
        "Can you help me update my profile?"
    )
    print(f"\nAgent response (PII was masked): {result[:200]}...")
    
    # Check what was detected
    print(f"\nPII detections: {len(shield.get_detections(agent._interceptors[0]._last_ctx)) if hasattr(shield, '_last_ctx') else 'tracked internally'}")


async def example_pii_shield_block():
    """Block requests containing credit card info."""
    print("\n" + "=" * 60)
    print("Example 7: PIIShield - Block Sensitive Requests")
    print("=" * 60)
    
    model = get_model()
    
    # Block if credit card detected
    shield = PIIShield(
        patterns=["credit_card"],
        action=PIIAction.BLOCK,
        block_message="Request blocked: Credit card information detected.",
    )
    
    agent = Agent(
        name="StrictBot",
        model=model,
        instructions="Help with user requests.",
        intercept=[shield],
    )
    
    print("\n--- Testing with credit card ---")
    # Note: This will be blocked before reaching the model
    try:
        result = await agent.run(
            "Here's my card: 4111111111111111. Please process payment."
        )
        print(f"Result: {result}")
    except Exception as e:
        print(f"Exception: {type(e).__name__}")


# =============================================================================
# Example 8: TokenLimiter - Hard Context Limit
# =============================================================================

async def example_token_limiter():
    """Stop execution if context gets too long."""
    print("\n" + "=" * 60)
    print("Example 8: TokenLimiter - Hard Context Limit")
    print("=" * 60)
    
    model = get_model()
    
    # Set a token limit (this is intentionally low for demo)
    limiter = TokenLimiter(
        max_tokens=500,
        message="Context too long. Please start a new conversation.",
    )
    
    agent = Agent(
        name="LimitedBot",
        model=model,
        instructions="Help with tasks.",
        intercept=[limiter],
    )
    
    # Short task - should work
    print("\n--- Short task (under limit) ---")
    result = await agent.run("Hello!")
    print(f"Result: {result[:100]}...")


# =============================================================================
# Example 9: ContentFilter - Block Specific Content
# =============================================================================

async def example_content_filter():
    """Filter out blocked content."""
    print("\n" + "=" * 60)
    print("Example 9: ContentFilter - Block Specific Content")
    print("=" * 60)
    
    model = get_model()
    
    filter_ = ContentFilter(
        blocked_words=["password", "hack"],
        blocked_patterns=[r"rm -rf"],
        action="block",
        message="Request contains blocked content.",
    )
    
    agent = Agent(
        name="FilteredBot",
        model=model,
        instructions="Help with tasks.",
        intercept=[filter_],
    )
    
    # Safe request
    print("\n--- Safe request ---")
    result = await agent.run("What's 2 + 2?")
    print(f"Result: {result[:100]}...")


# =============================================================================
# Example 10: RateLimiter - Control API Call Rate
# =============================================================================

async def example_rate_limiter():
    """Rate limit tool calls."""
    print("\n" + "=" * 60)
    print("Example 10: RateLimiter - Control API Call Rate")
    print("=" * 60)
    
    model = get_model()
    
    # Max 5 tool calls per 10 seconds
    limiter = RateLimiter(
        calls_per_window=5,
        window_seconds=10,
        action="wait",  # Wait if limit hit (vs "block")
    )
    
    agent = Agent(
        name="RateLimitedBot",
        model=model,
        tools=[search, calculate],
        instructions="Help with tasks.",
        intercept=[limiter],
    )
    
    print("\n--- Making multiple calls ---")
    result = await agent.run("Search for Python and calculate 5*5")
    print(f"Result: {result[:150]}...")
    
    usage = limiter.current_usage
    print(f"\nRate limit usage: {usage['global_calls']}/{usage['limit']} calls")


# =============================================================================
# Example 11: Auditor - Compliance Logging
# =============================================================================

async def example_auditor():
    """Log all agent actions for audit trail."""
    print("\n" + "=" * 60)
    print("Example 11: Auditor - Compliance Logging")
    print("=" * 60)
    
    model = get_model()
    
    # Create auditor with redaction
    auditor = Auditor(
        include_content=True,
        max_content_length=100,
        redact_patterns=[r"api[-_]?key\s*[:=]\s*\S+"],  # Redact API keys
    )
    
    agent = Agent(
        name="AuditedBot",
        model=model,
        tools=[search],
        instructions="Help with tasks.",
        intercept=[auditor],
    )
    
    print("\n--- Running audited task ---")
    await agent.run("Search for Python tutorials")
    
    print(f"\n--- Audit Summary ---")
    summary = auditor.summary()
    print(f"Total events: {summary['total_events']}")
    print(f"Event types: {summary.get('by_type', {})}")
    
    print(f"\n--- Event Log ---")
    for event in auditor.events[:5]:  # Show first 5 events
        print(f"  [{event.event_type.value}] {event.agent_name}")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all interceptor examples."""
    print("\n🔌 AgenticFlow Interceptors Examples\n")
    
    await example_budget_guard()
    await example_logging_interceptor()
    await example_query_enhancer()
    await example_chained_interceptors()
    await example_safety_guard()
    await example_pii_shield()
    await example_pii_shield_block()
    await example_token_limiter()
    await example_content_filter()
    await example_rate_limiter()
    await example_auditor()
    
    print("\n" + "=" * 60)
    print("✅ All interceptor examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
