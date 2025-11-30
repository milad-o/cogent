"""
Example 25: Interceptors

Demonstrates AgenticFlow's interceptor system for cross-cutting concerns
like cost control, security, and observability.

Key features:
- Composable interceptors that hook into agent execution
- BudgetGuard for limiting model/tool calls
- Custom interceptors for logging, validation, etc.

Run: uv run python examples/25_interceptors.py
"""

import asyncio
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
    
    print("\n" + "=" * 60)
    print("✅ All interceptor examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
