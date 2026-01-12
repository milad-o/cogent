#!/usr/bin/env python3
"""
Execution Strategies Demo
=========================

This example demonstrates the three execution strategies available in AgenticFlow:

1. NATIVE (default) - High-performance parallel tool execution
2. SEQUENTIAL - Sequential tool execution for ordered tasks
3. TREE_SEARCH - LATS-style Monte Carlo tree search with backtracking

Each strategy has different trade-offs:
- NATIVE: Fastest, executes tools in parallel
- SEQUENTIAL: More predictable, ordered execution
- TREE_SEARCH: Best accuracy for complex reasoning, but slower

Usage:
    python examples/executors_demo.py
    
Requires:
    Configure examples/.env with your preferred LLM_PROVIDER
"""

import asyncio
import sys
import time
from pathlib import Path

# Add examples dir to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from agenticflow import Agent, AgentConfig, ExecutionStrategy
from agenticflow.tools import tool

# Import from examples config
from config import get_model


# =============================================================================
# Define some tools for the agents to use
# =============================================================================

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Args:
        expression: A math expression like '2 + 2' or '10 * 5 / 2'
    """
    try:
        # Safe eval for simple math
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return f"Error: Invalid characters in expression"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.
    
    Args:
        city: Name of the city
    """
    # Simulated weather data
    weather_data = {
        "new york": "72°F, Sunny",
        "london": "58°F, Cloudy",
        "tokyo": "68°F, Partly cloudy",
        "paris": "65°F, Rainy",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool
def search_database(query: str) -> str:
    """Search a database for information.
    
    Args:
        query: Search query
    """
    # Simulated database search
    time.sleep(0.1)  # Simulate latency
    results = {
        "python": "Python is a high-level programming language.",
        "javascript": "JavaScript is a scripting language for the web.",
        "rust": "Rust is a systems programming language focused on safety.",
    }
    for key, value in results.items():
        if key in query.lower():
            return value
    return f"No results found for: {query}"


# =============================================================================
# Demo: Compare Execution Strategies
# =============================================================================

async def demo_native():
    """Demonstrate NATIVE execution strategy (parallel)."""
    print("\n" + "=" * 60)
    print("🚀 NATIVE Strategy (Parallel Execution)")
    print("=" * 60)
    
    model = get_model()
    
    config = AgentConfig(
        name="NativeAgent",
        model=model,
        execution_strategy="native",  # String format works!
        system_prompt="You are a helpful assistant. Use tools when appropriate.",
    )
    
    agent = Agent(config=config, tools=[calculate, get_weather, search_database])
    
    task = """
    I need you to:
    1. Calculate 25 * 4
    2. Get the weather in Tokyo
    3. Search for information about Python
    
    Give me a summary of all results.
    """
    
    start = time.time()
    result = await agent.run(task)
    elapsed = time.time() - start
    
    print(f"\n📊 Result:\n{result}")
    print(f"\n⏱️  Time: {elapsed:.2f}s (tools run in PARALLEL)")


async def demo_sequential():
    """Demonstrate SEQUENTIAL execution strategy."""
    print("\n" + "=" * 60)
    print("📋 SEQUENTIAL Strategy (Ordered Execution)")
    print("=" * 60)
    
    model = get_model()
    
    config = AgentConfig(
        name="SequentialAgent",
        model=model,
        execution_strategy=ExecutionStrategy.SEQUENTIAL,  # Enum format
        system_prompt="You are a helpful assistant. Use tools when appropriate.",
    )
    
    agent = Agent(config=config, tools=[calculate, get_weather, search_database])
    
    task = """
    I need you to:
    1. Calculate 25 * 4
    2. Get the weather in Tokyo
    3. Search for information about Python
    
    Give me a summary of all results.
    """
    
    start = time.time()
    result = await agent.run(task)
    elapsed = time.time() - start
    
    print(f"\n📊 Result:\n{result}")
    print(f"\n⏱️  Time: {elapsed:.2f}s (tools run SEQUENTIALLY)")


async def demo_tree_search():
    """Demonstrate TREE_SEARCH execution strategy (LATS)."""
    print("\n" + "=" * 60)
    print("🌳 TREE_SEARCH Strategy (LATS with Backtracking)")
    print("=" * 60)
    
    model = get_model()
    
    config = AgentConfig(
        name="TreeSearchAgent",
        model=model,
        execution_strategy="tree_search",  # String format
        system_prompt="""You are a precise assistant that thinks step by step.
        Break down complex problems and verify your work.""",
    )
    
    agent = Agent(config=config, tools=[calculate])
    
    # Tree search shines with complex reasoning
    task = """
    Solve this step by step:
    
    If I have 150 apples and give away 1/3 of them, then buy 20 more,
    how many apples do I have?
    
    Show your calculation steps.
    """
    
    start = time.time()
    result = await agent.run(task)
    elapsed = time.time() - start
    
    print(f"\n📊 Result:\n{result}")
    print(f"\n⏱️  Time: {elapsed:.2f}s (explores multiple paths, best ACCURACY)")


async def main():
    """Run all demos."""
    print("\n" + "🎯 " * 20)
    print("AGENTICFLOW EXECUTION STRATEGIES DEMO")
    print("🎯 " * 20)
    
    # Uses LLM_PROVIDER from examples/.env
    
    # Run NATIVE and SEQUENTIAL strategies
    await demo_native()
    await demo_sequential()
    
    # NOTE: TREE_SEARCH is experimental and best for complex reasoning tasks
    # await demo_tree_search()
    
    print("\n" + "=" * 60)
    print("✅ Demo Complete!")
    print("=" * 60)
    print("""
Summary:
- NATIVE: Fastest, parallel tool execution (default)
- SEQUENTIAL: Ordered execution, predictable
- TREE_SEARCH: Best accuracy, explores alternatives

Choose based on your needs:
- Speed → NATIVE
- Order matters → SEQUENTIAL  
- Complex reasoning → TREE_SEARCH
""")


if __name__ == "__main__":
    asyncio.run(main())
