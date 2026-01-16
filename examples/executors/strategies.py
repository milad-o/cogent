#!/usr/bin/env python3
"""
Execution Strategies Demo
=========================

This example demonstrates using different executors in AgenticFlow.

The simplified Agent API uses NativeExecutor by default (parallel tool execution).
For sequential execution or other strategies, use executors directly.

Usage:
    python examples/executors/strategies_demo.py
    
Requires:
    Configure examples/.env with your preferred LLM_PROVIDER
"""

import asyncio
import sys
import time
from pathlib import Path

# Add examples dir to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from agenticflow import Agent
from agenticflow.executors import NativeExecutor, SequentialExecutor
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
    """Demonstrate NATIVE execution (default - parallel tool execution)."""
    print("\\n" + "=" * 60)
    print("🚀 NATIVE Executor (Parallel - Default)")
    print("=" * 60)
    
    model = get_model()
    
    # By default, Agent uses NativeExecutor (parallel tool execution)
    agent = Agent(
        name="NativeAgent",
        model=model,
        tools=[calculate, get_weather, search_database],
        instructions="You are a helpful assistant. Use tools when appropriate.",
    )
    
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
    
    print(f"\\n📊 Result:\\n{result}")
    print(f"\\n⏱️  Time: {elapsed:.2f}s (tools run in PARALLEL)")


async def demo_sequential():
    """Demonstrate SEQUENTIAL execution using executor directly."""
    print("\\n" + "=" * 60)
    print("📋 SEQUENTIAL Executor (Ordered Execution)")
    print("=" * 60)
    
    model = get_model()
    
    # Create agent
    agent = Agent(
        name="SequentialAgent",
        model=model,
        tools=[calculate, get_weather, search_database],
        instructions="You are a helpful assistant. Use tools when appropriate.",
    )
    
    # Use SequentialExecutor directly for ordered execution
    executor = SequentialExecutor(agent)
    
    task = """
    I need you to:
    1. Calculate 25 * 4
    2. Get the weather in Tokyo
    3. Search for information about Python
    
    Give me a summary of all results.
    """
    
    start = time.time()
    result = await executor.execute(task)
    elapsed = time.time() - start
    
    print(f"\\n📊 Result:\\n{result}")
    print(f"\\n⏱️  Time: {elapsed:.2f}s (tools run SEQUENTIALLY)")


async def main():
    """Run all demos."""
    print("\\n" + "🎯 " * 20)
    print("AGENTICFLOW EXECUTION STRATEGIES DEMO")
    print("🎯 " * 20)
    
    # Uses LLM_PROVIDER from examples/.env
    
    # Run demos
    await demo_native()
    await demo_sequential()
    
    print("\\n" + "=" * 60)
    print("✅ Demo Complete!")
    print("=" * 60)
    print("""
Summary:
- NATIVE (default): Fastest, parallel tool execution
- SEQUENTIAL: Ordered execution, tools run one at a time
- TREE_SEARCH: Available via TreeSearchExecutor for complex reasoning

For most use cases, the default NATIVE executor is recommended.
Use SequentialExecutor when order of execution matters.
See tree_search_demo.py for advanced tree search examples.
""")


if __name__ == "__main__":
    asyncio.run(main())
