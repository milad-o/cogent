"""
Executor Strategies - Quick comparison of NativeExecutor vs SequentialExecutor.

This example shows the performance difference and when to use each strategy.
"""

import asyncio
import time

from cogent import Agent
from cogent.executors import ExecutionStrategy, SequentialExecutor, create_executor
from cogent.tools import tool


# Simulated tools with artificial delays
@tool
async def fetch_data_a(source: str) -> str:
    """Fetch data from source A."""
    await asyncio.sleep(0.3)  # Simulate API delay
    return f"Data from {source}: [A1, A2, A3]"


@tool
async def fetch_data_b(source: str) -> str:
    """Fetch data from source B."""
    await asyncio.sleep(0.3)  # Simulate API delay
    return f"Data from {source}: [B1, B2, B3]"


@tool
async def fetch_data_c(source: str) -> str:
    """Fetch data from source C."""
    await asyncio.sleep(0.3)  # Simulate API delay
    return f"Data from {source}: [C1, C2, C3]"


async def compare_executors():
    """Compare NativeExecutor vs SequentialExecutor.
    
    Key insight: NativeExecutor can achieve BOTH parallel and sequential execution.
    - Parallel: When LLM requests multiple tools in one turn
    - Sequential: When LLM naturally calls tools one at a time
    
    SequentialExecutor FORCES sequential execution regardless of LLM behavior.
    """
    print("=" * 80)
    print("Executor Strategy Comparison")
    print("=" * 80 + "\n")

    agent = Agent(
        name="executor_comparison",
        model="gpt-4o-mini",
        tools=[fetch_data_a, fetch_data_b, fetch_data_c],
    )

    # Task designed to trigger parallel execution
    task_parallel = """
    Fetch data from all three sources at once:
    - Source Alpha
    - Source Beta  
    - Source Gamma
    
    All three are independent - fetch them all now.
    """

    # Test 1: NativeExecutor - LLM likely requests all tools at once (parallel)
    print("Test 1: NativeExecutor with Parallel Task")
    print("-" * 80)
    print("Task encourages LLM to request all tools at once")
    print("Expected: LLM batches tools → parallel execution (~0.3s)\n")

    executor_parallel = create_executor(agent, ExecutionStrategy.NATIVE)
    start = time.time()
    result_parallel = await executor_parallel.execute(task_parallel)
    elapsed_parallel = time.time() - start

    print(f"✅ Result: {result_parallel[:100]}...")
    print(f"⏱️  Time: {elapsed_parallel:.2f}s")
    print("💡 LLM requested multiple tools → NativeExecutor ran them in parallel\n")

    # Test 2: SequentialExecutor - FORCES sequential even if LLM batches
    print("Test 2: SequentialExecutor with Same Parallel Task")
    print("-" * 80)
    print("Same task, but SequentialExecutor forces sequential execution")
    print("Expected: Even if LLM batches tools → forced sequential (~0.9s)\n")

    executor_sequential = create_executor(agent, ExecutionStrategy.SEQUENTIAL)
    start = time.time()
    result_sequential = await executor_sequential.execute(task_parallel)
    elapsed_sequential = time.time() - start

    print(f"✅ Result: {result_sequential[:100]}...")
    print(f"⏱️  Time: {elapsed_sequential:.2f}s")
    print("💡 SequentialExecutor forced one-at-a-time execution\n")

    # Comparison
    print("=" * 80)
    print("Performance Comparison")
    print("=" * 80)
    print(f"NativeExecutor (parallel):   {elapsed_parallel:.2f}s")
    print(f"SequentialExecutor (forced): {elapsed_sequential:.2f}s")
    print(f"Speedup:                     {elapsed_sequential / elapsed_parallel:.1f}×")
    print()
    print("Key Insights:")
    print("-------------")
    print("• NativeExecutor: LLM decides parallel vs sequential based on prompt")
    print("• SequentialExecutor: Always sequential, even if LLM wants parallel")
    print("• For sequential tasks: Use NativeExecutor + clear prompting (more flexible)")
    print("• For strict ordering: Use SequentialExecutor only when you must guarantee order")
    print("=" * 80 + "\n")


async def direct_executor_usage():
    """Direct usage without factory pattern.
    
    Demonstrates how NativeExecutor can achieve sequential execution
    through clear prompting (recommended approach).
    """
    print("=" * 80)
    print("NativeExecutor: Sequential Through Prompting")
    print("=" * 80 + "\n")

    agent = Agent(
        name="native_sequential",
        model="gpt-4o-mini",
        tools=[fetch_data_a, fetch_data_b, fetch_data_c],
    )

    # Task with clear sequential prompting
    task = """
    Fetch data step by step:
    1. FIRST fetch from Source X
    2. THEN fetch from Source Y  
    3. FINALLY fetch from Source Z
    
    Do these ONE AT A TIME, in order.
    """

    print("Task with sequential prompting:")
    print(task.strip())
    print("\nRunning with default NativeExecutor...")
    print("LLM should naturally call tools one at a time based on prompt\n")

    import time
    start = time.time()
    result = await agent.run(task)
    elapsed = time.time() - start

    print(f"✅ Result: {result.content[:100]}...")
    print(f"⏱️  Time: {elapsed:.2f}s")
    print()
    print("💡 NativeExecutor achieved sequential execution through prompting!")
    print("💡 No need for SequentialExecutor in most cases.")
    print("=" * 80 + "\n")


async def main():
    """Run all executor examples."""
    await compare_executors()
    await direct_executor_usage()


if __name__ == "__main__":
    asyncio.run(main())
