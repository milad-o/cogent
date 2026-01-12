#!/usr/bin/env python3
"""
Tree Search (LATS) Executor Example
====================================

Demonstrates LATS (Language Agent Tree Search) for complex reasoning tasks.

Tree search excels at:
- Complex multi-step problems
- Tasks requiring exploration of alternatives
- Problems where initial attempts may fail
- Tasks benefiting from self-reflection

How it works:
1. SELECT: Choose promising node using UCB1 scoring
2. EXPAND: Generate multiple candidate actions
3. EVALUATE: Use LLM to estimate path value
4. BACKPROPAGATE: Update values up the tree
5. REFLECT: Learn from failed paths

Usage:
    cd examples && python executors/tree_search_demo.py

Requires:
    Configure examples/.env with your preferred LLM_PROVIDER
"""

import asyncio
import sys
import time
from pathlib import Path

# Add examples dir to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from agenticflow import Agent, AgentConfig
from agenticflow.executors import TreeSearchExecutor
from agenticflow.tools import tool

from config import get_model


# =============================================================================
# Tools for the agent
# =============================================================================

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Args:
        expression: A math expression like '2 + 2' or '150 - 50'
    """
    try:
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return f"Error: Invalid characters in expression"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def lookup_fraction(fraction: str) -> str:
    """Convert a fraction to decimal.
    
    Args:
        fraction: A fraction like '1/3' or '2/5'
    """
    try:
        parts = fraction.split("/")
        if len(parts) != 2:
            return "Error: Invalid fraction format. Use 'a/b'"
        result = float(parts[0]) / float(parts[1])
        return f"{fraction} = {result:.4f}"
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# Tree Search Demo
# =============================================================================

async def demo_tree_search():
    """Demonstrate tree search for a multi-step reasoning problem."""
    print("\n" + "=" * 70)
    print("🌳 TREE SEARCH (LATS) EXECUTOR DEMO")
    print("=" * 70)
    
    model = get_model()
    
    # Create agent without execution_strategy (we'll use executor directly)
    config = AgentConfig(
        name="ReasoningAgent",
        model=model,
        system_prompt="""You are a precise mathematical reasoner.
Break down problems step by step and verify your work.
Use the calculate tool for arithmetic.""",
    )
    
    agent = Agent(config=config, tools=[calculate, lookup_fraction])
    
    # Create tree search executor with custom settings
    executor = TreeSearchExecutor(agent)
    executor.max_iterations = 5      # Number of MCTS iterations
    executor.max_depth = 4           # Maximum tree depth
    executor.num_candidates = 2      # Actions to generate per expansion
    executor.enable_reflection = True  # Learn from failures
    
    # Complex reasoning task
    task = """
    Word problem:
    
    A farmer has 150 apples. He gives away 1/3 of them to his neighbor.
    Then he sells 20 apples at the market.
    Finally, he picks 35 more apples from his orchard.
    
    How many apples does the farmer have now?
    
    Show step-by-step calculations using the calculate tool.
    """
    
    print(f"\n📋 Task:\n{task.strip()}")
    print("\n" + "-" * 70)
    print("🔍 Tree Search Progress:")
    print("-" * 70)
    
    # Track events
    def on_step(step_type: str, data: dict):
        if step_type == "mcts_iteration":
            print(f"\n  📊 MCTS Iteration {data.get('iteration')}")
        elif step_type == "expand":
            print(f"    └─ Expanding node at depth {data.get('depth')}")
        elif step_type == "simulate_tool":
            print(f"       └─ Calling {data.get('tool')}")
        elif step_type == "reflect":
            print(f"    💭 Reflecting on failure...")
    
    executor.on_step = on_step
    
    start = time.time()
    
    try:
        result = await executor.execute(task)
        elapsed = time.time() - start
        
        print("\n" + "=" * 70)
        print("✅ RESULT:")
        print("=" * 70)
        print(result)
        print(f"\n⏱️  Time: {elapsed:.2f}s")
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n❌ Error after {elapsed:.2f}s: {e}")
        import traceback
        traceback.print_exc()


async def demo_comparison():
    """Compare tree search vs native execution on same task."""
    print("\n" + "=" * 70)
    print("📊 COMPARING EXECUTORS: Native vs Tree Search")
    print("=" * 70)
    
    model = get_model()
    
    config = AgentConfig(
        name="ComparisonAgent",
        model=model,
        system_prompt="You are a helpful math assistant. Use tools when needed.",
    )
    
    agent = Agent(config=config, tools=[calculate])
    
    task = "What is 25 factorial divided by 24 factorial? (Hint: think about the relationship)"
    
    print(f"\n📋 Task: {task}\n")
    
    # Native execution
    print("🚀 Native Execution (single path):")
    start = time.time()
    native_result = await agent.run(task)
    native_time = time.time() - start
    print(f"   Result: {native_result[:200]}...")
    print(f"   Time: {native_time:.2f}s\n")
    
    # Tree search execution
    print("🌳 Tree Search (explores alternatives):")
    executor = TreeSearchExecutor(agent)
    executor.max_iterations = 3
    executor.num_candidates = 2
    
    start = time.time()
    tree_result = await executor.execute(task)
    tree_time = time.time() - start
    print(f"   Result: {tree_result[:200]}...")
    print(f"   Time: {tree_time:.2f}s\n")
    
    print(f"📈 Tree search took {tree_time/native_time:.1f}x longer but explored more reasoning paths")


async def main():
    """Run tree search demos."""
    print("\n" + "🌳 " * 20)
    print("AGENTICFLOW TREE SEARCH (LATS) DEMO")
    print("🌳 " * 20)
    
    print("""
Tree Search (LATS) is a Monte Carlo Tree Search algorithm for LLMs.
It explores multiple reasoning paths and backtracks on failures.

Best for:
- Complex multi-step reasoning
- Tasks where initial attempts may fail
- Problems requiring exploration
""")
    
    await demo_tree_search()
    
    print("\n" + "=" * 70)
    print("✅ Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
