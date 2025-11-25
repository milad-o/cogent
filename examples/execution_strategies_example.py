#!/usr/bin/env python3
"""
Execution Strategies Demo
=========================

Compares different execution strategies for complex tasks:

1. ReAct - Think-Act-Observe loop (baseline, slowest)
2. Plan & Execute - Plan upfront, then execute (medium)
3. DAG Executor - Parallel execution with dependencies (fastest)
4. Adaptive - Auto-selects best strategy

This demonstrates why DAG execution is faster for complex tasks
with multiple independent tool calls.
"""

import asyncio
import time
from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import tool

from agenticflow import (
    Agent,
    AgentConfig,
    AgentRole,
    EventBus,
    ToolRegistry,
    ExecutionStrategy,
    create_executor,
)


# ============================================================
# Define some example tools
# ============================================================

@tool
def search_news(topic: str) -> str:
    """Search for news articles on a topic."""
    # Simulate API latency
    time.sleep(0.5)
    return f"Found 3 news articles about {topic}: [Article 1, Article 2, Article 3]"


@tool
def search_papers(topic: str) -> str:
    """Search for academic papers on a topic."""
    # Simulate API latency
    time.sleep(0.5)
    return f"Found 2 academic papers about {topic}: [Paper A, Paper B]"


@tool
def search_social(topic: str) -> str:
    """Search social media for trending discussions."""
    # Simulate API latency
    time.sleep(0.5)
    return f"Found 5 social media discussions about {topic}: [Post 1, Post 2, ...]"


@tool
def summarize(content: str) -> str:
    """Summarize multiple pieces of content."""
    time.sleep(0.3)
    return f"Summary of {len(content)} chars: Key points extracted."


@tool
def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of text."""
    time.sleep(0.3)
    return "Sentiment: Mostly positive (72%), some concerns (28%)"


def create_research_agent() -> Agent:
    """Create an agent with research tools."""
    event_bus = EventBus()
    
    registry = ToolRegistry()
    registry.register_many([
        search_news,
        search_papers,
        search_social,
        summarize,
        analyze_sentiment,
    ])
    
    return Agent(
        config=AgentConfig(
            name="Researcher",
            role=AgentRole.RESEARCHER,
            model="gpt-4o-mini",
            system_prompt="""You are a research agent with access to multiple search tools.
When given a research task:
1. Identify what searches are needed
2. Use tools to gather information
3. Synthesize the results

Be efficient - identify independent searches that can be done in parallel.""",
            tools=["search_news", "search_papers", "search_social", "summarize", "analyze_sentiment"],
        ),
        event_bus=event_bus,
        tool_registry=registry,
    )


async def demo_execution_strategies():
    """Compare different execution strategies."""
    
    print("\n" + "=" * 70)
    print("EXECUTION STRATEGIES COMPARISON")
    print("=" * 70)
    
    task = """Research the topic "AI in Healthcare":
1. Search news for recent developments
2. Search academic papers for research
3. Search social media for public opinion
4. Summarize all findings
5. Analyze the overall sentiment"""
    
    print(f"\n📋 Task:\n{task}")
    print("\n" + "-" * 70)
    
    # Strategy comparison results
    results = {}
    
    # Test each strategy
    strategies = [
        ("ReAct (sequential)", "react"),
        ("Plan & Execute", "plan"),
        ("DAG (parallel)", "dag"),
    ]
    
    for name, strategy in strategies:
        print(f"\n🔄 Testing: {name}...")
        agent = create_research_agent()
        
        steps = []
        def on_step(step_type: str, data: dict):
            steps.append((step_type, data))
            if step_type == "executing_wave":
                print(f"   Wave {data['wave']}/{data['total_waves']}: {data['parallel_calls']} parallel calls")
            elif step_type == "act":
                print(f"   → {data['tool']}")
        
        start = time.perf_counter()
        try:
            result = await agent.run(task, strategy=strategy, on_step=on_step)
            elapsed = time.perf_counter() - start
            results[name] = {
                "time": elapsed,
                "steps": len(steps),
                "success": True,
            }
            print(f"   ✓ Completed in {elapsed:.2f}s ({len(steps)} steps)")
        except Exception as e:
            elapsed = time.perf_counter() - start
            results[name] = {
                "time": elapsed,
                "steps": len(steps),
                "success": False,
                "error": str(e),
            }
            print(f"   ✗ Failed after {elapsed:.2f}s: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    baseline_time = results.get("ReAct (sequential)", {}).get("time", 1)
    
    print(f"\n{'Strategy':<25} {'Time':>10} {'Speedup':>10} {'Steps':>8}")
    print("-" * 55)
    
    for name, data in results.items():
        speedup = baseline_time / data["time"] if data["time"] > 0 else 0
        status = "✓" if data["success"] else "✗"
        print(f"{status} {name:<23} {data['time']:>8.2f}s {speedup:>9.2f}x {data['steps']:>8}")
    
    print("\n" + "=" * 70)


async def demo_dag_execution_detail():
    """Show detailed DAG execution with waves."""
    
    print("\n" + "=" * 70)
    print("DAG EXECUTION DETAIL")
    print("=" * 70)
    
    print("""
The DAG executor builds a dependency graph:

    search_news ──────┐
                      │
    search_papers ────┼──► summarize ──► analyze_sentiment
                      │
    search_social ────┘

Wave 1: search_news, search_papers, search_social (PARALLEL)
Wave 2: summarize (depends on Wave 1)
Wave 3: analyze_sentiment (depends on Wave 2)

Sequential: 5 tool calls × ~0.5s each = ~2.5s
DAG:        3 waves = ~1.3s (Wave 1: 0.5s, Wave 2: 0.3s, Wave 3: 0.3s)
Speedup:    ~2x faster
""")
    
    agent = create_research_agent()
    
    task = "Search news, papers, and social media about 'quantum computing', then summarize."
    
    print(f"📋 Task: {task}")
    print("\n🔄 Executing with DAG strategy...\n")
    
    def on_step(step_type: str, data: dict):
        if step_type == "planning_dag":
            print("📊 Phase 1: Building dependency graph...")
        elif step_type == "dag_waves":
            print(f"📊 Plan: {data['waves']} waves, {data['total_calls']} total calls")
        elif step_type == "executing_wave":
            print(f"\n🔹 Wave {data['wave']}/{data['total_waves']}: Executing {data['parallel_calls']} calls in parallel")
    
    start = time.perf_counter()
    result = await agent.run(task, strategy="dag", on_step=on_step)
    elapsed = time.perf_counter() - start
    
    print(f"\n✅ Completed in {elapsed:.2f}s")
    print(f"\n📝 Result preview: {str(result)[:200]}...")


async def demo_adaptive_selection():
    """Show how adaptive executor chooses strategy."""
    
    print("\n" + "=" * 70)
    print("ADAPTIVE STRATEGY SELECTION")
    print("=" * 70)
    
    agent = create_research_agent()
    
    tasks = [
        ("What is 2+2?", "Simple - no tools needed"),
        ("Search news about AI", "Single tool"),
        ("Search news AND papers about AI, then summarize", "Parallel opportunity"),
    ]
    
    for task, description in tasks:
        print(f"\n📋 Task: {task}")
        print(f"   Expected: {description}")
        
        def on_step(step_type: str, data: dict):
            if step_type == "strategy_selected":
                print(f"   → Selected: {data['strategy']}")
        
        try:
            await agent.run(task, strategy="adaptive", on_step=on_step)
        except Exception:
            pass  # Ignore execution errors, we care about strategy selection


async def main():
    """Run all demos."""
    print("🚀 AgenticFlow - Execution Strategies Demo")
    print("=" * 70)
    
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found")
        return
    
    print("✅ OpenAI API key found")
    
    await demo_dag_execution_detail()
    await demo_execution_strategies()
    await demo_adaptive_selection()
    
    print("\n" + "=" * 70)
    print("🎉 ALL DEMOS COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
