"""Semantic Cache - Cache tool outputs by semantic similarity.

Achieves 83% hit rate vs 38% for traditional caching (arXiv:2601.16286).
"""

import asyncio
import time

from cogent import Agent
from cogent.capabilities import WebSearch
from cogent.observability import Observer
from cogent.tools import tool


@tool(cache=True)
async def expensive_analysis(topic: str) -> dict:
    """Expensive computation that benefits from caching."""
    await asyncio.sleep(2)  # Simulate expensive work
    return {"topic": topic, "analysis": f"Analysis of {topic}", "confidence": 0.95}


@tool(cache=True)
async def slow_lookup(query: str) -> str:
    """Slow database query that benefits from caching."""
    await asyncio.sleep(1.5)  # Simulate slow DB
    return f"Results for '{query}': 42 matching records"


async def demo_websearch_cache():
    """WebSearch with automatic caching."""
    print("\n--- WebSearch + Cache ---")
    
    agent = Agent(
        name="Researcher",
        model="gpt-4o-mini",
        capabilities=[WebSearch(max_results=3)],
        cache=True,
        observer=Observer.trace(),
    )

    # First search (cache miss)
    t0 = time.perf_counter()
    await agent.run("Search for Python async programming tutorials")
    t1 = time.perf_counter() - t0

    # Similar search (cache hit expected)
    t0 = time.perf_counter()
    await agent.run("Find tutorials about asynchronous Python programming")
    t2 = time.perf_counter() - t0

    print(f"First:   {t1:.2f}s")
    print(f"Similar: {t2:.2f}s {'(cache hit)' if t2 < t1 * 0.5 else ''}")
    _print_metrics(agent.cache)


async def demo_custom_tools_cache():
    """Custom tools with @tool(cache=True)."""
    print("\n--- Custom Tools + Cache ---")
    
    agent = Agent(
        name="Analyst",
        model="gpt-4o-mini",
        tools=[expensive_analysis, slow_lookup],
        cache=True,
        observer=Observer.trace(),
    )

    # First analysis (cache miss)
    t0 = time.perf_counter()
    await agent.run("Analyze Python async programming")
    t1 = time.perf_counter() - t0

    # Similar analysis (cache hit expected)
    t0 = time.perf_counter()
    await agent.run("Analyze asynchronous Python programming")
    t2 = time.perf_counter() - t0

    # Different topic (cache miss)
    t0 = time.perf_counter()
    await agent.run("Look up information about machine learning")
    t3 = time.perf_counter() - t0

    # Similar lookup (cache hit expected)
    t0 = time.perf_counter()
    await agent.run("Find data on ML and machine learning")
    t4 = time.perf_counter() - t0

    print(f"Analysis 1: {t1:.2f}s")
    print(f"Analysis 2: {t2:.2f}s {'(cache hit)' if t2 < t1 * 0.5 else ''}")
    print(f"Lookup 1:   {t3:.2f}s")
    print(f"Lookup 2:   {t4:.2f}s {'(cache hit)' if t4 < t3 * 0.5 else ''}")
    _print_metrics(agent.cache)


def _print_metrics(cache):
    """Print cache metrics."""
    if not cache:
        return
    m = cache.get_metrics()
    print(f"Cache: {m['cache_hit_rate']:.0%} hit rate, {m['cache_hits']} hits, {m['cache_misses']} misses")


async def main():
    print("=" * 50)
    print("SEMANTIC CACHE DEMO")
    print("=" * 50)
    
    await demo_websearch_cache()
    await demo_custom_tools_cache()
    
    print("\n✓ Done")


if __name__ == "__main__":
    asyncio.run(main())
