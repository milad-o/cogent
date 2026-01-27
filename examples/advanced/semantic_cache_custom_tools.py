"""Semantic Cache with Custom Tools - Automatic cache integration.

Shows how to use semantic caching with custom @tool functions using cache=True.

Key Points:
- Use @tool(cache=True) to enable automatic caching
- No manual cache.get() / cache.put() calls needed
- Cache uses semantic similarity (not exact string matching)
- Requires agent.cache to be enabled

Example tools that benefit from caching:
- Database queries
- External API calls
- Heavy computations
- File processing
"""

import asyncio
import time

from cogent import Agent
from cogent.observability import Observer
from cogent.tools import tool


@tool(cache=True)
async def expensive_analysis(topic: str) -> dict:
    """Analyze a topic with expensive computation (simulated).
    
    This tool simulates an expensive operation that benefits from caching.
    In reality, this could be:
    - Database query
    - External API call
    - Heavy computation
    - ML model inference
    """
    # Simulated expensive work - NO manual cache code needed!
    print(f"   ⏳ Computing analysis for: {topic} (expensive!)...")
    await asyncio.sleep(2)  # Simulate expensive computation
    
    result = {
        "topic": topic,
        "analysis": f"Detailed analysis of {topic} with insights and recommendations.",
        "confidence": 0.95,
        "sources": 10,
        "computed_at": time.time(),
    }
    
    return result


@tool(cache=True)
async def lookup_data(query: str) -> str:
    """Look up data from a slow database (simulated).
    
    Demonstrates caching for database queries.
    """
    # Simulated slow DB query - caching handled automatically
    print(f"   💾 Querying database for: {query} (slow!)...")
    await asyncio.sleep(1.5)  # Simulate slow DB query
    
    return f"Database results for '{query}': Found 42 matching records with relevant information."


async def main():
    print("=" * 60)
    print("SEMANTIC CACHE WITH CUSTOM TOOLS")
    print("=" * 60)
    print()
    
    # Create agent with cache enabled
    agent = Agent(
        name="Analyst",
        model="gpt-4o-mini",
        tools=[expensive_analysis, lookup_data],
        cache=True,  # Enable semantic cache
        observer=Observer.trace(),
    )
    
    print("✅ Agent created with custom tools")
    print("   Tools use @tool(cache=True) for automatic caching")
    print()
    
    # =========================================================================
    # Test 1: First analysis (cache cold)
    # =========================================================================
    print("-" * 60)
    print("TEST 1: First analysis (cache cold)")
    print("-" * 60)
    
    start = time.perf_counter()
    result1 = await agent.run("Analyze Python async programming")
    elapsed1 = time.perf_counter() - start
    
    print(f"\n📝 Query: 'Analyze Python async programming'")
    print(f"   Time: {elapsed1:.2f}s")
    print()
    
    # =========================================================================
    # Test 2: Similar query (semantic cache should hit)
    # =========================================================================
    print("-" * 60)
    print("TEST 2: Similar analysis (semantic match)")
    print("-" * 60)
    
    start = time.perf_counter()
    result2 = await agent.run("Analyze asynchronous Python programming")  # Different wording!
    elapsed2 = time.perf_counter() - start
    
    print(f"\n📝 Query: 'Analyze asynchronous Python programming'")
    print(f"   Time: {elapsed2:.2f}s")
    
    if elapsed2 < elapsed1 * 0.5:
        print(f"   🎯 CACHE HIT! Speedup: {elapsed1/elapsed2:.1f}x faster")
    print()
    
    # =========================================================================
    # Test 3: Database lookup (cache cold)
    # =========================================================================
    print("-" * 60)
    print("TEST 3: Database lookup (cache cold)")
    print("-" * 60)
    
    start = time.perf_counter()
    result3 = await agent.run("Look up information about machine learning")
    elapsed3 = time.perf_counter() - start
    
    print(f"\n📝 Query: 'Look up information about machine learning'")
    print(f"   Time: {elapsed3:.2f}s")
    print()
    
    # =========================================================================
    # Test 4: Similar lookup (semantic cache should hit)
    # =========================================================================
    print("-" * 60)
    print("TEST 4: Similar lookup (semantic match)")
    print("-" * 60)
    
    start = time.perf_counter()
    result4 = await agent.run("Find data on ML and machine learning")  # Similar!
    elapsed4 = time.perf_counter() - start
    
    print(f"\n📝 Query: 'Find data on ML and machine learning'")
    print(f"   Time: {elapsed4:.2f}s")
    
    if elapsed4 < elapsed3 * 0.5:
        print(f"   🎯 CACHE HIT! Speedup: {elapsed3/elapsed4:.1f}x faster")
    print()
    
    # =========================================================================
    # Test 5: Different topic (cache miss)
    # =========================================================================
    print("-" * 60)
    print("TEST 5: Different topic (cache miss)")
    print("-" * 60)
    
    start = time.perf_counter()
    result5 = await agent.run("Analyze chocolate cake recipes")
    elapsed5 = time.perf_counter() - start
    
    print(f"\n📝 Query: 'Analyze chocolate cake recipes'")
    print(f"   Time: {elapsed5:.2f}s")
    print("   ❌ CACHE MISS (expected - different topic)")
    print()
    
    # =========================================================================
    # Cache Metrics
    # =========================================================================
    print("=" * 60)
    print("CACHE METRICS")
    print("=" * 60)
    
    if agent.cache:
        metrics = agent.cache.get_metrics()
        print(f"""
Hit Rate:       {metrics['cache_hit_rate']:.1%}
Total Hits:     {metrics['cache_hits']}
Total Misses:   {metrics['cache_misses']}
Cache Entries:  {metrics['total_entries']}

💡 How It Works:
   1. Decorate tools with @tool(cache=True)
   2. Cache automatically checks before execution
   3. Results stored with semantic similarity matching
   4. No manual cache.get()/cache.put() needed!
   
📊 Cache Stages:
""")
        for stage, stage_cache in agent.cache._caches.items():
            print(f"   - {stage}: {len(stage_cache)} entries")
    else:
        print("\n⚠️  Cache not initialized")
    
    print()
    print("✅ Done! Use @tool(cache=True) for automatic semantic caching.")


if __name__ == "__main__":
    asyncio.run(main())
