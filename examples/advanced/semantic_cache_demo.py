"""Semantic Cache with Agent - Cache search results automatically.

Based on arXiv:2601.16286 - Achieves 83% hit rate vs 38% for traditional caching.

When you enable `cache=True` on an Agent with WebSearch:
- Search results are cached based on semantic similarity
- Similar queries reuse cached results (no repeated API calls)
- 7-10× speedup on cache hits

This example shows:
1. Enable semantic caching on Agent with WebSearch capability
2. Run similar queries - see cache hits
3. Check cache metrics (hit rate, saved calls)
"""

import asyncio
import time

from cogent import Agent
from cogent.capabilities import WebSearch
from cogent.observability import Observer


async def main():
    print("=" * 60)
    print("SEMANTIC CACHE WITH AGENT + WEBSEARCH")
    print("=" * 60)

    # Create agent with WebSearch capability and caching enabled
    agent = Agent(
        name="Researcher",
        model="gpt-4o-mini",
        capabilities=[WebSearch(max_results=3)],
        cache=True,  # 🔥 WebSearch automatically uses agent.cache
        observer=Observer.trace(),
    )

    print(f"\n✅ Agent created with cache=True")
    print(f"   WebSearch will cache results by semantic similarity\n")

    # =========================================================================
    # Test 1: First search - cache MISS
    # =========================================================================
    print("-" * 60)
    print("TEST 1: First search (cache cold)")
    print("-" * 60)

    start = time.perf_counter()
    result1 = await agent.run("Search for Python async programming tutorials")
    elapsed1 = time.perf_counter() - start

    print(f"\n📝 Query: 'Search for Python async programming tutorials'")
    print(f"   Time: {elapsed1:.2f}s")

    # =========================================================================
    # Test 2: Similar search - cache HIT expected
    # =========================================================================
    print("\n" + "-" * 60)
    print("TEST 2: Similar search (cache should hit)")
    print("-" * 60)

    start = time.perf_counter()
    # Different wording, same intent!
    result2 = await agent.run("Find tutorials about asynchronous Python programming")
    elapsed2 = time.perf_counter() - start

    print(f"\n📝 Query: 'Find tutorials about asynchronous Python programming'")
    print(f"   Time: {elapsed2:.2f}s")

    if elapsed2 < elapsed1 * 0.5:
        print(f"   🎯 CACHE HIT! Speedup: {elapsed1/elapsed2:.1f}x faster")
    else:
        print(f"   Note: Cache may have missed (embedding threshold 85%)")

    # =========================================================================
    # Test 2.5: EXACT same search - cache MUST HIT
    # =========================================================================
    print("\n" + "-" * 60)
    print("TEST 2.5: Exact duplicate (cache MUST hit)")
    print("-" * 60)

    start = time.perf_counter()
    # Exact same query as Test 1!
    result25 = await agent.run("Search for Python async programming tutorials")
    elapsed25 = time.perf_counter() - start

    print(f"\n📝 Query: 'Search for Python async programming tutorials'")
    print(f"   Time: {elapsed25:.2f}s")

    if elapsed25 < elapsed1 * 0.5:
        print(f"   🎯 CACHE HIT! Speedup: {elapsed1/elapsed25:.1f}x faster")
    else:
        print(f"   ❌ UNEXPECTED MISS (same query should hit cache)")

    # =========================================================================
    # Test 3: Different search - cache MISS expected
    # =========================================================================
    print("\n" + "-" * 60)
    print("TEST 3: Different search (cache should miss)")
    print("-" * 60)

    start = time.perf_counter()
    result3 = await agent.run("Search for chocolate cake recipes")
    elapsed3 = time.perf_counter() - start

    print(f"\n📝 Query: 'Search for chocolate cake recipes'")
    print(f"   Time: {elapsed3:.2f}s")
    print(f"   ❌ CACHE MISS (expected - completely different topic)")

    # =========================================================================
    # Cache Metrics
    # =========================================================================
    print("\n" + "=" * 60)
    print("CACHE METRICS")
    print("=" * 60)

    if agent.cache:
        metrics = agent.cache.get_metrics()
        print(f"""
Hit Rate:       {metrics['cache_hit_rate']:.1%}
Total Hits:     {metrics['cache_hits']}
Total Misses:   {metrics['cache_misses']}
Cache Entries:  {metrics['total_entries']}

📊 What's Cached:
   - WebSearch API results (semantic similarity matching)
   - Agent reasoning is NOT cached (LLM still runs)
   - Speedup from avoiding redundant web searches
""")
    else:
        print("\n⚠️  Cache not initialized")

    print("✅ Done! Semantic caching reduces repeated searches automatically.")


if __name__ == "__main__":
    asyncio.run(main())
