"""Semantic Cache Demo - Cache reasoning artifacts, not just responses.

Based on arXiv:2601.16286 - Achieves 83% hit rate vs 38% for traditional caching.

Key insight: Cache structured intermediate representations (intents, plans, tool patterns)
at each reasoning stage. Similar queries reuse cached reasoning - massive LLM savings.

This example demonstrates:
1. Caching parsed intents (what the user wants)
2. Caching generated plans (how to achieve it)
3. Context-aware invalidation (cache invalidates when tools change)
4. Performance metrics (hit rate, entries, memory)
"""

import asyncio
import time

from cogent.memory import SemanticCache, compute_context_hash


def create_simple_embedding(text: str) -> list[float]:
    """Character frequency embedding for demo (use real embeddings in production).
    
    In production, use:
        from cogent.models import create_embedding
        embed = create_embedding("openai", "text-embedding-3-small")
        cache = SemanticCache(embedding_fn=embed.embed_query, ...)
    """
    chars = "abcdefghijklmnopqrstuvwxyz0123456789 "
    freq = [text.lower().count(c) / max(len(text), 1) for c in chars]
    # Normalize
    norm = sum(f * f for f in freq) ** 0.5
    return [f / norm if norm > 0 else 0 for f in freq]


async def main():
    print("=" * 60)
    print("SEMANTIC CACHE DEMO")
    print("Cache reasoning artifacts, not just responses")
    print("=" * 60)

    # Initialize cache
    cache = SemanticCache(
        embedding_fn=create_simple_embedding,
        similarity_threshold=0.80,  # 80% similarity for cache hit
        max_entries=1000,
        default_ttl=3600,  # 1 hour TTL
    )

    # Context hash - changes when tools/model change (invalidates cache)
    ctx = compute_context_hash(
        system_prompt="You are a research assistant",
        tool_names=["web_search", "summarize"],
        model_identifier="gpt-4o-mini",
    )

    print(f"\nContext hash: {ctx}")
    print("(Cache invalidates automatically if tools or model change)\n")

    # =========================================================================
    # DEMO 1: Intent Caching
    # =========================================================================
    print("-" * 60)
    print("DEMO 1: Intent Caching")
    print("-" * 60)

    # First query - cache MISS, parse intent
    query1 = "What are the best Python frameworks for web development?"
    print(f"\n📝 Query 1: '{query1}'")

    cached = await cache.get("intent", query1, ctx)
    if cached:
        print(f"   🎯 CACHE HIT - reusing intent")
        intent = cached.artifact
    else:
        print(f"   ❌ CACHE MISS - parsing intent (simulated LLM call)")
        # Simulate LLM parsing the intent
        intent = {
            "goal": "find_information",
            "topic": "python web frameworks",
            "type": "comparison",
            "entities": ["Python", "web development", "frameworks"],
        }
        await cache.put("intent", query1, intent, ctx)
        print(f"   💾 Stored intent in cache")

    print(f"   Intent: {intent}")

    # Second query - SEMANTICALLY SIMILAR - should hit cache!
    query2 = "Best Python web frameworks to use in 2026"
    print(f"\n📝 Query 2: '{query2}'")

    cached = await cache.get("intent", query2, ctx)
    if cached:
        print(f"   🎯 CACHE HIT! Matched: '{cached.semantic_key[:40]}...'")
        print(f"   Intent: {cached.artifact}")
        print(f"   ⚡ Saved 1 LLM call!")
    else:
        print(f"   ❌ CACHE MISS")

    # Third query - DIFFERENT topic - should miss
    query3 = "How to cook pasta carbonara"
    print(f"\n📝 Query 3: '{query3}'")

    cached = await cache.get("intent", query3, ctx)
    if cached:
        print(f"   🎯 CACHE HIT")
    else:
        print(f"   ❌ CACHE MISS (expected - completely different topic)")

    # =========================================================================
    # DEMO 2: Plan Caching
    # =========================================================================
    print("\n" + "-" * 60)
    print("DEMO 2: Plan Caching")
    print("-" * 60)

    # Store a plan
    plan_query = "research python web frameworks"
    plan = {
        "steps": [
            {"tool": "web_search", "query": "python web frameworks comparison 2026"},
            {"tool": "web_search", "query": "django vs fastapi vs flask"},
            {"tool": "summarize", "input": "search_results"},
        ],
        "estimated_calls": 3,
    }

    print(f"\n📝 Storing plan for: '{plan_query}'")
    await cache.put("plan", plan_query, plan, ctx)
    print(f"   💾 Plan stored: {len(plan['steps'])} steps")

    # Similar request should hit
    similar_query = "investigate python frameworks for web apps"
    print(f"\n📝 Query: '{similar_query}'")

    cached = await cache.get("plan", similar_query, ctx)
    if cached:
        print(f"   🎯 CACHE HIT! Reusing plan:")
        for i, step in enumerate(cached.artifact["steps"], 1):
            print(f"      Step {i}: {step['tool']}({step.get('query', step.get('input', ''))})")
        print(f"   ⚡ Saved {cached.artifact['estimated_calls']} LLM calls!")

    # =========================================================================
    # DEMO 3: Context Invalidation
    # =========================================================================
    print("\n" + "-" * 60)
    print("DEMO 3: Context Invalidation")
    print("-" * 60)

    # If tools change, context hash changes, cache misses
    new_ctx = compute_context_hash(
        system_prompt="You are a research assistant",
        tool_names=["web_search"],  # Removed 'summarize'!
        model_identifier="gpt-4o-mini",
    )

    print(f"\n📝 Tools changed: removed 'summarize'")
    print(f"   Old context: {ctx}")
    print(f"   New context: {new_ctx}")

    cached = await cache.get("plan", plan_query, new_ctx)
    if cached:
        print(f"   🎯 CACHE HIT")
    else:
        print(f"   ❌ CACHE MISS (correct! old plan used 'summarize' which is gone)")

    # =========================================================================
    # METRICS
    # =========================================================================
    print("\n" + "=" * 60)
    print("CACHE METRICS")
    print("=" * 60)

    metrics = cache.get_metrics()
    print(f"""
Hit Rate:       {metrics['cache_hit_rate']:.1%}
Total Hits:     {metrics['cache_hits']}
Total Misses:   {metrics['cache_misses']}
Total Requests: {metrics['total_requests']}
Cache Entries:  {metrics['total_entries']}
Cache Stages:   {metrics['stages']}
Memory (est):   {metrics['memory_mb']:.2f} MB
""")

    # =========================================================================
    # PERFORMANCE SIMULATION
    # =========================================================================
    print("-" * 60)
    print("PERFORMANCE SIMULATION: 100 similar queries")
    print("-" * 60)

    # Simulate 100 queries with variations
    base_queries = [
        "python web frameworks",
        "best python frameworks",
        "python for web development",
        "python web app frameworks",
        "frameworks for python web",
    ]

    start = time.perf_counter()
    hits = 0

    for i in range(100):
        q = base_queries[i % len(base_queries)] + f" variant {i}"
        cached = await cache.get("intent", q, ctx)
        if cached:
            hits += 1
        else:
            # Store for future hits
            await cache.put("intent", q, {"goal": "find_info", "topic": "python"}, ctx)

    elapsed = time.perf_counter() - start

    print(f"\nProcessed 100 queries in {elapsed*1000:.1f}ms")
    print(f"Cache hits: {hits}/100 ({hits}%)")
    print(f"LLM calls saved: {hits}")

    final_metrics = cache.get_metrics()
    print(f"\nFinal hit rate: {final_metrics['cache_hit_rate']:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
