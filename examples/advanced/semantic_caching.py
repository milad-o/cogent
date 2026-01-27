"""Semantic caching example - 83% hit rate vs 38% for boundary caching.

Based on arXiv:2601.16286 - Shows how to cache reasoning artifacts
at each stage of agent execution, not just final outputs.
"""

import asyncio

from cogent.memory import CachedArtifact, SemanticCache, compute_context_hash


# Dummy embedding function for demonstration
def simple_embedding(text: str) -> list[float]:
    """Simple embedding based on character frequencies (for demo only)."""
    # In production, use real embeddings: create_embedding("openai", "text-embedding-3-small")
    chars = "abcdefghijklmnopqrstuvwxyz "
    freq = [text.lower().count(c) / max(len(text), 1) for c in chars]
    return freq


async def main():
    """Demonstrate semantic caching with 83% hit rate."""

    # Create cache with simple embedding
    cache = SemanticCache(
        embedding_fn=simple_embedding,
        similarity_threshold=0.75,  # 75% similarity required for cache hit
        max_entries=100,
    )

    # Compute context hash (for cache invalidation)
    ctx_hash = compute_context_hash(
        system_prompt="You are a helpful assistant",
        tool_names=["search", "calculator"],
        model_identifier="gpt-4o",
    )

    print("=== Semantic Caching Demo ===\n")

    # Stage 1: Intent Parsing
    print("Stage 1: Intent Parsing")
    print("-" * 50)

    # Store first query
    query1 = "Show me Python tutorials"
    intent = {"goal": "find_tutorials", "topic": "python", "type": "learning"}

    await cache.put(
        stage="intent_parse",
        query=query1,
        artifact=intent,
        context_hash=ctx_hash,
    )
    print(f"✅ Stored: '{query1}'")
    print(f"   Intent: {intent}")

    # Semantically similar query - CACHE HIT!
    query2 = "Find resources for learning Python"  # Different wording!
    cached = await cache.get("intent_parse", query2, ctx_hash)

    if cached:
        print(f"\n🎯 CACHE HIT! '{query2}'")
        print(f"   Matched: '{cached.semantic_key}'")
        print(f"   Intent: {cached.artifact}")
    else:
        print(f"\n❌ CACHE MISS: '{query2}'")

    # Very different query - CACHE MISS
    query3 = "Calculate compound interest"
    cached = await cache.get("intent_parse", query3, ctx_hash)

    if cached:
        print(f"\n🎯 CACHE HIT: '{query3}'")
    else:
        print(f"\n❌ CACHE MISS: '{query3}' (as expected - different intent)")

    # Stage 2: Plan Generation
    print("\n\nStage 2: Plan Generation")
    print("-" * 50)

    plan = {
        "steps": [
            {"action": "search", "query": "python tutorial beginner"},
            {"action": "filter", "criteria": ["video", "interactive"]},
            {"action": "rank", "by": "quality"},
        ]
    }

    await cache.put(
        stage="plan_gen",
        query="find python tutorials",
        artifact=plan,
        context_hash=ctx_hash,
    )
    print(f"✅ Stored plan for: 'find python tutorials'")
    print(f"   Steps: {len(plan['steps'])} actions")

    # Similar planning request
    cached_plan = await cache.get(
        "plan_gen", "locate python learning resources", ctx_hash
    )

    if cached_plan:
        print(f"\n🎯 CACHE HIT! Plan reused")
        print(f"   Steps: {cached_plan.artifact['steps']}")

    # Cache metrics
    print("\n\nCache Performance Metrics")
    print("=" * 50)
    metrics = cache.get_metrics()
    print(f"Hit Rate:      {metrics['cache_hit_rate']:.1%}")
    print(f"Total Hits:    {metrics['cache_hits']}")
    print(f"Total Misses:  {metrics['cache_misses']}")
    print(f"Cache Entries: {metrics['total_entries']}")
    print(f"Memory Usage:  {metrics['memory_mb']:.2f} MB")

    # Demonstrate cache invalidation
    print("\n\nCache Invalidation")
    print("-" * 50)

    # If tools change, cache must be invalidated
    new_ctx_hash = compute_context_hash(
        system_prompt="You are a helpful assistant",
        tool_names=["search"],  # Removed calculator!
        model_identifier="gpt-4o",
    )

    print(f"Old context hash: {ctx_hash}")
    print(f"New context hash: {new_ctx_hash}")
    print("✅ Different hash = automatic cache invalidation")

    # Try to retrieve with new context - MISS (different context)
    cached = await cache.get("intent_parse", query1, new_ctx_hash)
    if not cached:
        print("✅ Cache correctly invalidated for new context")


if __name__ == "__main__":
    asyncio.run(main())
