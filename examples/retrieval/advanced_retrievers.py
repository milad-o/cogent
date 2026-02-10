"""Advanced Retrievers - Specialized retrieval patterns.

Demonstrates advanced retrieval strategies:
1. HyDERetriever - Hypothetical Document Embeddings
2. SelfQueryRetriever - Natural language to structured filters
3. SentenceWindowRetriever - Sentence-level with context
4. TimeBasedRetriever - Time-decay scoring for recency

Each example showcases unique capabilities for specific use cases.

Usage:
    uv run python examples/retrieval/advanced_retrievers.py
"""

import asyncio
from datetime import UTC, datetime, timedelta

from _shared import (
    LONG_DOCUMENT,
    SAMPLE_DOCS,
    create_vectorstore,
    print_header,
    print_results,
)

from cogent import create_chat
from cogent.retriever import (
    AttributeInfo,
    DecayFunction,
    DenseRetriever,
    Document,
    HyDERetriever,
    SelfQueryRetriever,
    SentenceWindowRetriever,
    TimeBasedRetriever,
)


async def demo_hyde_retriever() -> None:
    """Generate hypothetical answer, then search with it."""
    print_header("1. HyDE Retriever (Hypothetical Documents)")

    # Create and populate vectorstore
    vs = create_vectorstore()
    await vs.add_documents(SAMPLE_DOCS)
    base = DenseRetriever(vs)

    # Wrap with HyDE
    hyde = HyDERetriever(
        base_retriever=base,
        model=create_chat("gpt-4o-mini"),
    )

    query = "How do coral reefs protect coastlines?"
    print(f"\nQuery: '{query}'")
    print("(HyDE generates hypothetical answer, then searches)")

    results = await hyde.retrieve(query, k=2, include_scores=True)
    print_results(results)


async def demo_self_query_retriever() -> None:
    """Parse natural language into semantic query + metadata filters."""
    print_header("2. Self-Query Retriever (NL → Filters)")

    # Create and populate vectorstore
    vs = create_vectorstore()
    await vs.add_documents(SAMPLE_DOCS)

    retriever = SelfQueryRetriever(
        vectorstore=vs,
        llm=create_chat("gpt-4o-mini"),
        attribute_info=[
            AttributeInfo(
                "category",
                "Type: astronomy, nature, history, biology, physics",
                "string",
            ),
            AttributeInfo("author", "Author or team name", "string"),
            AttributeInfo(
                "severity", "Severity level: low, medium, high, critical", "string"
            ),
        ],
    )

    query = "What are the critical biological processes?"
    print(f"\nQuery: '{query}'")
    print("(Automatically extracts filter: severity='critical')")

    results = await retriever.retrieve(query, k=2, include_scores=True)
    print_results(results)


async def demo_sentence_window_retriever() -> None:
    """Index sentences, return with surrounding context window."""
    print_header("3. Sentence Window Retriever")

    vs = create_vectorstore()
    retriever = SentenceWindowRetriever(
        vectorstore=vs,
        window_size=2,  # 2 sentences before + after
    )

    await retriever.add_documents([LONG_DOCUMENT])

    query = "zooxanthellae symbiotic relationship"
    print(f"\nQuery: '{query}'")
    print("(Matches sentence, returns with ±2 sentence context)")

    results = await retriever.retrieve(query, k=1, include_scores=True)
    for r in results:
        print(f"  Score: {r.score:.3f}")
        print(f"  Context window: {r.document.text[:200]}...")


async def demo_time_based_retriever() -> None:
    """Score documents with time-decay for recency."""
    print_header("4. Time-Based Retriever (Recency Scoring)")

    # Create docs with different timestamps
    now = datetime.now(UTC)
    time_docs = [
        Document(
            text="Recent discovery: exoplanet with water vapor detected",
            metadata={"timestamp": (now - timedelta(hours=2)).isoformat()},
        ),
        Document(
            text="Monthly climate report: coral bleaching patterns",
            metadata={"timestamp": (now - timedelta(days=30)).isoformat()},
        ),
        Document(
            text="Ancient Roman engineering: aqueduct construction methods",
            metadata={"timestamp": (now - timedelta(days=180)).isoformat()},
        ),
    ]

    time_retriever = TimeBasedRetriever(
        vectorstore=create_vectorstore(),
        decay_function=DecayFunction.EXPONENTIAL,
        decay_rate=0.1,  # Faster decay = more recent bias
    )

    await time_retriever.add_documents(time_docs)

    query = "exoplanet discoveries"
    print(f"\nQuery: '{query}'")
    print("(Recent docs scored higher with exponential decay)")

    results = await time_retriever.retrieve(query, k=3, include_scores=True)
    for r in results:
        ts_str = r.document.metadata.get("timestamp")
        if ts_str:
            ts = datetime.fromisoformat(ts_str)
            age = (now - ts).days
        else:
            age = 0
        print(f"  [{r.score:.3f}] {r.document.text} (age: {age} days)")


async def main() -> None:
    """Run all advanced retriever examples."""
    await demo_hyde_retriever()
    await demo_self_query_retriever()
    await demo_sentence_window_retriever()
    await demo_time_based_retriever()

    print("\n" + "=" * 60)
    print("✓ All advanced retriever examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
