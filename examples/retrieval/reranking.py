"""Reranking - Improve retrieval precision with rerankers.

Demonstrates different reranking strategies to refine initial retrieval results:
1. CrossEncoderReranker - Local neural reranker
2. CohereReranker - Cohere's rerank API
3. LLMReranker - Use any LLM for scoring

Rerankers are typically applied after initial retrieval to improve precision.

Usage:
    uv run python examples/retrieval/reranking.py
"""

import asyncio

from _shared import SAMPLE_DOCS, create_vectorstore, print_header

from cogent import create_chat
from cogent.retriever import (
    BM25Retriever,
    CohereReranker,
    CrossEncoderReranker,
    DenseRetriever,
    LLMReranker,
)


async def demo_cross_encoder_reranker() -> None:
    """Use local cross-encoder model for reranking."""
    print_header("1. Cross-Encoder Reranker (Local)")

    # Initial retrieval with BM25
    retriever = BM25Retriever(documents=SAMPLE_DOCS)
    query = "database performance issues"

    initial = await retriever.retrieve(query, k=4)
    print(f"\nQuery: '{query}'")
    print(f"Initial retrieval: {len(initial)} results")

    # Rerank with cross-encoder
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    reranked = await reranker.rerank(query, initial, top_k=2)

    print("\nReranked results:")
    for i, r in enumerate(reranked, 1):
        source = r.document.metadata.get("source", "unknown")
        print(f"  {i}. [{r.score:.3f}] {source}")
        print(f"     {r.document.text[:80]}...")


async def demo_cohere_reranker() -> None:
    """Use Cohere's rerank API for precision."""
    print_header("2. Cohere Reranker (API)")

    # Create and populate vectorstore
    vs = create_vectorstore()
    await vs.add_documents(SAMPLE_DOCS)
    retriever = DenseRetriever(vs)

    query = "security vulnerabilities and fixes"
    initial = await retriever.retrieve(query, k=5)

    print(f"\nQuery: '{query}'")
    print(f"Initial retrieval: {len(initial)} results")

    # Rerank with Cohere
    reranker = CohereReranker(model="rerank-english-v3.0")

    reranked = await reranker.rerank(query, initial, top_k=3)

    print("\nReranked results:")
    for i, r in enumerate(reranked, 1):
        source = r.document.metadata.get("source", "unknown")
        print(f"  {i}. [{r.score:.3f}] {source}")
        print(f"     {r.document.text[:80]}...")


async def demo_llm_reranker() -> None:
    """Use any LLM to score and rerank results."""
    print_header("3. LLM Reranker (Flexible)")

    retriever = BM25Retriever(documents=SAMPLE_DOCS)

    query = "Python programming improvements"
    initial = await retriever.retrieve(query, k=4)

    print(f"\nQuery: '{query}'")
    print(f"Initial retrieval: {len(initial)} results")

    # Rerank using LLM
    reranker = LLMReranker(
        model=create_chat("gpt-4o-mini"),
        prompt_template="""Rate how relevant this document is to the query on a scale of 0-100.
        
Query: {query}
Document: {document}

Relevance score (0-100):""",
    )

    reranked = await reranker.rerank(query, initial, top_k=2)

    print("\nLLM reranked results:")
    for i, r in enumerate(reranked, 1):
        source = r.document.metadata.get("source", "unknown")
        print(f"  {i}. [{r.score:.3f}] {source}")
        print(f"     {r.document.text[:80]}...")


async def demo_retrieval_with_reranking() -> None:
    """Complete workflow: retrieve → rerank."""
    print_header("4. Complete Workflow (Retrieve + Rerank)")

    # Step 1: Initial broad retrieval
    vs = create_vectorstore()
    await vs.add_documents(SAMPLE_DOCS)
    retriever = DenseRetriever(vs)

    query = "incident response and fixes"
    print(f"\nQuery: '{query}'")

    # Retrieve top-5 candidates for reranking
    candidates = await retriever.retrieve(query, k=5, include_scores=True)

    print(f"\nStep 1: Retrieved {len(candidates)} candidates")
    for i, r in enumerate(candidates[:3], 1):
        print(f"  {i}. [{r.score:.3f}] {r.document.metadata['source']}")

    # Step 2: Rerank to top-3
    reranker = LLMReranker(model=create_chat("gpt-4o-mini"))
    final = await reranker.rerank(query, candidates, top_k=2)

    print(f"\nStep 2: Reranked to top {len(final)}")
    for i, r in enumerate(final, 1):
        print(f"  {i}. [{r.score:.3f}] {r.document.metadata['source']}")
        print(f"     {r.document.text[:80]}...")


async def main() -> None:
    """Run all reranking examples."""
    await demo_cross_encoder_reranker()
    await demo_cohere_reranker()
    await demo_llm_reranker()
    await demo_retrieval_with_reranking()

    print("\n" + "=" * 60)
    print("✓ All reranking examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
