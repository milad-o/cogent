"""Basic Retrievers - Core retrieval patterns.

Demonstrates the fundamental retrieval strategies:
1. DenseRetriever - Semantic vector search
2. BM25Retriever - Keyword-based sparse retrieval
3. TFIDFRetriever - TF-IDF sparse retrieval
4. HybridRetriever - Metadata + content fusion
5. EnsembleRetriever - Combine multiple retrievers
6. ParentDocumentRetriever - Index chunks, return full docs

Each example is compact (~15-25 lines) and runnable independently.

Usage:
    uv run python examples/retrieval/basic_retrievers.py
"""

import asyncio

from cogent.retriever import (
    BM25Retriever,
    DenseRetriever,
    EnsembleRetriever,
    FusionStrategy,
    HybridRetriever,
    MetadataMatchMode,
    ParentDocumentRetriever,
    TFIDFRetriever,
)

from _shared import (
    LONG_DOCS,
    SAMPLE_DOCS,
    create_vectorstore,
    print_header,
    print_results,
)


async def demo_dense_retriever() -> None:
    """Basic semantic search using vector embeddings."""
    print_header("1. Dense Retriever (Semantic Search)")
    
    # Create vectorstore and add documents (explicit!)
    vs = create_vectorstore()
    await vs.add_documents(SAMPLE_DOCS)
    
    retriever = DenseRetriever(vs)
    
    query = "What happens when massive stars collapse?"
    print(f"\nQuery: '{query}'")
    
    results = await retriever.retrieve(query, k=2, include_scores=True)
    print_results(results)


async def demo_bm25_retriever() -> None:
    """Keyword-based sparse retrieval (BM25 algorithm)."""
    print_header("2. BM25 Retriever (Keyword Search)")
    
    retriever = BM25Retriever(documents=SAMPLE_DOCS)
    
    query = "rainforest biodiversity and species"
    print(f"\nQuery: '{query}'")
    
    results = await retriever.retrieve(query, k=2, include_scores=True)
    print_results(results)


async def demo_tfidf_retriever() -> None:
    """TF-IDF sparse retrieval (simpler than BM25)."""
    print_header("3. TF-IDF Retriever (Term Weighting)")
    
    retriever = TFIDFRetriever(documents=SAMPLE_DOCS)
    
    query = "photosynthesis and oxygen production"
    print(f"\nQuery: '{query}'")
    
    results = await retriever.retrieve(query, k=2, include_scores=True)
    print_results(results)


async def demo_hybrid_retriever() -> None:
    """Combine metadata filtering with content search."""
    print_header("4. Hybrid Retriever (Metadata + Content)")
    
    # Create vectorstore with documents
    vs = create_vectorstore()
    await vs.add_documents(SAMPLE_DOCS)
    dense = DenseRetriever(vs)
    
    # Use metadata fields that ALL documents have
    hybrid = HybridRetriever(
        retriever=dense,
        metadata_fields=["category", "department", "author"],
        metadata_weight=0.3,  # 30% metadata, 70% content
        content_weight=0.7,
        mode=MetadataMatchMode.BOOST,
    )
    
    query = "ecosystems from the Environmental Science department"
    print(f"\nQuery: '{query}'")
    print("(Boosts results matching 'Environmental Science' in department field)")
    
    results = await hybrid.retrieve(query, k=2, include_scores=True)
    print_results(results)


async def demo_ensemble_retriever() -> None:
    """Combine multiple retrieval strategies with fusion."""
    print_header("4. Ensemble Retriever (Multi-Strategy)")
    
    # Dense retriever with vectorstore
    vs = create_vectorstore()
    await vs.add_documents(SAMPLE_DOCS)
    dense = DenseRetriever(vs)
    
    # BM25 retriever with same documents
    bm25 = BM25Retriever(documents=SAMPLE_DOCS)
    
    ensemble = EnsembleRetriever(
        retrievers=[dense, bm25],
        weights=[0.6, 0.4],
        fusion=FusionStrategy.RRF,  # Reciprocal Rank Fusion
    )
    
    query = "planets and solar system formation"
    print(f"\nQuery: '{query}'")
    
    results = await ensemble.retrieve(query, k=3, include_scores=True)
    print_results(results)


async def demo_parent_document_retriever() -> None:
    """Index small chunks but retrieve full parent documents."""
    print_header("5. Parent Document Retriever")
    
    vs = create_vectorstore()
    retriever = ParentDocumentRetriever(
        vectorstore=vs,
        chunk_size=150,  # Small chunks for precise matching
        chunk_overlap=20,
    )
    
    # Add multiple long documents - chunks indexed, full docs returned
    await retriever.add_documents(LONG_DOCS)
    
    query = "Roman military roads and infrastructure"
    print(f"\nQuery: '{query}'")
    print("(Matches small chunk but returns entire parent document)")
    
    results = await retriever.retrieve(query, k=2, include_scores=True)
    for i, r in enumerate(results, 1):
        chunk_count = r.metadata.get('matching_chunks', '?')
        print(f"\n  {i}. Score: {r.score:.3f} | Matched {chunk_count} chunks")
        print(f"     Source: {r.document.metadata.get('source')}")
        print(f"     Full doc length: {len(r.document.text)} chars")
        print(f"     Preview: {r.document.text[:140]}...")


async def main() -> None:
    """Run all basic retriever examples."""
    await demo_dense_retriever()
    await demo_bm25_retriever()
    await demo_tfidf_retriever()
    await demo_hybrid_retriever()
    await demo_ensemble_retriever()
    await demo_parent_document_retriever()
    
    print("\n" + "=" * 60)
    print("✓ All basic retriever examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
