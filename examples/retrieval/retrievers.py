"""
Advanced Retrievers

Demonstrates core retriever types in Cogent:
- Dense/BM25/Hybrid/Ensemble retrievers
- Contextual retrievers (ParentDocument, SentenceWindow)
- LLM-powered indexes (Summary, Keyword, SelfQuery)
- Specialized indexes (Time-based, Multi-representation)
- Rerankers

Usage: uv run python examples/retrieval/retrievers.py
"""

import asyncio

from cogent.models import OpenAIEmbedding
from cogent.retriever import (
    AttributeInfo,
    BM25Retriever,
    DecayFunction,
    DenseRetriever,
    EnsembleRetriever,
    FusionStrategy,
    HybridRetriever,
    KeywordTableIndex,
    LLMReranker,
    MetadataMatchMode,
    MultiRepresentationIndex,
    ParentDocumentRetriever,
    SelfQueryRetriever,
    SentenceWindowRetriever,
    SummaryIndex,
    TimeBasedIndex,
)
from cogent.vectorstore import Document, VectorStore

DOCUMENTS = [
    Document(
        text="""Python 3.13 introduces free-threading (experimental no-GIL mode),
improved error messages, and new REPL with multiline editing. Performance
improvements averaging 5% faster execution.""",
        metadata={"source": "python_release.md", "category": "release", "date": "2024-12-01"},
    ),
    Document(
        text="""ML Best Practices: Clean data, handle missing values, start with
simple baselines, use cross-validation, monitor for drift in production.""",
        metadata={"source": "ml_guide.md", "category": "guide", "author": "Data Science Team"},
    ),
    Document(
        text="""API Rate Limiting: Implement token bucket algorithm. Return 429 status
with Retry-After header. Log violations. Consider tiered limits by user plan.""",
        metadata={"source": "api_policy.md", "category": "policy", "date": "2024-06-01"},
    ),
    Document(
        text="""Incident Report (March 2024): Database connection pool exhaustion caused
30-min outage. Root cause: missing connection timeout. Fix: Added timeout, increased
pool size, updated code review checklist.""",
        metadata={"source": "incident_march.md", "category": "incident", "severity": "high", "date": "2024-03-15"},
    ),
]

LONG_DOCUMENT = Document(
    text="""Understanding Transformer Architecture

The transformer architecture revolutionized NLP. Unlike RNNs, transformers process
all tokens in parallel using self-attention mechanisms.

Self-Attention Mechanism

Self-attention allows each token to attend to all others. The attention score is
computed as the dot product of query and key vectors, scaled by sqrt(d_k).

Multi-Head Attention

Multi-head attention runs multiple attention operations in parallel. Each head learns
different aspects of relationships. Typically models use 8-16 heads.

Position Encoding

Since transformers process tokens in parallel, they need positional information.
Original paper used sinusoidal encodings; modern models often use learned embeddings
or RoPE.

Feed-Forward Networks

After attention, each position passes through a feed-forward network with two linear
transformations and ReLU/GELU activation. Hidden dimension is usually 4x model dimension.

Layer Normalization

Transformers use layer normalization to stabilize training. Original architecture used
post-norm; modern architectures often use pre-norm for better convergence.""",
    metadata={"source": "transformer_guide.md", "category": "education"},
)


async def demo_dense_retriever() -> None:
    """Basic dense (semantic) retrieval."""
    print("\n" + "=" * 60)
    print("1. DENSE RETRIEVER (Semantic Search)")
    print("=" * 60)

    embeddings = OpenAIEmbedding(model="text-embedding-3-small")
    vectorstore = VectorStore(embeddings=embeddings)
    await vectorstore.add_documents(DOCUMENTS)

    retriever = DenseRetriever(vectorstore)
    query = "How to build machine learning models?"
    print(f"\nQuery: '{query}'")

    results = await retriever.retrieve(query, k=2, include_scores=True)
    for r in results:
        print(f"  [{r.score:.3f}] {r.document.metadata['source']}")


async def demo_bm25_retriever() -> None:
    """BM25 lexical retrieval."""
    print("\n" + "=" * 60)
    print("2. BM25 RETRIEVER (Keyword Search)")
    print("=" * 60)

    retriever = BM25Retriever(k1=1.5, b=0.75)
    retriever.add_documents(DOCUMENTS)

    query = "rate limits API requests"
    print(f"\nQuery: '{query}'")

    results = await retriever.retrieve(query, k=2, include_scores=True)
    for r in results:
        print(f"  [{r.score:.3f}] {r.document.metadata['source']}")


async def demo_hybrid_retriever() -> None:
    """Hybrid retrieval (metadata + content)."""
    print("\n" + "=" * 60)
    print("3. HYBRID RETRIEVER (Metadata + Content)")
    print("=" * 60)

    embeddings = OpenAIEmbedding(model="text-embedding-3-small")
    vectorstore = VectorStore(embeddings=embeddings)
    await vectorstore.add_documents(DOCUMENTS)

    dense = DenseRetriever(vectorstore)
    retriever = HybridRetriever(
        retriever=dense,
        metadata_fields=["category", "author"],
        metadata_weight=0.3,
        content_weight=0.7,
        mode=MetadataMatchMode.BOOST,
    )

    query = "machine learning best practices"
    print(f"\nQuery: '{query}'")

    results = await retriever.retrieve(query, k=2, include_scores=True)
    for r in results:
        print(f"  [{r.score:.3f}] {r.document.metadata['source']}")


async def demo_ensemble_retriever() -> None:
    """Ensemble retrieval with multiple strategies."""
    print("\n" + "=" * 60)
    print("4. ENSEMBLE RETRIEVER (Dense + Sparse)")
    print("=" * 60)

    embeddings = OpenAIEmbedding(model="text-embedding-3-small")
    vectorstore = VectorStore(embeddings=embeddings)
    await vectorstore.add_documents(DOCUMENTS)

    dense = DenseRetriever(vectorstore)
    sparse = BM25Retriever()
    sparse.add_documents(DOCUMENTS)

    ensemble = EnsembleRetriever(
        retrievers=[dense, sparse],
        weights=[0.6, 0.4],
        fusion=FusionStrategy.RRF,
    )

    query = "database connection issues"
    print(f"\nQuery: '{query}'")

    results = await ensemble.retrieve(query, k=2, include_scores=True)
    for r in results:
        print(f"  [{r.score:.3f}] {r.document.metadata['source']}")


async def demo_parent_document_retriever() -> None:
    """Parent document retrieval - index small chunks, return full documents."""
    print("\n" + "=" * 60)
    print("5. PARENT DOCUMENT RETRIEVER")
    print("=" * 60)

    embeddings = OpenAIEmbedding(model="text-embedding-3-small")
    vectorstore = VectorStore(embeddings=embeddings)

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        chunk_size=200,
        chunk_overlap=30,
    )

    await retriever.add_documents([LONG_DOCUMENT])

    query = "What is the attention formula?"
    print(f"\nQuery: '{query}'")

    results = await retriever.retrieve(query, k=1, include_scores=True)
    for r in results:
        print(f"  Score: {r.score:.3f}")
        print(f"  Full document returned ({len(r.document.text)} chars)")


async def demo_sentence_window_retriever() -> None:
    """Sentence window retrieval - index sentences, return with context."""
    print("\n" + "=" * 60)
    print("6. SENTENCE WINDOW RETRIEVER")
    print("=" * 60)

    embeddings = OpenAIEmbedding(model="text-embedding-3-small")
    vectorstore = VectorStore(embeddings=embeddings)

    retriever = SentenceWindowRetriever(
        vectorstore=vectorstore,
        window_size=2,
    )

    await retriever.add_documents([LONG_DOCUMENT])

    query = "number of attention heads"
    print(f"\nQuery: '{query}'")

    results = await retriever.retrieve(query, k=1, include_scores=True)
    for r in results:
        print(f"  Score: {r.score:.3f}")
        print(f"  Window: {r.document.text[:150]}...")


async def demo_summary_index() -> None:
    """LLM-powered summary index."""
    print("\n" + "=" * 60)
    print("7. SUMMARY INDEX (LLM-Generated)")
    print("=" * 60)

    embeddings = OpenAIEmbedding(model="text-embedding-3-small")
    vectorstore = VectorStore(embeddings=embeddings)

    index = SummaryIndex(
        llm="gpt4",
        vectorstore=vectorstore,
        extract_entities=True,
        extract_keywords=True,
    )

    print("\nGenerating summaries...")
    await index.add_documents(DOCUMENTS[:2])

    query = "What are the Python 3.13 features?"
    print(f"Query: '{query}'")

    results = await index.retrieve(query, k=1, include_scores=True)
    for r in results:
        print(f"  Score: {r.score:.3f}, Source: {r.document.metadata['source']}")


async def demo_keyword_table_index() -> None:
    """Keyword table index with LLM extraction."""
    print("\n" + "=" * 60)
    print("8. KEYWORD TABLE INDEX")
    print("=" * 60)

    index = KeywordTableIndex(llm="gpt4", max_keywords_per_doc=8)

    print("\nExtracting keywords...")
    await index.add_documents(DOCUMENTS[:2])

    query = "Python features"
    print(f"Query: '{query}'")

    results = await index.retrieve(query, k=2, include_scores=True)
    for r in results:
        print(f"  [{r.score:.3f}] {r.document.metadata['source']}")


async def demo_self_query_retriever() -> None:
    """Self-query retrieval with natural language filters."""
    print("\n" + "=" * 60)
    print("9. SELF-QUERY RETRIEVER (NL → Filters)")
    print("=" * 60)

    embeddings = OpenAIEmbedding(model="text-embedding-3-small")
    vectorstore = VectorStore(embeddings=embeddings)
    await vectorstore.add_documents(DOCUMENTS)

    retriever = SelfQueryRetriever(
        vectorstore=vectorstore,
        llm="gpt4",
        attribute_info=[
            AttributeInfo("category", "Document type: release, guide, policy, incident", "string"),
            AttributeInfo("author", "Author or team name", "string"),
        ],
    )

    query = "What are the guidelines from the Data Science Team?"
    print(f"\nQuery: '{query}'")

    results = await retriever.retrieve(query, k=2, include_scores=True)
    for r in results:
        print(f"  [{r.score:.3f}] {r.document.metadata['source']}")


async def demo_time_based_index() -> None:
    """Time-aware retrieval with recency boost."""
    print("\n" + "=" * 60)
    print("10. TIME-BASED INDEX (Recency-Aware)")
    print("=" * 60)

    embeddings = OpenAIEmbedding(model="text-embedding-3-small")
    vectorstore = VectorStore(embeddings=embeddings)

    index = TimeBasedIndex(
        vectorstore=vectorstore,
        decay_function=DecayFunction.EXPONENTIAL,
        decay_rate=0.01,
        auto_extract_timestamps=True,
    )

    await index.add_documents(DOCUMENTS)

    query = "API documentation"
    print(f"\nQuery: '{query}'")

    results = await index.retrieve(query, k=2, include_scores=True)
    for r in results:
        date = r.document.metadata.get("date", "unknown")
        print(f"  [{r.score:.3f}] {date} - {r.document.metadata['source']}")


async def demo_multi_representation_index() -> None:
    """Multi-representation retrieval - multiple embeddings per doc."""
    print("\n" + "=" * 60)
    print("11. MULTI-REPRESENTATION INDEX")
    print("=" * 60)

    embeddings = OpenAIEmbedding(model="text-embedding-3-small")
    vectorstore = VectorStore(embeddings=embeddings)

    index = MultiRepresentationIndex(
        vectorstore=vectorstore,
        llm="gpt4",
        representations=["original", "summary", "detailed"],
    )

    print("\nGenerating representations...")
    await index.add_documents(DOCUMENTS[:2])

    query = "What is Python?"
    print(f"\nQuery: '{query}'")

    results = await index.retrieve(query, k=1, include_scores=True)
    for r in results:
        print(f"  [{r.score:.3f}] {r.document.metadata['source']}")


async def demo_reranker() -> None:
    """Reranking for improved relevance."""
    print("\n" + "=" * 60)
    print("12. LLM RERANKER")
    print("=" * 60)

    embeddings = OpenAIEmbedding(model="text-embedding-3-small")
    vectorstore = VectorStore(embeddings=embeddings)
    await vectorstore.add_documents(DOCUMENTS)

    retriever = DenseRetriever(vectorstore)
    reranker = LLMReranker(model="gpt4")

    query = "How do I handle too many API requests?"

    initial = await retriever.retrieve(query, k=3, include_scores=True)
    print(f"\nQuery: '{query}'")
    print("Before reranking:")
    for i, r in enumerate(initial, 1):
        print(f"  {i}. [{r.score:.3f}] {r.document.metadata['source']}")

    docs = [r.document for r in initial]
    reranked = await reranker.rerank(query, docs, top_n=2)
    print("\nAfter reranking:")
    for i, r in enumerate(reranked, 1):
        print(f"  {i}. [{r.score:.3f}] {r.document.metadata['source']}")


async def main() -> None:
    """Run all retriever demos."""
    print("\n" + "=" * 60)
    print("AGENTICFLOW RETRIEVERS DEMO")
    print("=" * 60)

    await demo_dense_retriever()
    await demo_bm25_retriever()
    await demo_hybrid_retriever()
    await demo_ensemble_retriever()
    await demo_parent_document_retriever()
    await demo_sentence_window_retriever()
    await demo_summary_index()
    await demo_keyword_table_index()
    await demo_self_query_retriever()
    await demo_time_based_index()
    await demo_multi_representation_index()
    await demo_reranker()

    print("\n" + "=" * 60)
    print("✓ All demos completed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
