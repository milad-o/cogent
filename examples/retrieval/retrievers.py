"""
Example 37: Advanced Retrievers

Demonstrates all retriever types available in AgenticFlow:

1. Core Retrievers (Dense, BM25, Hybrid, Ensemble)
2. Contextual Retrievers (ParentDocument, SentenceWindow)
3. LLM-Powered Indexes (Summary, Keyword, SelfQuery)
4. Specialized Indexes (Time-Based, Multi-Representation)
5. Rerankers (CrossEncoder, LLM-based)

Usage:
    uv run python examples/retrieval/retrievers.py
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from config import get_embeddings, get_model

from agenticflow.vectorstore import VectorStore, Document
from agenticflow.retriever import (
    # Core retrievers
    DenseRetriever,
    BM25Retriever,
    HybridRetriever,
    MetadataMatchMode,
    EnsembleRetriever,
    FusionStrategy,
    # Contextual retrievers
    ParentDocumentRetriever,
    SentenceWindowRetriever,
    # LLM-powered indexes
    SummaryIndex,
    KeywordTableIndex,
    SelfQueryRetriever,
    AttributeInfo,
    # Specialized indexes
    TimeBasedIndex,
    TimeRange,
    DecayFunction,
    MultiRepresentationIndex,
    QueryType,
    # Rerankers
    LLMReranker,
    # Utilities
    fuse_results,
    normalize_scores,
)


# =============================================================================
# Sample Data: Tech Company Knowledge Base
# =============================================================================

DOCUMENTS = [
    Document(
        text="""Python 3.13 Release Notes (December 2024)

Python 3.13 introduces several exciting features including free-threading 
(experimental no-GIL mode), improved error messages, and the new REPL with 
multiline editing support. The release also includes performance improvements
averaging 5% faster execution.

Key Features:
- Free-threaded Python (PEP 703) - experimental
- Improved error messages with color support
- New interactive REPL with multiline editing
- JIT compiler foundation (internal)
- Deprecation of many legacy features

Migration Guide:
To upgrade from Python 3.12, ensure your dependencies support 3.13.
Test thoroughly with the free-threaded build if using multi-threading.""",
        metadata={
            "source": "python_release_notes.md",
            "category": "release",
            "version": "3.13",
            "date": "2024-12-01",
            "author": "Python Core Team",
        },
    ),
    Document(
        text="""Machine Learning Best Practices

When building ML pipelines, follow these guidelines:

1. Data Preparation
- Clean and validate input data
- Handle missing values appropriately  
- Normalize numerical features
- Encode categorical variables

2. Model Selection
- Start with simple baselines (linear models)
- Use cross-validation for hyperparameter tuning
- Consider ensemble methods for better performance

3. Evaluation
- Use appropriate metrics for your task
- Test on held-out data
- Monitor for data drift in production

4. Deployment
- Version your models
- Implement A/B testing
- Set up monitoring and alerting""",
        metadata={
            "source": "ml_best_practices.md",
            "category": "guide",
            "topic": "machine learning",
            "date": "2024-06-15",
            "author": "Data Science Team",
        },
    ),
    Document(
        text="""API Rate Limiting Policy (Updated January 2024)

Our API enforces the following rate limits:

Free Tier:
- 100 requests per minute
- 1,000 requests per day
- Max 10 concurrent connections

Pro Tier:
- 1,000 requests per minute
- Unlimited daily requests
- Max 100 concurrent connections

Enterprise Tier:
- Custom limits based on contract
- Dedicated infrastructure available
- SLA guarantees

Rate Limit Headers:
X-RateLimit-Limit: Your tier's limit
X-RateLimit-Remaining: Requests remaining
X-RateLimit-Reset: Unix timestamp for reset

Exceeding limits returns HTTP 429 with Retry-After header.""",
        metadata={
            "source": "api_rate_limits.md",
            "category": "policy",
            "topic": "api",
            "date": "2024-01-15",
            "author": "Platform Team",
        },
    ),
    Document(
        text="""Vector Database Comparison (2024)

We evaluated three vector databases for our RAG system:

Pinecone:
- Fully managed, easy setup
- Good performance up to 1M vectors
- Higher cost at scale
- Best for: Quick prototypes, small-medium scale

Qdrant:
- Self-hosted or cloud options
- Excellent filtering capabilities
- Lower cost, more control
- Best for: Production systems with filtering needs

Milvus:
- Open source, highly scalable
- Complex setup and operations
- Best performance at billion-scale
- Best for: Large enterprise deployments

Recommendation: Start with Qdrant for most use cases.""",
        metadata={
            "source": "vector_db_comparison.md",
            "category": "evaluation",
            "topic": "infrastructure",
            "date": "2024-08-20",
            "author": "Infrastructure Team",
        },
    ),
    Document(
        text="""Incident Report: Database Outage (March 2024)

Summary: Primary database became unresponsive for 45 minutes.

Timeline:
- 14:32 UTC: Alerts triggered for high latency
- 14:35 UTC: On-call engineer engaged
- 14:42 UTC: Root cause identified (connection pool exhaustion)
- 14:55 UTC: Connection pool increased, service recovering
- 15:17 UTC: Full recovery confirmed

Root Cause: A new feature released that morning had a connection 
leak bug. Each request opened a new connection without releasing it.

Resolution:
1. Hotfix deployed to fix connection leak
2. Connection pool size increased as temporary measure
3. Added connection pool monitoring

Prevention:
- Added integration tests for connection handling
- Implemented connection pool metrics dashboard
- Updated code review checklist""",
        metadata={
            "source": "incident_report_march.md",
            "category": "incident",
            "severity": "high",
            "date": "2024-03-15",
            "author": "SRE Team",
        },
    ),
]


# Long document for parent/sentence window demos
LONG_DOCUMENT = Document(
    text="""Understanding Transformer Architecture

The transformer architecture, introduced in 2017, revolutionized natural language 
processing. Unlike RNNs, transformers process all tokens in parallel using 
self-attention mechanisms.

Self-Attention Mechanism

Self-attention allows each token to attend to all other tokens in the sequence. 
The attention score is computed as the dot product of query and key vectors, 
scaled by the square root of the dimension. This enables the model to capture 
long-range dependencies efficiently.

The formula is: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

Multi-Head Attention

Instead of computing a single attention, multi-head attention runs multiple 
attention operations in parallel. Each "head" learns different aspects of 
the relationships between tokens. The outputs are concatenated and projected.

Typically, models use 8-16 attention heads. More heads allow finer-grained 
attention patterns but increase computation cost.

Position Encoding

Since transformers process tokens in parallel, they need positional information. 
The original paper used sinusoidal position encodings. Modern models often use 
learned position embeddings or rotary position embeddings (RoPE).

Feed-Forward Networks

After attention, each position passes through a feed-forward network. This 
typically consists of two linear transformations with a ReLU or GELU activation 
in between. The hidden dimension is usually 4x the model dimension.

Layer Normalization

Transformers use layer normalization to stabilize training. The original 
architecture placed LayerNorm after each sub-layer (post-norm). Modern 
architectures often use pre-norm, placing LayerNorm before attention and FFN.

Scaling Laws

Research has shown that transformer performance scales predictably with model 
size, dataset size, and compute. The scaling laws suggest that larger models 
trained on more data consistently perform better, following power-law curves.""",
    metadata={
        "source": "transformer_guide.md",
        "category": "education",
        "topic": "deep learning",
    },
)


async def demo_dense_retriever() -> None:
    """Demonstrate basic dense (semantic) retrieval."""
    print("\n" + "=" * 70)
    print("1. DENSE RETRIEVER (Semantic Search)")
    print("=" * 70)

    embeddings = get_embeddings()
    vectorstore = VectorStore(embeddings=embeddings)
    await vectorstore.add_documents(DOCUMENTS)

    retriever = DenseRetriever(vectorstore)

    # Semantic search - finds conceptually similar content
    query = "How to build machine learning models?"
    print(f"\nQuery: '{query}'")

    # Unified API: include_scores=True returns RetrievalResult with scores
    results = await retriever.retrieve(query, k=2, include_scores=True)
    for r in results:
        print(f"\n  [{r.score:.3f}] {r.document.metadata['source']}")
        print(f"  {r.document.text[:100]}...")


async def demo_bm25_retriever() -> None:
    """Demonstrate BM25 lexical retrieval."""
    print("\n" + "=" * 70)
    print("2. BM25 RETRIEVER (Keyword Search)")
    print("=" * 70)

    retriever = BM25Retriever(k1=1.5, b=0.75)
    retriever.add_documents(DOCUMENTS)

    # Keyword search - exact term matching
    query = "rate limits API requests"
    print(f"\nQuery: '{query}'")

    results = await retriever.retrieve(query, k=2, include_scores=True)
    for r in results:
        print(f"\n  [{r.score:.3f}] {r.document.metadata['source']}")
        print(f"  {r.document.text[:100]}...")


async def demo_hybrid_retriever() -> None:
    """Demonstrate hybrid retrieval (metadata + content)."""
    print("\n" + "=" * 70)
    print("3. HYBRID RETRIEVER (Metadata + Content)")
    print("=" * 70)

    embeddings = get_embeddings()
    vectorstore = VectorStore(embeddings=embeddings)
    await vectorstore.add_documents(DOCUMENTS)

    # Wrap a dense retriever with metadata search
    dense = DenseRetriever(vectorstore)
    retriever = HybridRetriever(
        retriever=dense,
        metadata_fields=["category", "author"],
        metadata_weight=0.3,
        content_weight=0.7,
        mode=MetadataMatchMode.BOOST,
    )

    # Query that benefits from both content AND metadata matching
    query = "Data Science machine learning best practices"
    print(f"\nQuery: '{query}'")
    print("(Boosts results where metadata fields match query terms)")

    results = await retriever.retrieve(query, k=3, include_scores=True)
    for r in results:
        print(f"\n  [{r.score:.3f}] {r.document.metadata['source']}")
        print(f"    Content: {r.metadata.get('content_score', 0):.3f}, Metadata: {r.metadata.get('metadata_score', 0):.3f}")
        print(f"    Author: {r.document.metadata.get('author', 'N/A')}")


async def demo_ensemble_retriever() -> None:
    """Demonstrate ensemble retrieval with multiple strategies."""
    print("\n" + "=" * 70)
    print("4. ENSEMBLE RETRIEVER (Multiple Retrievers)")
    print("=" * 70)

    embeddings = get_embeddings()
    vectorstore = VectorStore(embeddings=embeddings)
    await vectorstore.add_documents(DOCUMENTS)

    # Create multiple retrievers
    dense = DenseRetriever(vectorstore)
    sparse = BM25Retriever()
    sparse.add_documents(DOCUMENTS)

    # Combine them
    ensemble = EnsembleRetriever(
        retrievers=[dense, sparse],
        weights=[0.6, 0.4],
        fusion=FusionStrategy.RRF,
    )

    query = "database connection issues incident"
    print(f"\nQuery: '{query}'")
    print("(Combining dense + sparse with weighted RRF)")

    results = await ensemble.retrieve(query, k=2, include_scores=True)
    for r in results:
        print(f"\n  [{r.score:.3f}] {r.document.metadata['source']}")
        print(f"  Retriever: {r.retriever_name}")


async def demo_parent_document_retriever() -> None:
    """Demonstrate parent document retrieval."""
    print("\n" + "=" * 70)
    print("5. PARENT DOCUMENT RETRIEVER")
    print("=" * 70)
    print("(Index small chunks, return full documents)")

    embeddings = get_embeddings()
    vectorstore = VectorStore(embeddings=embeddings)

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        chunk_size=200,  # Small chunks for precise matching
        chunk_overlap=30,
    )

    await retriever.add_documents([LONG_DOCUMENT])

    query = "What is the attention formula?"
    print(f"\nQuery: '{query}'")

    results = await retriever.retrieve(query, k=1, include_scores=True)
    for r in results:
        print(f"\n  Score: {r.score:.3f}")
        print(f"  Matching chunks: {r.metadata.get('matching_chunks', 1)}")
        print(f"  Full document returned ({len(r.document.text)} chars)")
        print(f"  Preview: {r.document.text[:150]}...")


async def demo_sentence_window_retriever() -> None:
    """Demonstrate sentence window retrieval."""
    print("\n" + "=" * 70)
    print("6. SENTENCE WINDOW RETRIEVER")
    print("=" * 70)
    print("(Index sentences, return with context)")

    embeddings = get_embeddings()
    vectorstore = VectorStore(embeddings=embeddings)

    retriever = SentenceWindowRetriever(
        vectorstore=vectorstore,
        window_size=2,  # 2 sentences before/after
    )

    await retriever.add_documents([LONG_DOCUMENT])

    query = "number of attention heads"
    print(f"\nQuery: '{query}'")

    results = await retriever.retrieve(query, k=1, include_scores=True)
    for r in results:
        print(f"\n  Score: {r.score:.3f}")
        matched = r.document.metadata.get("matched_sentence", "")
        print(f"  Matched sentence: {matched[:80]}...")
        print(f"  Window context: {r.document.text[:200]}...")


async def demo_summary_index() -> None:
    """Demonstrate LLM-powered summary index."""
    print("\n" + "=" * 70)
    print("7. SUMMARY INDEX (LLM-Generated Summaries)")
    print("=" * 70)

    embeddings = get_embeddings()
    model = get_model()
    vectorstore = VectorStore(embeddings=embeddings)

    index = SummaryIndex(
        llm=model,  # Auto-adapts chat models internally
        vectorstore=vectorstore,
        extract_entities=True,
        extract_keywords=True,
    )

    # Add a subset (summarization is slow)
    print("\nGenerating summaries...")
    await index.add_documents(DOCUMENTS[:2])

    query = "What are the Python 3.13 features?"
    print(f"\nQuery: '{query}'")

    results = await index.retrieve(query, k=1, include_scores=True)
    for r in results:
        print(f"\n  Score: {r.score:.3f}")
        doc_id = r.document.metadata.get("id")
        if doc_id and doc_id in index.summaries:
            summary = index.summaries[doc_id]
            print(f"  Keywords: {summary.keywords[:5]}")
            if summary.entities:
                print(f"  Entities: {summary.entities[:3]}")


async def demo_keyword_table_index() -> None:
    """Demonstrate keyword table index."""
    print("\n" + "=" * 70)
    print("8. KEYWORD TABLE INDEX (LLM Keyword Extraction)")
    print("=" * 70)

    model = get_model()

    index = KeywordTableIndex(
        llm=model,  # Auto-adapts chat models internally
        max_keywords_per_doc=8,
    )

    print("\nExtracting keywords...")
    await index.add_documents(DOCUMENTS[:2])

    # Show extracted keywords
    print("\nKeyword table sample:")
    keyword_table = index.get_keyword_table()
    for keyword, doc_ids in list(keyword_table.items())[:5]:
        print(f"  '{keyword}' → {len(doc_ids)} docs")

    query = "Python features"
    print(f"\nQuery: '{query}'")

    results = await index.retrieve(query, k=2, include_scores=True)
    for r in results:
        print(f"\n  [{r.score:.3f}] {r.document.metadata.get('source', 'unknown')}")


async def demo_self_query_retriever() -> None:
    """Demonstrate self-query retrieval with natural language filters."""
    print("\n" + "=" * 70)
    print("9. SELF-QUERY RETRIEVER (Natural Language → Filters)")
    print("=" * 70)

    embeddings = get_embeddings()
    model = get_model()
    vectorstore = VectorStore(embeddings=embeddings)
    await vectorstore.add_documents(DOCUMENTS)

    retriever = SelfQueryRetriever(
        vectorstore=vectorstore,
        llm=model,  # Auto-adapts chat models internally
        attribute_info=[
            AttributeInfo("category", "Document type: release, guide, policy, incident, evaluation", "string"),
            AttributeInfo("date", "Publication date in YYYY-MM-DD format", "string"),
            AttributeInfo("author", "Author or team name", "string"),
        ],
    )

    # Natural language query with implicit filter
    query = "What are the guidelines from the Data Science Team?"
    print(f"\nQuery: '{query}'")
    print("(LLM extracts: semantic='guidelines' + filter={'author': 'Data Science Team'})")

    results = await retriever.retrieve(query, k=2, include_scores=True)
    for r in results:
        print(f"\n  [{r.score:.3f}] {r.document.metadata.get('source', 'unknown')}")
        print(f"  Author: {r.document.metadata.get('author', 'unknown')}")


async def demo_time_based_index() -> None:
    """Demonstrate time-aware retrieval."""
    print("\n" + "=" * 70)
    print("10. TIME-BASED INDEX (Recency-Aware Retrieval)")
    print("=" * 70)

    embeddings = get_embeddings()
    vectorstore = VectorStore(embeddings=embeddings)

    index = TimeBasedIndex(
        vectorstore=vectorstore,
        decay_function=DecayFunction.EXPONENTIAL,
        decay_rate=0.01,  # Gentle decay
        auto_extract_timestamps=True,
    )

    await index.add_documents(DOCUMENTS)

    query = "API documentation and limits"
    print(f"\nQuery: '{query}'")

    # Regular search (recent docs score higher)
    print("\n  With time decay (recent = higher score):")
    results = await index.retrieve(query, k=2, include_scores=True)
    for r in results:
        date = r.document.metadata.get("date", "unknown")
        print(f"    [{r.score:.3f}] {date} - {r.document.metadata.get('source', 'unknown')}")

    # Filtered to specific time range
    print("\n  Filtered to 2024 only:")
    results = await index.retrieve(
        query,
        k=2,
        time_range=TimeRange.year(2024),
        include_scores=True,
    )
    for r in results:
        date = r.document.metadata.get("date", "unknown")
        print(f"    [{r.score:.3f}] {date} - {r.document.metadata.get('source', 'unknown')}")


async def demo_multi_representation_index() -> None:
    """Demonstrate multi-representation retrieval."""
    print("\n" + "=" * 70)
    print("11. MULTI-REPRESENTATION INDEX")
    print("=" * 70)
    print("(Multiple embeddings per document)")

    embeddings = get_embeddings()
    model = get_model()
    vectorstore = VectorStore(embeddings=embeddings)

    index = MultiRepresentationIndex(
        vectorstore=vectorstore,
        llm=model,  # Auto-adapts chat models internally
        representations=["original", "summary", "detailed"],
    )

    print("\nGenerating multiple representations...")
    await index.add_documents(DOCUMENTS[:2])

    # Broad conceptual query
    query = "What is Python?"
    print(f"\nBroad query: '{query}'")
    print("  → Routes to SUMMARY representation")

    results = await index.retrieve(query, k=1, include_scores=True)
    for r in results:
        print(f"    [{r.score:.3f}] {r.document.metadata.get('source', 'unknown')}")

    # Specific technical query
    query = "free-threading PEP 703 no-GIL experimental"
    print(f"\nSpecific query: '{query}'")
    print("  → Routes to DETAILED representation")

    results = await index.retrieve(query, k=1, include_scores=True)
    for r in results:
        print(f"    [{r.score:.3f}] {r.document.metadata.get('source', 'unknown')}")


async def demo_reranker() -> None:
    """Demonstrate reranking for improved relevance."""
    print("\n" + "=" * 70)
    print("12. LLM RERANKER")
    print("=" * 70)
    print("(Improve ranking with LLM scoring)")

    embeddings = get_embeddings()
    model = get_model()
    vectorstore = VectorStore(embeddings=embeddings)
    await vectorstore.add_documents(DOCUMENTS)

    retriever = DenseRetriever(vectorstore)
    reranker = LLMReranker(model=model)  # Auto-adapts chat models internally

    query = "How do I handle too many API requests?"

    # Initial retrieval
    print(f"\nQuery: '{query}'")
    print("\n  Before reranking:")
    initial = await retriever.retrieve(query, k=4, include_scores=True)
    for i, r in enumerate(initial, 1):
        print(f"    {i}. [{r.score:.3f}] {r.document.metadata.get('source', 'unknown')}")

    # Rerank (reranker takes Documents, not RetrievalResults)
    print("\n  After LLM reranking:")
    docs_to_rerank = [r.document for r in initial]
    reranked = await reranker.rerank(query, docs_to_rerank, top_n=3)
    for i, r in enumerate(reranked, 1):
        print(f"    {i}. [{r.score:.3f}] {r.document.metadata.get('source', 'unknown')}")


async def demo_fusion_utilities() -> None:
    """Demonstrate result fusion utilities."""
    print("\n" + "=" * 70)
    print("13. FUSION UTILITIES")
    print("=" * 70)

    embeddings = get_embeddings()
    vectorstore = VectorStore(embeddings=embeddings)
    await vectorstore.add_documents(DOCUMENTS)

    dense = DenseRetriever(vectorstore)
    sparse = BM25Retriever()
    sparse.add_documents(DOCUMENTS)

    query = "machine learning best practices"

    # Get results from both
    dense_results = await dense.retrieve(query, k=3, include_scores=True)
    sparse_results = await sparse.retrieve(query, k=3, include_scores=True)

    print(f"\nQuery: '{query}'")
    print("\n  Dense results:")
    for r in dense_results:
        print(f"    [{r.score:.3f}] {r.document.metadata.get('source', 'unknown')}")

    print("\n  Sparse results:")
    for r in sparse_results:
        print(f"    [{r.score:.3f}] {r.document.metadata.get('source', 'unknown')}")

    # Fuse with RRF
    print("\n  After RRF fusion:")
    fused = fuse_results(
        [dense_results, sparse_results],
        strategy=FusionStrategy.RRF,
        weights=[0.7, 0.3],
        k=3,
    )
    for r in fused:
        print(f"    [{r.score:.3f}] {r.document.metadata.get('source', 'unknown')}")


async def main() -> None:
    """Run all retriever demos."""
    print("\n" + "=" * 70)
    print("AGENTICFLOW RETRIEVERS - COMPREHENSIVE DEMO")
    print("=" * 70)

    # Core retrievers
    await demo_dense_retriever()
    await demo_bm25_retriever()
    await demo_hybrid_retriever()
    await demo_ensemble_retriever()

    # Contextual retrievers
    await demo_parent_document_retriever()
    await demo_sentence_window_retriever()

    # LLM-powered indexes (require model, slower)
    await demo_summary_index()
    await demo_keyword_table_index()
    await demo_self_query_retriever()

    # Specialized indexes
    await demo_time_based_index()
    await demo_multi_representation_index()

    # Reranking
    await demo_reranker()

    # Utilities
    await demo_fusion_utilities()

    print("\n" + "=" * 70)
    print("✓ All retriever demos completed")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
