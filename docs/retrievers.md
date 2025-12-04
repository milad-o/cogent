# Retriever Guide

AgenticFlow provides a comprehensive retrieval system with multiple strategies for different use cases. This guide covers all available retrievers and when to use each.

## Unified API

All retrievers share a unified `retrieve()` API with optional scoring:

```python
# Get documents only (default)
docs = await retriever.retrieve("query", k=5)

# Get documents with relevance scores
results = await retriever.retrieve("query", k=5, include_scores=True)
for r in results:
    print(f"{r.score:.3f}: {r.document.text[:50]}")

# With metadata filter
results = await retriever.retrieve(
    "query",
    k=10,
    filter={"category": "docs"},
    include_scores=True,
)

# Retriever-specific args (e.g., TimeBasedIndex)
results = await retriever.retrieve(
    "recent news",
    k=5,
    time_range=TimeRange.last_days(30),
    include_scores=True,
)
```

## Overview

| Category | Retriever | Best For |
|----------|-----------|----------|
| **Core** | `DenseRetriever` | Semantic similarity search |
| | `BM25Retriever` | Keyword/lexical matching |
| | `HybridRetriever` | Best of both (dense + sparse) |
| | `EnsembleRetriever` | Combining multiple retrievers |
| **Contextual** | `ParentDocumentRetriever` | Precise chunks → full context |
| | `SentenceWindowRetriever` | Sentence-level → paragraph context |
| **LLM-Powered** | `SummaryIndex` | Document summaries |
| | `TreeIndex` | Hierarchical summary tree |
| | `KeywordTableIndex` | Keyword extraction + lookup |
| | `KnowledgeGraphIndex` | Entity-based retrieval |
| | `SelfQueryRetriever` | Natural language → filters |
| **Specialized** | `HierarchicalIndex` | Structured docs (markdown/html) |
| | `TimeBasedIndex` | Recency-aware retrieval |
| | `MultiRepresentationIndex` | Multiple embeddings per doc |

---

## Core Retrievers

### DenseRetriever

Semantic search using vector embeddings. The most common retriever for general RAG applications.

```python
from agenticflow.retriever import DenseRetriever
from agenticflow.vectorstore import VectorStore

# Create vectorstore and retriever
vectorstore = VectorStore(embeddings=embeddings)
await vectorstore.add_texts([
    "Python is a programming language",
    "Machine learning uses algorithms",
    "Neural networks learn from data",
])

retriever = DenseRetriever(vectorstore)
results = await retriever.retrieve("AI and deep learning", k=2)
```

**When to use:**
- General semantic search
- Finding conceptually similar content
- When exact keyword matching isn't required

---

### BM25Retriever

Lexical retrieval using the BM25 algorithm. Fast, interpretable, and excellent for keyword queries.

```python
from agenticflow.retriever import BM25Retriever
from agenticflow.vectorstore import Document

# Create BM25 index
retriever = BM25Retriever(k1=1.5, b=0.75)
retriever.add_documents([
    Document(text="Python programming tutorial", metadata={"type": "tutorial"}),
    Document(text="JavaScript web development", metadata={"type": "tutorial"}),
    Document(text="Machine learning with Python", metadata={"type": "guide"}),
])

# Keyword-based search
results = await retriever.retrieve("Python tutorial", k=2)
```

**When to use:**
- Exact keyword matching is important
- Domain-specific terminology
- Fast, interpretable results needed
- No embedding model available

---

### HybridRetriever

Combines dense (semantic) and sparse (BM25) retrieval for best of both worlds.

```python
from agenticflow.retriever import HybridRetriever

# Hybrid with automatic BM25 sync
retriever = HybridRetriever(
    vectorstore=vectorstore,
    dense_weight=0.7,   # 70% semantic
    sparse_weight=0.3,  # 30% keyword
    fusion="rrf",       # Reciprocal Rank Fusion
)

# Add documents (syncs to both indexes)
await retriever.add_documents(documents)

# Search combines both approaches
results = await retriever.retrieve("machine learning Python", k=5)
```

**When to use:**
- Best default choice for production RAG
- Mixed queries (concepts + keywords)
- When you need high recall without sacrificing precision

---

### EnsembleRetriever

Combine any number of retrievers with configurable fusion strategies.

```python
from agenticflow.retriever import (
    EnsembleRetriever,
    DenseRetriever,
    BM25Retriever,
    FusionStrategy,
)

# Combine multiple retrievers
ensemble = EnsembleRetriever(
    retrievers=[
        DenseRetriever(vectorstore_openai),    # OpenAI embeddings
        DenseRetriever(vectorstore_cohere),    # Cohere embeddings
        BM25Retriever(documents),              # Lexical
    ],
    weights=[0.4, 0.4, 0.2],
    fusion=FusionStrategy.RRF,  # or LINEAR, MAX, VOTING
)

results = await ensemble.retrieve("query", k=10)
```

**Fusion strategies:**
- `RRF` (Reciprocal Rank Fusion): Best for diverse retrievers
- `LINEAR`: Weighted score combination
- `MAX`: Take highest score per document
- `VOTING`: Count how many retrievers found each doc

---

## Contextual Retrievers

### ParentDocumentRetriever

Index small chunks for precise matching, but return full parent documents for context.

```python
from agenticflow.retriever import ParentDocumentRetriever

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    chunk_size=500,     # Small chunks for precise matching
    chunk_overlap=50,
)

# Add full documents (automatically chunked)
await retriever.add_documents(large_documents)

# Search finds chunks, returns parents
results = await retriever.retrieve("specific concept", k=3)
# Each result is a full document, not a chunk
```

**When to use:**
- LLM needs more context than a single chunk
- Documents have interconnected information
- You want precise matching with comprehensive results

---

### SentenceWindowRetriever

Index individual sentences, but return with surrounding context.

```python
from agenticflow.retriever import SentenceWindowRetriever

retriever = SentenceWindowRetriever(
    vectorstore=vectorstore,
    window_size=2,  # 2 sentences before and after
)

await retriever.add_documents(documents)

# Precise sentence match with context
results = await retriever.retrieve("specific fact", k=3, include_scores=True)
for r in results:
    print(f"Matched: {r.metadata['matched_sentence']}")
    print(f"Context: {r.document.text}")  # Full window
```

**When to use:**
- Need precise sentence-level matching
- Want to return paragraph-level context
- Fact-checking or citation tasks

---

## LLM-Powered Indexes

### SummaryIndex

Generate LLM summaries of documents for efficient high-level retrieval.

```python
from agenticflow.retriever import SummaryIndex

index = SummaryIndex(
    llm=model,
    vectorstore=vectorstore,
    extract_entities=True,   # For knowledge graph
    extract_keywords=True,
)

await index.add_documents(long_documents)

# Search by summary
results = await index.retrieve("machine learning concepts", k=3)

# Access extracted entities for KG integration
for doc_id, summary in index.summaries.items():
    print(f"Keywords: {summary.keywords}")
    print(f"Entities: {summary.entities}")
```

**When to use:**
- Long documents that don't fit in embeddings well
- Need document-level topics quickly
- Building knowledge graphs from documents

---

### TreeIndex

Hierarchical tree of summaries for very large documents or corpora.

```python
from agenticflow.retriever import TreeIndex

index = TreeIndex(
    llm=model,
    vectorstore=vectorstore,
    max_children=5,      # Children per node
    max_depth=3,         # Tree depth
)

await index.add_documents(very_large_documents)

# Efficient tree traversal
results = await index.retrieve("specific topic", k=5)
```

**When to use:**
- Very large documents (books, manuals)
- Corpus-level search across many documents
- When full indexing is too slow/expensive

---

### KeywordTableIndex

Extract keywords with LLM and build inverted index for fast lookup.

```python
from agenticflow.retriever import KeywordTableIndex

index = KeywordTableIndex(
    llm=model,
    max_keywords_per_doc=10,
)

await index.add_documents(documents)

# Fast keyword-based lookup
results = await index.retrieve("Python machine learning", k=5)

# Access keyword table
print(index.keyword_table)  # {"python": [doc_ids...], "ml": [...]}
```

**When to use:**
- Domain with specific terminology
- Fast keyword lookup needed
- Interpretable retrieval wanted

---

### SelfQueryRetriever

LLM parses natural language queries into semantic search + metadata filters.

```python
from agenticflow.retriever import SelfQueryRetriever, AttributeInfo

retriever = SelfQueryRetriever(
    vectorstore=vectorstore,
    llm=model,
    attribute_info=[
        AttributeInfo("category", "Document category", "string"),
        AttributeInfo("year", "Publication year", "integer"),
        AttributeInfo("author", "Author name", "string"),
    ],
)

# Natural language with implicit filters
results = await retriever.retrieve(
    "research papers about AI from 2024 by OpenAI"
)
# LLM extracts: semantic="AI research papers"
#              filter={"year": 2024, "author": "OpenAI"}
```

**When to use:**
- Users query in natural language
- Documents have filterable metadata
- Want to combine semantic + structured search

---

## Specialized Indexes

### HierarchicalIndex

Respect and leverage document structure (headers, sections).

```python
from agenticflow.retriever import HierarchicalIndex

index = HierarchicalIndex(
    vectorstore=vectorstore,
    llm=model,
    structure_type="markdown",  # or "html"
    top_k_sections=3,
    chunks_per_section=3,
)

await index.add_documents(structured_docs)

# Find section first, then relevant chunks
results = await index.retrieve("installation", k=5)
for r in results:
    print(f"Section: {r.metadata['section_title']}")
    print(f"Path: {r.metadata['hierarchy_path']}")
```

**When to use:**
- Well-structured documents (docs, manuals, specs)
- Want to respect document organization
- Need section-level context

---

### TimeBasedIndex

Prioritize recent information with time-decay scoring.

```python
from agenticflow.retriever import TimeBasedIndex, TimeRange, DecayFunction

index = TimeBasedIndex(
    vectorstore=vectorstore,
    decay_function=DecayFunction.EXPONENTIAL,
    decay_rate=0.01,  # Halve score every ~70 days
    auto_extract_timestamps=True,
)

await index.add_documents(news_articles)

# Recent docs score higher
results = await index.retrieve("market trends", k=5)

# Filter by time range
results = await index.retrieve(
    "company policy",
    time_range=TimeRange.last_days(30),
)

# Point-in-time query
results = await index.retrieve(
    "regulations",
    time_range=TimeRange.year(2023),
)
```

**Decay functions:**
- `EXPONENTIAL`: Smooth decay over time
- `LINEAR`: Linear decrease
- `STEP`: Full score within window, zero outside
- `LOGARITHMIC`: Slow initial decay
- `NONE`: No decay, just filtering

**When to use:**
- News, articles, changelogs
- Evolving knowledge bases
- Time-sensitive information

---

### MultiRepresentationIndex

Store multiple embeddings per document for diverse query handling.

```python
from agenticflow.retriever import MultiRepresentationIndex, QueryType

index = MultiRepresentationIndex(
    vectorstore=vectorstore,
    llm=model,
    representations=["original", "summary", "detailed", "questions"],
)

await index.add_documents(documents)

# Auto-detect query type
results = await index.retrieve("What is machine learning?")

# Force specific representation
results = await index.retrieve(
    "backpropagation gradient calculation",
    query_type=QueryType.SPECIFIC,  # Uses detailed representation
)

# Search all and fuse
results = await index.retrieve(
    "AI applications",
    search_all=True,
)
```

**Representations:**
- `original`: Raw document embedding
- `summary`: Conceptual summary
- `detailed`: Technical details
- `keywords`: Key terms
- `questions`: Hypothetical Q&A
- `entities`: Named entities

**When to use:**
- Diverse query styles expected
- Technical/specialized domains
- Want maximum recall

---

## Rerankers

Rerankers improve retrieval quality by re-scoring initial results.

```python
from agenticflow.retriever import (
    DenseRetriever,
    CrossEncoderReranker,
    CohereReranker,
    LLMReranker,
)

# Initial retrieval
retriever = DenseRetriever(vectorstore)
initial_docs = await retriever.retrieve(query, k=20)  # Get documents

# Rerank with cross-encoder (local)
reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranked = await reranker.rerank(query, initial_docs, top_n=5)

# Or Cohere Rerank API
reranker = CohereReranker(api_key="...")
reranked = await reranker.rerank(query, initial_docs, top_n=5)

# Or any LLM
reranker = LLMReranker(llm=model)
reranked = await reranker.rerank(query, initial_docs, top_n=5)
```

**Available rerankers:**
- `CrossEncoderReranker`: Local cross-encoder models
- `FlashRankReranker`: Lightweight, fast local reranker
- `CohereReranker`: Cohere Rerank API
- `LLMReranker`: Any LLM for pointwise scoring
- `ListwiseLLMReranker`: LLM ranks all docs at once

---

## Utilities

### Fusion Functions

```python
from agenticflow.retriever import fuse_results, FusionStrategy

# Fuse results from multiple retrievers
fused = fuse_results(
    [results_1, results_2, results_3],
    strategy=FusionStrategy.RRF,
    weights=[0.5, 0.3, 0.2],
    k=10,
)
```

### Score Normalization

```python
from agenticflow.retriever import normalize_scores

# Normalize scores to 0-1 range
normalized = normalize_scores(results)
```

### Deduplication

```python
from agenticflow.retriever import deduplicate_results

# Remove duplicate documents
unique = deduplicate_results(results, by="content")  # or "id"
```

---

## Choosing a Retriever

```
┌─────────────────────────────────────────────────────────────┐
│                    What's your use case?                     │
└─────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
     General RAG        Specialized        Advanced
            │                 │                 │
            ▼                 │                 │
    ┌───────────────┐        │                 │
    │ HybridRetriever│◄──────┤                 │
    │   (default)    │        │                 │
    └───────────────┘        │                 │
                              ▼                 │
                    ┌─────────────────────┐    │
                    │  Time-sensitive?    │    │
                    │  → TimeBasedIndex   │    │
                    ├─────────────────────┤    │
                    │  Structured docs?   │    │
                    │  → HierarchicalIndex│    │
                    ├─────────────────────┤    │
                    │  Need full context? │    │
                    │  → ParentDocument   │    │
                    └─────────────────────┘    │
                                               ▼
                              ┌─────────────────────────────┐
                              │ Multiple embedding models?  │
                              │ → EnsembleRetriever         │
                              ├─────────────────────────────┤
                              │ Natural language filters?   │
                              │ → SelfQueryRetriever        │
                              ├─────────────────────────────┤
                              │ Very long documents?        │
                              │ → SummaryIndex / TreeIndex  │
                              └─────────────────────────────┘
```

---

## Performance Tips

1. **Start with HybridRetriever** - Best default for most cases
2. **Use rerankers** - Cheap way to improve quality
3. **Retrieve more, rerank less** - Get top 20-50, rerank to top 5
4. **Cache embeddings** - Reuse for similar queries
5. **Batch operations** - Add documents in batches
6. **Filter first** - Use metadata filters before semantic search
