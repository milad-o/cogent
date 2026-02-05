# Retrieval Examples

Focused, minimal examples demonstrating each retrieval pattern in Cogent.

## 📁 Files

### Core Files

- **[_shared.py](_shared.py)** - Common data, factories, and utilities
  - Sample documents and datasets
  - Vectorstore factories with embeddings
  - Output formatting helpers

### Retrieval Patterns

- **[basic_retrievers.py](basic_retrievers.py)** - Fundamental retrieval (5 examples)
  - `DenseRetriever` - Semantic vector search
  - `BM25Retriever` - Keyword-based sparse retrieval
  - `HybridRetriever` - Metadata + content fusion
  - `EnsembleRetriever` - Combine multiple strategies
  - `ParentDocumentRetriever` - Index chunks, return full docs

- **[advanced_retrievers.py](advanced_retrievers.py)** - Specialized patterns (4 examples)
  - `HyDERetriever` - Hypothetical Document Embeddings
  - `SelfQueryRetriever` - Natural language → filters
  - `SentenceWindowRetriever` - Sentence-level with context
  - `TimeBasedIndex` - Time-decay for recency scoring

- **[indexes.py](indexes.py)** - LLM-powered indexes (5 examples)
  - `SummaryIndex` - Query-time summarization
  - `TreeIndex` - Hierarchical summary trees
  - `KeywordTableIndex` - LLM-extracted keywords
  - `KnowledgeGraphIndex` - Graph-based retrieval
  - `MultiRepresentationIndex` - Multiple embeddings per doc

- **[reranking.py](reranking.py)** - Precision refinement (4 examples)
  - `CrossEncoderReranker` - Local neural reranker
  - `CohereReranker` - Cohere API reranking
  - `LLMReranker` - Use any LLM for scoring
  - Complete workflow (retrieve → rerank)

### Legacy/Reference

- **[hyde.py](hyde.py)** - Original detailed HyDE example
- **[retrievers.py.bak](retrievers.py.bak)** - Original comprehensive example (archived)

## 🚀 Usage

Each file is independently runnable:

```bash
# Basic retrievers
uv run python examples/retrieval/basic_retrievers.py

# Advanced patterns
uv run python examples/retrieval/advanced_retrievers.py

# LLM-powered indexes
uv run python examples/retrieval/indexes.py

# Reranking strategies
uv run python examples/retrieval/reranking.py
```

## 📊 Comparison

| Pattern | Use Case | Example |
|---------|----------|---------|
| **Dense** | Semantic search, general RAG | Find conceptually similar docs |
| **BM25** | Keyword matching, exact terms | Search by specific terminology |
| **Hybrid** | Metadata + content | Filter by category, search content |
| **Ensemble** | Best of multiple strategies | Combine dense + BM25 |
| **HyDE** | Abstract queries | "How to prevent bugs?" |
| **SelfQuery** | Natural language filters | "Critical security issues" |
| **Summary Index** | Large doc collections | Query-time summarization |
| **Knowledge Graph** | Entity relationships | "Landmarks in capitals?" |
| **Reranking** | Improve precision | Retrieve 20 → rerank to 3 |

## 🎯 Design Principles

Each example is:
- ✅ **Compact** - 15-25 lines per demo
- ✅ **Minimal** - No unnecessary setup
- ✅ **Reproducible** - Uses shared synthetic data
- ✅ **Focused** - One retriever/pattern per function
- ✅ **Runnable** - Works independently

## 📚 Learn More

- [Retrievers Documentation](../../docs/retrievers.md)
- [Main Examples README](../README.md)
