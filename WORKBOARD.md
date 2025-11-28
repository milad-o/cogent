# WORKBOARD: AgenticFlow Development

## Goal
Build a comprehensive, native agentic AI framework without external dependencies.
Simple by default, powerful when needed.

## Status: Phase 11 - Final Cleanup ✅ COMPLETE

### Completed ✅
- [x] Native message types (`core/messages.py`)
- [x] Native model wrappers for all providers
- [x] Native tools (`@tool` decorator, `BaseTool`)
- [x] Agent class uses native types only
- [x] All capabilities use native tools
- [x] Examples updated to native models
- [x] Prebuilt components updated
- [x] **Native Topologies** - Removed LangGraph completely!
- [x] **Memory System Complete** ✅
- [x] **Vector Store Complete** ✅
  - 5 backends: InMemory, FAISS, Chroma, Qdrant, pgvector
  - SQLite memory persistence backend
  - 56 tests for vector store + backends
- [x] **Retriever Module Complete** ✅
- [x] **Final Cleanup Complete** ✅

### Phase 10 ✅ - Retriever Module (COMPLETE)

> **Design Philosophy**: Composable retrievers with pluggable components.
> Protocol-based design for maximum extensibility.

#### Retriever Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RETRIEVER MODULE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Retriever (Protocol)                             │  │
│  │  async def retrieve(query, k, filter) -> list[Document]              │  │
│  │  async def retrieve_with_scores(...) -> list[RetrievalResult]        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│           ┌────────────────────────┼────────────────────────┐               │
│           ▼                        ▼                        ▼               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  DenseRetriever │    │ SparseRetriever │    │ HybridRetriever │         │
│  │  (VectorStore)  │    │  (BM25/TF-IDF)  │    │ (Dense+Sparse)  │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│           │                        │                        │               │
│           └────────────────────────┼────────────────────────┘               │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      EnsembleRetriever                                │  │
│  │  retrievers: list[Retriever], weights: list[float]                   │  │
│  │  fusion: "rrf" | "linear" | "max" | "voting"                         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Rerankers                                        │  │
│  │  CrossEncoder | Cohere | LLM | FlashRank (all optional)              │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  ContextualRetriever          │  SelfQueryRetriever                  │  │
│  │  parent_child, sentence_window│  LLM-parsed filters from query       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Tasks (All Complete ✅)

**Core** (`retriever/`)
- [x] `retriever/base.py` - Retriever protocol, FusionStrategy enum, RetrievalResult
- [x] `retriever/dense.py` - DenseRetriever (wraps VectorStore)
- [x] `retriever/sparse.py` - BM25Retriever, TFIDFRetriever
- [x] `retriever/hybrid.py` - HybridRetriever (dense + sparse fusion)
- [x] `retriever/ensemble.py` - EnsembleRetriever (multi-retriever fusion)
- [x] `retriever/contextual.py` - ParentDocumentRetriever, SentenceWindowRetriever
- [x] `retriever/self_query.py` - SelfQueryRetriever (LLM-parsed filters)

**Rerankers** (`retriever/rerankers/`)
- [x] `rerankers/base.py` - Reranker protocol, BaseReranker
- [x] `rerankers/cross_encoder.py` - sentence-transformers (optional)
- [x] `rerankers/cohere.py` - Cohere rerank API (optional)
- [x] `rerankers/llm.py` - LLMReranker, ListwiseLLMReranker
- [x] `rerankers/flashrank.py` - FlashRank lightweight (optional)

**Utils** (`retriever/utils/`)
- [x] `utils/fusion.py` - RRF, linear, max, voting strategies
- [x] `utils/tokenizers.py` - Tokenization utilities for BM25

**Integration**
- [x] Update `prebuilt/rag.py` to use retrievers (supports hybrid, reranking)
- [x] Tests for retriever module (33 tests)

#### API Design (Working ✅)

```python
# SIMPLE: Dense retrieval (default)
from agenticflow.retriever import DenseRetriever

retriever = DenseRetriever(vectorstore)
docs = await retriever.retrieve("What is Python?", k=5)

# HYBRID: Dense + BM25
from agenticflow.retriever import HybridRetriever

retriever = HybridRetriever(
    vectorstore=store,
    documents=docs,
    sparse_weight=0.3,   # 30% BM25, 70% dense
)
docs = await retriever.retrieve("Python programming tutorials")

# ENSEMBLE: Multiple retrievers
from agenticflow.retriever import EnsembleRetriever, DenseRetriever, BM25Retriever

retriever = EnsembleRetriever(
    retrievers=[
        DenseRetriever(store1),
        DenseRetriever(store2),
        BM25Retriever(documents),
    ],
    weights=[0.4, 0.4, 0.2],
    fusion="rrf",
)

# WITH RERANKING: Two-stage retrieval
from agenticflow.retriever.rerankers import LLMReranker

reranker = LLMReranker(model=llm)
results = await retriever.retrieve_with_scores("query", k=20)
reranked = await reranker.rerank("query", [r.document for r in results], top_n=5)

# CONTEXTUAL: Parent-child chunking
from agenticflow.retriever import ParentDocumentRetriever

retriever = ParentDocumentRetriever(
    vectorstore=store,
    child_chunk_size=200,
    parent_chunk_size=1000,
)

# SELF-QUERY: LLM-parsed filters
from agenticflow.retriever import SelfQueryRetriever

retriever = SelfQueryRetriever(
    vectorstore=store,
    model=llm,
    document_description="Technical papers",
    metadata_fields=[
        AttributeInfo(name="author", type="string"),
        AttributeInfo(name="year", type="integer"),
    ],
)
# "papers by Smith from 2023" → filter={author:"Smith", year:2023}

# PREBUILT RAG with hybrid retrieval
from agenticflow.prebuilt import RAGAgent

rag = RAGAgent(
    model=llm,
    retrieval_mode="hybrid",  # dense + BM25
    reranker="llm",           # LLM-based reranking
)
await rag.load_documents(["doc1.pdf", "doc2.txt"])
answer = await rag.query("What is the main finding?")
```

#### Optional Dependencies

| Feature | Package | Size |
|---------|---------|------|
| BM25 | `rank-bm25` | ~10KB |
| Cross-encoder | `sentence-transformers` | ~500MB |
| Cohere rerank | `cohere` | ~100KB |
| FlashRank | `flashrank` | ~50MB |

### Phase 11 ✅ - Final Cleanup (COMPLETE)

**Completed:**
- [x] Renamed `chunk_from_langchain` to `chunk_from_message` (backward-compat alias kept)
- [x] Renamed `_langchain_tools` to `_native_tools` in MCP capability
- [x] Renamed `_create_langchain_tool` to `_create_native_tool`
- [x] Updated all LangChain-specific comments to be generic
- [x] Kept `models/adapter.py` - intentional compatibility layer for migration
- [x] All tests passing: 859 passed, 36 skipped

**Notes:**
- The `models/adapter.py` module is intentionally kept as a bridge for users migrating from LangChain
- The `chunk_from_langchain` function is aliased to `chunk_from_message` for backward compatibility

## Current Test Status
- **859 passed**, 36 skipped
- Memory system: 71 tests
- Vector store: 51 tests  
- Backend tests: 28 tests
- Retriever module: 33 tests (24 passed, 9 skipped for optional deps)
- Core library is native-only

## Quick API (Working ✅)
```python
from agenticflow import run, tool

@tool
def search(query: str) -> str:
    '''Search the web.'''
    return f"Results for {query}"

result = await run(
    "Search for Python tutorials",
    tools=[search],
    model="gpt-4o-mini",
)
```

