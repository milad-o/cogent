# RAG Architecture: Three-Layer API Design

## Overview

AgenticFlow provides RAG (Retrieval-Augmented Generation) at three abstraction levels:

| Level | API | Use Case |
|-------|-----|----------|
| **High** | `agent.run()` with RAG capability | Agentic RAG with tool use |
| **Mid** | `rag.search()` | Citation-aware retrieval |
| **Low** | `vectorstore.search()` | Raw document retrieval |

## Two Usage Patterns

### Pattern 1: Managed Mode (RAG loads documents)

Let RAG handle document loading and indexing:

```python
from agenticflow import Agent
from agenticflow.capabilities import RAG

# RAG creates its own vectorstore
rag = RAG(embeddings=embeddings)

# Load documents - required in managed mode
await rag.load("docs/", "report.pdf")

# Now ready to search
results = await rag.search("key findings")
```

### Pattern 2: Pre-configured Mode (Bring your own retriever)

Provide a pre-loaded retriever - no `load()` needed:

```python
from agenticflow.capabilities import RAG
from agenticflow.vectorstore import VectorStore
from agenticflow.retriever import DenseRetriever, BM25Retriever, HybridRetriever

# 1. Prepare your components
store = VectorStore(embeddings=embeddings)
await store.add_documents(documents)

dense = DenseRetriever(store)
sparse = BM25Retriever(documents)
retriever = HybridRetriever(dense=dense, sparse=sparse)

# 2. Pass to RAG - ready immediately!
rag = RAG(embeddings=embeddings, retriever=retriever)

# No load() needed - search works right away
results = await rag.search("key findings")
```

---

## Layer 1: High-Level (Agentic RAG)

Agent + RAG capability for autonomous document Q&A:

```python
from agenticflow import Agent
from agenticflow.capabilities import RAG

rag = RAG(embeddings=embeddings)

agent = Agent(
    name="DocAssistant",
    model=model,
    capabilities=[rag],
)

await rag.load("docs/", "report.pdf", "data.csv")
answer = await agent.run("What are the key findings?")
```

### What it provides:
- Agent uses `search_documents` tool automatically
- Auto-detects file types → picks optimal loader/splitter
- Default vectorstore (inmemory, configurable)
- LLM generates answer based on retrieved context

---

## Layer 2: Mid-Level (Citation-Aware Search)

Direct `rag.search()` returns `CitedPassage` objects with source/score:

```python
from agenticflow.capabilities import RAG

rag = RAG(embeddings=embeddings)
await rag.load("docs/")

# Get passages with citation metadata
passages = await rag.search("key findings", k=5)

for p in passages:
    print(f"{p.format_reference()} {p.source} (score: {p.score:.2f})")
    print(f"  {p.text[:100]}...")

# Build custom prompt with citations
context = "\n".join(f"[{p.citation_id}] {p.text}" for p in passages)
answer = await agent.run(f"Based on:\n{context}\n\nAnswer: {question}")
```

### What it provides:
- `CitedPassage` with `citation_id`, `source`, `page`, `score`, `text`
- `format_reference()` → `[1]` or `[1, p.5]`
- `format_full()` → `[1] source.pdf, p.5 (score: 0.92)`
- No LLM - just retrieval with citation metadata

---

## Layer 3: Low-Level (Raw Vectorstore)

Direct vectorstore access for full control:

```python
from agenticflow.vectorstore import VectorStore

vectorstore = VectorStore(embeddings=embeddings)
await vectorstore.add_documents(chunks)

# Raw search - returns SearchResult objects
results = await vectorstore.search("query", k=10)

for result in results:
    doc = result.document
    print(f"{doc.metadata['source']} (score: {result.score:.2f})")
    print(f"  {doc.text}")
```

---

## Custom Pipelines

For fine-grained control over document processing:

```python
from agenticflow.capabilities import RAG, DocumentPipeline, PipelineRegistry
from agenticflow.document import PDFLoader, SemanticSplitter, MarkdownSplitter

# Custom pipelines per file type
pipelines = PipelineRegistry()

pipelines.register(".pdf", DocumentPipeline(
    loader=PDFLoader(extract_images=True),
    splitter=SemanticSplitter(embeddings=embeddings, threshold=0.8),
    metadata={"source_type": "pdf"},
))

pipelines.register(".md", DocumentPipeline(
    splitter=MarkdownSplitter(chunk_size=500),
    metadata={"source_type": "documentation"},
))

# RAG with custom pipelines
rag = RAG(
    embeddings=embeddings,
    pipelines=pipelines,
)

await rag.load("report.pdf", "docs/*.md")
```

---

## Advanced Patterns

### Custom Vectorstore Backend

```python
from agenticflow.vectorstore import VectorStore, FAISSBackend, ChromaBackend

# FAISS for high-performance
vectorstore = VectorStore(
    embeddings=embeddings,
    backend=FAISSBackend(dimension=1536),
)

# Chroma for persistence
vectorstore = VectorStore(
    embeddings=embeddings,
    backend=ChromaBackend(path="./chroma_db"),
)

rag = RAG(embeddings=embeddings, vectorstore=vectorstore)
```

### Combining RAG with Other Capabilities

```python
from agenticflow.capabilities import RAG, WebSearch, CodeSandbox

agent = Agent(
    model=model,
    tools=[
        my_database_tool,
        my_api_tool,
    ],
    capabilities=[
        RAG(embeddings=embeddings),      # Document search
        WebSearch(),                       # Live web search
        CodeSandbox(),                     # Execute code
    ],
)

# Agent has access to:
# - search_documents (from RAG)
# - web_search, fetch_page (from WebSearch)
# - execute_code (from CodeSandbox)
# - my_database_tool, my_api_tool (custom)
```

### Per-Document Type Configuration

```python
from agenticflow.capabilities import RAG, PipelineRegistry, DocumentPipeline
from agenticflow.document import (
    PDFLoader, MarkdownSplitter, CodeSplitter, SemanticSplitter,
)

pipelines = PipelineRegistry()

# PDFs: Use vision-based loader, semantic splitting
pipelines.register(".pdf", DocumentPipeline(
    loader=PDFLoader(use_vision=True, model=vision_model),
    splitter=SemanticSplitter(embeddings=embeddings),
    metadata={"type": "document"},
))

# Code: Language-aware splitting
for ext in [".py", ".js", ".ts", ".go", ".rs"]:
    lang = ext[1:]  # Remove dot
    pipelines.register(ext, DocumentPipeline(
        splitter=CodeSplitter(language=lang, chunk_size=1500),
        metadata={"type": "code", "language": lang},
    ))

# Markdown: Structure-aware
pipelines.register(".md", DocumentPipeline(
    splitter=MarkdownSplitter(chunk_size=1000),
    metadata={"type": "documentation"},
))

# Custom post-processing
def add_embeddings(chunks):
    for chunk in chunks:
        chunk.metadata["indexed_at"] = datetime.now().isoformat()
    return chunks

pipelines.register(".txt", DocumentPipeline(
    post_process=add_embeddings,
))

rag = RAG(embeddings=embeddings, pipelines=pipelines)
```

---

## Component Reference

### Loaders
| Loader | Formats | Notes |
|--------|---------|-------|
| `TextLoader` | .txt | Simple text |
| `MarkdownLoader` | .md, .mdx | Preserves structure |
| `PDFLoader` | .pdf | Text + optional OCR |
| `PDFMarkdownLoader` | .pdf | LLM-based, better formatting |
| `WordLoader` | .docx | Microsoft Word |
| `HTMLLoader` | .html, .htm | Web pages |
| `CSVLoader` | .csv | Tabular data |
| `JSONLoader` | .json, .jsonl | Structured data |
| `XLSXLoader` | .xlsx | Excel spreadsheets |
| `CodeLoader` | .py, .js, etc. | Source code |

### Splitters
| Splitter | Strategy | Best For |
|----------|----------|----------|
| `RecursiveCharacterSplitter` | Hierarchical separators | General text |
| `SentenceSplitter` | Sentence boundaries | Prose |
| `MarkdownSplitter` | Headers, lists | Documentation |
| `HTMLSplitter` | HTML tags | Web content |
| `CodeSplitter` | AST-aware | Source code |
| `SemanticSplitter` | Embedding similarity | Mixed content |
| `TokenSplitter` | Token count | LLM context |

### Backends
| Backend | Persistence | Scale | Notes |
|---------|-------------|-------|-------|
| `InMemoryBackend` | No | Small | Default, fast |
| `FAISSBackend` | Optional | Large | Facebook's FAISS |
| `ChromaBackend` | Yes | Medium | Simple persistent |
| `QdrantBackend` | Yes | Large | Production-ready |
| `PgVectorBackend` | Yes | Large | PostgreSQL extension |

### Retrievers
| Retriever | Type | Notes |
|-----------|------|-------|
| `DenseRetriever` | Vector | Wraps VectorStore |
| `BM25Retriever` | Sparse | Lexical matching |
| `HybridRetriever` | Combined | Dense + Sparse |
| `EnsembleRetriever` | Multi | N retrievers with fusion |
| `ParentDocumentRetriever` | Contextual | Return full documents |
| `SentenceWindowRetriever` | Contextual | Return surrounding context |

### Rerankers
| Reranker | Type | Notes |
|----------|------|-------|
| `CrossEncoderReranker` | Neural | Local models |
| `FlashRankReranker` | Neural | Lightweight |
| `CohereReranker` | API | Cohere Rerank |
| `LLMReranker` | LLM | Any chat model |
