# RAG Architecture

## Overview

RAG (Retrieval-Augmented Generation) is a **thin capability** that provides document search tools to agents. Document loading and indexing happens **outside** the capability.

| Component | Responsibility |
|-----------|---------------|
| `DocumentLoader` | Load files (PDF, DOCX, etc.) |
| `Splitter` | Chunk documents |
| `VectorStore` | Index and store embeddings |
| `Retriever` | Search documents |
| **`RAG`** | Provide search tools to agent |

## Quick Start

```python
from agenticflow import Agent
from agenticflow.capabilities import RAG
from agenticflow.document import DocumentLoader, RecursiveCharacterSplitter
from agenticflow.retriever import DenseRetriever
from agenticflow.vectorstore import VectorStore

# 1. Load and chunk documents
loader = DocumentLoader()
docs = await loader.load_directory("docs/")
chunks = RecursiveCharacterSplitter(chunk_size=1000).split_documents(docs)

# 2. Create vectorstore and index
store = VectorStore(embeddings=embeddings)
await store.add_documents(chunks)

# 3. Create RAG with retriever
rag = RAG(DenseRetriever(store))

# 4. Add to agent
agent = Agent(model=model, capabilities=[rag])
answer = await agent.run("What are the key findings?")
```

---

## Two API Styles

### Single Retriever

```python
from agenticflow.retriever import DenseRetriever

rag = RAG(DenseRetriever(store))
```

### Multiple Retrievers with Fusion

```python
from agenticflow.retriever import DenseRetriever, BM25Retriever

# RAG creates EnsembleRetriever internally
rag = RAG(
    retrievers=[DenseRetriever(store), BM25Retriever(chunks)],
    weights=[0.6, 0.4],
    fusion="rrf",  # or "linear", "max", "voting"
)
```

**Fusion Strategies:**

| Strategy | Description |
|----------|-------------|
| `rrf` | Reciprocal Rank Fusion (default, robust) |
| `linear` | Weighted score combination |
| `max` | Maximum score per document |
| `voting` | Count appearances across retrievers |

---

## Three Access Levels

### High-Level: Agentic RAG

Agent uses `search_documents` tool automatically:

```python
agent = Agent(model=model, capabilities=[rag])
answer = await agent.run("What are the key findings?")
```

### Mid-Level: Citation-Aware Search

Direct `rag.search()` returns `CitedPassage` objects:

```python
passages = await rag.search("key findings", k=5)

for p in passages:
    print(f"{p.format_reference()} {p.source} (score: {p.score:.2f})")
    print(f"  {p.text[:100]}...")

# Format bibliography
print(rag.format_bibliography(passages))
```

### Low-Level: Direct Retriever

Access the underlying retriever:

```python
results = await rag.retriever.retrieve("query", k=5, include_scores=True)

for result in results:
    print(f"{result.score:.3f}: {result.document.text[:100]}")
```

---

## With Reranking

Add a reranker for two-stage retrieval:

```python
from agenticflow.retriever.rerankers import CrossEncoderReranker

rag = RAG(
    DenseRetriever(store),
    reranker=CrossEncoderReranker(),
)
```

---

## Configuration

```python
from agenticflow.capabilities import RAG, RAGConfig, CitationStyle

rag = RAG(
    retriever,
    config=RAGConfig(
        top_k=6,
        citation_style=CitationStyle.NUMERIC,
        include_page_in_citation=True,
        include_score_in_bibliography=True,
    ),
)
```

---

## Document Loading (Outside RAG)

RAG doesn't load documents - use `DocumentLoader` and splitters:

```python
from agenticflow.document import (
    DocumentLoader,
    RecursiveCharacterSplitter,
    SemanticSplitter,
    CodeSplitter,
)

# Load any file type
loader = DocumentLoader()
docs = await loader.load("report.pdf")
docs = await loader.load_directory("docs/", glob="**/*.md")

# Choose a splitter
splitter = RecursiveCharacterSplitter(chunk_size=1000, chunk_overlap=200)
# or
splitter = SemanticSplitter(embeddings=embeddings, threshold=0.8)
# or
splitter = CodeSplitter(language="python", chunk_size=1000)

chunks = splitter.split_documents(docs)
```

---

## Complete Example

```python
import asyncio
from agenticflow import Agent
from agenticflow.capabilities import RAG
from agenticflow.document import DocumentLoader, RecursiveCharacterSplitter
from agenticflow.retriever import DenseRetriever, BM25Retriever
from agenticflow.vectorstore import VectorStore

async def main():
    # Load documents
    loader = DocumentLoader()
    docs = await loader.load_directory("knowledge_base/")
    
    # Chunk
    splitter = RecursiveCharacterSplitter(chunk_size=1000)
    chunks = splitter.split_documents(docs)
    
    # Index
    store = VectorStore(embeddings=embeddings)
    await store.add_documents(chunks)
    
    # Create RAG with dense + sparse retrieval
    rag = RAG(
        retrievers=[DenseRetriever(store), BM25Retriever(chunks)],
        weights=[0.7, 0.3],
        fusion="rrf",
    )
    
    # Create agent
    agent = Agent(
        name="ResearchAssistant",
        model=model,
        capabilities=[rag],
    )
    
    # Ask questions
    answer = await agent.run("Summarize the main findings")
    print(answer)

asyncio.run(main())
```
