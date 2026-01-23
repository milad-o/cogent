# VectorStore Module

The `agenticflow.vectorstore` module provides semantic search and vector storage for RAG applications.

## Overview

VectorStore provides:
- Document storage with embeddings
- Similarity search
- Multiple backend support (InMemory, FAISS, Chroma, Qdrant, pgvector)
- Multiple embedding providers (OpenAI, Ollama, etc.)

```python
from agenticflow.vectorstore import VectorStore

# Simple: in-memory with OpenAI embeddings
store = VectorStore()
await store.add_texts(["Python is great", "JavaScript is popular"])
results = await store.search("programming language", k=2)
```

---

## Quick Start

### Basic Usage

```python
from agenticflow.vectorstore import VectorStore

# Create store (uses OpenAI embeddings by default)
store = VectorStore()

# Add texts
await store.add_texts([
    "Python is a programming language",
    "Machine learning uses algorithms",
    "Neural networks learn from data",
])

# Search
results = await store.search("AI and deep learning", k=2)
for r in results:
    print(f"{r.score:.3f}: {r.text[:50]}")
```

### With Documents

```python
from agenticflow.vectorstore import VectorStore, Document, DocumentMetadata

# Create documents with structured metadata
docs = [
    Document(
        text="Python guide", 
        metadata=DocumentMetadata(
            source="tutorial.md",
            source_type="markdown",
            custom={"type": "tutorial", "lang": "python"}
        )
    ),
    Document(
        text="JavaScript intro", 
        metadata=DocumentMetadata(
            source="intro.md",
            custom={"type": "tutorial", "lang": "js"}
        )
    ),
    Document(
        text="ML basics", 
        metadata=DocumentMetadata(
            source="guide.pdf",
            source_type="pdf",
            page=1,
            custom={"type": "guide", "topic": "ml"}
        )
    ),
]

store = VectorStore()
await store.add_documents(docs)

# Access structured metadata in results
results = await store.search("programming tutorial", k=5)
for r in results:
    print(f"Source: {r.document.source}")  # Convenience property
    print(f"Type: {r.document.metadata.source_type}")
    print(f"Custom: {r.document.metadata.custom.get('type')}")

# Search with metadata filter (use custom dict for app-specific filters)
results = await store.search(
    "programming tutorial",
    k=5,
    filter={"custom": {"type": "tutorial"}},
)
```

**Note on Metadata Filtering:**

VectorStore backends have different filtering capabilities:

- **InMemory/FAISS**: Full dict matching, nested filtering supported
- **Chroma**: Only supports primitive types (str, int, float, bool) - nested dicts like `custom` are stringified
- **Qdrant/PGVector**: Full JSON filtering supported

For maximum compatibility, use standard DocumentMetadata fields (source, page, etc.) or keep custom dict values simple.

---

## Embedding Providers

All embedding providers support a **standardized API**:

**Primary Methods (with metadata):**
- `embed(texts)` / `aembed(texts)` → Returns `EmbeddingResult` with metadata
- `embed_one(text)` / `aembed_one(text)` → Returns vector only

**VectorStore Protocol Methods (async, no metadata):**
- `embed_texts(texts)` → Returns `list[list[float]]` 
- `embed_query(text)` → Returns `list[float]`

### OpenAI (Default)

```python
from agenticflow.vectorstore import VectorStore
from agenticflow.models import OpenAIEmbedding

# Uses text-embedding-3-small by default
store = VectorStore()

# Or specify model
embeddings = OpenAIEmbedding(model="text-embedding-3-large")
store = VectorStore(embeddings=embeddings)

# Use embeddings directly
result = await embeddings.aembed(["Hello", "World"])
print(result.metadata.tokens)  # Track token usage
```

### Ollama (Local)

```python
from agenticflow.models import OllamaEmbedding

embeddings = OllamaEmbedding(
    model="nomic-embed-text",
    host="http://localhost:11434",
)
store = VectorStore(embeddings=embeddings)
```

### Other Providers

```python
from agenticflow.models import (
    GeminiEmbedding,
    CohereEmbedding,
    MistralEmbedding,
    AzureOpenAIEmbedding,
    CloudflareEmbedding,
    CustomEmbedding,
)

# Gemini
embeddings = GeminiEmbedding(model="text-embedding-004")

# Cohere
embeddings = CohereEmbedding(model="embed-english-v3.0")

# Mistral
embeddings = MistralEmbedding(model="mistral-embed")

# Azure OpenAI
from agenticflow.models.azure import AzureEntraAuth
embeddings = AzureOpenAIEmbedding(
    azure_endpoint="https://your-resource.openai.azure.com",
    deployment="text-embedding-ada-002",
)

# Cloudflare Workers AI
embeddings = CloudflareEmbedding(
    model="@cf/baai/bge-base-en-v1.5",
    account_id="your-account-id",
)

# Custom OpenAI-compatible
embeddings = CustomEmbedding(
    base_url="http://localhost:8000/v1",
    model="custom-model",
)
```

### Mock (Testing)

```python
from agenticflow.models import MockEmbedding

embeddings = MockEmbedding(dimensions=384)
store = VectorStore(embeddings=embeddings)

# Fast, deterministic embeddings for tests
```

---

## Storage Backends

### InMemory (Default)

NumPy-based, no external dependencies:

```python
from agenticflow.vectorstore import VectorStore
from agenticflow.vectorstore.backends.inmemory import SimilarityMetric

store = VectorStore(
    metric=SimilarityMetric.COSINE,  # or DOT_PRODUCT, EUCLIDEAN
)
```

### FAISS

Large-scale similarity search:

```python
# pip install faiss-cpu (or faiss-gpu)
from agenticflow.vectorstore.backends import FAISSBackend

backend = FAISSBackend(
    index_type="IVF",      # or "Flat", "HNSW"
    nlist=100,             # Number of clusters
    nprobe=10,             # Search clusters
)
store = VectorStore(backend=backend)
```

### Chroma

Persistent vector database:

```python
# pip install chromadb
from agenticflow.vectorstore.backends import ChromaBackend

backend = ChromaBackend(
    collection_name="my_docs",
    persist_directory="./chroma_db",
)
store = VectorStore(backend=backend)
```

### Qdrant

Production-ready vector database:

```python
# pip install qdrant-client
from agenticflow.vectorstore.backends import QdrantBackend

backend = QdrantBackend(
    collection_name="my_docs",
    url="http://localhost:6333",
    # Or cloud:
    # url="https://xxx.qdrant.io",
    # api_key="...",
)
store = VectorStore(backend=backend)
```

### pgvector

PostgreSQL with vector extension:

```python
# pip install psycopg[pool]
from agenticflow.vectorstore.backends import PgVectorBackend

backend = PgVectorBackend(
    connection_string="postgresql://user:pass@localhost/db",
    table_name="embeddings",
    dimension=1536,
)
store = VectorStore(backend=backend)
```

---

## Document Management

### Adding Documents

```python
from agenticflow.vectorstore import VectorStore, Document

store = VectorStore()

# Add texts (simple)
ids = await store.add_texts([
    "First document",
    "Second document",
])

# Add texts with metadata
ids = await store.add_texts(
    texts=["Doc 1", "Doc 2"],
    metadatas=[{"type": "a"}, {"type": "b"}],
)

# Add Document objects
docs = [
    Document(text="Content", metadata={"source": "file.txt"}),
]
ids = await store.add_documents(docs)
```

### Document Utilities

```python
from agenticflow.vectorstore import (
    create_documents,
    split_text,
    split_documents,
)

# Create documents from texts
docs = create_documents(
    texts=["Text 1", "Text 2"],
    metadatas=[{"id": 1}, {"id": 2}],
)

# Split text into chunks
chunks = split_text(
    text=long_text,
    chunk_size=1000,
    chunk_overlap=200,
)

# Split documents into chunks
chunks = split_documents(
    documents=docs,
    chunk_size=1000,
    chunk_overlap=200,
)
```

---

## Searching

### Basic Search

```python
results = await store.search("query", k=5)

for r in results:
    print(f"Score: {r.score}")
    print(f"Text: {r.text}")
    print(f"Metadata: {r.metadata}")
```

### Filtered Search

```python
# Filter by metadata
results = await store.search(
    "query",
    k=10,
    filter={"type": "tutorial"},
)

# Multiple filter conditions
results = await store.search(
    "query",
    k=10,
    filter={
        "type": "tutorial",
        "language": "python",
    },
)
```

### Search with Threshold

```python
# Only return results above score threshold
results = await store.search(
    "query",
    k=10,
    score_threshold=0.7,
)
```

---

## SearchResult

Search results contain:

```python
from agenticflow.vectorstore import SearchResult

@dataclass
class SearchResult:
    text: str              # Document text
    score: float           # Similarity score
    metadata: dict         # Document metadata
    id: str | None         # Document ID
    embedding: list | None # Vector (if requested)
```

---

## Factory Function

Create vectorstore with specific configuration:

```python
from agenticflow.vectorstore import create_vectorstore

# Simple
store = create_vectorstore()

# With backend
store = create_vectorstore(
    backend="chroma",
    collection_name="docs",
    persist_directory="./data",
)

# With embeddings
store = create_vectorstore(
    embeddings="ollama",
    model="nomic-embed-text",
)
```

---

## Similarity Metrics

```python
from agenticflow.vectorstore.backends.inmemory import SimilarityMetric

SimilarityMetric.COSINE       # Normalized dot product (default)
SimilarityMetric.DOT_PRODUCT  # Raw dot product
SimilarityMetric.EUCLIDEAN    # L2 distance
```

---

## Integration with Retrievers

```python
from agenticflow.retriever import DenseRetriever
from agenticflow.vectorstore import VectorStore

# Create vectorstore and add documents
store = VectorStore()
await store.add_texts(docs)

# Create retriever
retriever = DenseRetriever(store)

# Search directly
results = await retriever.retrieve("What is X?", k=5)
for r in results:
    print(r.document.text)
```

---

## Persistence

### Save/Load (InMemory)

```python
# Save to disk
await store.save("vectorstore.pkl")

# Load from disk
store = await VectorStore.load("vectorstore.pkl")
```

### Persistent Backends

Chroma, Qdrant, and pgvector automatically persist:

```python
# Chroma persists to disk
backend = ChromaBackend(persist_directory="./data")
store = VectorStore(backend=backend)

# Data persists across sessions
```

---

## Batch Operations

```python
# Add in batches for large datasets
batch_size = 100
for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    await store.add_texts(batch)

# Parallel embedding
await store.add_texts(
    texts=large_list,
    batch_size=50,
    parallel=True,
)
```

---

## API Reference

### VectorStore

| Method | Description |
|--------|-------------|
| `add_texts(texts, metadatas?)` | Add texts to store |
| `add_documents(docs)` | Add Document objects |
| `search(query, k?, filter?)` | Similarity search |
| `delete(ids)` | Delete by IDs |
| `clear()` | Remove all documents |
| `save(path)` | Save to disk |
| `load(path)` | Load from disk |

### Embedding Providers

All embedding providers implement the `EmbeddingProvider` protocol:

```python
from agenticflow.vectorstore.base import EmbeddingProvider

class EmbeddingProvider(Protocol):
    """Protocol for embedding providers used by VectorStore."""
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
    
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
    
    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a query text."""
```

**Available providers:**
- `OpenAIEmbedding` - OpenAI API (default)
- `AzureOpenAIEmbedding` - Azure OpenAI
- `OllamaEmbedding` - Local Ollama
- `GeminiEmbedding` - Google Gemini
- `CohereEmbedding` - Cohere
- `MistralEmbedding` - Mistral AI
- `CloudflareEmbedding` - Cloudflare Workers AI
- `CustomEmbedding` - Custom OpenAI-compatible APIs
- `MockEmbedding` - Testing

All providers are imported from `agenticflow.models`.

### Backends

| Backend | Use Case |
|---------|----------|
| `InMemoryBackend` | Development, small datasets |
| `FAISSBackend` | Large-scale, local |
| `ChromaBackend` | Persistent, easy setup |
| `QdrantBackend` | Production, distributed |
| `PgVectorBackend` | PostgreSQL integration |

### Utilities

| Function | Description |
|----------|-------------|
| `create_documents(texts, metadatas)` | Create Document list |
| `split_text(text, chunk_size)` | Split text into chunks |
| `split_documents(docs, chunk_size)` | Split documents into chunks |
