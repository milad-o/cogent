# Document Module

The `cogent.documents` module provides comprehensive document loading, text splitting, and summarization capabilities for RAG (Retrieval Augmented Generation) systems.

## Overview

The document module includes:
- **Document & DocumentMetadata**: Type-safe document structure with rich metadata
- **Loaders**: Load documents from various file formats with automatic metadata population
- **Splitters**: Chunk text for embedding and retrieval while preserving metadata
- **Summarizers**: Handle documents exceeding LLM context limits

```python
from cogent.documents import (
    Document,
    DocumentMetadata,
    DocumentLoader,
    RecursiveCharacterSplitter,
)

# Load documents - metadata automatically populated
loader = DocumentLoader()
docs = await loader.load_directory("./documents")

# Access structured metadata
for doc in docs:
    print(f"Source: {doc.source}")
    print(f"Type: {doc.metadata.source_type}")
    print(f"Page: {doc.metadata.page}")

# Split into chunks - metadata preserved
splitter = RecursiveCharacterSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
```

---

## Document & Metadata

### Document

Core document type used throughout cogent:

```python
from cogent.documents import Document, DocumentMetadata

# Create with structured metadata
doc = Document(
    text="Document content...",
    metadata=DocumentMetadata(
        source="report.pdf",
        source_type="pdf",
        page=5,
        loader="PDFMarkdownLoader"
    )
)

# Convenience properties
print(doc.id)        # doc_a1b2c3d4e5f6g7h8
print(doc.source)    # report.pdf
print(doc.page)      # 5

# Access full metadata
print(doc.metadata.char_count)  # Auto-populated
print(doc.metadata.timestamp)   # Creation time
```

### DocumentMetadata

Structured metadata with type safety and IDE autocomplete:

```python
@dataclass
class DocumentMetadata:
    """Structured metadata for documents with provenance and tracking."""
    
    # Identification & timing
    id: str                        # Auto-generated: doc_a1b2c3d4
    timestamp: float               # Unix timestamp
    
    # Source information
    source: str                    # File path, URL, etc.
    source_type: str | None        # "pdf", "markdown", "web", "api"
    
    # Content positioning (for chunked documents)
    page: int | None               # Page number
    chunk_index: int | None        # Chunk position (0-based)
    chunk_total: int | None        # Total chunks from parent
    start_char: int | None         # Start position in parent
    end_char: int | None           # End position in parent
    
    # Content metrics
    token_count: int | None        # Token count (if computed)
    char_count: int | None         # Character count (auto-populated)
    
    # Provenance & relationships
    loader: str | None             # Loader that created document
    created_by: str | None         # Agent/tool name
    parent_id: str | None          # Parent document ID (for chunks)
    
    # Custom fields (extensibility)
    custom: dict[str, Any]         # User-defined metadata
```

**Key Features:**

1. **Type Safety**: Properties with proper types, no dict key errors
2. **Auto-population**: `char_count`, `id`, `timestamp` set automatically
3. **Provenance**: Track `loader`, `created_by`, `parent_id` for observability
4. **Chunking-aware**: `chunk_index`, `chunk_total`, `parent_id` for chunk relationships
5. **Extensible**: Use `custom` dict for application-specific fields

**Example - Chunk Metadata:**

```python
# Loaders automatically populate metadata
docs = await PDFMarkdownLoader().load("report.pdf")
doc = docs[0]

print(doc.metadata.loader)       # "PDFMarkdownLoader"
print(doc.metadata.source_type)  # "pdf"
print(doc.metadata.page)         # 1

# Splitters preserve and extend metadata
splitter = RecursiveCharacterSplitter(chunk_size=500)
chunks = splitter.split_documents(docs)

chunk = chunks[0]
print(chunk.metadata.parent_id)    # ID of original document
print(chunk.metadata.chunk_index)  # 0
print(chunk.metadata.chunk_total)  # 10
print(chunk.metadata.start_char)   # 0
print(chunk.metadata.end_char)     # 500
```

**Serialization:**

```python
# To dict (for storage/API)
data = doc.to_dict()
# {
#   "text": "...",
#   "metadata": {
#     "id": "doc_...",
#     "source": "file.txt",
#     "char_count": 1234,
#     ...
#   }
# }

# From dict (backward compatible - collects unknown fields into custom)
doc = Document.from_dict({
    "text": "...",
    "metadata": {
        "source": "file.txt",
        "custom_field": "value"  # Goes into metadata.custom
    }
})
print(doc.metadata.custom["custom_field"])  # "value"
```

---

## Document Loaders

### DocumentLoader

The main loader class that auto-detects file types:

```python
from cogent.documents.loaders import DocumentLoader

loader = DocumentLoader()

# Load single file
docs = await loader.load("document.pdf")

# Load directory
docs = await loader.load_directory("./docs", glob="**/*.md")

# Load with options
docs = await loader.load("data.csv", encoding="utf-8")
```

### Supported Formats

| Format | Extensions | Loader |
|--------|------------|--------|
| Text | .txt, .rst, .log | `TextLoader` |
| Markdown | .md | `MarkdownLoader` |
| HTML | .html, .htm | `HTMLLoader` |
| PDF | .pdf | `PDFLoader` |
| Word | .docx | `WordLoader` |
| CSV | .csv | `CSVLoader` |
| JSON | .json, .jsonl | `JSONLoader` |
| Excel | .xlsx | `XLSXLoader` |
| Code | .py, .js, .ts, etc. | `CodeLoader` |

### PDFMarkdownLoader

High-performance PDF loader optimized for LLM/RAG with parallel processing:

```python
from cogent.documents.loaders import PDFMarkdownLoader

loader = PDFMarkdownLoader(
    max_workers=4,      # CPU workers for parallel processing
    batch_size=10,      # Pages per batch
)

# Standard API - returns list[Document] (consistent with other loaders)
docs = await loader.load("large_document.pdf")
print(f"Loaded {len(docs)} pages")

# With tracking - returns PDFProcessingResult with metrics
result = await loader.load("large_document.pdf", tracking=True)
print(f"Success rate: {result.success_rate:.0%}")
print(f"Time: {result.total_time_ms:.0f}ms")
docs = result.documents
```

**PDFProcessingResult (from `load(tracking=True)`):**
```python
@dataclass
class PDFProcessingResult:
    file_path: Path
    status: PDFProcessingStatus
    total_pages: int
    successful_pages: int
    failed_pages: int
    empty_pages: int
    page_results: list[PageResult]
    total_time_ms: float
    
    @property
    def success_rate(self) -> float:
        """Returns ratio 0.0-1.0 for percentage formatting."""
        ...
    
    @property
    def documents(self) -> list[Document]:
        """Convert page results to Document list."""
        ...
```

### Convenience Functions

```python
from cogent.documents.loaders import load_documents, load_documents_sync

# Async loading
docs = await load_documents("./data")

# Sync loading (for scripts)
docs = load_documents_sync("./data")
```

### Custom Loaders

Register custom file handlers:

```python
from cogent.documents.loaders import BaseLoader, register_loader

class XMLLoader(BaseLoader):
    EXTENSIONS = [".xml"]
    
    async def load(self, path, **kwargs) -> list[Document]:
        # Custom loading logic
        ...

register_loader(XMLLoader)
```

---

## Text Splitters

### RecursiveCharacterSplitter

The recommended splitter for most use cases:

```python
from cogent.documents.splitters import RecursiveCharacterSplitter

splitter = RecursiveCharacterSplitter(
    chunk_size=1000,      # Target chunk size
    chunk_overlap=200,    # Overlap between chunks
    separators=["\n\n", "\n", " ", ""],  # Hierarchy
)

chunks = splitter.split_text(text)
chunks = splitter.split_documents(docs)
```

### Available Splitters

| Splitter | Use Case |
|----------|----------|
| `RecursiveCharacterSplitter` | General text, preserves structure |
| `CharacterSplitter` | Simple single-separator splitting |
| `SentenceSplitter` | Sentence boundary detection |
| `MarkdownSplitter` | Markdown structure-aware |
| `HTMLSplitter` | HTML tag-based splitting |
| `CodeSplitter` | Language-aware code splitting |
| `SemanticSplitter` | Embedding-based semantic chunking |
| `TokenSplitter` | Token count-based (for LLMs) |

### SentenceSplitter

Split by sentence boundaries:

```python
from cogent.documents.splitters import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

chunks = splitter.split_text(text)
```

### MarkdownSplitter

Preserve Markdown structure:

```python
from cogent.documents.splitters import MarkdownSplitter

splitter = MarkdownSplitter(
    chunk_size=1000,
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ],
)

chunks = splitter.split_text(markdown_text)
```

### CodeSplitter

Language-aware code splitting:

```python
from cogent.documents.splitters import CodeSplitter

splitter = CodeSplitter(
    language="python",
    chunk_size=1000,
)

chunks = splitter.split_text(python_code)
```

**Supported languages:** Python, JavaScript, TypeScript, Java, C++, Go, Rust, Ruby, and more.

### SemanticSplitter

Split based on semantic similarity:

```python
from cogent.documents.splitters import SemanticSplitter

splitter = SemanticSplitter(
    embedding_model=my_embeddings,
    breakpoint_threshold=0.5,  # Similarity threshold
)

chunks = await splitter.split_text(text)
```

### TokenSplitter

Split by token count (for LLM context limits):

```python
from cogent.documents.splitters import TokenSplitter

splitter = TokenSplitter(
    chunk_size=512,       # Max tokens per chunk
    chunk_overlap=50,     # Token overlap
    encoding="cl100k_base",  # Tokenizer encoding
)

chunks = splitter.split_text(text)
```

### Convenience Function

```python
from cogent.documents.splitters import split_text

chunks = split_text(
    text,
    chunk_size=1000,
    chunk_overlap=200,
    splitter_type="recursive",  # or "sentence", "markdown", etc.
)
```

---

## Document Type

The standard document container:

```python
from cogent.documents import Document

doc = Document(
    text="Document content...",
    metadata={
        "source": "file.pdf",
        "page": 1,
        "author": "John Doe",
    },
)

# Access
print(doc.text)
print(doc.metadata["source"])
```

### TextChunk

Chunk with position information:

```python
from cogent.documents import TextChunk

chunk = TextChunk(
    text="Chunk content...",
    start=0,
    end=1000,
    metadata={"chunk_index": 0},
)
```

---

## Document Summarization

For documents exceeding LLM context limits, use summarization strategies:

### MapReduceSummarizer

Parallel chunk summarization, then combine:

```python
from cogent.documents.summarizer import MapReduceSummarizer

summarizer = MapReduceSummarizer(model=my_model)
result = await summarizer.summarize(long_text)
```

### RefineSummarizer

Sequential refinement through chunks:

```python
from cogent.documents.summarizer import RefineSummarizer

summarizer = RefineSummarizer(model=my_model)
result = await summarizer.summarize(long_text)
```

### HierarchicalSummarizer

Tree-based recursive summarization:

```python
from cogent.documents.summarizer import HierarchicalSummarizer

summarizer = HierarchicalSummarizer(
    model=my_model,
    levels=3,
)
result = await summarizer.summarize(very_long_text)
```

---

## Exports

```python
from cogent.documents import (
    # Types
    Document,
    TextChunk,
    FileType,
    SplitterType,
    # Loaders
    BaseLoader,
    DocumentLoader,
    TextLoader,
    MarkdownLoader,
    HTMLLoader,
    PDFLoader,
    PDFMarkdownLoader,
    WordLoader,
    CSVLoader,
    JSONLoader,
    XLSXLoader,
    CodeLoader,
    load_documents,
    load_documents_sync,
    register_loader,
    # Splitters
    BaseSplitter,
    RecursiveCharacterSplitter,
    CharacterSplitter,
    SentenceSplitter,
    MarkdownSplitter,
    HTMLSplitter,
    CodeSplitter,
    SemanticSplitter,
    TokenSplitter,
    split_text,
)

from cogent.documents.loaders import (
    # PDF LLM types
    PDFProcessingResult,
    PDFProcessingStatus,
    PageResult,
    PageStatus,
    PDFConfig,
    ProcessingMetrics,
    OutputFormat,
)
```
