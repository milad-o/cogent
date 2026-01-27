"""Document processing module for Cogent.

This module provides comprehensive document loading, text splitting, and
summarization capabilities for building RAG (Retrieval Augmented Generation) systems.

**Document Loading:**
The loaders submodule supports loading documents from various file formats:
- Text: .txt, .md, .rst
- Documents: .pdf, .docx
- Data: .csv, .json, .jsonl, .xlsx
- Web: .html, .htm
- Code: .py, .js, .ts, .java, and many more

**Text Splitting:**
The splitters submodule provides multiple strategies for chunking text:
- RecursiveCharacterSplitter: Hierarchical separator-based splitting
- SentenceSplitter: Sentence boundary detection
- MarkdownSplitter: Markdown structure-aware splitting
- HTMLSplitter: HTML tag-based splitting
- CodeSplitter: Language-aware code splitting
- SemanticSplitter: Embedding-based semantic chunking
- TokenSplitter: Token count-based splitting

**Document Summarization:**
The summarizer module handles documents that exceed LLM context limits:
- MapReduceSummarizer: Parallel chunk summarization, then combine
- RefineSummarizer: Sequential refinement with each chunk
- HierarchicalSummarizer: Tree-based recursive summarization

Example:
    >>> from cogent.documents import (
    ...     Document,
    ...     DocumentLoader,
    ...     RecursiveCharacterSplitter,
    ...     MapReduceSummarizer,
    ... )
    >>>
    >>> # Load documents
    >>> loader = DocumentLoader()
    >>> docs = await loader.load_directory("./documents")
    >>>
    >>> # Split into chunks
    >>> splitter = RecursiveCharacterSplitter(chunk_size=1000, chunk_overlap=200)
    >>> chunks = splitter.split_documents(docs)
    >>>
    >>> # Summarize a large document
    >>> summarizer = MapReduceSummarizer(model=my_model)
    >>> result = await summarizer.summarize(docs[0].text)
"""

# Core types (re-exported from core for convenience)
from cogent.core import Document, DocumentMetadata

# Enricher
from cogent.documents.enricher import (
    EnricherConfig,
    MetadataEnricher,
    enrich_documents,
)

# Loaders
from cogent.documents.loaders import (
    LOADERS,
    BaseLoader,
    CodeLoader,
    CSVLoader,
    DocumentLoader,
    HTMLLoader,
    JSONLoader,
    MarkdownLoader,
    OutputFormat,
    PageResult,
    PageStatus,
    PDFConfig,
    PDFLoader,
    PDFMarkdownLoader,
    PDFProcessingResult,
    PDFProcessingStatus,
    PDFVisionLoader,
    PDFVisionOptions,
    ProcessingMetrics,
    TextLoader,
    WordLoader,
    XLSXLoader,
    load_documents,
    load_documents_sync,
    register_loader,
)

# Splitters
from cogent.documents.splitters import (
    BaseSplitter,
    CharacterSplitter,
    CodeSplitter,
    HTMLSplitter,
    MarkdownSplitter,
    RecursiveCharacterSplitter,
    SemanticSplitter,
    SentenceSplitter,
    TokenSplitter,
    split_text,
)

# Types
from cogent.documents.types import FileType, SplitterType

__all__ = [
    # Types
    "Document",
    "DocumentMetadata",
    "FileType",
    "SplitterType",
    # Loaders
    "BaseLoader",
    "DocumentLoader",
    "TextLoader",
    "MarkdownLoader",
    "HTMLLoader",
    "PDFLoader",
    "PDFMarkdownLoader",
    "PDFVisionLoader",
    "WordLoader",
    "CSVLoader",
    "JSONLoader",
    "XLSXLoader",
    "CodeLoader",
    "LOADERS",
    "load_documents",
    "load_documents_sync",
    "register_loader",
    # PDF types
    "PDFConfig",
    "PDFProcessingResult",
    "PageResult",
    "ProcessingMetrics",
    "PDFProcessingStatus",
    "PageStatus",
    "OutputFormat",
    "PDFVisionOptions",
    # Splitters
    "BaseSplitter",
    "RecursiveCharacterSplitter",
    "CharacterSplitter",
    "SentenceSplitter",
    "MarkdownSplitter",
    "HTMLSplitter",
    "CodeSplitter",
    "SemanticSplitter",
    "TokenSplitter",
    "split_text",
    # Enricher
    "EnricherConfig",
    "MetadataEnricher",
    "enrich_documents",
]
