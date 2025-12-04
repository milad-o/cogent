"""
RAG (Retrieval-Augmented Generation) capability.

Provides document ingestion, vector search, and citation-aware retrieval
as a composable capability that can be added to any agent.

Example:
    ```python
    from agenticflow import Agent
    from agenticflow.capabilities import RAG
    from agenticflow.vectorstore import OpenAIEmbeddings
    
    agent = Agent(
        name="ResearchAssistant",
        model=model,
        capabilities=[
            RAG(embeddings=OpenAIEmbeddings(), top_k=5)
        ],
    )
    
    # Load documents
    await agent.rag.load("report.pdf", "notes.md")
    
    # Query with citations
    response = await agent.rag.query("What are the key findings?")
    print(response.answer)
    print(response.citations)
    
    # Direct semantic search
    passages = await agent.rag.search("machine learning")
    for p in passages:
        print(f"{p.format_reference()}: {p.text[:100]}...")
    ```

Features:
- Per-file-type processing pipelines (markdown, code, PDF, etc.)
- Configurable chunking and overlap
- Multiple retriever backends (dense, hybrid, reranking)
- Structured citations with source, page, and score
- Unified query/search API
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agenticflow.capabilities.base import BaseCapability
from agenticflow.document import BaseSplitter, Document, DocumentLoader, TextChunk
from agenticflow.tools.base import tool
from agenticflow.vectorstore import VectorStore

if TYPE_CHECKING:
    from agenticflow.document.loaders import BaseLoader
    from agenticflow.models.base import BaseChatModel
    from agenticflow.retriever.base import Retriever
    from agenticflow.retriever.rerankers.base import Reranker
    from agenticflow.vectorstore.base import EmbeddingProvider


# ============================================================
# Citation & Response Types
# ============================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class CitedPassage:
    """A cited passage from a retrieved document.
    
    Contains the source reference, relevance score, and text excerpt
    used to support an answer.
    
    Attributes:
        citation_id: Citation reference number (e.g., 1, 2, 3).
        source: Source document name or path.
        page: Page number (1-based) if available.
        chunk_index: Index of the chunk within the document.
        score: Relevance/similarity score (0.0-1.0 typically).
        text: The actual text content of the passage.
        
    Example:
        >>> passage = CitedPassage(
        ...     citation_id=1,
        ...     source="report.pdf",
        ...     page=5,
        ...     chunk_index=12,
        ...     score=0.89,
        ...     text="The key finding was...",
        ... )
        >>> print(passage.format_reference())  # "[1, p.5]"
    """
    
    citation_id: int
    source: str
    page: int | None = None
    chunk_index: int | None = None
    score: float = 0.0
    text: str = ""
    
    def format_reference(self) -> str:
        """Format as inline citation reference.
        
        Returns:
            Formatted reference like "[1]" or "[1, p.5]".
        """
        if self.page is not None:
            return f"[{self.citation_id}, p.{self.page}]"
        return f"[{self.citation_id}]"
    
    def format_full(self) -> str:
        """Format as full citation entry.
        
        Returns:
            Formatted citation like "[1] Source: report.pdf, Page 5 (Score: 0.89)".
        """
        parts = [f"[{self.citation_id}] Source: {self.source}"]
        if self.page is not None:
            parts.append(f"Page {self.page}")
        if self.score > 0:
            parts.append(f"(Score: {self.score:.2f})")
        return ", ".join(parts)


@dataclass(slots=True, kw_only=True)
class RAGResponse:
    """Structured response from a RAG query with citations.
    
    Contains the answer, supporting citations, and metadata about
    the retrieval process.
    
    Attributes:
        answer: The generated answer text.
        citations: List of cited passages used to support the answer.
        sources_used: Unique source documents referenced.
        query: The original query.
        
    Example:
        >>> response = await rag.query("What are the findings?")
        >>> print(response.answer)
        >>> for cite in response.citations:
        ...     print(cite.format_full())
        >>> print(response.format_full_response())
    """
    
    answer: str
    citations: list[CitedPassage] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)
    query: str = ""
    
    @property
    def has_citations(self) -> bool:
        """Whether the response has any citations."""
        return len(self.citations) > 0
    
    @property
    def citation_count(self) -> int:
        """Number of citations."""
        return len(self.citations)
    
    def format_bibliography(self) -> str:
        """Format all citations as a bibliography.
        
        Returns:
            Formatted bibliography section.
        """
        if not self.citations:
            return ""
        
        lines = ["", "---", "**References:**"]
        for cite in self.citations:
            lines.append(cite.format_full())
        return "\n".join(lines)
    
    def format_full(self) -> str:
        """Format answer with appended citations.
        
        Returns:
            Complete response with answer and bibliography.
        """
        return self.answer + self.format_bibliography()
    
    def __str__(self) -> str:
        """String representation returns just the answer."""
        return self.answer


# ============================================================
# Pipeline Configuration
# ============================================================


@dataclass
class DocumentPipeline:
    """Processing pipeline for a specific file type.
    
    Defines how to load, split, and optionally post-process documents
    of a particular type.
    
    Example:
        >>> from agenticflow.document import MarkdownSplitter
        >>> pipeline = DocumentPipeline(
        ...     splitter=MarkdownSplitter(chunk_size=500),
        ...     metadata={"preserve_headers": True},
        ... )
    """
    
    loader: BaseLoader | None = None
    splitter: BaseSplitter | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    post_process: Callable[[list[Document]], list[Document]] | None = None


@dataclass
class PipelineRegistry:
    """Registry of document processing pipelines per file extension.
    
    Maps file extensions to their processing pipelines, enabling
    type-specific handling for different document formats.
    """
    
    pipelines: dict[str, DocumentPipeline] = field(default_factory=dict)
    default: DocumentPipeline | None = None
    
    def get(self, extension: str) -> DocumentPipeline | None:
        """Get pipeline for a file extension."""
        ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        return self.pipelines.get(ext, self.default)
    
    def register(self, extension: str, pipeline: DocumentPipeline) -> None:
        """Register a pipeline for a file extension."""
        ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        self.pipelines[ext] = pipeline
    
    @classmethod
    def create_defaults(
        cls,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> PipelineRegistry:
        """Create a registry with sensible defaults for common file types."""
        from agenticflow.document import (
            CodeSplitter,
            HTMLSplitter,
            MarkdownSplitter,
            RecursiveCharacterSplitter,
        )
        
        default_splitter = RecursiveCharacterSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        return cls(
            pipelines={
                ".md": DocumentPipeline(splitter=MarkdownSplitter(chunk_size=chunk_size)),
                ".mdx": DocumentPipeline(splitter=MarkdownSplitter(chunk_size=chunk_size)),
                ".html": DocumentPipeline(splitter=HTMLSplitter(chunk_size=chunk_size)),
                ".htm": DocumentPipeline(splitter=HTMLSplitter(chunk_size=chunk_size)),
                ".py": DocumentPipeline(splitter=CodeSplitter(language="python", chunk_size=chunk_size)),
                ".js": DocumentPipeline(splitter=CodeSplitter(language="javascript", chunk_size=chunk_size)),
                ".ts": DocumentPipeline(splitter=CodeSplitter(language="typescript", chunk_size=chunk_size)),
                ".java": DocumentPipeline(splitter=CodeSplitter(language="java", chunk_size=chunk_size)),
                ".go": DocumentPipeline(splitter=CodeSplitter(language="go", chunk_size=chunk_size)),
                ".rs": DocumentPipeline(splitter=CodeSplitter(language="rust", chunk_size=chunk_size)),
                ".cpp": DocumentPipeline(splitter=CodeSplitter(language="cpp", chunk_size=chunk_size)),
                ".c": DocumentPipeline(splitter=CodeSplitter(language="c", chunk_size=chunk_size)),
            },
            default=DocumentPipeline(splitter=default_splitter),
        )


# ============================================================
# RAG Configuration
# ============================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class RAGConfig:
    """Configuration for RAG capability.
    
    Attributes:
        chunk_size: Target size for text chunks.
        chunk_overlap: Overlap between chunks.
        top_k: Number of passages to retrieve.
        backend: Vector store backend name.
        show_progress: Print progress messages during loading.
    """
    
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4
    backend: str = "inmemory"
    show_progress: bool = True


# ============================================================
# RAG Capability
# ============================================================


class RAG(BaseCapability):
    """
    RAG (Retrieval-Augmented Generation) capability.
    
    Adds document loading, vector search, and citation-aware retrieval
    to any agent. Documents are chunked, embedded, and indexed for
    semantic search. Queries use the agent's LLM with retrieved context.
    
    Args:
        embeddings: Embedding provider for vector search. Required.
        config: RAG configuration options.
        pipelines: Per-file-type processing pipelines.
        vectorstore: Pre-configured vector store (optional).
        retriever: Pre-configured retriever (optional).
        reranker: Pre-configured reranker for two-stage retrieval.
        
    Tools provided:
        - search_documents: Search for relevant passages
        - get_document_info: Get info about loaded documents
        
    Methods:
        - load(): Load documents from files/URLs/text
        - query(): Ask a question, get RAGResponse with citations
        - search(): Direct semantic search, get list[CitedPassage]
        
    Example:
        ```python
        from agenticflow import Agent
        from agenticflow.capabilities import RAG
        from agenticflow.vectorstore import OpenAIEmbeddings
        
        agent = Agent(
            name="Assistant",
            model=model,
            capabilities=[RAG(embeddings=OpenAIEmbeddings())],
        )
        
        # Load documents
        await agent.rag.load("docs/", "report.pdf")
        
        # Query with structured citations
        response = await agent.rag.query("What is the main topic?")
        print(response.answer)
        print(response.format_bibliography())
        
        # Or use the agent directly (has search_documents tool)
        answer = await agent.run("What is the main topic?")
        ```
    """
    
    DEFAULT_INSTRUCTIONS = """When answering questions:
1. Use search_documents to find relevant information first
2. Base answers ONLY on retrieved passages - do not make up information
3. Cite sources using [1], [2], etc. matching the search results
4. Include page numbers when available: "According to [1, p.5]..."
5. If multiple sources support a point, cite all: "...confirmed in [1][2]"
6. If information isn't in documents, explicitly say so"""
    
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        *,
        config: RAGConfig | None = None,
        pipelines: PipelineRegistry | None = None,
        vectorstore: VectorStore | None = None,
        retriever: Retriever | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._config = config or RAGConfig()
        self._pipelines = pipelines or PipelineRegistry.create_defaults(
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
        )
        self._vector_store = vectorstore
        self._retriever = retriever
        self._reranker = reranker
        
        # Internal state
        self._loader = DocumentLoader()
        self._documents: list[Document] = []
        self._doc_info: dict[str, Any] = {}
        self._initialized = False
        self._last_citations: list[CitedPassage] = []
    
    @property
    def name(self) -> str:
        return "rag"
    
    @property
    def description(self) -> str:
        return "Document retrieval and question answering with citations"
    
    @property
    def tools(self) -> list:
        """RAG tools for the agent."""
        return self._create_tools()
    
    @property
    def document_count(self) -> int:
        """Number of document chunks loaded."""
        return len(self._documents)
    
    @property
    def sources(self) -> list[str]:
        """List of loaded document sources."""
        return self._doc_info.get("sources", [])
    
    @property
    def citations(self) -> list[CitedPassage]:
        """Citations from the last query/search."""
        return list(self._last_citations)
    
    @property
    def is_ready(self) -> bool:
        """Whether documents have been loaded."""
        return self._initialized
    
    # ================================================================
    # Document Loading
    # ================================================================
    
    async def load(
        self,
        *sources: str | Path,
        glob: str | None = None,
    ) -> None:
        """Load documents from files, directories, or URLs.
        
        Supports all file types via the extensible loader system:
        - Text: .txt, .md, .rst
        - Documents: .pdf, .docx
        - Data: .csv, .json, .jsonl, .xlsx
        - Web: .html, .htm
        - Code: .py, .js, .ts, .java, and many more
        
        Args:
            *sources: File paths, directory paths, or URLs.
            glob: Glob pattern when loading from directories.
            
        Example:
            >>> await rag.load("report.pdf", "notes/", glob="**/*.md")
        """
        show = self._config.show_progress
        files_by_ext: dict[str, list[Path]] = {}
        source_names: list[str] = []
        
        for source in sources:
            path = Path(source)
            
            if str(source).startswith(("http://", "https://")):
                # URL - load directly
                await self._load_url(str(source))
                source_names.append(str(source))
            elif path.is_dir():
                if show:
                    print(f"  📁 Scanning: {path}")
                for file_path in path.glob(glob or "**/*"):
                    if file_path.is_file():
                        ext = file_path.suffix.lower()
                        files_by_ext.setdefault(ext, []).append(file_path)
                source_names.append(f"{path.name}/")
            elif path.exists():
                ext = path.suffix.lower()
                files_by_ext.setdefault(ext, []).append(path)
                source_names.append(path.name)
            else:
                if show:
                    print(f"  ⚠ Not found: {path}")
        
        if not files_by_ext and not source_names:
            raise ValueError("No files found to load")
        
        # Process files by type
        all_chunks: list[Document] = []
        pipelines_used: set[str] = set()
        
        for ext, files in files_by_ext.items():
            if show:
                print(f"  📄 Processing {len(files)} {ext} file(s)...")
            
            pipeline = self._get_pipeline(files[0])
            splitter = self._get_splitter(files[0])
            pipelines_used.add(f"{ext}→{type(splitter).__name__}")
            
            docs: list[Document] = []
            for file_path in files:
                try:
                    if pipeline.loader:
                        loaded = await pipeline.loader.load(file_path)
                    else:
                        loaded = await self._loader.load(file_path)
                    
                    for doc in loaded:
                        doc.metadata.update(pipeline.metadata)
                    docs.extend(loaded)
                except Exception as e:
                    if show:
                        print(f"    ⚠ Error: {file_path.name}: {e}")
            
            if docs:
                chunks = splitter.split_documents(docs)
                if pipeline.post_process:
                    chunks = pipeline.post_process(chunks)
                all_chunks.extend(chunk.to_document() for chunk in chunks)
                
                if show:
                    print(f"    ✓ {len(chunks)} chunks from {len(docs)} docs")
        
        if all_chunks:
            await self._index_chunks(all_chunks, source_names, list(pipelines_used))
    
    async def load_text(
        self,
        text: str,
        source: str = "text",
    ) -> None:
        """Load text content directly.
        
        Args:
            text: Text content to load.
            source: Source name for citations.
        """
        if self._config.show_progress:
            print(f"  📝 Loading text: {source}")
        
        doc = Document(text=text, metadata={"source": source})
        splitter = self._get_default_splitter()
        chunks = splitter.split_documents([doc])
        await self._index_chunks(
            [c.to_document() for c in chunks],
            [source],
            ["text→" + type(splitter).__name__],
        )
    
    async def _load_url(self, url: str) -> None:
        """Load content from a URL."""
        from agenticflow.document import HTMLLoader
        import httpx
        
        if self._config.show_progress:
            print(f"  🌐 Fetching: {url}")
        
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            html_content = response.text
        
        loader = HTMLLoader()
        try:
            text, _ = loader._extract_with_bs4(html_content)
        except ImportError:
            text, _ = loader._extract_with_regex(html_content)
        
        await self.load_text(text, source=url)
    
    async def _index_chunks(
        self,
        chunks: list[Document],
        sources: list[str],
        pipelines_used: list[str],
    ) -> None:
        """Index chunks into vector store."""
        show = self._config.show_progress
        
        if show:
            print(f"  🧮 Creating embeddings for {len(chunks)} chunks...")
        
        vectorstore = self._get_vectorstore()
        await vectorstore.add_documents(chunks)
        self._documents.extend(chunks)
        
        if show:
            print("  ✓ Vector store ready")
        
        # Update metadata
        existing_sources = self._doc_info.get("sources", [])
        existing_pipelines = self._doc_info.get("pipelines_used", [])
        
        self._doc_info = {
            "sources": existing_sources + sources,
            "total_chunks": len(self._documents),
            "pipelines_used": list(set(existing_pipelines + pipelines_used)),
        }
        self._initialized = True
    
    # ================================================================
    # Query & Search
    # ================================================================
    
    async def query(
        self,
        question: str,
        *,
        k: int | None = None,
    ) -> RAGResponse:
        """Ask a question about loaded documents.
        
        Uses the agent's LLM with retrieved context to generate
        an answer with citations.
        
        Args:
            question: Your question.
            k: Number of passages to retrieve (default: config.top_k).
            
        Returns:
            RAGResponse with answer, citations, and metadata.
            
        Example:
            >>> response = await rag.query("What are the key findings?")
            >>> print(response.answer)
            >>> print(response.format_bibliography())
        """
        if not self._initialized:
            raise RuntimeError("No documents loaded. Call load() first.")
        
        if self._agent is None:
            raise RuntimeError("Capability not attached to an agent.")
        
        # Clear previous citations
        self._last_citations = []
        k = k or self._config.top_k
        
        # Retrieve relevant passages
        passages = await self.search(question, k=k)
        
        if not passages:
            return RAGResponse(
                answer="No relevant information found in the documents.",
                citations=[],
                sources_used=[],
                query=question,
            )
        
        # Build context from passages
        context_parts = []
        for p in passages:
            header = f"[{p.citation_id}] Source: {p.source}"
            if p.page:
                header += f", Page {p.page}"
            context_parts.append(f"{header}\n{p.text}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate answer using agent's LLM
        prompt = f"""Based on the following retrieved passages, answer the question.
Cite sources using [1], [2], etc. If page numbers are available, use [n, p.X].

PASSAGES:
{context}

QUESTION: {question}

ANSWER:"""
        
        answer = await self._agent.run(prompt)
        
        # Build response
        sources_used = list(dict.fromkeys(p.source for p in passages))
        
        return RAGResponse(
            answer=answer,
            citations=passages,
            sources_used=sources_used,
            query=question,
        )
    
    async def search(
        self,
        query: str,
        *,
        k: int | None = None,
    ) -> list[CitedPassage]:
        """Direct semantic search over documents.
        
        Returns the most relevant passages without LLM generation.
        Useful for exploring document content.
        
        Args:
            query: Search query.
            k: Number of results (default: config.top_k).
            
        Returns:
            List of CitedPassage objects with source info and scores.
            
        Example:
            >>> passages = await rag.search("machine learning")
            >>> for p in passages:
            ...     print(f"{p.format_reference()}: {p.text[:100]}...")
        """
        if not self._initialized:
            raise RuntimeError("No documents loaded. Call load() first.")
        
        k = k or self._config.top_k
        results = await self._retrieve(query, k=k)
        
        citations = []
        for i, (doc, score) in enumerate(results, 1):
            metadata = doc.metadata
            source = metadata.get("source", "unknown")
            if "/" in source or "\\" in source:
                source = Path(source).name
            
            citation = CitedPassage(
                citation_id=i,
                source=source,
                page=metadata.get("page"),
                chunk_index=metadata.get("chunk_index") or metadata.get("_chunk_index"),
                score=score,
                text=doc.text.strip(),
            )
            citations.append(citation)
        
        self._last_citations = citations
        return citations
    
    async def _retrieve(
        self,
        query: str,
        k: int,
    ) -> list[tuple[Document, float]]:
        """Internal retrieval with optional reranking."""
        retriever = self._get_retriever()
        
        if self._reranker:
            initial_k = k * 3
            results = await retriever.retrieve_with_scores(query, k=initial_k)
            docs = [r.document for r in results]
            reranked = await self._reranker.rerank(query, docs, top_n=k)
            return [(r.document, r.score) for r in reranked]
        
        results = await retriever.retrieve_with_scores(query, k=k)
        return [(r.document, r.score) for r in results]
    
    # ================================================================
    # Tool Creation
    # ================================================================
    
    def _create_tools(self) -> list:
        """Create RAG tools for the agent."""
        cap = self
        
        @tool
        async def search_documents(query: str, num_results: int = 4) -> str:
            """Search documents for relevant passages with citations.
            
            Returns passages with citation numbers, source info,
            page numbers (when available), and relevance scores.
            
            Args:
                query: Search query to find relevant passages.
                num_results: Number of passages to return (default: 4).
                
            Returns:
                Formatted passages with citation metadata.
            """
            if not cap._initialized:
                return "Error: No documents loaded."
            
            k = min(num_results, cap._config.top_k, 10)
            passages = await cap.search(query, k=k)
            
            if not passages:
                return "No relevant passages found."
            
            formatted = []
            for p in passages:
                header_parts = [f"[{p.citation_id}]", f"Source: {p.source}"]
                if p.page is not None:
                    header_parts.append(f"Page: {p.page}")
                header_parts.append(f"Score: {p.score:.3f}")
                header = " | ".join(header_parts)
                formatted.append(f"{header}\n{p.text}")
            
            return "\n\n---\n\n".join(formatted)
        
        @tool
        def get_document_info() -> str:
            """Get information about loaded documents.
            
            Returns:
                Document metadata (sources, chunks, pipelines).
            """
            if not cap._doc_info:
                return "No documents loaded."
            
            sources = cap._doc_info.get("sources", [])
            pipelines = cap._doc_info.get("pipelines_used", [])
            return f"""Loaded Documents:
- Sources: {', '.join(sources)}
- Total chunks: {cap._doc_info.get('total_chunks', 0)}
- Pipelines: {', '.join(pipelines) if pipelines else 'default'}"""
        
        return [search_documents, get_document_info]
    
    # ================================================================
    # Pipeline Helpers
    # ================================================================
    
    def register_pipeline(
        self,
        extension: str,
        pipeline: DocumentPipeline,
    ) -> None:
        """Register a processing pipeline for a file extension.
        
        Args:
            extension: File extension (e.g., ".pdf", ".custom").
            pipeline: DocumentPipeline with loader and splitter.
        """
        self._pipelines.register(extension, pipeline)
    
    def _get_pipeline(self, path: Path) -> DocumentPipeline:
        """Get pipeline for a file."""
        pipeline = self._pipelines.get(path.suffix)
        if pipeline:
            return pipeline
        return DocumentPipeline()
    
    def _get_splitter(self, path: Path) -> BaseSplitter:
        """Get splitter for a file."""
        pipeline = self._get_pipeline(path)
        if pipeline.splitter:
            return pipeline.splitter
        return self._get_default_splitter()
    
    def _get_default_splitter(self) -> BaseSplitter:
        """Get default splitter."""
        if self._pipelines.default and self._pipelines.default.splitter:
            return self._pipelines.default.splitter
        from agenticflow.document import RecursiveCharacterSplitter
        return RecursiveCharacterSplitter(
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
        )
    
    def _get_vectorstore(self) -> VectorStore:
        """Get or create vector store."""
        if self._vector_store is not None:
            return self._vector_store
        
        backend = self._create_backend()
        self._vector_store = VectorStore(embeddings=self._embeddings, backend=backend)
        return self._vector_store
    
    def _get_retriever(self):
        """Get retriever."""
        if self._retriever is not None:
            return self._retriever
        
        from agenticflow.retriever import DenseRetriever
        self._retriever = DenseRetriever(self._get_vectorstore())
        return self._retriever
    
    def _create_backend(self):
        """Create vector store backend."""
        from agenticflow.vectorstore.backends import InMemoryBackend
        
        backend_name = self._config.backend
        
        if backend_name == "inmemory":
            return InMemoryBackend()
        elif backend_name == "faiss":
            from agenticflow.vectorstore.backends import FAISSBackend
            dim = getattr(self._embeddings, "dimension", 1536)
            return FAISSBackend(dimension=dim)
        elif backend_name == "chroma":
            from agenticflow.vectorstore.backends import ChromaBackend
            return ChromaBackend(collection_name="rag_documents")
        elif backend_name == "qdrant":
            from agenticflow.vectorstore.backends import QdrantBackend
            return QdrantBackend(collection_name="rag_documents", location=":memory:")
        else:
            raise ValueError(f"Unknown backend: {backend_name}")


# ============================================================
# Exports
# ============================================================

__all__ = [
    "RAG",
    "RAGConfig",
    "RAGResponse",
    "CitedPassage",
    "DocumentPipeline",
    "PipelineRegistry",
]
