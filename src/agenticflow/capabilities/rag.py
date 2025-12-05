"""
RAG (Retrieval-Augmented Generation) capability.

API Layers:
- **Low-level**: VectorStore.search() - raw document retrieval
- **Mid-level**: rag.search() - citation-aware retrieval
- **High-level**: agent.run() with RAG capability - agentic RAG

Example (agentic RAG):
    ```python
    from agenticflow import Agent
    from agenticflow.capabilities import RAG
    
    agent = Agent(
        model=model,
        capabilities=[RAG(embeddings=embeddings)],
    )
    
    await agent.rag.load("docs/", "report.pdf")
    answer = await agent.run("What are the key findings?")
    ```

Example (citation-aware search):
    ```python
    rag = RAG(embeddings=embeddings)
    await rag.load("docs/")
    
    passages = await rag.search("key findings", k=5)
    for p in passages:
        print(f"{p.format_reference()}: {p.text[:100]}...")
    ```

Example (full control):
    ```python
    from agenticflow.capabilities import RAG, DocumentPipeline, PipelineRegistry
    from agenticflow.document import PDFLoader, SemanticSplitter
    from agenticflow.vectorstore import VectorStore, FAISSBackend
    from agenticflow.retriever import HybridRetriever
    
    # Custom pipelines
    pipelines = PipelineRegistry()
    pipelines.register(".pdf", DocumentPipeline(
        loader=PDFLoader(use_vision=True),
        splitter=SemanticSplitter(embeddings=embeddings),
    ))
    
    # Custom vectorstore
    vectorstore = VectorStore(
        embeddings=embeddings,
        backend=FAISSBackend(dimension=1536),
    )
    
    rag = RAG(
        embeddings=embeddings,
        pipelines=pipelines,
        vectorstore=vectorstore,
    )
    ```
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agenticflow.capabilities.base import BaseCapability
from agenticflow.document import BaseSplitter, Document, DocumentLoader
from agenticflow.tools.base import tool
from agenticflow.vectorstore import VectorStore

if TYPE_CHECKING:
    from agenticflow.document.loaders import BaseLoader
    from agenticflow.memory import Memory
    from agenticflow.retriever.base import Retriever
    from agenticflow.retriever.rerankers.base import Reranker
    from agenticflow.vectorstore.base import EmbeddingProvider


# ============================================================
# Citation Types
# ============================================================


class CitationStyle(Enum):
    """Citation formatting styles."""
    
    NUMERIC = "numeric"      # [1], [2], [3]
    AUTHOR_YEAR = "author_year"  # (Smith, 2023)
    FOOTNOTE = "footnote"    # ¹, ², ³
    INLINE = "inline"        # [source.pdf], [doc.md]


@dataclass(frozen=True, slots=True, kw_only=True)
class CitedPassage:
    """A cited passage from a retrieved document.
    
    Attributes:
        citation_id: Citation reference number (1, 2, 3...).
        source: Source document name.
        page: Page number if available.
        chunk_index: Chunk index within document.
        score: Relevance score (0.0-1.0).
        text: The passage text.
    """
    
    citation_id: int
    source: str
    page: int | None = None
    chunk_index: int | None = None
    score: float = 0.0
    text: str = ""
    
    def format_reference(
        self, 
        style: CitationStyle = CitationStyle.NUMERIC,
        include_page: bool = True,
    ) -> str:
        """Format citation reference.
        
        Args:
            style: Citation style (NUMERIC, AUTHOR_YEAR, FOOTNOTE, INLINE).
            include_page: Include page number if available.
            
        Returns:
            Formatted citation string.
            
        Examples:
            >>> p.format_reference(CitationStyle.NUMERIC)
            '[1]' or '[1, p.5]'
            >>> p.format_reference(CitationStyle.FOOTNOTE)
            '¹' or '¹ (p.5)'
            >>> p.format_reference(CitationStyle.INLINE)
            '[report.pdf]' or '[report.pdf, p.5]'
        """
        page_suffix = ""
        if include_page and self.page is not None:
            page_suffix = f", p.{self.page}"
        
        match style:
            case CitationStyle.NUMERIC:
                if page_suffix:
                    return f"[{self.citation_id}{page_suffix}]"
                return f"[{self.citation_id}]"
            case CitationStyle.FOOTNOTE:
                superscripts = "⁰¹²³⁴⁵⁶⁷⁸⁹"
                sup = "".join(superscripts[int(d)] for d in str(self.citation_id))
                if page_suffix:
                    return f"{sup} (p.{self.page})"
                return sup
            case CitationStyle.INLINE:
                if page_suffix:
                    return f"[{self.source}{page_suffix}]"
                return f"[{self.source}]"
            case CitationStyle.AUTHOR_YEAR:
                # Extract author-like name from source
                name = Path(self.source).stem.replace("_", " ").title()
                if page_suffix:
                    return f"({name}{page_suffix})"
                return f"({name})"
            case _:
                return f"[{self.citation_id}]"
    
    def format_full(
        self,
        include_score: bool = True,
        include_chunk: bool = False,
    ) -> str:
        """Format as full citation entry for bibliography.
        
        Args:
            include_score: Include relevance score.
            include_chunk: Include chunk index.
            
        Returns:
            Full citation string.
        """
        parts = [f"[{self.citation_id}] {self.source}"]
        if self.page is not None:
            parts.append(f"p.{self.page}")
        if include_chunk and self.chunk_index is not None:
            parts.append(f"chunk {self.chunk_index}")
        if include_score and self.score > 0:
            parts.append(f"(score: {self.score:.2f})")
        return ", ".join(parts)


@dataclass(slots=True, kw_only=True)
class RAGResponse:
    """Structured response from RAG with citations.
    
    Attributes:
        answer: The generated answer.
        citations: Supporting citations.
        sources_used: Unique sources referenced.
        query: Original query.
    """
    
    answer: str
    citations: list[CitedPassage] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)
    query: str = ""
    
    @property
    def has_citations(self) -> bool:
        return len(self.citations) > 0
    
    @property
    def citation_count(self) -> int:
        return len(self.citations)
    
    def format_bibliography(self) -> str:
        """Format citations as bibliography."""
        if not self.citations:
            return ""
        lines = ["", "---", "**References:**"]
        for cite in self.citations:
            lines.append(cite.format_full())
        return "\n".join(lines)
    
    def format_full(self) -> str:
        """Answer with bibliography appended."""
        return self.answer + self.format_bibliography()
    
    def __str__(self) -> str:
        return self.answer


# ============================================================
# Pipeline Configuration
# ============================================================


@dataclass
class DocumentPipeline:
    """Processing pipeline for a document type.
    
    Allows full control over how documents are loaded and split.
    
    Attributes:
        loader: Custom loader instance (e.g., PDFLoader with options).
        splitter: Custom splitter instance (e.g., SemanticSplitter).
        metadata: Extra metadata to add to all documents.
        post_process: Optional callback to transform chunks after splitting.
        vectorstore: Optional separate vectorstore for this doc type.
        
    Example:
        ```python
        from agenticflow.document import PDFLoader, SemanticSplitter
        
        # PDF-specific pipeline with custom settings
        pdf_pipeline = DocumentPipeline(
            loader=PDFLoader(extract_images=True, use_ocr=True),
            splitter=SemanticSplitter(embeddings=embeddings, threshold=0.8),
            metadata={"source_type": "pdf", "department": "research"},
            post_process=lambda chunks: [c for c in chunks if len(c.text) > 50],
        )
        ```
    """
    loader: BaseLoader | None = None
    splitter: BaseSplitter | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    post_process: Callable[[list[Document]], list[Document]] | None = None
    vectorstore: VectorStore | None = None  # Optional per-type vectorstore


@dataclass
class PipelineRegistry:
    """Registry of pipelines per file extension.
    
    Provides per-file-type control over loading and splitting.
    
    Example:
        ```python
        from agenticflow.capabilities import PipelineRegistry, DocumentPipeline
        from agenticflow.document import (
            PDFLoader, CodeSplitter, SemanticSplitter, MarkdownSplitter,
        )
        
        pipelines = PipelineRegistry()
        
        # Custom PDF handling
        pipelines.register(".pdf", DocumentPipeline(
            loader=PDFLoader(use_vision=True),
            splitter=SemanticSplitter(embeddings=embeddings),
        ))
        
        # Custom code handling  
        for ext, lang in [(".py", "python"), (".js", "javascript")]:
            pipelines.register(ext, DocumentPipeline(
                splitter=CodeSplitter(language=lang, chunk_size=1500),
            ))
        
        # Set default for unregistered types
        pipelines.set_default(DocumentPipeline(
            splitter=RecursiveCharacterSplitter(chunk_size=1000),
        ))
        ```
    """
    pipelines: dict[str, DocumentPipeline] = field(default_factory=dict)
    default: DocumentPipeline | None = None
    
    def get(self, extension: str) -> DocumentPipeline | None:
        """Get pipeline for extension."""
        ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        return self.pipelines.get(ext, self.default)
    
    def register(self, extension: str, pipeline: DocumentPipeline) -> None:
        """Register pipeline for extension."""
        ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        self.pipelines[ext] = pipeline
    
    def set_default(self, pipeline: DocumentPipeline) -> None:
        """Set default pipeline for unregistered extensions."""
        self.default = pipeline
    
    def register_many(self, extensions: list[str], pipeline: DocumentPipeline) -> None:
        """Register same pipeline for multiple extensions."""
        for ext in extensions:
            self.register(ext, pipeline)
    
    @classmethod
    def create_defaults(cls, chunk_size: int = 1000, chunk_overlap: int = 200) -> PipelineRegistry:
        """Create registry with sensible defaults for common file types."""
        from agenticflow.document import (
            CodeSplitter, HTMLSplitter, MarkdownSplitter, RecursiveCharacterSplitter,
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
            default=DocumentPipeline(splitter=RecursiveCharacterSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            )),
        )
    
    @classmethod
    def empty(cls) -> PipelineRegistry:
        """Create empty registry (use with custom pipelines only)."""
        return cls()


# ============================================================
# RAG Configuration
# ============================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class RAGConfig:
    """Configuration for RAG capability.
    
    Attributes:
        chunk_size: Default chunk size for splitting.
        chunk_overlap: Default overlap between chunks.
        top_k: Default number of results to retrieve.
        backend: Default vectorstore backend name.
        show_progress: Print progress during loading.
        store_to_memory: Also store documents in long-term Memory.
        memory_namespace: Namespace for memory storage.
        
    Example:
        ```python
        config = RAGConfig(
            chunk_size=500,
            chunk_overlap=50,
            top_k=6,
            backend="faiss",
            citation_style=CitationStyle.NUMERIC,
            store_to_memory=True,
            memory_namespace="documents",
        )
        
        rag = RAG(embeddings=embeddings, config=config)
        ```
    """
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4
    backend: str = "inmemory"
    show_progress: bool = True
    store_to_memory: bool = False
    memory_namespace: str = "rag:documents"
    # Citation formatting
    citation_style: CitationStyle = CitationStyle.NUMERIC
    include_page_in_citation: bool = True
    include_score_in_bibliography: bool = True


# ============================================================
# RAG Capability
# ============================================================


class RAG(BaseCapability):
    """
    RAG (Retrieval-Augmented Generation) capability.
    
    **Two usage patterns:**
    
    1. **Managed mode** - Let RAG handle document loading:
        ```python
        rag = RAG(embeddings=embeddings)
        await rag.load("docs/", "report.pdf")
        results = await rag.search("query")
        ```
    
    2. **Pre-configured mode** - Provide ready retriever:
        ```python
        # Prepare your retriever
        store = VectorStore(embeddings=embeddings)
        await store.add_documents(documents)
        retriever = HybridRetriever(
            dense=DenseRetriever(store),
            sparse=BM25Retriever(documents),
        )
        
        # Pass to RAG - no load() needed!
        rag = RAG(embeddings=embeddings, retriever=retriever)
        results = await rag.search("query")  # Works immediately
        ```
    
    **Access components directly:**
    - `rag.vectorstore` - VectorStore for search
    - `rag.retriever` - Retriever with optional reranking
    - `rag.embeddings` - Embedding provider
    
    **Customization options:**
    
    1. **Config-level** (simple settings):
        ```python
        rag = RAG(
            embeddings=embeddings,
            config=RAGConfig(chunk_size=500, top_k=6),
        )
        await rag.load("docs/")
        ```
    
    2. **Pipeline-level** (per file type):
        ```python
        pipelines = PipelineRegistry()
        pipelines.register(".pdf", DocumentPipeline(
            loader=PDFMarkdownLoader(),
            splitter=SemanticSplitter(embeddings),
        ))
        rag = RAG(embeddings=embeddings, pipelines=pipelines)
        await rag.load("docs/")
        ```
    
    3. **Component-level** (full control, no load needed):
        ```python
        rag = RAG(
            embeddings=embeddings,
            retriever=my_hybrid_retriever,  # Pre-loaded
            reranker=CrossEncoderReranker(),
        )
        # Ready to search immediately!
        ```
    """
    
    DEFAULT_INSTRUCTIONS = """When answering questions:
1. Use search_documents to find relevant information first
2. Base answers ONLY on retrieved passages - do not make up information
3. Cite sources using [1], [2], etc. matching the search results
4. Include page numbers when available: [1, p.5]
5. If information isn't in documents, say so explicitly"""
    
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        *,
        config: RAGConfig | None = None,
        pipelines: PipelineRegistry | None = None,
        vectorstore: VectorStore | None = None,
        retriever: Retriever | None = None,
        reranker: Reranker | None = None,
        memory: Memory | None = None,
        extra_tools: list | None = None,
    ) -> None:
        """
        Create RAG capability.
        
        **Two usage patterns:**
        
        1. **Managed mode** - RAG manages loading:
            ```python
            rag = RAG(embeddings=embeddings)
            await rag.load("docs/", "report.pdf")  # Required
            ```
        
        2. **Pre-configured mode** - You provide ready components:
            ```python
            # Prepare retriever with documents
            store = VectorStore(embeddings=embeddings)
            await store.add_documents(documents)
            retriever = DenseRetriever(store)
            
            # Pass to RAG - no load() needed!
            rag = RAG(embeddings=embeddings, retriever=retriever)
            ```
        
        Args:
            embeddings: Embedding provider for vectorization.
            config: RAGConfig for chunk_size, backend, etc.
            pipelines: Per-file-type processing pipelines (managed mode).
            vectorstore: Pre-populated VectorStore (pre-configured mode).
            retriever: Pre-configured retriever (pre-configured mode).
            reranker: Reranker for two-stage retrieval.
            memory: Memory instance for long-term storage.
            extra_tools: Additional tools to include with RAG tools.
            
        Note:
            If `retriever` or `vectorstore` is provided, RAG is immediately
            ready for search - no `load()` call needed.
        """
        self._embeddings = embeddings
        self._config = config or RAGConfig()
        self._pipelines = pipelines or PipelineRegistry.create_defaults(
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
        )
        self._vector_store = vectorstore
        self._retriever = retriever
        self._reranker = reranker
        self._memory = memory
        self._extra_tools = extra_tools or []
        
        self._loader = DocumentLoader()
        self._documents: list[Document] = []
        self._doc_info: dict[str, Any] = {}
        self._last_citations: list[CitedPassage] = []
        
        # If retriever is provided, RAG is ready immediately (no load needed)
        # If vectorstore is provided, also ready (user pre-populated it)
        self._initialized = retriever is not None or vectorstore is not None
    
    # ================================================================
    # Properties
    # ================================================================
    
    @property
    def name(self) -> str:
        return "rag"
    
    @property
    def description(self) -> str:
        return "Document retrieval and search"
    
    @property
    def tools(self) -> list:
        return self._create_tools() + self._extra_tools
    
    @property
    def vectorstore(self) -> VectorStore:
        """Access the vector store directly for search operations."""
        return self._get_vectorstore()
    
    @property
    def retriever(self):
        """Access the retriever (with optional reranking)."""
        return self._get_retriever()
    
    @property
    def embeddings(self) -> EmbeddingProvider:
        """Access the embedding provider."""
        return self._embeddings
    
    @property
    def pipelines(self) -> PipelineRegistry:
        """Access the pipeline registry for customization."""
        return self._pipelines
    
    @property
    def document_count(self) -> int:
        """Number of indexed document chunks."""
        return len(self._documents)
    
    @property
    def sources(self) -> list[str]:
        """List of loaded source names."""
        return self._doc_info.get("sources", [])
    
    @property
    def citations(self) -> list[CitedPassage]:
        """Citations from last search (via tools)."""
        return list(self._last_citations)
    
    @property
    def is_ready(self) -> bool:
        """Whether documents have been loaded."""
        return self._initialized
    
    @property
    def top_k(self) -> int:
        """Default number of results to retrieve."""
        return self._config.top_k
    
    # ================================================================
    # Document Loading
    # ================================================================
    
    async def load(
        self,
        *sources: str | Path | Document,
        glob: str | None = None,
    ) -> None:
        """Load documents from files, directories, URLs, or Document objects.

        Args:
            *sources: Paths to files/directories, URLs, or Document objects.
            glob: Glob pattern for directory scanning (default: "**/*").

        Examples:
            await rag.load("docs/")  # Directory
            await rag.load("file.pdf", "other.txt")  # Files
            await rag.load("https://example.com")  # URL
            await rag.load(Document(text="...", metadata={"source": "api"}))  # Raw text
        """
        show = self._config.show_progress
        files_by_ext: dict[str, list[Path]] = {}
        source_names: list[str] = []
        
        for source in sources:
            # Handle Document objects directly
            if isinstance(source, Document):
                doc_source = source.metadata.get("source", "document")
                await self._load_text(source.text, source=doc_source)
                source_names.append(doc_source)
                continue

            path = Path(source)

            if str(source).startswith(("http://", "https://")):
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
            elif show:
                print(f"  ⚠ Not found: {path}")
        
        if not files_by_ext and not source_names:
            raise ValueError("No files found to load")
        
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
                all_chunks.extend(chunks)
                if show:
                    print(f"    ✓ {len(chunks)} chunks from {len(docs)} docs")
        
        if all_chunks:
            await self._index_chunks(all_chunks, source_names, list(pipelines_used))
    
    async def _load_text(self, text: str, source: str = "text") -> None:
        """Load text content directly (internal)."""
        if self._config.show_progress:
            print(f"  📝 Loading text: {source}")

        doc = Document(text=text, metadata={"source": source})
        splitter = self._get_default_splitter()
        chunks = splitter.split_documents([doc])
        await self._index_chunks(
            chunks,
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
        
        await self._load_text(text, source=url)
    
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
        
        existing_sources = self._doc_info.get("sources", [])
        existing_pipelines = self._doc_info.get("pipelines_used", [])
        
        self._doc_info = {
            "sources": existing_sources + sources,
            "total_chunks": len(self._documents),
            "pipelines_used": list(set(existing_pipelines + pipelines_used)),
        }
        self._initialized = True
    
    # ================================================================
    # Search (converts results to CitedPassage)
    # ================================================================
    
    async def search(self, query: str, *, k: int | None = None) -> list[CitedPassage]:
        """Search documents and return CitedPassage objects with citations.
        
        This is the primary citation-aware API. Use the returned passages
        to build prompts for agent.run() or display references to users.
        
        Args:
            query: Search query.
            k: Number of results (default: config.top_k).
            
        Returns:
            List of CitedPassage objects with source, score, text, and citation_id.
            
        Example:
            ```python
            # Get citations
            passages = await rag.search("machine learning", k=5)
            
            # Display with citations
            for p in passages:
                print(f"{p.format_reference()}: {p.text[:100]}...")
            
            # Use for agent prompts
            context = "\\n".join(f"[{p.citation_id}] {p.text}" for p in passages)
            answer = await agent.run(f"Based on:\\n{context}\\n\\nAnswer: {question}")
            ```
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
            
            citations.append(CitedPassage(
                citation_id=i,
                source=source,
                page=metadata.get("page"),
                chunk_index=metadata.get("chunk_index") or metadata.get("_chunk_index"),
                score=score,
                text=doc.text.strip(),
            ))
        
        self._last_citations = citations
        return citations
    
    async def _retrieve(self, query: str, k: int) -> list[tuple[Document, float]]:
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
    # Citation Formatting Helpers
    # ================================================================
    
    def format_context(
        self,
        passages: list[CitedPassage] | None = None,
        style: CitationStyle | None = None,
        include_source: bool = True,
    ) -> str:
        """Format passages as context for LLM prompts.
        
        Args:
            passages: Passages to format (default: last search results).
            style: Citation style (default: from config).
            include_source: Include source name in header.
            
        Returns:
            Formatted context string.
            
        Example:
            ```python
            passages = await rag.search("key findings")
            context = rag.format_context(passages)
            answer = await agent.run(f"Based on:\\n{context}\\n\\nAnswer: ...")
            ```
        """
        passages = passages or self._last_citations
        style = style or self._config.citation_style
        
        if not passages:
            return ""
        
        formatted = []
        for p in passages:
            ref = p.format_reference(style, self._config.include_page_in_citation)
            if include_source:
                header = f"{ref} ({p.source})"
            else:
                header = ref
            formatted.append(f"{header}\n{p.text}")
        
        return "\n\n".join(formatted)
    
    def format_bibliography(
        self,
        passages: list[CitedPassage] | None = None,
        title: str = "References",
    ) -> str:
        """Format passages as bibliography/references section.
        
        Args:
            passages: Passages to format (default: last search results).
            title: Section title.
            
        Returns:
            Formatted bibliography string.
            
        Example:
            ```python
            passages = await rag.search("findings")
            answer = await agent.run("...")
            print(answer)
            print(rag.format_bibliography(passages))
            ```
        """
        passages = passages or self._last_citations
        
        if not passages:
            return ""
        
        # Deduplicate by source
        seen_sources: dict[str, CitedPassage] = {}
        for p in passages:
            if p.source not in seen_sources:
                seen_sources[p.source] = p
        
        lines = [f"\n---\n**{title}:**"]
        for p in seen_sources.values():
            lines.append(
                p.format_full(
                    include_score=self._config.include_score_in_bibliography,
                )
            )
        
        return "\n".join(lines)
    
    # ================================================================
    # Tools
    # ================================================================
    
    def _create_tools(self) -> list:
        """Create RAG tools for the agent."""
        cap = self
        
        @tool
        async def search_documents(query: str, num_results: int = 4) -> str:
            """Search documents for relevant passages.
            
            Args:
                query: Search query.
                num_results: Number of passages (default: 4).
                
            Returns:
                Formatted passages with citations [1], [2], etc.
            """
            if not cap._initialized:
                return "Error: No documents loaded."
            
            k = min(num_results, 10)
            passages = await cap.search(query, k=k)
            
            if not passages:
                return "No relevant passages found."
            
            formatted = []
            for p in passages:
                header_parts = [f"[{p.citation_id}]", f"Source: {p.source}"]
                if p.page is not None:
                    header_parts.append(f"Page: {p.page}")
                header_parts.append(f"Score: {p.score:.3f}")
                formatted.append(f"{' | '.join(header_parts)}\n{p.text}")
            
            return "\n\n---\n\n".join(formatted)
        
        @tool
        def get_document_info() -> str:
            """Get information about loaded documents."""
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
    # Helpers
    # ================================================================
    
    def register_pipeline(self, extension: str, pipeline: DocumentPipeline) -> None:
        """Register a processing pipeline for a file extension."""
        self._pipelines.register(extension, pipeline)
    
    def _get_pipeline(self, path: Path) -> DocumentPipeline:
        pipeline = self._pipelines.get(path.suffix)
        return pipeline if pipeline else DocumentPipeline()
    
    def _get_splitter(self, path: Path) -> BaseSplitter:
        pipeline = self._get_pipeline(path)
        if pipeline.splitter:
            return pipeline.splitter
        return self._get_default_splitter()
    
    def _get_default_splitter(self) -> BaseSplitter:
        if self._pipelines.default and self._pipelines.default.splitter:
            return self._pipelines.default.splitter
        from agenticflow.document import RecursiveCharacterSplitter
        return RecursiveCharacterSplitter(
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
        )
    
    def _get_vectorstore(self) -> VectorStore:
        if self._vector_store is not None:
            return self._vector_store
        
        backend = self._create_backend()
        self._vector_store = VectorStore(embeddings=self._embeddings, backend=backend)
        return self._vector_store
    
    def _get_retriever(self):
        if self._retriever is not None:
            return self._retriever
        
        from agenticflow.retriever import DenseRetriever
        self._retriever = DenseRetriever(self._get_vectorstore())
        return self._retriever
    
    def _create_backend(self):
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


__all__ = [
    "RAG",
    "RAGConfig",
    "RAGResponse",
    "CitedPassage",
    "DocumentPipeline",
    "PipelineRegistry",
]
