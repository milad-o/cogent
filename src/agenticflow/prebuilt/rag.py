"""
Prebuilt RAG (Retrieval-Augmented Generation) Agent.

A composable RAG system that inherits from Agent and adds:
- Per-file-type processing pipelines (loader + splitter per extension)
- DocumentLoader: Extensible document loading
- BaseSplitter: Configurable text splitting
- EmbeddingProvider: Vector embeddings
- Retriever: Configurable retrieval strategies
- Reranker: Optional two-stage retrieval
- Structured citations with source references

This module composes the agenticflow document, vectorstore, and retriever modules.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agenticflow import Agent
from agenticflow.document import BaseSplitter, Document, DocumentLoader, TextChunk
from agenticflow.tools.base import tool
from agenticflow.vectorstore import SearchResult, VectorStore
from agenticflow.vectorstore.base import EmbeddingProvider

if TYPE_CHECKING:
    from agenticflow.agent.memory import AgentMemory, MemorySaver, MemoryStore
    from agenticflow.agent.resilience import ResilienceConfig
    from agenticflow.context import RunContext
    from agenticflow.document.loaders import BaseLoader
    from agenticflow.models.base import BaseChatModel
    from agenticflow.retriever.base import Retriever
    from agenticflow.retriever.rerankers.base import Reranker


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
        retrieval_count: Number of passages retrieved.
        
    Example:
        >>> response = await rag.query_with_citations("What are the findings?")
        >>> print(response.answer)
        >>> for cite in response.citations:
        ...     print(cite.format_full())
    """
    
    answer: str
    citations: list[CitedPassage] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)
    query: str = ""
    retrieval_count: int = 0
    
    @property
    def has_citations(self) -> bool:
        """Whether the response has any citations."""
        return len(self.citations) > 0
    
    @property
    def citation_count(self) -> int:
        """Number of citations."""
        return len(self.citations)
    
    def format_citations(self) -> str:
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
    
    def format_full_response(self) -> str:
        """Format answer with appended citations.
        
        Returns:
            Complete response with answer and bibliography.
        """
        return self.answer + self.format_citations()
    
    def __str__(self) -> str:
        """String representation shows answer with citation count."""
        cite_info = f" [{self.citation_count} citations]" if self.has_citations else ""
        return f"{self.answer}{cite_info}"


# ============================================================
# Document Processing Pipeline
# ============================================================

@dataclass
class DocumentPipeline:
    """
    Processing pipeline for a specific file type.
    
    Defines how to load, split, and optionally post-process documents
    of a particular type.
    
    Example:
        ```python
        from agenticflow.document import MarkdownLoader, MarkdownSplitter
        
        # Pipeline for markdown files
        md_pipeline = DocumentPipeline(
            loader=MarkdownLoader(),
            splitter=MarkdownSplitter(chunk_size=500),
            metadata={"preserve_headers": True},
        )
        ```
    """
    loader: BaseLoader | None = None  # None = use default loader
    splitter: BaseSplitter | None = None  # None = use default splitter
    metadata: dict[str, Any] = field(default_factory=dict)  # Extra metadata to add
    post_process: Callable[[list[Document]], list[Document]] | None = None  # Optional transform


@dataclass
class PipelineRegistry:
    """
    Registry of document processing pipelines per file extension.
    
    Maps file extensions to their processing pipelines, enabling
    type-specific handling for different document formats.
    
    Example:
        ```python
        from agenticflow.document import (
            MarkdownSplitter, CodeSplitter, SemanticSplitter,
            MarkdownLoader, CodeLoader, PDFLoader,
        )
        from agenticflow.prebuilt.rag import PipelineRegistry, DocumentPipeline
        
        # Create registry with type-specific pipelines
        registry = PipelineRegistry(
            pipelines={
                ".md": DocumentPipeline(
                    loader=MarkdownLoader(),
                    splitter=MarkdownSplitter(chunk_size=500),
                ),
                ".py": DocumentPipeline(
                    loader=CodeLoader(language="python"),
                    splitter=CodeSplitter(language="python"),
                ),
                ".pdf": DocumentPipeline(
                    loader=PDFLoader(),
                    splitter=SemanticSplitter(model=model),
                ),
            },
            default=DocumentPipeline(
                splitter=RecursiveCharacterSplitter(chunk_size=1000),
            ),
        )
        
        rag = RAGAgent(model=model, embeddings=emb, pipelines=registry)
        ```
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
    
    def set_default(self, pipeline: DocumentPipeline) -> None:
        """Set the default pipeline for unregistered extensions."""
        self.default = pipeline
    
    @classmethod
    def create_defaults(
        cls,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> PipelineRegistry:
        """
        Create a registry with sensible defaults for common file types.
        
        Args:
            chunk_size: Default chunk size.
            chunk_overlap: Default chunk overlap.
            
        Returns:
            PipelineRegistry with default pipelines.
        """
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
                # Markdown - preserve structure
                ".md": DocumentPipeline(splitter=MarkdownSplitter(chunk_size=chunk_size)),
                ".mdx": DocumentPipeline(splitter=MarkdownSplitter(chunk_size=chunk_size)),
                
                # HTML - strip tags intelligently  
                ".html": DocumentPipeline(splitter=HTMLSplitter(chunk_size=chunk_size)),
                ".htm": DocumentPipeline(splitter=HTMLSplitter(chunk_size=chunk_size)),
                
                # Code - preserve functions/classes
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


class RAGAgent(Agent):
    """
    A composable RAG (Retrieval-Augmented Generation) agent.
    
    Inherits from Agent, so you get full access to all agent capabilities:
    - tools, capabilities, streaming, reasoning, structured output
    - memory, store, interceptors, resilience, observability
    - run(), chat(), think(), and all other Agent methods
    
    Plus RAG-specific functionality:
    - load_documents(): Load and index documents
    - query(): Ask questions about documents
    - search(): Direct semantic search
    
    **Per-File-Type Processing:**
    Different file types often need different processing:
    - Markdown → MarkdownSplitter (preserves headers)
    - Code → CodeSplitter (preserves functions)
    - PDF → SemanticSplitter (handles mixed content)
    - HTML → HTMLSplitter (strips tags intelligently)
    
    Use `pipelines=` to configure per-extension processing.
    
    Example - Simple:
        ```python
        from agenticflow.models import ChatModel
        from agenticflow.vectorstore import OpenAIEmbeddings
        from agenticflow.prebuilt import RAGAgent
        
        rag = RAGAgent(
            model=ChatModel(model="gpt-4o-mini"),
            embeddings=OpenAIEmbeddings(),
        )
        await rag.load_documents(["report.pdf", "notes.md"])
        answer = await rag.query("What are the key findings?")
        ```
    
    Example - Per-file-type pipelines:
        ```python
        from agenticflow.document import MarkdownSplitter, CodeSplitter, SemanticSplitter
        from agenticflow.prebuilt.rag import PipelineRegistry, DocumentPipeline
        
        # Custom pipelines per file type
        pipelines = PipelineRegistry(
            pipelines={
                ".md": DocumentPipeline(splitter=MarkdownSplitter(chunk_size=500)),
                ".py": DocumentPipeline(splitter=CodeSplitter(language="python")),
                ".pdf": DocumentPipeline(splitter=SemanticSplitter(model=model)),
            },
        )
        
        rag = RAGAgent(
            model=model,
            embeddings=embeddings,
            pipelines=pipelines,
        )
        ```
    
    Example - With additional tools:
        ```python
        from agenticflow.tools import tool
        
        @tool
        def calculate(expression: str) -> str:
            '''Evaluate a math expression.'''
            return str(eval(expression))
        
        rag = RAGAgent(
            model=model,
            embeddings=embeddings,
            tools=[calculate],  # Add custom tools alongside RAG tools!
        )
        await rag.load_documents(["financial_report.pdf"])
        # RAG agent can now search docs AND do calculations
        ```
    """
    
    DEFAULT_INSTRUCTIONS = """You are a helpful assistant that answers questions based on the provided documents.

When answering questions:
1. ALWAYS use search_documents to find relevant information first
2. Base your answers ONLY on the retrieved passages - do not make up information
3. ALWAYS cite sources using the citation format [1], [2], etc. from the search results
4. Include page numbers when available: "According to [1, p.5]..."
5. If multiple sources support a point, cite all of them: "...confirmed in [1][2]"
6. If the information isn't in the documents, explicitly say so
7. Be precise and thorough, synthesizing information from multiple passages when relevant

Citation Format:
- Use [n] to reference passage n from search results
- When page numbers are available, use [n, p.X] format
- Group citations at the end of sentences or claims they support
- Ensure every factual claim has a corresponding citation

{tools}"""
    
    def __init__(
        self,
        model: BaseChatModel,
        *,
        # RAG components (pass instances for full control)
        pipelines: PipelineRegistry | None = None,  # Per-file-type processing
        loader: DocumentLoader | None = None,  # Fallback loader
        splitter: BaseSplitter | None = None,  # Fallback splitter
        embeddings: EmbeddingProvider | None = None,
        vectorstore: VectorStore | None = None,
        retriever: Retriever | None = None,
        reranker: Reranker | None = None,
        # RAG config (used when components not provided)
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 4,
        backend: str = "inmemory",
        # All Agent parameters
        name: str = "RAG_Assistant",
        instructions: str | None = None,
        tools: Sequence[Any] | None = None,  # Additional tools beyond RAG tools
        capabilities: Sequence[Any] | None = None,
        memory: bool | MemorySaver | AgentMemory | None = None,
        store: MemoryStore | None = None,
        intercept: Sequence[Any] | None = None,
        stream: bool = False,
        reasoning: bool | Any = False,
        output: type | dict | None = None,
        verbose: bool | str = False,
        resilience: ResilienceConfig | None = None,
        interrupt_on: dict[str, Any] | None = None,
        observer: Any | None = None,
        taskboard: bool | Any = None,
    ) -> None:
        """
        Create a RAG agent.
        
        Args:
            model: Chat model for answer generation.
            
            **RAG Components (for full control):**
            pipelines: PipelineRegistry for per-file-type processing.
                Maps extensions to (loader, splitter) pairs. If not provided,
                uses sensible defaults for common file types.
            loader: Fallback DocumentLoader for unregistered extensions.
            splitter: Fallback text splitter for unregistered extensions.
            embeddings: Embedding provider. Required if vectorstore not provided.
            vectorstore: Pre-configured VectorStore. If None, creates from embeddings.
            retriever: Pre-configured retriever. If None, creates DenseRetriever.
            reranker: Pre-configured reranker (optional).
            
            **RAG Config (when components not provided):**
            chunk_size: Chunk size for default splitters (default: 1000).
            chunk_overlap: Chunk overlap for default splitters (default: 200).
            top_k: Number of passages to retrieve (default: 4).
            backend: Vector store backend ("inmemory", "faiss", "chroma", etc.).
            
            **All Agent parameters are supported:**
            name: Agent name.
            instructions: Custom system prompt (or uses default RAG prompt).
            tools: Additional tools beyond the built-in RAG tools.
            capabilities: Capabilities to attach.
            memory: Enable conversation persistence.
            store: Long-term memory store.
            intercept: Interceptors (gates, guards, prompt adapters).
            stream: Enable streaming responses.
            reasoning: Enable extended thinking.
            output: Structured output schema.
            verbose: Observability level.
            resilience: Retry/fallback configuration.
            interrupt_on: HITL tool approval rules.
            observer: Custom observer.
            taskboard: Enable task tracking.
        """
        # Store RAG config
        self._top_k = top_k
        self._backend_name = backend
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        
        # Pipeline registry for per-file-type processing
        self._pipelines = pipelines or PipelineRegistry.create_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        # Fallback loader/splitter for unregistered extensions
        self._fallback_loader = loader or DocumentLoader()
        self._fallback_splitter = splitter  # None = use default from registry
        
        # Other RAG components
        self._embeddings = embeddings
        self._vector_store = vectorstore
        self._retriever = retriever
        self._reranker = reranker
        
        # RAG state
        self._documents: list[Document] = []
        self._doc_info: dict[str, Any] = {}
        self._rag_initialized = False
        self._last_citations: list[CitedPassage] = []  # Track citations from last search
        
        # Store user's additional tools
        self._user_tools = list(tools) if tools else []
        
        # Initialize Agent with RAG tools + user tools
        rag_tools = self._create_rag_tools()
        all_tools = rag_tools + self._user_tools
        
        super().__init__(
            name=name,
            model=model,
            instructions=instructions or self.DEFAULT_INSTRUCTIONS,
            tools=all_tools,
            capabilities=capabilities,
            memory=memory,
            store=store,
            intercept=intercept,
            stream=stream,
            reasoning=reasoning,
            output=output,
            verbose=verbose,
            resilience=resilience,
            interrupt_on=interrupt_on,
            observer=observer,
            taskboard=taskboard,
        )
    
    def _get_embeddings(self) -> EmbeddingProvider:
        """Get embeddings provider.
        
        Raises:
            ValueError: If no embeddings configured.
        """
        if self._embeddings is not None:
            return self._embeddings
        
        raise ValueError(
            "No embeddings provider configured. Pass embeddings= parameter:\n"
            "  from agenticflow.vectorstore import OpenAIEmbeddings, OllamaEmbeddings\n"
            "  RAGAgent(model=..., embeddings=OpenAIEmbeddings())\n"
            "  RAGAgent(model=..., embeddings=OllamaEmbeddings())"
        )
    
    def _get_pipeline(self, path: Path) -> DocumentPipeline:
        """Get the processing pipeline for a file based on its extension."""
        ext = path.suffix.lower()
        pipeline = self._pipelines.get(ext)
        if pipeline:
            return pipeline
        # Return fallback pipeline
        return DocumentPipeline(
            loader=None,  # Will use fallback loader
            splitter=self._fallback_splitter,
        )
    
    def _get_loader_for_file(self, path: Path) -> Any:
        """Get the appropriate loader for a file."""
        pipeline = self._get_pipeline(path)
        if pipeline.loader is not None:
            return pipeline.loader
        return self._fallback_loader
    
    def _get_splitter_for_file(self, path: Path) -> BaseSplitter:
        """Get the appropriate splitter for a file."""
        pipeline = self._get_pipeline(path)
        if pipeline.splitter is not None:
            return pipeline.splitter
        if self._fallback_splitter is not None:
            return self._fallback_splitter
        # Use default from registry
        if self._pipelines.default and self._pipelines.default.splitter:
            return self._pipelines.default.splitter
        # Ultimate fallback
        from agenticflow.document import RecursiveCharacterSplitter
        return RecursiveCharacterSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
    
    def _get_vectorstore(self) -> VectorStore:
        """Get or create vector store."""
        if self._vector_store is not None:
            return self._vector_store
        
        embeddings = self._get_embeddings()
        backend = self._create_backend()
        self._vector_store = VectorStore(embeddings=embeddings, backend=backend)
        return self._vector_store
    
    def _get_retriever(self) -> Retriever:
        """Get retriever (provided or default DenseRetriever)."""
        if self._retriever is not None:
            return self._retriever
        
        from agenticflow.retriever import DenseRetriever
        vectorstore = self._get_vectorstore()
        self._retriever = DenseRetriever(vectorstore)
        return self._retriever
    
    def _create_backend(self):
        """Create backend instance from name."""
        from agenticflow.vectorstore.backends import InMemoryBackend
        
        if self._backend_name == "inmemory":
            return InMemoryBackend()
        elif self._backend_name == "faiss":
            from agenticflow.vectorstore.backends import FAISSBackend
            embeddings = self._get_embeddings()
            dim = getattr(embeddings, "dimension", 1536)
            return FAISSBackend(dimension=dim)
        elif self._backend_name == "chroma":
            from agenticflow.vectorstore.backends import ChromaBackend
            return ChromaBackend(collection_name="rag_documents")
        elif self._backend_name == "qdrant":
            from agenticflow.vectorstore.backends import QdrantBackend
            return QdrantBackend(collection_name="rag_documents", location=":memory:")
        elif self._backend_name == "pgvector":
            from agenticflow.vectorstore.backends import PgVectorBackend
            import os
            conninfo = os.environ.get("DATABASE_URL", "")
            return PgVectorBackend(conninfo=conninfo, table_name="rag_documents")
        else:
            raise ValueError(f"Unknown backend: {self._backend_name}")
    
    async def _retrieve(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Internal retrieval using configured retriever and optional reranker."""
        retriever = self._get_retriever()
        
        # Use reranker if provided (two-stage retrieval)
        if self._reranker:
            initial_k = k * 3  # Get more candidates for reranking
            results = await retriever.retrieve_with_scores(query, k=initial_k)
            docs = [r.document for r in results]
            reranked = await self._reranker.rerank(query, docs, top_n=k)
            return [(r.document, r.score) for r in reranked]
        
        # Direct retrieval
        results = await retriever.retrieve_with_scores(query, k=k)
        return [(r.document, r.score) for r in results]
    
    def _create_rag_tools(self) -> list:
        """Create RAG tools that reference this instance."""
        # Use 'rag_self' to avoid closure issues - we want live references
        rag_self = self
        
        @tool
        async def search_documents(query: str, num_results: int = 4) -> str:
            """Search documents for relevant passages with citation metadata.
            
            Returns passages formatted with citation numbers, source info,
            page numbers (when available), and relevance scores.
            
            Args:
                query: The search query to find relevant passages.
                num_results: Number of passages to return (default: 4).
                
            Returns:
                Formatted string with cited passages including:
                - Citation number [1], [2], etc.
                - Source document name
                - Page number (if available)
                - Relevance score
                - Passage text
            """
            if rag_self._vector_store is None:
                return "Error: No documents loaded."
            
            num_results = min(num_results, rag_self._top_k, 10)
            
            # Use internal retrieve method (handles retriever + reranker)
            results = await rag_self._retrieve(query, k=num_results)
            
            if not results:
                return "No relevant passages found."
            
            # Store citations for potential structured access
            rag_self._last_citations = []
            
            formatted = []
            for i, (doc, score) in enumerate(results, 1):
                content = doc.text.strip()
                metadata = doc.metadata
                
                # Extract citation info from metadata
                source = metadata.get("source", "unknown")
                # Handle both filename and full path
                if "/" in source or "\\" in source:
                    source = Path(source).name
                
                page = metadata.get("page")
                chunk_index = metadata.get("chunk_index") or metadata.get("_chunk_index")
                
                # Create CitedPassage for structured access
                citation = CitedPassage(
                    citation_id=i,
                    source=source,
                    page=page,
                    chunk_index=chunk_index,
                    score=score,
                    text=content[:500] + "..." if len(content) > 500 else content,
                )
                rag_self._last_citations.append(citation)
                
                # Format citation header with all available metadata
                header_parts = [f"[{i}]"]
                header_parts.append(f"Source: {source}")
                if page is not None:
                    header_parts.append(f"Page: {page}")
                header_parts.append(f"Score: {score:.3f}")
                header = " | ".join(header_parts)
                
                formatted.append(f"{header}\n{content}")
            
            return "\n\n---\n\n".join(formatted)
        
        @tool
        def get_document_info() -> str:
            """Get information about loaded documents.
            
            Returns:
                Document metadata (sources, chunks, etc.).
            """
            if not rag_self._doc_info:
                return "No documents loaded."
            
            sources = rag_self._doc_info.get("sources", [])
            pipelines_used = rag_self._doc_info.get("pipelines_used", [])
            return f"""Loaded Documents:
- Sources: {', '.join(sources)}
- Total chunks: {rag_self._doc_info.get('total_chunks', 0)}
- Pipelines used: {', '.join(pipelines_used) if pipelines_used else 'default'}"""
        
        return [search_documents, get_document_info]
    
    def register_pipeline(
        self,
        extension: str,
        pipeline: DocumentPipeline,
    ) -> None:
        """Register a processing pipeline for a file extension.
        
        Args:
            extension: File extension (e.g., ".xyz", ".custom").
            pipeline: DocumentPipeline with loader and splitter.
            
        Example:
            ```python
            from agenticflow.document import SemanticSplitter
            from agenticflow.prebuilt.rag import DocumentPipeline
            
            # Use semantic splitting for research papers
            rag.register_pipeline(
                ".pdf",
                DocumentPipeline(
                    splitter=SemanticSplitter(model=model),
                    metadata={"type": "research_paper"},
                ),
            )
            ```
        """
        self._pipelines.register(extension, pipeline)
    
    def register_loader(
        self,
        extension: str,
        loader: Any,
    ) -> None:
        """Register a custom document loader for a file extension.
        
        Note: For full control, use register_pipeline() instead.
        
        Args:
            extension: File extension (e.g., ".xyz", ".custom").
            loader: Loader class (subclass of BaseLoader) or async function.
            
        Example:
            ```python
            from agenticflow.document import BaseLoader
            
            class MyFormatLoader(BaseLoader):
                async def load(self, path):
                    # Custom loading logic
                    return [Document(text="...", metadata={"source": str(path)})]
            
            rag.register_loader(".myformat", MyFormatLoader)
            await rag.load_documents(["data.myformat"])
            ```
        """
        self._fallback_loader.register_loader(extension, loader)
    
    async def load_documents(
        self,
        paths: list[str | Path],
        *,
        glob: str | None = None,
        show_progress: bool = True,
    ) -> None:
        """
        Load documents from files using per-file-type pipelines.
        
        Each file type can have its own loader and splitter:
        - Markdown → MarkdownSplitter (preserves headers)
        - Code → CodeSplitter (preserves functions)
        - PDF → configured splitter (default or custom)
        - etc.
        
        Supports all formats via the extensible loader system:
        - Text: .txt, .md, .rst
        - Documents: .pdf, .docx
        - Data: .csv, .json, .jsonl, .xlsx
        - Web: .html, .htm
        - Code: .py, .js, .ts, .java, and many more
        
        Custom pipelines can be added via register_pipeline().
        
        Args:
            paths: List of file paths or directories to load.
            glob: Optional glob pattern when loading from directories.
            show_progress: Print progress messages.
        """
        # Group files by extension for per-type processing
        files_by_ext: dict[str, list[Path]] = {}
        sources: list[str] = []
        
        for path in paths:
            path = Path(path)
            
            if path.is_dir():
                if show_progress:
                    pattern = glob or "**/*"
                    print(f"  📁 Scanning directory: {path} ({pattern})")
                # Collect files from directory
                for file_path in path.glob(glob or "**/*"):
                    if file_path.is_file():
                        ext = file_path.suffix.lower()
                        if ext not in files_by_ext:
                            files_by_ext[ext] = []
                        files_by_ext[ext].append(file_path)
                sources.append(f"{path.name}/")
            elif path.exists():
                ext = path.suffix.lower()
                if ext not in files_by_ext:
                    files_by_ext[ext] = []
                files_by_ext[ext].append(path)
                sources.append(path.name)
            else:
                if show_progress:
                    print(f"  ⚠ File not found: {path}")
        
        if not files_by_ext:
            raise ValueError("No files found to load")
        
        # Process each file type with its appropriate pipeline
        all_chunks: list[Document] = []
        pipelines_used: set[str] = set()
        
        for ext, files in files_by_ext.items():
            if show_progress:
                print(f"  📄 Processing {len(files)} {ext} file(s)...")
            
            # Get pipeline for this extension
            pipeline = self._get_pipeline(files[0])
            splitter = self._get_splitter_for_file(files[0])
            pipelines_used.add(f"{ext}→{type(splitter).__name__}")
            
            # Load documents
            docs: list[Document] = []
            for file_path in files:
                try:
                    # Use pipeline loader or fallback
                    if pipeline.loader is not None:
                        loaded = await pipeline.loader.load(file_path)
                    else:
                        loaded = await self._fallback_loader.load(file_path)
                    
                    # Add pipeline metadata
                    for doc in loaded:
                        doc.metadata.update(pipeline.metadata)
                    
                    docs.extend(loaded)
                except Exception as e:
                    if show_progress:
                        print(f"    ⚠ Error loading {file_path.name}: {e}")
            
            if not docs:
                continue
            
            # Split with type-specific splitter
            text_chunks = splitter.split_documents(docs)
            
            # Apply post-processing if configured
            if pipeline.post_process:
                text_chunks = pipeline.post_process(text_chunks)
            
            # Convert TextChunk to Document for vector store
            for chunk in text_chunks:
                all_chunks.append(chunk.to_document())
            
            if show_progress:
                print(f"    ✓ {len(text_chunks)} chunks from {len(docs)} docs")
        
        if not all_chunks:
            raise ValueError("No documents were loaded successfully")
        
        await self._index_chunks(all_chunks, sources, list(pipelines_used), show_progress)
    
    async def load_text(
        self,
        text: str,
        source: str = "text",
        *,
        show_progress: bool = True,
    ) -> None:
        """
        Load text directly (no file needed).
        
        Args:
            text: The text content to load.
            source: Source name for citations.
            show_progress: Print progress messages.
        """
        if show_progress:
            print(f"  📝 Loading text from: {source}")
        
        doc = Document(text=text, metadata={"source": source})
        await self._index_documents([doc], [source], show_progress)
    
    async def load_url(
        self,
        url: str,
        *,
        show_progress: bool = True,
    ) -> None:
        """
        Load content from a URL.
        
        Args:
            url: The URL to fetch and load.
            show_progress: Print progress messages.
        """
        from agenticflow.document import HTMLLoader
        import httpx
        
        if show_progress:
            print(f"  🌐 Fetching: {url}")
        
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            html_content = response.text
        
        # Use HTMLLoader to extract text (tries BeautifulSoup, falls back to regex)
        loader = HTMLLoader()
        try:
            text, _ = loader._extract_with_bs4(html_content)
        except ImportError:
            text, _ = loader._extract_with_regex(html_content)
        
        await self.load_text(text, source=url, show_progress=show_progress)
    
    async def _index_chunks(
        self,
        chunks: list[Document],
        sources: list[str],
        pipelines_used: list[str],
        show_progress: bool = True,
    ) -> None:
        """Internal: embed and index pre-split chunks."""
        if show_progress:
            print(f"  🧮 Creating embeddings for {len(chunks)} chunks...")
        
        # Get/create vector store and add documents
        vectorstore = self._get_vectorstore()
        await vectorstore.add_documents(chunks)
        self._documents.extend(chunks)
        
        if show_progress:
            print("  ✓ Vector store ready")
        
        # Update doc info
        retriever_name = type(self._get_retriever()).__name__
        reranker_name = type(self._reranker).__name__ if self._reranker else None
        
        # Merge with existing doc info
        existing_sources = self._doc_info.get("sources", [])
        existing_pipelines = self._doc_info.get("pipelines_used", [])
        
        self._doc_info = {
            "sources": existing_sources + sources,
            "total_chunks": len(self._documents),
            "pipelines_used": list(set(existing_pipelines + pipelines_used)),
            "retriever": retriever_name,
            "reranker": reranker_name,
        }
        
        self._rag_initialized = True
    
    async def _index_documents(
        self,
        docs: list[Document],
        sources: list[str],
        show_progress: bool = True,
    ) -> None:
        """Internal: split, embed, and index documents (legacy method)."""
        # Use fallback splitter for all docs
        if show_progress:
            print("  ✂️  Splitting into chunks...")
        
        splitter = self._fallback_splitter
        if splitter is None:
            from agenticflow.document import RecursiveCharacterSplitter
            splitter = RecursiveCharacterSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
        
        text_chunks = splitter.split_documents(docs)
        # Convert TextChunk to Document for vector store
        chunks = [chunk.to_document() for chunk in text_chunks]
        
        if show_progress:
            print(f"  ✓ {len(chunks)} chunks created")
        
        await self._index_chunks(
            chunks, sources, [f"*→{type(splitter).__name__}"], show_progress
        )
    
    async def query(
        self,
        question: str,
        *,
        context: dict[str, Any] | RunContext | None = None,
        strategy: str = "dag",
        verbose: bool = False,
        max_iterations: int = 10,
    ) -> str:
        """
        Ask a question about the loaded documents.
        
        Args:
            question: Your question.
            context: Optional context dict or RunContext for tools/interceptors.
            strategy: Execution strategy ("dag", "react", "plan").
            verbose: Show detailed progress with tool calls.
            max_iterations: Maximum LLM iterations (default: 10).
            
        Returns:
            The answer with inline citations.
        """
        if not self._rag_initialized:
            raise RuntimeError("No documents loaded. Call load_documents() first.")
        
        # Clear previous citations
        self._last_citations = []
        
        if verbose:
            from agenticflow.observability import OutputConfig, ProgressTracker
            tracker = ProgressTracker(OutputConfig.verbose())
            return await self.run(
                question,
                context=context,
                strategy=strategy,
                tracker=tracker,
                max_iterations=max_iterations,
            )
        
        return await self.run(
            question,
            context=context,
            strategy=strategy,
            max_iterations=max_iterations,
        )
    
    async def query_with_citations(
        self,
        question: str,
        *,
        context: dict[str, Any] | RunContext | None = None,
        strategy: str = "dag",
        verbose: bool = False,
        max_iterations: int = 10,
    ) -> RAGResponse:
        """
        Ask a question and get a structured response with citations.
        
        Like query(), but returns a RAGResponse object containing:
        - The answer text with inline citations
        - List of CitedPassage objects with full source info
        - Metadata about sources and retrieval
        
        Args:
            question: Your question.
            context: Optional context dict or RunContext for tools/interceptors.
            strategy: Execution strategy ("dag", "react", "plan").
            verbose: Show detailed progress with tool calls.
            max_iterations: Maximum LLM iterations (default: 10).
            
        Returns:
            RAGResponse with answer, citations, and metadata.
            
        Example:
            >>> response = await rag.query_with_citations("What are the key findings?")
            >>> print(response.answer)
            >>> for cite in response.citations:
            ...     print(f"{cite.format_reference()}: {cite.source}")
            >>> print(response.format_full_response())  # Answer + bibliography
        """
        # Clear previous citations
        self._last_citations = []
        
        # Run the query
        answer = await self.query(
            question,
            context=context,
            strategy=strategy,
            verbose=verbose,
            max_iterations=max_iterations,
        )
        
        # Get unique sources from citations
        sources_used = list(dict.fromkeys(c.source for c in self._last_citations))
        
        return RAGResponse(
            answer=answer,
            citations=list(self._last_citations),  # Copy to prevent mutation
            sources_used=sources_used,
            query=question,
            retrieval_count=len(self._last_citations),
        )
    
    async def search_with_citations(
        self,
        query: str,
        k: int | None = None,
    ) -> list[CitedPassage]:
        """
        Direct semantic search returning structured citations.
        
        Like search(), but returns CitedPassage objects instead of SearchResult.
        
        Args:
            query: Search query.
            k: Number of results (default: top_k).
            
        Returns:
            List of CitedPassage objects with source info and scores.
        """
        if self._vector_store is None:
            raise RuntimeError("No documents loaded.")
        
        k = k or self._top_k
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
        
        return citations
    
    async def search(
        self,
        query: str,
        k: int | None = None,
    ) -> list[SearchResult]:
        """
        Direct semantic search (without LLM).
        
        Args:
            query: Search query.
            k: Number of results (default: top_k).
            
        Returns:
            List of SearchResult objects with document and score.
        """
        if self._vector_store is None:
            raise RuntimeError("No documents loaded.")
        
        k = k or self._top_k
        return await self._vector_store.search(query, k=k)
    
    @property
    def pipelines(self) -> PipelineRegistry:
        """Access the pipeline registry."""
        return self._pipelines
    
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
        """Citations from the last query/search operation.
        
        Returns a copy to prevent accidental mutation.
        """
        return list(self._last_citations)
    
    @property
    def last_citations(self) -> list[CitedPassage]:
        """Alias for citations property."""
        return self.citations


def create_rag_agent(
    model: BaseChatModel,
    *,
    # Pipeline registry for per-file-type processing
    pipelines: PipelineRegistry | None = None,
    # Component instances (fallbacks)
    loader: DocumentLoader | None = None,
    splitter: BaseSplitter | None = None,
    embeddings: EmbeddingProvider | None = None,
    vectorstore: VectorStore | None = None,
    retriever: Retriever | None = None,
    reranker: Reranker | None = None,
    # RAG config
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    top_k: int = 4,
    backend: str = "inmemory",
    # All Agent parameters
    name: str = "RAG_Assistant",
    instructions: str | None = None,
    tools: Sequence[Any] | None = None,
    capabilities: Sequence[Any] | None = None,
    memory: bool | MemorySaver | AgentMemory | None = None,
    store: MemoryStore | None = None,
    intercept: Sequence[Any] | None = None,
    stream: bool = False,
    reasoning: bool | Any = False,
    output: type | dict | None = None,
    verbose: bool | str = False,
    resilience: ResilienceConfig | None = None,
    interrupt_on: dict[str, Any] | None = None,
    observer: Any | None = None,
    taskboard: bool | Any = None,
) -> RAGAgent:
    """
    Create a composable RAG agent for document Q&A.
    
    Args:
        model: Chat model for answer generation.
        
        **Per-File-Type Processing:**
        pipelines: PipelineRegistry for per-extension (loader, splitter) pairs.
        
        **Component Instances (fallbacks):**
        loader: Fallback DocumentLoader for unregistered extensions.
        splitter: Fallback text splitter for unregistered extensions.
        embeddings: Embedding provider (required if vectorstore not provided).
        vectorstore: Pre-configured VectorStore.
        retriever: Pre-configured retriever.
        reranker: Pre-configured reranker.
        
        **RAG Config:**
        chunk_size: Default chunk size for splitters.
        chunk_overlap: Default chunk overlap for splitters.
        top_k: Number of passages to retrieve.
        backend: Vector store backend.
        
        **All Agent parameters are supported:**
        name, instructions, tools, capabilities, memory, store,
        intercept, stream, reasoning, output, verbose, resilience,
        interrupt_on, observer, taskboard.
        
    Returns:
        Configured RAGAgent instance (which is an Agent).
        
    Example - Simple:
        ```python
        from agenticflow.models import ChatModel
        from agenticflow.vectorstore import OpenAIEmbeddings
        
        rag = create_rag_agent(
            model=ChatModel(model="gpt-4o-mini"),
            embeddings=OpenAIEmbeddings(),
        )
        await rag.load_documents(["report.pdf"])
        answer = await rag.query("What are the key findings?")
        ```
    
    Example - Per-file-type pipelines:
        ```python
        from agenticflow.prebuilt.rag import PipelineRegistry, DocumentPipeline
        from agenticflow.document import MarkdownSplitter, CodeSplitter
        
        pipelines = PipelineRegistry(
            pipelines={
                ".md": DocumentPipeline(splitter=MarkdownSplitter(chunk_size=500)),
                ".py": DocumentPipeline(splitter=CodeSplitter(language="python")),
            },
        )
        
        rag = create_rag_agent(
            model=model,
            embeddings=embeddings,
            pipelines=pipelines,
        )
        ```
        
    Example - With additional tools:
        ```python
        @tool
        def calculate(expr: str) -> str:
            return str(eval(expr))
        
        rag = create_rag_agent(
            model=model,
            embeddings=embeddings,
            tools=[calculate],  # Extra tools alongside RAG tools
        )
        ```
    """
    return RAGAgent(
        model=model,
        pipelines=pipelines,
        loader=loader,
        splitter=splitter,
        embeddings=embeddings,
        vectorstore=vectorstore,
        retriever=retriever,
        reranker=reranker,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        backend=backend,
        name=name,
        instructions=instructions,
        tools=tools,
        capabilities=capabilities,
        memory=memory,
        store=store,
        intercept=intercept,
        stream=stream,
        reasoning=reasoning,
        output=output,
        verbose=verbose,
        resilience=resilience,
        interrupt_on=interrupt_on,
        observer=observer,
        taskboard=taskboard,
    )
