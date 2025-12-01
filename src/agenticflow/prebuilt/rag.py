"""
Prebuilt RAG (Retrieval-Augmented Generation) Agent.

A composable RAG system that accepts pre-configured components:
- DocumentLoader: Extensible document loading
- BaseSplitter: Configurable text splitting
- EmbeddingProvider: Vector embeddings
- Retriever: Configurable retrieval strategies
- Reranker: Optional two-stage retrieval

This module composes the agenticflow document, vectorstore, and retriever modules.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from agenticflow import Agent
from agenticflow.document import Document, DocumentLoader, BaseSplitter
from agenticflow.tools.base import tool
from agenticflow.vectorstore import SearchResult, VectorStore
from agenticflow.vectorstore.base import EmbeddingProvider

if TYPE_CHECKING:
    from agenticflow.agent.resilience import ResilienceConfig
    from agenticflow.context import RunContext
    from agenticflow.models.base import BaseChatModel
    from agenticflow.retriever.base import Retriever
    from agenticflow.retriever.rerankers.base import Reranker


class RAGAgent:
    """
    A composable RAG (Retrieval-Augmented Generation) agent.
    
    Accepts pre-configured components for maximum flexibility:
    - loader: DocumentLoader instance for loading files
    - splitter: BaseSplitter instance for chunking text
    - embeddings: EmbeddingProvider for vectorization
    - retriever: Retriever instance for search (optional, auto-created if not provided)
    - reranker: Reranker instance for two-stage retrieval (optional)
    
    Example - Simple (auto-configured):
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
    
    Example - Composable (full control):
        ```python
        from agenticflow.document import (
            DocumentLoader,
            SemanticSplitter,
            MarkdownSplitter,
        )
        from agenticflow.vectorstore import OllamaEmbeddings, VectorStore
        from agenticflow.retriever import HybridRetriever
        from agenticflow.retriever.rerankers import LLMReranker
        
        # Configure each component
        loader = DocumentLoader()
        loader.register_loader(".custom", MyCustomLoader)
        
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        splitter = MarkdownSplitter(chunk_size=500)
        
        rag = RAGAgent(
            model=model,
            loader=loader,
            splitter=splitter,
            embeddings=embeddings,
            reranker=LLMReranker(model=model),
        )
        ```
    """
    
    def __init__(
        self,
        model: BaseChatModel,
        *,
        # Core components (pass instances for full control)
        loader: DocumentLoader | None = None,
        splitter: BaseSplitter | None = None,
        embeddings: EmbeddingProvider | None = None,
        vectorstore: VectorStore | None = None,
        retriever: Retriever | None = None,
        reranker: Reranker | None = None,
        # Simple config (used when components not provided)
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 4,
        backend: str = "inmemory",
        # Agent configuration
        name: str = "RAG_Assistant",
        instructions: str | None = None,
        intercept: Sequence[Any] | None = None,
        stream: bool = False,
        reasoning: bool | Any = False,
        output: type | dict | None = None,
        verbose: bool | str = False,
        resilience: ResilienceConfig | None = None,
    ) -> None:
        """
        Create a RAG agent.
        
        Args:
            model: Chat model for answer generation.
            
            **Component Instances (recommended for full control):**
            loader: Pre-configured DocumentLoader. If None, creates default.
            splitter: Pre-configured text splitter (RecursiveCharacterSplitter, 
                SemanticSplitter, MarkdownSplitter, etc.). If None, creates default.
            embeddings: Pre-configured embedding provider (OpenAIEmbeddings,
                OllamaEmbeddings, etc.). Required if vectorstore not provided.
            vectorstore: Pre-configured VectorStore. If None, creates from embeddings.
            retriever: Pre-configured retriever (DenseRetriever, HybridRetriever,
                EnsembleRetriever). If None, creates DenseRetriever.
            reranker: Pre-configured reranker (LLMReranker, CohereReranker, etc.).
            
            **Simple Config (used when components not provided):**
            chunk_size: Chunk size for default splitter (default: 1000).
            chunk_overlap: Chunk overlap for default splitter (default: 200).
            top_k: Number of passages to retrieve (default: 4).
            backend: Vector store backend when auto-creating ("inmemory", "faiss", etc.).
            
            **Agent Configuration:**
            name: Agent name.
            instructions: Custom system prompt.
            intercept: Interceptors (gates, guards, prompt adapters).
            stream: Enable streaming responses.
            reasoning: Enable extended thinking.
            output: Structured output schema.
            verbose: Observability level.
            resilience: Retry/fallback configuration.
        """
        self._model = model
        self._top_k = top_k
        self._name = name
        self._custom_instructions = instructions
        self._backend_name = backend
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        
        # Agent configuration
        self._intercept = intercept
        self._stream = stream
        self._reasoning = reasoning
        self._output = output
        self._verbose = verbose
        self._resilience = resilience
        
        # Store provided components (or None to auto-create)
        self._loader = loader or DocumentLoader()
        self._splitter = splitter  # None = create default on first use
        self._embeddings = embeddings
        self._vector_store = vectorstore
        self._retriever = retriever
        self._reranker = reranker
        
        # State
        self._documents: list[Document] = []
        self._doc_info: dict[str, Any] = {}
        self._agent: Agent | None = None
    
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
    
    def _get_splitter(self) -> BaseSplitter:
        """Get text splitter (provided or default)."""
        if self._splitter is not None:
            return self._splitter
        # Lazy import to avoid circular dependency
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
    
    def _create_tools(self) -> list:
        """Create RAG tools that reference this instance."""
        # Use 'rag_self' to avoid closure issues - we want live references
        rag_self = self
        
        @tool
        async def search_documents(query: str, num_results: int = 4) -> str:
            """Search documents for relevant passages.
            
            Args:
                query: The search query to find relevant passages.
                num_results: Number of passages to return (default: 4).
                
            Returns:
                Formatted string with relevant passages.
            """
            if rag_self._vector_store is None:
                return "Error: No documents loaded."
            
            num_results = min(num_results, rag_self._top_k, 10)
            
            # Use internal retrieve method (handles retriever + reranker)
            results = await rag_self._retrieve(query, k=num_results)
            
            if not results:
                return "No relevant passages found."
            
            formatted = []
            for i, (doc, score) in enumerate(results, 1):
                content = doc.text.strip()
                source = doc.metadata.get("source", "unknown")
                formatted.append(f"[{i}] (Source: {source}, Score: {score:.3f})\n{content}")
            
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
            return f"""Loaded Documents:
- Sources: {', '.join(sources)}
- Total chunks: {rag_self._doc_info.get('total_chunks', 0)}
- Chunk size: {rag_self._doc_info.get('chunk_size', 0)} characters
- Chunk overlap: {rag_self._doc_info.get('overlap', 0)} characters"""
        
        return [search_documents, get_document_info]
    
    def _create_agent(self) -> Agent:
        """Create the RAG agent with tools."""
        default_instructions = """You are a helpful assistant that answers questions based on the provided documents.

When answering questions:
1. ALWAYS use search_documents to find relevant information
2. Base your answers ONLY on the retrieved passages
3. Cite sources using [1], [2], etc.
4. If the information isn't in the documents, say so
5. Be concise but thorough

{tools}"""
        
        instructions = self._custom_instructions or default_instructions
        
        return Agent(
            name=self._name,
            model=self._model,
            instructions=instructions,
            tools=self._create_tools(),
            intercept=self._intercept,
            stream=self._stream,
            reasoning=self._reasoning,
            output=self._output,
            verbose=self._verbose,
            resilience=self._resilience,
        )
    
    def register_loader(
        self,
        extension: str,
        loader: Any,
    ) -> None:
        """Register a custom document loader for a file extension.
        
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
        self._loader.register_loader(extension, loader)
    
    async def load_documents(
        self,
        paths: list[str | Path],
        *,
        glob: str | None = None,
        show_progress: bool = True,
    ) -> None:
        """
        Load documents from files.
        
        Supports all formats via the extensible loader system:
        - Text: .txt, .md, .rst
        - Documents: .pdf, .docx
        - Data: .csv, .json, .jsonl, .xlsx
        - Web: .html, .htm
        - Code: .py, .js, .ts, .java, and many more
        
        Custom formats can be added via register_loader() or by passing
        a pre-configured DocumentLoader instance.
        
        Args:
            paths: List of file paths or directories to load.
            glob: Optional glob pattern when loading from directories.
            show_progress: Print progress messages.
        """
        all_docs: list[Document] = []
        sources: list[str] = []
        
        for path in paths:
            path = Path(path)
            
            if path.is_dir():
                if show_progress:
                    pattern = glob or "**/*"
                    print(f"  📁 Loading directory: {path} ({pattern})")
                try:
                    docs = await self._loader.load_directory(path, glob=glob)
                    all_docs.extend(docs)
                    sources.append(f"{path.name}/ ({len(docs)} files)")
                except Exception as e:
                    if show_progress:
                        print(f"  ⚠ Error loading directory {path}: {e}")
            elif path.exists():
                if show_progress:
                    print(f"  📄 Loading: {path.name}")
                try:
                    docs = await self._loader.load(path)
                    all_docs.extend(docs)
                    sources.append(path.name)
                except Exception as e:
                    if show_progress:
                        print(f"  ⚠ Error loading {path.name}: {e}")
            else:
                if show_progress:
                    print(f"  ⚠ File not found: {path}")
        
        if not all_docs:
            raise ValueError("No documents were loaded successfully")
        
        await self._index_documents(all_docs, sources, show_progress)
    
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
    
    async def _index_documents(
        self,
        docs: list[Document],
        sources: list[str],
        show_progress: bool = True,
    ) -> None:
        """Internal: split, embed, and index documents."""
        # Split documents
        if show_progress:
            print("  ✂️  Splitting into chunks...")
        
        splitter = self._get_splitter()
        chunks = splitter.split_documents(docs)
        
        if show_progress:
            print(f"  ✓ {len(chunks)} chunks created")
            print("  🧮 Creating embeddings...")
        
        # Get/create vector store and add documents
        vectorstore = self._get_vectorstore()
        await vectorstore.add_documents(chunks)
        self._documents = chunks
        
        if show_progress:
            print("  ✓ Vector store ready")
        
        # Update doc info
        splitter_name = type(splitter).__name__
        retriever_name = type(self._get_retriever()).__name__
        reranker_name = type(self._reranker).__name__ if self._reranker else None
        
        self._doc_info = {
            "sources": sources,
            "total_chunks": len(chunks),
            "splitter": splitter_name,
            "retriever": retriever_name,
            "reranker": reranker_name,
        }
        
        # Create agent with tools
        self._agent = self._create_agent()
    
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
            The answer with citations.
        """
        if self._agent is None:
            raise RuntimeError("No documents loaded. Call load_documents() first.")
        
        if verbose:
            from agenticflow.observability import OutputConfig, ProgressTracker
            tracker = ProgressTracker(OutputConfig.verbose())
            return await self._agent.run(
                question,
                context=context,
                strategy=strategy,
                tracker=tracker,
                max_iterations=max_iterations,
            )
        
        return await self._agent.run(
            question,
            context=context,
            strategy=strategy,
            max_iterations=max_iterations,
        )
    
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
    def agent(self) -> Agent | None:
        """Access the underlying Agent if needed."""
        return self._agent
    
    @property
    def document_count(self) -> int:
        """Number of document chunks loaded."""
        return len(self._documents)
    
    @property
    def sources(self) -> list[str]:
        """List of loaded document sources."""
        return self._doc_info.get("sources", [])


def create_rag_agent(
    model: BaseChatModel,
    *,
    # Component instances
    loader: DocumentLoader | None = None,
    splitter: BaseSplitter | None = None,
    embeddings: EmbeddingProvider | None = None,
    vectorstore: VectorStore | None = None,
    retriever: Retriever | None = None,
    reranker: Reranker | None = None,
    # Simple config
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    top_k: int = 4,
    backend: str = "inmemory",
    # Agent configuration
    name: str = "RAG_Assistant",
    instructions: str | None = None,
    intercept: Sequence[Any] | None = None,
    stream: bool = False,
    reasoning: bool | Any = False,
    output: type | dict | None = None,
    verbose: bool | str = False,
    resilience: ResilienceConfig | None = None,
) -> RAGAgent:
    """
    Create a composable RAG agent for document Q&A.
    
    Args:
        model: Chat model for answer generation.
        
        **Component Instances (for full control):**
        loader: Pre-configured DocumentLoader.
        splitter: Pre-configured text splitter.
        embeddings: Pre-configured embedding provider.
        vectorstore: Pre-configured VectorStore.
        retriever: Pre-configured retriever.
        reranker: Pre-configured reranker.
        
        **Simple Config (when components not provided):**
        chunk_size: Chunk size for default splitter.
        chunk_overlap: Chunk overlap for default splitter.
        top_k: Number of passages to retrieve.
        backend: Vector store backend.
        
        **Agent Configuration:**
        name: Agent name.
        instructions: Custom system prompt.
        intercept: Interceptors.
        stream: Enable streaming.
        reasoning: Enable extended thinking.
        output: Structured output schema.
        verbose: Observability level.
        resilience: Retry/fallback config.
        
    Returns:
        Configured RAGAgent instance.
        
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
    
    Example - Composable:
        ```python
        from agenticflow.document import MarkdownSplitter
        from agenticflow.retriever import HybridRetriever
        from agenticflow.retriever.rerankers import LLMReranker
        
        rag = create_rag_agent(
            model=model,
            splitter=MarkdownSplitter(chunk_size=500),
            embeddings=embeddings,
            reranker=LLMReranker(model=model),
        )
        ```
    """
    return RAGAgent(
        model=model,
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
        intercept=intercept,
        stream=stream,
        reasoning=reasoning,
        output=output,
        verbose=verbose,
        resilience=resilience,
    )
