"""
Prebuilt RAG (Retrieval-Augmented Generation) Agent.

A complete, ready-to-use RAG system with:
- Document loading (PDF, TXT, MD, HTML)
- Text splitting
- Vector embeddings (native)
- Semantic search with optional hybrid retrieval
- Reranking support
- LLM-powered Q&A

This module uses the native agenticflow vector store and retriever.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from agenticflow import Agent
from agenticflow.tools.base import tool
from agenticflow.vectorstore import (
    Document,
    SearchResult,
    VectorStore,
    split_text,
)
from agenticflow.vectorstore.base import EmbeddingProvider

if TYPE_CHECKING:
    from agenticflow.agent.resilience import ResilienceConfig
    from agenticflow.context import RunContext
    from agenticflow.models.base import BaseChatModel
    from agenticflow.retriever.base import Retriever
    from agenticflow.retriever.rerankers.base import Reranker


class RAGAgent:
    """
    A prebuilt RAG (Retrieval-Augmented Generation) agent.
    
    This provides a complete document Q&A system out of the box:
    - Load documents from files or text
    - Automatically chunk and embed
    - Semantic search retrieval with optional hybrid/reranking
    - LLM-powered answer generation with citations
    
    Uses the native agenticflow vector store and retriever - no external dependencies.
    
    Example:
        ```python
        from agenticflow.models import ChatModel
        from agenticflow.prebuilt import RAGAgent
        
        rag = RAGAgent(
            model=ChatModel(model="gpt-4o-mini"),
        )
        
        # Load documents
        await rag.load_documents(["report.pdf", "notes.txt"])
        
        # Or load from text directly
        await rag.load_text("This is my document content...", source="manual")
        
        # Query
        answer = await rag.query("What are the key findings?")
        
        # Advanced: hybrid retrieval with reranking
        rag = RAGAgent(
            model=ChatModel(model="gpt-4o-mini"),
            retrieval_mode="hybrid",  # dense + BM25
            reranker="llm",  # LLM-based reranking
        )
        ```
    """
    
    def __init__(
        self,
        model: BaseChatModel,
        embeddings: EmbeddingProvider | None = None,
        *,
        name: str = "RAG_Assistant",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 4,
        instructions: str | None = None,
        backend: str = "inmemory",
        retrieval_mode: Literal["dense", "hybrid", "ensemble"] = "dense",
        sparse_weight: float = 0.3,
        reranker: Literal["none", "llm", "cross_encoder", "cohere"] | Reranker | None = None,
        initial_k: int | None = None,
        # Agent configuration
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
            model: Native chat model for answer generation.
            embeddings: Embedding provider for vectorization.
                If None, uses OpenAI text-embedding-3-small.
            name: Agent name.
            chunk_size: Size of text chunks (default: 1000 chars).
            chunk_overlap: Overlap between chunks (default: 200 chars).
            top_k: Number of passages to retrieve (default: 4).
            instructions: Custom instructions for the agent.
            backend: Vector store backend ("inmemory", "faiss", "chroma", "qdrant", "pgvector").
            retrieval_mode: Retrieval strategy:
                - "dense": Vector similarity only (default)
                - "hybrid": Dense + BM25 sparse retrieval
                - "ensemble": Multiple retrievers with RRF fusion
            sparse_weight: Weight for sparse retrieval in hybrid mode (0-1, default: 0.3).
            reranker: Optional reranker for two-stage retrieval:
                - "none": No reranking (default)
                - "llm": Use the same LLM for reranking
                - "cross_encoder": sentence-transformers cross-encoder (requires: sentence-transformers)
                - "cohere": Cohere Rerank API (requires: cohere, COHERE_API_KEY)
                - Or pass a custom Reranker instance
            initial_k: Documents to retrieve before reranking (default: top_k * 3).
            intercept: Interceptors for execution hooks (gates, guards, prompt adapters).
            stream: Enable streaming responses.
            reasoning: Enable extended thinking mode.
            output: Structured output schema (Pydantic model, dataclass, etc.).
            verbose: Observability level (False, True, "verbose", "debug", "trace").
            resilience: Retry and fallback configuration.
        """
        self._model = model
        self._embeddings = embeddings
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._top_k = top_k
        self._name = name
        self._custom_instructions = instructions
        self._backend_name = backend
        self._retrieval_mode = retrieval_mode
        self._sparse_weight = sparse_weight
        self._reranker_config = reranker
        self._initial_k = initial_k or (top_k * 3)
        
        # Agent configuration
        self._intercept = intercept
        self._stream = stream
        self._reasoning = reasoning
        self._output = output
        self._verbose = verbose
        self._resilience = resilience
        
        # Vector store (lazy initialized) - native type
        self._vector_store: VectorStore | None = None
        self._documents: list[Document] = []
        self._doc_info: dict[str, Any] = {}
        
        # Retriever (lazy initialized)
        self._retriever: Retriever | None = None
        self._reranker: Reranker | None = None
        
        # Agent (lazy initialized after documents loaded)
        self._agent: Agent | None = None
    
    def _get_embeddings(self) -> EmbeddingProvider:
        """Get or create embeddings model (native EmbeddingProvider).
        
        Auto-detects the provider based on available environment variables:
        - OPENAI_API_KEY or AZURE_OPENAI_API_KEY → OpenAIEmbeddings
        - OLLAMA_HOST or local ollama → OllamaEmbeddings
        
        Raises:
            ValueError: If no embeddings provided and can't auto-detect.
        """
        if self._embeddings is not None:
            return self._embeddings
        
        import os
        
        # Try OpenAI first (most common)
        if os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY"):
            from agenticflow.vectorstore import OpenAIEmbeddings
            self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            return self._embeddings
        
        # Try Ollama
        if os.environ.get("OLLAMA_HOST"):
            from agenticflow.vectorstore import OllamaEmbeddings
            self._embeddings = OllamaEmbeddings(model="nomic-embed-text")
            return self._embeddings
        
        # No auto-detection possible - raise helpful error
        raise ValueError(
            "No embeddings provider configured. Either:\n"
            "1. Pass embeddings= parameter explicitly:\n"
            "   RAGAgent(model=..., embeddings=OpenAIEmbeddings())\n"
            "   RAGAgent(model=..., embeddings=OllamaEmbeddings())\n"
            "2. Set environment variable:\n"
            "   OPENAI_API_KEY=sk-...  (for OpenAI)\n"
            "   OLLAMA_HOST=http://localhost:11434  (for Ollama)"
        )
    
    def _create_backend(self):
        """Create backend instance from name."""
        from agenticflow.vectorstore.backends import InMemoryBackend
        
        if self._backend_name == "inmemory":
            return InMemoryBackend()
        elif self._backend_name == "faiss":
            from agenticflow.vectorstore.backends import FAISSBackend
            # FAISS needs dimension - get from embeddings
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
    
    def _create_vector_store(self) -> VectorStore:
        """Create vector store with configured backend."""
        embeddings = self._get_embeddings()
        backend = self._create_backend()
        return VectorStore(embeddings=embeddings, backend=backend)
    
    def _create_retriever(self, documents: list[Document]) -> Retriever:
        """Create retriever based on retrieval_mode."""
        from agenticflow.retriever import DenseRetriever
        
        # Always need the vector store for dense retrieval
        assert self._vector_store is not None
        
        if self._retrieval_mode == "dense":
            return DenseRetriever(self._vector_store)
        
        elif self._retrieval_mode == "hybrid":
            from agenticflow.retriever import HybridRetriever
            return HybridRetriever(
                vectorstore=self._vector_store,
                documents=documents,
                sparse_weight=self._sparse_weight,
            )
        
        elif self._retrieval_mode == "ensemble":
            from agenticflow.retriever import BM25Retriever, DenseRetriever, EnsembleRetriever
            
            dense = DenseRetriever(self._vector_store)
            sparse = BM25Retriever(documents)
            
            return EnsembleRetriever(
                retrievers=[dense, sparse],
                weights=[1.0 - self._sparse_weight, self._sparse_weight],
            )
        
        else:
            raise ValueError(f"Unknown retrieval mode: {self._retrieval_mode}")
    
    def _create_reranker(self) -> Reranker | None:
        """Create reranker based on configuration."""
        if self._reranker_config is None or self._reranker_config == "none":
            return None
        
        # If already a Reranker instance, return it
        from agenticflow.retriever.rerankers.base import Reranker as RerankerProtocol
        if isinstance(self._reranker_config, RerankerProtocol):
            return self._reranker_config
        
        if self._reranker_config == "llm":
            from agenticflow.retriever.rerankers import LLMReranker
            return LLMReranker(model=self._model)
        
        elif self._reranker_config == "cross_encoder":
            from agenticflow.retriever.rerankers import CrossEncoderReranker
            return CrossEncoderReranker()
        
        elif self._reranker_config == "cohere":
            from agenticflow.retriever.rerankers import CohereReranker
            return CohereReranker()
        
        else:
            raise ValueError(f"Unknown reranker: {self._reranker_config}")
    
    async def _retrieve(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Internal retrieval method using configured retriever and reranker."""
        if self._retriever is None:
            # Fallback to direct vector store search
            if self._vector_store is None:
                return []
            results = await self._vector_store.search(query, k=k)
            return [(r.document, r.score) for r in results]
        
        # Use retriever
        if self._reranker:
            # Two-stage retrieval: get more candidates, then rerank
            results = await self._retriever.retrieve_with_scores(query, k=self._initial_k)
            docs = [r.document for r in results]
            reranked = await self._reranker.rerank(query, docs, top_n=k)
            return [(r.document, r.score) for r in reranked]
        else:
            results = await self._retriever.retrieve_with_scores(query, k=k)
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
    
    async def load_documents(
        self,
        paths: list[str | Path],
        *,
        show_progress: bool = True,
    ) -> None:
        """
        Load documents from files.
        
        Supports: .txt, .md, .pdf, .html, .csv, .json
        
        Args:
            paths: List of file paths to load.
            show_progress: Print progress messages.
        """
        all_texts: list[tuple[str, str]] = []  # (text, source)
        sources: list[str] = []
        
        for path in paths:
            path = Path(path)
            if not path.exists():
                if show_progress:
                    print(f"  ⚠ File not found: {path}")
                continue
            
            if show_progress:
                print(f"  📄 Loading: {path.name}")
            
            sources.append(path.name)
            
            # Select loader based on extension
            ext = path.suffix.lower()
            try:
                if ext in (".txt", ".md"):
                    text = path.read_text(encoding="utf-8")
                    all_texts.append((text, path.name))
                elif ext == ".pdf":
                    text = await self._load_pdf(path)
                    if text:
                        all_texts.append((text, path.name))
                    else:
                        if show_progress:
                            print(f"  ⚠ Could not extract text from PDF: {path.name}")
                elif ext in (".html", ".htm"):
                    text = await self._load_html(path)
                    all_texts.append((text, path.name))
                elif ext == ".csv":
                    text = await self._load_csv(path)
                    all_texts.append((text, path.name))
                elif ext == ".json":
                    text = await self._load_json(path)
                    all_texts.append((text, path.name))
                else:
                    # Try as text
                    text = path.read_text(encoding="utf-8")
                    all_texts.append((text, path.name))
            except Exception as e:
                if show_progress:
                    print(f"  ⚠ Error loading {path.name}: {e}")
        
        if not all_texts:
            raise ValueError("No documents were loaded successfully")
        
        # Split documents into chunks
        if show_progress:
            print("  ✂️  Splitting into chunks...")
        
        all_chunks: list[Document] = []
        for text, source in all_texts:
            chunks = split_text(
                text,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
            for chunk in chunks:
                all_chunks.append(Document(text=chunk, metadata={"source": source}))
        
        if show_progress:
            print(f"  ✓ {len(all_chunks)} chunks created")
        
        # Create vector store and add documents
        if show_progress:
            print("  🧮 Creating embeddings...")
        
        self._vector_store = self._create_vector_store()
        await self._vector_store.add_documents(all_chunks)
        self._documents = all_chunks
        
        if show_progress:
            print("  ✓ Vector store ready")
        
        # Create retriever if using advanced retrieval
        if self._retrieval_mode != "dense" or self._reranker_config:
            if show_progress:
                print(f"  🔍 Setting up {self._retrieval_mode} retriever...")
            self._retriever = self._create_retriever(all_chunks)
            self._reranker = self._create_reranker()
            if show_progress:
                if self._reranker:
                    print(f"  ✓ Retriever ready with {self._reranker_config} reranking")
                else:
                    print("  ✓ Retriever ready")
        
        # Update doc info
        self._doc_info = {
            "sources": sources,
            "total_chunks": len(all_chunks),
            "chunk_size": self._chunk_size,
            "overlap": self._chunk_overlap,
            "retrieval_mode": self._retrieval_mode,
            "reranker": str(self._reranker_config) if self._reranker_config else None,
        }
        
        # Create agent with tools
        self._agent = self._create_agent()
    
    async def _load_pdf(self, path: Path) -> str | None:
        """Load PDF file text."""
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return "\n\n".join(text_parts) if text_parts else None
        except ImportError:
            print("  ⚠ PDF support requires: uv add pypdf")
            return None
    
    async def _load_html(self, path: Path) -> str:
        """Load HTML file and extract text."""
        html = path.read_text(encoding="utf-8")
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator="\n", strip=True)
        except ImportError:
            # Fallback: basic regex extraction
            import re
            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()
    
    async def _load_csv(self, path: Path) -> str:
        """Load CSV file as text."""
        import csv
        rows = []
        with path.open(encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(", ".join(row))
        return "\n".join(rows)
    
    async def _load_json(self, path: Path) -> str:
        """Load JSON file as text."""
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(data, indent=2)
    
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
        
        # Split into chunks
        chunks = split_text(
            text,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        documents = [Document(text=chunk, metadata={"source": source}) for chunk in chunks]
        
        if show_progress:
            print(f"  ✓ {len(documents)} chunks created")
            print("  🧮 Creating embeddings...")
        
        self._vector_store = self._create_vector_store()
        await self._vector_store.add_documents(documents)
        self._documents = documents
        
        if show_progress:
            print("  ✓ Vector store ready")
        
        # Create retriever if using advanced retrieval
        if self._retrieval_mode != "dense" or self._reranker_config:
            if show_progress:
                print(f"  🔍 Setting up {self._retrieval_mode} retriever...")
            self._retriever = self._create_retriever(documents)
            self._reranker = self._create_reranker()
            if show_progress:
                if self._reranker:
                    print(f"  ✓ Retriever ready with {self._reranker_config} reranking")
                else:
                    print("  ✓ Retriever ready")
        
        self._doc_info = {
            "sources": [source],
            "total_chunks": len(documents),
            "chunk_size": self._chunk_size,
            "overlap": self._chunk_overlap,
            "retrieval_mode": self._retrieval_mode,
            "reranker": str(self._reranker_config) if self._reranker_config else None,
        }
        
        self._agent = self._create_agent()
    
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
        import httpx
        
        if show_progress:
            print(f"  🌐 Fetching: {url}")
        
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            text = response.text
        
        # Extract text from HTML if needed
        if "text/html" in response.headers.get("content-type", ""):
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(text, "html.parser")
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text(separator="\n", strip=True)
            except ImportError:
                import re
                text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text)
                text = text.strip()
        
        await self.load_text(text, source=url, show_progress=show_progress)
    
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
    embeddings: EmbeddingProvider | None = None,
    *,
    documents: list[str | Path] | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    top_k: int = 4,
    name: str = "RAG_Assistant",
    instructions: str | None = None,
    backend: str = "inmemory",
    retrieval_mode: Literal["dense", "hybrid", "ensemble"] = "dense",
    sparse_weight: float = 0.3,
    reranker: Literal["none", "llm", "cross_encoder", "cohere"] | None = None,
    initial_k: int | None = None,
    # Agent configuration
    intercept: Sequence[Any] | None = None,
    stream: bool = False,
    reasoning: bool | Any = False,
    output: type | dict | None = None,
    verbose: bool | str = False,
    resilience: ResilienceConfig | None = None,
) -> RAGAgent:
    """
    Create a RAG agent for document Q&A.
    
    This is a convenience function that creates and optionally
    initializes a RAGAgent.
    
    Uses the native agenticflow vector store and retriever - no external dependencies.
    
    Args:
        model: Native chat model.
        embeddings: Embedding provider (default: OpenAI text-embedding-3-small).
        documents: Optional list of document paths to load immediately.
        chunk_size: Text chunk size (default: 1000).
        chunk_overlap: Chunk overlap (default: 200).
        top_k: Passages to retrieve (default: 4).
        name: Agent name.
        instructions: Custom instructions.
        backend: Vector store backend ("inmemory", "faiss", "chroma", "qdrant", "pgvector").
        retrieval_mode: Retrieval strategy ("dense", "hybrid", "ensemble").
        sparse_weight: Weight for sparse retrieval in hybrid mode.
        reranker: Reranker type ("none", "llm", "cross_encoder", "cohere").
        initial_k: Documents to retrieve before reranking.
        intercept: Interceptors for execution hooks.
        stream: Enable streaming responses.
        reasoning: Enable extended thinking mode.
        output: Structured output schema.
        verbose: Observability level.
        resilience: Retry and fallback configuration.
        
    Returns:
        Configured RAGAgent instance.
        
    Example:
        ```python
        from agenticflow.models import ChatModel
        from agenticflow.prebuilt import create_rag_agent
        
        # Simple: dense retrieval (default)
        rag = create_rag_agent(
            model=ChatModel(model="gpt-4o-mini"),
            documents=["report.pdf", "data.csv"],
        )
        
        # Advanced: hybrid retrieval with LLM reranking
        rag = create_rag_agent(
            model=ChatModel(model="gpt-4o-mini"),
            retrieval_mode="hybrid",
            reranker="llm",
        )
        await rag.load_documents(["doc1.txt", "doc2.txt"])
        
        # Query
        answer = await rag.query("What are the main findings?")
        ```
    """
    rag = RAGAgent(
        model=model,
        embeddings=embeddings,
        name=name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        instructions=instructions,
        backend=backend,
        retrieval_mode=retrieval_mode,
        sparse_weight=sparse_weight,
        reranker=reranker,
        initial_k=initial_k,
        intercept=intercept,
        stream=stream,
        reasoning=reasoning,
        output=output,
        verbose=verbose,
        resilience=resilience,
    )
    
    # Note: If documents provided, user needs to await load_documents()
    # We can't do async in a regular function
    if documents:
        # Return with a hint that documents need loading
        rag._pending_documents = documents  # type: ignore
    
    return rag
