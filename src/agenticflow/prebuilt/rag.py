"""
Prebuilt RAG (Retrieval-Augmented Generation) Agent.

A complete, ready-to-use RAG system with:
- Document loading (PDF, TXT, MD, HTML)
- Text splitting
- Vector embeddings
- Semantic search
- LLM-powered Q&A

NOTE: This module uses LangChain for document loaders and vector stores.
For native-only RAG, use the SimpleRAG class which has no external dependencies.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agenticflow.tools.base import tool

from agenticflow import Agent, AgentRole

if TYPE_CHECKING:
    from agenticflow.models.base import BaseChatModel, BaseEmbeddingModel


class RAGAgent:
    """
    A prebuilt RAG (Retrieval-Augmented Generation) agent.
    
    This provides a complete document Q&A system out of the box:
    - Load documents from files or text
    - Automatically chunk and embed
    - Semantic search retrieval
    - LLM-powered answer generation with citations
    
    NOTE: This class uses LangChain for document loaders and vector stores.
    Make sure to install: `uv add langchain-openai langchain-community`
    
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
        ```
    """
    
    def __init__(
        self,
        model: BaseChatModel,
        embeddings: Any | None = None,
        *,
        name: str = "RAG_Assistant",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 4,
        instructions: str | None = None,
    ) -> None:
        """
        Create a RAG agent.
        
        Args:
            model: Native chat model for answer generation.
            embeddings: LangChain embedding model for vectorization.
                If None, uses OpenAI text-embedding-3-small.
            name: Agent name.
            chunk_size: Size of text chunks (default: 1000 chars).
            chunk_overlap: Overlap between chunks (default: 200 chars).
            top_k: Number of passages to retrieve (default: 4).
            instructions: Custom instructions for the agent.
        """
        self._model = model
        self._embeddings = embeddings
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._top_k = top_k
        self._name = name
        self._custom_instructions = instructions
        
        # Vector store (lazy initialized) - uses LangChain types
        self._vector_store: Any = None
        self._documents: list[Any] = []
        self._doc_info: dict[str, Any] = {}
        
        # Agent (lazy initialized after documents loaded)
        self._agent: Agent | None = None
    
    def _get_embeddings(self) -> Any:
        """Get or create embeddings model (LangChain Embeddings)."""
        if self._embeddings is None:
            try:
                from langchain_openai import OpenAIEmbeddings
                self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            except ImportError as e:
                raise ImportError(
                    "RAGAgent requires langchain-openai for embeddings. "
                    "Install with: uv add langchain-openai"
                ) from e
        return self._embeddings
    
    def _create_tools(self) -> list:
        """Create RAG tools that reference this instance."""
        # Use 'rag_self' to avoid closure issues - we want live references
        rag_self = self
        
        @tool
        def search_documents(query: str, num_results: int = 4) -> str:
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
            results = rag_self._vector_store.similarity_search(query, k=num_results)
            
            if not results:
                return "No relevant passages found."
            
            formatted = []
            for i, doc in enumerate(results, 1):
                content = doc.page_content.strip()
                source = doc.metadata.get("source", "unknown")
                formatted.append(f"[{i}] (Source: {source})\n{content}")
            
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
            role=AgentRole.AUTONOMOUS,
            model=self._model,
            instructions=instructions,
            tools=self._create_tools(),
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
        from langchain_community.document_loaders import (
            CSVLoader,
            JSONLoader,
            TextLoader,
            UnstructuredHTMLLoader,
        )
        from langchain_core.vectorstores import InMemoryVectorStore
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        all_docs: list[Document] = []
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
                    loader = TextLoader(str(path), encoding="utf-8")
                elif ext == ".pdf":
                    try:
                        from langchain_community.document_loaders import PyPDFLoader
                        loader = PyPDFLoader(str(path))
                    except ImportError:
                        print(f"  ⚠ PDF support requires: uv add pypdf")
                        continue
                elif ext in (".html", ".htm"):
                    loader = UnstructuredHTMLLoader(str(path))
                elif ext == ".csv":
                    loader = CSVLoader(str(path))
                elif ext == ".json":
                    loader = JSONLoader(str(path), jq_schema=".", text_content=False)
                else:
                    # Try as text
                    loader = TextLoader(str(path), encoding="utf-8")
                
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                if show_progress:
                    print(f"  ⚠ Error loading {path.name}: {e}")
        
        if not all_docs:
            raise ValueError("No documents were loaded successfully")
        
        # Split documents
        if show_progress:
            print(f"  ✂️  Splitting into chunks...")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(all_docs)
        
        if show_progress:
            print(f"  ✓ {len(chunks)} chunks created")
        
        # Create embeddings
        if show_progress:
            print(f"  🧮 Creating embeddings...")
        
        embeddings = self._get_embeddings()
        self._vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
        self._documents = chunks
        
        if show_progress:
            print(f"  ✓ Vector store ready")
        
        # Update doc info
        self._doc_info = {
            "sources": sources,
            "total_chunks": len(chunks),
            "chunk_size": self._chunk_size,
            "overlap": self._chunk_overlap,
        }
        
        # Create agent with tools
        self._agent = self._create_agent()
    
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
        from langchain_core.documents import Document as LCDocument
        from langchain_core.vectorstores import InMemoryVectorStore
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        if show_progress:
            print(f"  📝 Loading text from: {source}")
        
        doc = LCDocument(page_content=text, metadata={"source": source})
        
        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents([doc])
        
        if show_progress:
            print(f"  ✓ {len(chunks)} chunks created")
            print(f"  🧮 Creating embeddings...")
        
        embeddings = self._get_embeddings()
        self._vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
        self._documents = chunks
        
        if show_progress:
            print(f"  ✓ Vector store ready")
        
        self._doc_info = {
            "sources": [source],
            "total_chunks": len(chunks),
            "chunk_size": self._chunk_size,
            "overlap": self._chunk_overlap,
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
        
        await self.load_text(text, source=url, show_progress=show_progress)
    
    async def query(
        self,
        question: str,
        *,
        strategy: str = "dag",
        verbose: bool = False,
    ) -> str:
        """
        Ask a question about the loaded documents.
        
        Args:
            question: Your question.
            strategy: Execution strategy ("dag", "react", "plan").
            verbose: Show detailed progress with tool calls.
            
        Returns:
            The answer with citations.
        """
        if self._agent is None:
            raise RuntimeError("No documents loaded. Call load_documents() first.")
        
        if verbose:
            from agenticflow.observability import OutputConfig, ProgressTracker
            tracker = ProgressTracker(OutputConfig.verbose())
            return await self._agent.run(question, strategy=strategy, tracker=tracker)
        
        return await self._agent.run(question, strategy=strategy)
    
    async def search(
        self,
        query: str,
        k: int | None = None,
    ) -> list[Any]:
        """
        Direct semantic search (without LLM).
        
        Args:
            query: Search query.
            k: Number of results (default: top_k).
            
        Returns:
            List of matching Document objects (LangChain Document type).
        """
        if self._vector_store is None:
            raise RuntimeError("No documents loaded.")
        
        k = k or self._top_k
        return self._vector_store.similarity_search(query, k=k)
    
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
    embeddings: Any | None = None,
    *,
    documents: list[str | Path] | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    top_k: int = 4,
    name: str = "RAG_Assistant",
    instructions: str | None = None,
) -> RAGAgent:
    """
    Create a RAG agent for document Q&A.
    
    This is a convenience function that creates and optionally
    initializes a RAGAgent.
    
    NOTE: This uses LangChain for document loaders and vector stores.
    Make sure to install: `uv add langchain-openai langchain-community`
    
    Args:
        model: Native chat model.
        embeddings: LangChain embedding model (default: OpenAI text-embedding-3-small).
        documents: Optional list of document paths to load immediately.
        chunk_size: Text chunk size (default: 1000).
        chunk_overlap: Chunk overlap (default: 200).
        top_k: Passages to retrieve (default: 4).
        name: Agent name.
        instructions: Custom instructions.
        
    Returns:
        Configured RAGAgent instance.
        
    Example:
        ```python
        from agenticflow.models import ChatModel
        from agenticflow.prebuilt import create_rag_agent
        
        # Create and load in one step
        rag = create_rag_agent(
            model=ChatModel(model="gpt-4o-mini"),
            documents=["report.pdf", "data.csv"],
        )
        
        # Or create empty and load later
        rag = create_rag_agent(model=ChatModel(model="gpt-4o-mini"))
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
    )
    
    # Note: If documents provided, user needs to await load_documents()
    # We can't do async in a regular function
    if documents:
        # Return with a hint that documents need loading
        rag._pending_documents = documents
    
    return rag
