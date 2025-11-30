#!/usr/bin/env python3
"""
Example 10: Agentic RAG with Nature-themed Document
====================================================

A complete Agentic RAG system with a single AUTONOMOUS agent that has
tools to search and retrieve from a vector store.

This demonstrates:
1. Document loading and text splitting (built-in)
2. Embeddings (OpenAI via our EmbeddingModel)
3. Vector store (our InMemoryVectorStore)
4. Single autonomous agent with RAG tools
5. Agent-driven retrieval and answer generation

The agent has access to:
- search_documents: Semantic search over document chunks
- get_document_info: Get metadata about the loaded document

Text Source: "The Secret Garden" by Frances Hodgson Burnett (Public Domain)

Usage:
    export OPENAI_API_KEY="your-key"
    python examples/10_agentic_rag.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from config import get_model, settings

from agenticflow import Agent, AgentRole
from agenticflow.tools.base import tool
from agenticflow.visualization import AgentDiagram, MermaidConfig


# =============================================================================
# Configuration
# =============================================================================

NATURE_TEXT_URL = "https://www.gutenberg.org/cache/epub/113/pg113.txt"
NATURE_TEXT_NAME = "the_secret_garden.txt"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4


# =============================================================================
# Simple Document class (no langchain dependency)
# =============================================================================

@dataclass
class Document:
    """Simple document chunk with content and metadata."""
    page_content: str
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# =============================================================================
# Global vector store (accessible by tools)
# =============================================================================

_vector_store = None
_document_info: dict = {}


# =============================================================================
# RAG Tools for the Agent
# =============================================================================


@tool
async def search_documents(query: str, num_results: int = 4) -> str:
    """Search the document for relevant passages.

    Args:
        query: The search query to find relevant passages.
        num_results: Number of passages to return (default: 4).

    Returns:
        Formatted string with relevant passages from the document.
    """
    if _vector_store is None:
        return "Error: No document loaded. Please load a document first."

    num_results = min(num_results, 10)  # Cap at 10
    results = await _vector_store.search(query, k=num_results)

    if not results:
        return "No relevant passages found for your query."

    formatted = []
    for i, result in enumerate(results, 1):
        content = result.document.text.strip()
        score = result.score
        formatted.append(f"[Passage {i}] (relevance: {score:.2f})\n{content}")

    return "\n\n---\n\n".join(formatted)


@tool
def get_document_info() -> str:
    """Get information about the loaded document.

    Returns:
        String with document metadata (name, chunks, size).
    """
    if not _document_info:
        return "No document loaded."

    return f"""Document Information:
- Name: {_document_info.get('name', 'Unknown')}
- Total chunks: {_document_info.get('chunks', 0)}
- Chunk size: {_document_info.get('chunk_size', 0)} characters
- Chunk overlap: {_document_info.get('overlap', 0)} characters"""


# =============================================================================
# Document Loading & Processing (no langchain!)
# =============================================================================


async def download_document(url: str, filename: str) -> Path:
    """Download a text document from URL."""
    import httpx

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    text_path = data_dir / filename

    if text_path.exists():
        print(f"  ✓ Document cached: {text_path.name}")
        return text_path

    print(f"  ⬇ Downloading from {url}...")
    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        text_path.write_bytes(response.content)

    size_kb = text_path.stat().st_size / 1024
    print(f"  ✓ Downloaded: {text_path.name} ({size_kb:.1f} KB)")
    return text_path


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: list[str] | None = None,
) -> list[str]:
    """Split text into chunks with overlap.
    
    Simple recursive character text splitter implementation.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]
    
    chunks = []
    
    def _split_recursive(text: str, seps: list[str]) -> list[str]:
        if not text:
            return []
        
        # If text is small enough, return it
        if len(text) <= chunk_size:
            return [text]
        
        # Try each separator
        for i, sep in enumerate(seps):
            if sep and sep in text:
                parts = text.split(sep)
                result = []
                current_chunk = ""
                
                for part in parts:
                    test_chunk = current_chunk + (sep if current_chunk else "") + part
                    
                    if len(test_chunk) <= chunk_size:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            result.append(current_chunk)
                        # If part itself is too big, split with next separator
                        if len(part) > chunk_size and i + 1 < len(seps):
                            result.extend(_split_recursive(part, seps[i + 1:]))
                            current_chunk = ""
                        else:
                            current_chunk = part
                
                if current_chunk:
                    result.append(current_chunk)
                
                return result
        
        # No separator worked, just split by size
        result = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            result.append(text[i:i + chunk_size])
        return result
    
    raw_chunks = _split_recursive(text, separators)
    
    # Add overlap by including end of previous chunk
    for i, chunk in enumerate(raw_chunks):
        if i > 0 and chunk_overlap > 0:
            # Add overlap from previous chunk
            prev_chunk = raw_chunks[i - 1]
            overlap_text = prev_chunk[-chunk_overlap:] if len(prev_chunk) > chunk_overlap else prev_chunk
            # Only add if it doesn't make chunk too big
            if len(overlap_text) + len(chunk) <= chunk_size * 1.5:
                chunks.append(overlap_text + chunk)
            else:
                chunks.append(chunk)
        else:
            chunks.append(chunk)
    
    return chunks


def load_and_split_document(file_path: Path) -> list[Document]:
    """Load document and split into chunks."""
    print(f"\n📄 Loading: {file_path.name}")

    text = file_path.read_text(encoding="utf-8")
    
    # Split into chunks
    chunk_texts = split_text(
        text,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    
    # Convert to Document objects
    chunks = [
        Document(
            page_content=chunk,
            metadata={"source": file_path.name, "chunk_index": i}
        )
        for i, chunk in enumerate(chunk_texts)
    ]
    
    print(f"  ✓ Split into {len(chunks)} chunks")
    return chunks


async def create_vector_store(chunks: list[Document]):
    """Create vector store with OpenAI embeddings."""
    from agenticflow.vectorstore import VectorStore, OpenAIEmbeddings, backends

    print("\n🧮 Creating embeddings...")
    
    # Create embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.openai_api_key,
    )
    
    # Create vector store with in-memory backend
    vector_store = VectorStore(
        backend=backends.InMemoryBackend(),
        embeddings=embeddings,
    )
    
    # Add documents
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    
    await vector_store.add_texts(texts, metadatas=metadatas)
    
    print(f"  ✓ Vector store ready ({len(chunks)} embeddings)")
    return vector_store


# =============================================================================
# Agent Setup
# =============================================================================


def create_rag_agent(model) -> Agent:
    """Create a single autonomous RAG agent with tools."""

    agent = Agent(
        name="RAG_Assistant",
        role=AgentRole.AUTONOMOUS,
        model=model,
        instructions="""You are a helpful assistant that answers questions about "The Secret Garden" 
by Frances Hodgson Burnett.

You have access to the full text of the book through your search tool. When answering questions:

1. ALWAYS use the search_documents tool to find relevant passages
2. Base your answers ONLY on the retrieved passages
3. Cite passage numbers [1], [2], etc. to support your claims
4. If the passages don't contain the answer, say so honestly
5. Be concise but thorough

{tools}""",
        tools=[search_documents, get_document_info],
    )

    return agent


def visualize_agent(agent: Agent) -> str:
    """Generate Mermaid diagram for the agent."""
    config = MermaidConfig(
        show_tools=True,
        show_roles=True,
        title="Agentic RAG Assistant",
    )
    diagram = AgentDiagram(agent, config=config)
    return diagram.to_mermaid()


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Run the Agentic RAG example with a single agent."""
    global _vector_store, _document_info

    print("=" * 60)
    print("🌿 Agentic RAG - The Secret Garden")
    print("=" * 60)

    # Check for OpenAI API key (needed for embeddings)
    if not settings.openai_api_key:
        print("\n❌ Error: OPENAI_API_KEY not found in environment")
        print("   OpenAI embeddings are required for RAG.")
        print("   Please set it in .env file or export it")
        return

    # Step 1: Download and process document
    print("\n📥 Step 1: Loading document...")
    text_path = await download_document(NATURE_TEXT_URL, NATURE_TEXT_NAME)
    chunks = load_and_split_document(text_path)

    # Step 2: Create vector store
    _vector_store = await create_vector_store(chunks)
    _document_info = {
        "name": text_path.name,
        "chunks": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "overlap": CHUNK_OVERLAP,
    }

    # Step 3: Create agent with LLM
    print("\n🤖 Step 3: Creating RAG agent...")

    model = get_model()
    agent = create_rag_agent(model)
    print(f"  ✓ Agent: {agent.name} ({agent.role.value})")
    print(f"  ✓ Tools: {', '.join(t.name for t in agent._direct_tools)}")
    print(f"  ✓ Model: {settings.get_preferred_provider()}")

    # Step 4: Generate visualization
    print("\n📊 Step 4: Generating diagram...")
    mermaid = visualize_agent(agent)
    diagram_path = Path(__file__).parent / "diagrams" / "7_agentic_rag.mmd"
    diagram_path.parent.mkdir(exist_ok=True)
    diagram_path.write_text(mermaid)
    print(f"  ✓ Saved to: {diagram_path}")

    # Step 5: Run interactive RAG queries
    print("\n" + "=" * 60)
    print("🔍 Step 5: Running RAG queries with agent...")
    print("=" * 60)

    sample_questions = [
        "What does Mary discover when she first enters the secret garden?",
        "How does working in the garden affect Colin's health?",
        "Who is Dickon and what is his relationship with animals?",
    ]

    for question in sample_questions:
        print(f"\n{'='*60}")
        print(f"❓ Question: {question}")
        print("=" * 60)

        # Use agent.run() with DAG strategy for tool execution
        response = await agent.run(
            f"Answer this question about The Secret Garden: {question}",
            strategy="dag",
        )

        print(f"\n💬 Agent Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)

    # Summary
    print("\n" + "=" * 60)
    print("✅ Agentic RAG Complete!")
    print("=" * 60)

    print(f"\n📊 Summary:")
    print(f"  • Document: {text_path.name}")
    print(f"  • Chunks: {len(chunks)}")
    print(f"  • Agent: {agent.name} (autonomous)")
    print(f"  • Tools: search_documents, get_document_info")
    print(f"  • Model: {settings.get_preferred_provider()}")

    print(f"\n📝 Mermaid Diagram:")
    print(mermaid)


if __name__ == "__main__":
    asyncio.run(main())
