#!/usr/bin/env python3
"""
Example 10: Agentic RAG with Nature-themed Document
====================================================

A complete Agentic RAG system with a single AUTONOMOUS agent that has
tools to search and retrieve from a vector store.

This demonstrates:
1. Document loading (TextLoader)
2. Text splitting (RecursiveCharacterTextSplitter)
3. Embeddings (OpenAI text-embedding-3-small)
4. Vector store (InMemoryVectorStore)
5. Single autonomous agent with RAG tools
6. Agent-driven retrieval and answer generation

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
import os
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from agenticflow.tools.base import tool

# Load .env file for API keys
load_dotenv()

from agenticflow import Agent, AgentRole
from agenticflow.visualization import AgentDiagram, MermaidConfig

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.vectorstores import VectorStore


# =============================================================================
# Configuration
# =============================================================================

NATURE_TEXT_URL = "https://www.gutenberg.org/cache/epub/113/pg113.txt"
NATURE_TEXT_NAME = "the_secret_garden.txt"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4


# =============================================================================
# Global vector store (accessible by tools)
# =============================================================================

_vector_store: VectorStore | None = None
_document_info: dict = {}


# =============================================================================
# RAG Tools for the Agent
# =============================================================================


@tool
def search_documents(query: str, num_results: int = 4) -> str:
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
    results = _vector_store.similarity_search(query, k=num_results)

    if not results:
        return "No relevant passages found for your query."

    formatted = []
    for i, doc in enumerate(results, 1):
        content = doc.page_content.strip()
        formatted.append(f"[Passage {i}]\n{content}")

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
# Document Loading & Processing
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


def load_and_split_document(file_path: Path) -> list[Document]:
    """Load document and split into chunks."""
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print(f"\n📄 Loading: {file_path.name}")

    loader = TextLoader(str(file_path), encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  ✓ Split into {len(chunks)} chunks")

    return chunks


def create_vector_store(chunks: list[Document]) -> VectorStore:
    """Create vector store with OpenAI embeddings."""
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.vectorstores import InMemoryVectorStore

    print("\n🧮 Creating embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
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
        # {tools} placeholder is auto-replaced with tool descriptions!
        # Or if you omit it, tools are appended automatically.
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

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY not found in environment")
        print("   Please set it in .env file or export it")
        return

    # Step 1: Download and process document
    print("\n📥 Step 1: Loading document...")
    text_path = await download_document(NATURE_TEXT_URL, NATURE_TEXT_NAME)
    chunks = load_and_split_document(text_path)

    # Step 2: Create vector store
    _vector_store = create_vector_store(chunks)
    _document_info = {
        "name": text_path.name,
        "chunks": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "overlap": CHUNK_OVERLAP,
    }

    # Step 3: Create agent with LLM
    print("\n🤖 Step 3: Creating RAG agent...")
    from agenticflow.models import ChatModel

    model = ChatModel(model="gpt-4o-mini", temperature=0)
    agent = create_rag_agent(model)
    print(f"  ✓ Agent: {agent.name} ({agent.role.value})")
    print(f"  ✓ Tools: {', '.join(t.name for t in agent._direct_tools)}")
    print(f"  ✓ Model: gpt-4o-mini")

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
    print(f"  • Model: gpt-4o-mini")

    print(f"\n📝 Mermaid Diagram:")
    print(mermaid)


if __name__ == "__main__":
    asyncio.run(main())
