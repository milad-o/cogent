"""LLM-Powered Indexes - Advanced indexing with LLMs.

Demonstrates specialized indexes that use LLMs for enhanced retrieval:
1. SummaryIndex - Query-time summarization
2. TreeIndex - Hierarchical summary tree
3. KeywordTableIndex - LLM-extracted keyword index
4. KnowledgeGraphIndex - Graph-based retrieval
5. HierarchicalIndex - Multi-level structured retrieval
6. MultiRepresentationIndex - Multiple embeddings per doc

These indexes are ideal for large documents and complex queries.

Usage:
    uv run python examples/retrieval/indexes.py
"""

import asyncio

from cogent import create_chat
from cogent.retriever import (
    HierarchicalIndex,
    KeywordTableIndex,
    KnowledgeGraphIndex,
    MultiRepresentationIndex,
    QueryType,
    RepresentationType,
    SummaryIndex,
    TreeIndex,
    Document,
)

from _shared import (
    KNOWLEDGE_BASE,
    LONG_DOCUMENT,
    create_vectorstore,
    load_company_knowledge,
    print_header,
)


async def demo_summary_index() -> None:
    """Query-time summarization of all documents."""
    print_header("1. Summary Index (LLM Summaries)")
    
    docs = [
        Document(text=f"Section {i}: " + KNOWLEDGE_BASE[i * 200 : (i + 1) * 200])
        for i in range(5)
    ]
    
    index = SummaryIndex(
        llm=create_chat("gpt-4o-mini"),
        vectorstore=create_vectorstore(),
    )
    await index.add_documents(docs)
    
    query = "How does water return to the ocean?"
    print(f"\nQuery: '{query}'")
    print("(LLM generates summary from all relevant docs)")
    
    results = await index.retrieve(query, k=2)
    for r in results:
        print(f"\n  Summary: {r.text[:150]}...")


async def demo_tree_index() -> None:
    """Hierarchical tree of summaries for large documents."""
    print_header("2. Tree Index (Hierarchical Summaries)")
    
    # Simulate large document with multiple sections
    sections = [
        f"Part {i}: " + KNOWLEDGE_BASE[i * 150 : (i + 1) * 150] for i in range(6)
    ]
    docs = [Document(text=section) for section in sections]
    
    tree = TreeIndex(
        llm=create_chat("gpt-4o-mini"),
        chunk_size=200,
        max_children=3,
    )
    await tree.add_documents(docs)
    
    query = "Explain the complete water cycle process"
    print(f"\nQuery: '{query}'")
    print("(Builds summary tree, traverses from root to leaves)")
    
    results = await tree.retrieve(query, k=2)
    for r in results:
        print(f"\n  Leaf summary: {r.text[:120]}...")


async def demo_keyword_table_index() -> None:
    """LLM-extracted keyword inverted index."""
    print_header("3. Keyword Table Index (LLM Keywords)")
    
    docs = [
        Document(
            text="Black holes warp spacetime so severely that nothing can escape their event horizon.",
            metadata={"source": "black_holes.md"},
        ),
        Document(
            text="Quantum entanglement allows particles to exhibit correlated behavior across vast distances.",
            metadata={"source": "quantum_physics.md"},
        ),
        Document(
            text="Photosynthesis converts solar energy into chemical energy using chlorophyll molecules.",
            metadata={"source": "photosynthesis.md"},
        ),
    ]
    
    index = KeywordTableIndex(
        llm=create_chat("gpt-4o-mini"),
    )
    await index.add_documents(docs)
    
    query = "quantum particle behavior and correlation"
    print(f"\nQuery: '{query}'")
    print("(LLM extracts keywords from query, matches to doc keywords)")
    
    results = await index.retrieve(query, k=2, include_scores=True)
    for r in results:
        print(f"  [{r.score:.3f}] {r.document.metadata['source']}: {r.document.text}")


async def demo_knowledge_graph_index() -> None:
    """Extract and query knowledge graph from documents."""
    print_header("4. Knowledge Graph Index")
    
    docs = [
        Document(text="The Amazon rainforest is located in South America. It produces 20% of Earth's oxygen."),
        Document(text="The Great Barrier Reef is in Australia. It contains over 400 species of coral."),
        Document(text="Mount Everest is in the Himalayas. It is the tallest mountain on Earth."),
    ]
    
    index = KnowledgeGraphIndex(
        llm=create_chat("gpt-4o-mini"),
    )
    await index.add_documents(docs)
    
    query = "What natural ecosystems are found in different regions?"
    print(f"\nQuery: '{query}'")
    print("(Extracts entities/relations, queries graph)")
    
    results = await index.retrieve(query, k=3)
    for r in results:
        print(f"  {r.text}")


async def demo_hierarchical_index() -> None:
    """Structured document retrieval respecting hierarchy."""
    print_header("6. Hierarchical Index (Structured Docs)")
    
    # Markdown document with headers
    md_doc = Document(text="""# Astronomy Guide

## Stars and Galaxies
Stars form from collapsing gas clouds. Galaxies contain billions of stars.

## Planetary Science
Planets orbit stars. Rocky planets form close, gas giants form farther out.

### Planet Formation
Dust particles collide and stick together, gradually forming planetesimals.

### Orbital Mechanics
Gravity keeps planets in elliptical orbits following Kepler's laws.

## Black Holes
Massive stars collapse into black holes. Nothing escapes their gravity.
""", metadata={"source": "astronomy_guide.md"})
    
    vs = create_vectorstore()
    index = HierarchicalIndex(
        vectorstore=vs,
        llm=create_chat("gpt-4o-mini"),
        structure_type="markdown",
        chunk_size=200,
    )
    
    await index.add_documents([md_doc])
    
    query = "how do planets form and orbit"
    print(f"\nQuery: '{query}'")
    print("(Finds relevant section, then retrieves chunks)")
    
    results = await index.retrieve(query, k=2, include_scores=True)
    for r in results:
        section = r.metadata.get("section_title", "Unknown")
        print(f"\n  [{r.score:.3f}] Section: {section}")
        print(f"  Content: {r.document.text[:100]}...")


async def demo_multi_representation_index() -> None:
    """Multiple embeddings per document for better retrieval."""
    print_header("6. Multi-Representation Index")
    
    vs = create_vectorstore()
    
    index = MultiRepresentationIndex(
        vectorstore=vs,
        llm=create_chat("gpt-4o-mini"),
        representations=[
            RepresentationType.ORIGINAL,  # Original text
            RepresentationType.SUMMARY,  # LLM summary
            RepresentationType.KEYWORDS,  # Extracted keywords
        ],
    )
    
    docs = [
        Document(text=KNOWLEDGE_BASE),
        Document(text=LONG_DOCUMENT.text),
    ]
    await index.add_documents(docs)
    
    query = "coral reef biodiversity and species"
    print(f"\nQuery: '{query}'")
    print("(Searches across original, summary, and keyword embeddings)")
    
    results = await index.retrieve(query, k=2, query_type=QueryType.AUTO, include_scores=True)
    for r in results:
        print(f"\n  Matched: {r.document.text[:100]}...")
        print(f"  Metadata: {r.document.metadata}")


async def main() -> None:
    """Run all index examples."""
    await demo_summary_index()
    await demo_tree_index()
    await demo_keyword_table_index()
    await demo_knowledge_graph_index()
    await demo_multi_representation_index()
    
    print("\n" + "=" * 60)
    print("✓ All index examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
