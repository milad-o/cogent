"""
Example 10: RAG (Retrieval-Augmented Generation)

Two usage patterns:
- **Managed mode**: RAG loads documents, creates vectorstore
- **Pre-configured mode**: You provide a ready retriever (no load needed)

Three API levels:
1. **High-level**: agent.run() with RAG capability (agentic RAG)
2. **Mid-level**: rag.search() with citations
3. **Low-level**: vectorstore.search() raw results

Usage:
    uv run python examples/10_rag.py
"""

import asyncio
from pathlib import Path

from config import get_embeddings, get_model

from agenticflow import Agent
from agenticflow.capabilities import (
    RAG,
    RAGConfig,
    CitationStyle,
    DocumentPipeline,
    PipelineRegistry,
)
from agenticflow.vectorstore import VectorStore, Document
from agenticflow.retriever import DenseRetriever, BM25Retriever, HybridRetriever


# Sample text for demo (The Secret Garden excerpt)
SAMPLE_TEXT = """
The Secret Garden by Frances Hodgson Burnett

Chapter 1: There Is No One Left

When Mary Lennox was sent to Misselthwaite Manor to live with her uncle, 
everybody said she was the most disagreeable-looking child ever seen. 
It was true, too. She had a little thin face and a little thin body, 
thin light hair and a sour expression. Her hair was yellow, and her 
face was yellow because she had been born in India and had always been ill.

Chapter 3: Across the Moor

The moor was a vast stretch of wild land, covered with brown heather 
and gorse bushes. It stretched for miles in every direction. The sky 
seemed so high above, and the air was so fresh and pure.

"It's the moor," said Martha. "It's called the moor. Does tha' like it?"

Mary looked at it and thought she did not like it at all.

Chapter 4: Martha

Martha was a good-natured Yorkshire girl who had been hired to wait on Mary.
She was different from any servant Mary had ever known. She talked and 
laughed and seemed not to know that a servant should be silent.

"Th' fresh air an' th' skippin' rope will make thee strong," Martha said.
"Mother says there's naught like th' moor air."

Chapter 8: The Robin and the Key

One day, Mary was walking along the path by the wall when she heard a 
chirping sound. A robin was sitting on a branch, looking at her with 
his bright eyes. He seemed to be trying to tell her something.
"""


async def main() -> None:
    model = get_model()
    embeddings = get_embeddings()

    # =========================================================================
    # Pattern 1: Managed Mode (RAG loads documents)
    # =========================================================================
    print("=" * 60)
    print("Pattern 1: Managed Mode (RAG loads documents)")
    print("=" * 60)

    # Create RAG - it will manage its own vectorstore
    rag = RAG(embeddings=embeddings)

    # Add to agent - agent gets search_documents tool automatically
    agent = Agent(
        name="BookAssistant",
        model=model,
        capabilities=[rag],
    )

    # Load documents - required in managed mode
    # Can load files, directories, URLs, or Document objects
    await rag.load(Document(text=SAMPLE_TEXT, metadata={"source": "the_secret_garden.txt"}))
    print(f"Loaded {rag.document_count} chunks")

    # Agent uses tools autonomously to search and answer
    answer = await agent.run("What was Mary Lennox like when she arrived?")
    print(f"\n{answer}")

    # =========================================================================
    # Pattern 2: Pre-configured Mode (Bring your own retriever)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Pattern 2: Pre-configured Mode (no load needed)")
    print("=" * 60)

    # 1. Prepare documents
    docs = [
        Document(text="Python is great for data science.", metadata={"source": "python.txt"}),
        Document(text="JavaScript runs in browsers.", metadata={"source": "js.txt"}),
        Document(text="Rust provides memory safety.", metadata={"source": "rust.txt"}),
    ]

    # 2. Create and populate vectorstore
    store = VectorStore(embeddings=embeddings)
    await store.add_documents(docs)

    # 3. Create retriever
    dense = DenseRetriever(store)
    sparse = BM25Retriever(docs)
    retriever = HybridRetriever(dense=dense, sparse=sparse)

    # 4. Pass to RAG - ready immediately, no load() needed!
    rag2 = RAG(embeddings=embeddings, retriever=retriever)
    print(f"RAG ready: {rag2.is_ready}")  # True

    # Search works right away
    passages = await rag2.search("memory safety", k=2)
    for p in passages:
        print(f"  {p.format_reference()} {p.source}: {p.text[:50]}...")

    # =========================================================================
    # Mid-Level API: Citation-Aware Search
    # =========================================================================
    print("\n" + "=" * 60)
    print("Mid-Level API: Citation-Aware Search (rag.search)")
    print("=" * 60)

    # Get CitedPassage objects with source, score, citation_id
    passages = await rag.search("Describe the moor and Martha.", k=3)

    print("Retrieved passages:")
    for p in passages:
        print(f"  {p.format_reference()} {p.source} (score: {p.score:.2f})")
        print(f"    {p.text[:80]}...")

    # Demonstrate different citation styles
    print("\nCitation style examples:")
    for style in CitationStyle:
        formatted = passages[0].format_reference(style)
        print(f"  {style.name:12} → {formatted}")

    # Use helper methods for formatted output
    print("\nFormatted context (NUMERIC style):")
    print(rag.format_context(passages[:2], style=CitationStyle.NUMERIC))

    print("\nBibliography:")
    print(rag.format_bibliography(passages))

    # Build your own prompt with citations
    context = "\n\n".join(f"[{p.citation_id}] {p.text}" for p in passages)
    answer = await agent.run(
        f"Based on these passages, describe the moor and Martha:\n\n{context}"
    )
    print(f"\nAnswer with citations:\n{answer}")

    # =========================================================================
    # Low-Level API: Raw Vectorstore Search
    # =========================================================================
    print("\n" + "=" * 60)
    print("Low-Level API: Raw Vectorstore Search")
    print("=" * 60)

    # Direct vectorstore access - returns SearchResult objects
    results = await rag.vectorstore.search("robin bird key", k=2)

    for i, result in enumerate(results, 1):
        doc = result.document
        source = doc.metadata.get("source", "unknown")
        print(f"[{i}] {source} (score: {result.score:.2f})")
        print(f"    {doc.text[:100].strip()}...")

    # =========================================================================
    # Bonus: Custom Pipelines (per file type)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Bonus: Custom Pipelines")
    print("=" * 60)

    from agenticflow.document import MarkdownSplitter, CodeSplitter

    # Create custom pipelines for different file types
    pipelines = PipelineRegistry()
    pipelines.register(
        ".md",
        DocumentPipeline(
            splitter=MarkdownSplitter(chunk_size=500),
            metadata={"type": "documentation"},
        ),
    )
    pipelines.register(
        ".py",
        DocumentPipeline(
            splitter=CodeSplitter(language="python", chunk_size=1000),
            metadata={"type": "code"},
        ),
    )

    # Create RAG with custom pipelines
    rag2 = RAG(
        embeddings=embeddings,
        pipelines=pipelines,
        config=RAGConfig(chunk_size=500, top_k=3),
    )

    # Agent with custom pipelines
    agent2 = Agent(
        name="SmartHomeHelper",
        model=model,
        capabilities=[rag2],
    )

    # Load mixed content
    data_dir = Path(__file__).parent / "data"
    if (data_dir / "smarthome_docs.md").exists():
        await rag2.load(
            data_dir / "smarthome_docs.md",
            data_dir / "smarthome_devices.py",
        )
        print(f"Loaded {rag2.document_count} chunks with custom pipelines")

        # Use citation-aware search
        passages = await rag2.search("How do I control smart lights?", k=3)
        for p in passages:
            print(f"  {p.format_reference()} {p.source}")

        # Then use agent for full answer
        answer = await agent2.run("How do I control smart lights?")
        print(f"\nAgent answer:\n{answer}")
    else:
        print("(Skipped - data files not found)")

    print("\n✓ Done")


if __name__ == "__main__":
    asyncio.run(main())
