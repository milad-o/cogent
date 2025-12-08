"""
Example 10: RAG (Retrieval-Augmented Generation)

All RAG patterns in agenticflow:

┌─────────────────────────────────────────────────────────────────────────┐
│ AGENTIC RAG (Agent decides when to search)                              │
├─────────────────────────────────────────────────────────────────────────┤
│ Pattern 1: RAG Capability           → search_documents tool             │
│            - Agent autonomously decides when to search                  │
│            - Citations: [1], [2] (numeric style)                        │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ NAIVE RAG (Context injected before agent thinks)                        │
├─────────────────────────────────────────────────────────────────────────┤
│ Pattern 2: RAG Interceptor          → Pre-think context injection       │
│            - With citations: «1», «2» (guillemet style)                 │
│ Pattern 3: RAG Interceptor (no citations)                               │
│            - Plain context, no citation markers                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ PROGRAMMATIC RAG (Full control)                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ Pattern 4: Direct Retriever + Utilities                                 │
│            - Manual retrieval, formatting, citation                     │
│ Pattern 5: Ensemble Retriever (hybrid search)                           │
│            - Dense + Sparse with RRF/linear fusion                      │
└─────────────────────────────────────────────────────────────────────────┘

Usage:
    uv run python examples/10_rag.py
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_embeddings, get_model

from agenticflow import Agent
from agenticflow.capabilities import RAG
from agenticflow.interceptors import RAGInterceptor
from agenticflow.document import RecursiveCharacterSplitter
from agenticflow.retriever import (
    DenseRetriever,
    BM25Retriever,
    EnsembleRetriever,
    # Result utilities
    add_citations,
    format_context,
    format_citations_reference,
    filter_by_score,
)
from agenticflow.vectorstore import VectorStore, Document


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
    # Setup: Prepare documents and retriever
    # =========================================================================
    print("=" * 70)
    print("Setup: Prepare documents and retriever")
    print("=" * 70)

    # Create document and split into chunks
    doc = Document(text=SAMPLE_TEXT, metadata={"source": "the_secret_garden.txt"})
    splitter = RecursiveCharacterSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents([doc])
    print(f"Created {len(chunks)} chunks")

    # Create vectorstore and index
    store = VectorStore(embeddings=embeddings)
    await store.add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks in vectorstore")

    # Create retrievers
    dense = DenseRetriever(store)
    sparse = BM25Retriever(chunks)

    # =========================================================================
    # AGENTIC RAG
    # =========================================================================
    print("\n")
    print("═" * 70)
    print("  AGENTIC RAG - Agent decides when to search")
    print("═" * 70)

    # -------------------------------------------------------------------------
    # Pattern 1: RAG Capability (Agentic RAG with tool)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Pattern 1: RAG Capability")
    print("  → Agent gets search_documents tool")
    print("  → Citations: [1], [2], [3] (numeric)")
    print("-" * 70)

    rag = RAG(dense)
    agent = Agent(
        name="BookAssistant",
        model=model,
        capabilities=[rag],
    )

    answer = await agent.run("What was Mary Lennox like when she arrived?")
    print(f"\n{answer}")

    # =========================================================================
    # NAIVE RAG
    # =========================================================================
    print("\n")
    print("═" * 70)
    print("  NAIVE RAG - Context injected before agent thinks")
    print("═" * 70)

    # -------------------------------------------------------------------------
    # Pattern 2: RAG Interceptor WITH citations
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Pattern 2: RAG Interceptor (with citations)")
    print("  → Context injected before first LLM call")
    print("  → Citations: «1», «2» (guillemet)")
    print("-" * 70)

    rag_interceptor = RAGInterceptor(
        retriever=dense,
        k=3,
        min_score=0.3,
        include_sources=True,  # Show sources reference
        # Uses default template with citation instructions
    )

    agent2 = Agent(
        name="BookExpert",
        model=model,
        intercept=[rag_interceptor],
    )

    answer = await agent2.run("What did the moor look like?")
    print(f"\n{answer}")

    # -------------------------------------------------------------------------
    # Pattern 3: RAG Interceptor WITHOUT citations
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Pattern 3: RAG Interceptor (no citations)")
    print("  → Plain context, no citation markers")
    print("  → Simpler output for some use cases")
    print("-" * 70)

    # Custom template without citation instructions
    no_citation_template = """Answer the question based on the following context.

Context:
{context}

Question: {question}"""

    rag_no_cite = RAGInterceptor(
        retriever=dense,
        k=3,
        min_score=0.3,
        include_sources=False,  # No sources reference
        context_template=no_citation_template,
    )

    agent3 = Agent(
        name="SimpleAssistant",
        model=model,
        intercept=[rag_no_cite],
    )

    answer = await agent3.run("Who was Martha?")
    print(f"\n{answer}")

    # =========================================================================
    # PROGRAMMATIC RAG
    # =========================================================================
    print("\n")
    print("═" * 70)
    print("  PROGRAMMATIC RAG - Full control over retrieval")
    print("═" * 70)

    # -------------------------------------------------------------------------
    # Pattern 4: Direct Retriever + Utilities
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Pattern 4: Direct Retriever + Utilities")
    print("  → Manual retrieval and formatting")
    print("  → Full control over pipeline")
    print("-" * 70)

    # Retrieve
    results = await dense.retrieve("Describe the moor", k=5, include_scores=True)
    
    # Filter
    results = filter_by_score(results, min_score=0.3)
    
    # Add citations
    results = add_citations(results)
    
    print("\nRetrieved passages with citations:")
    for r in results:
        citation = r.metadata.get("citation", "")
        source = r.document.metadata.get("source", "unknown")
        print(f"  {citation} {source} (score: {r.score:.2f})")
        print(f"      {r.document.text[:70]}...")

    # Format for LLM
    context = format_context(results)
    sources = format_citations_reference(results)
    
    print("\n--- Context for LLM ---")
    print(context[:400] + "..." if len(context) > 400 else context)
    print("\n--- Sources Reference ---")
    print(sources)

    # -------------------------------------------------------------------------
    # Pattern 5: Ensemble Retriever (Hybrid Search)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Pattern 5: Ensemble Retriever (Hybrid Search)")
    print("  → Dense + Sparse with RRF fusion")
    print("  → Better recall than single retriever")
    print("-" * 70)

    ensemble = EnsembleRetriever(
        retrievers=[dense, sparse],
        weights=[0.6, 0.4],
        fusion="rrf",  # Reciprocal Rank Fusion
    )

    results = await ensemble.retrieve("Martha Yorkshire girl", k=3, include_scores=True)
    results = add_citations(results)
    
    print("\nEnsemble results (RRF fusion):")
    for r in results:
        source = r.document.metadata.get("source", "unknown")
        print(f"  {r.metadata['citation']} {source} (score: {r.score:.3f})")
        print(f"      {r.document.text[:60]}...")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n")
    print("═" * 70)
    print("  SUMMARY")
    print("═" * 70)
    print("""
┌──────────────────┬─────────────────────────────────────────────────────┐
│ Pattern          │ Use Case                                            │
├──────────────────┼─────────────────────────────────────────────────────┤
│ RAG Capability   │ Agent autonomously decides when to search           │
│                  │ Best for: Complex queries, multi-turn conversations │
├──────────────────┼─────────────────────────────────────────────────────┤
│ RAG Interceptor  │ Always retrieve, inject context, single LLM call    │
│ (with citations) │ Best for: Q&A, cost-sensitive, traceable answers    │
├──────────────────┼─────────────────────────────────────────────────────┤
│ RAG Interceptor  │ Simple context injection, no citation overhead      │
│ (no citations)   │ Best for: Summarization, simple Q&A                 │
├──────────────────┼─────────────────────────────────────────────────────┤
│ Direct Retriever │ Full programmatic control                           │
│                  │ Best for: Custom pipelines, batch processing        │
├──────────────────┼─────────────────────────────────────────────────────┤
│ Ensemble         │ Hybrid search (dense + sparse)                      │
│                  │ Best for: Better recall, domain-specific terms      │
└──────────────────┴─────────────────────────────────────────────────────┘
""")
    print("✓ Done")


if __name__ == "__main__":
    asyncio.run(main())
