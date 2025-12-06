"""
Example 10: RAG (Retrieval-Augmented Generation)

Two approaches for RAG:

1. **RAG Capability** - For agents (provides search_documents tool)
2. **Direct Retriever + Utilities** - For programmatic use

The RAG capability is a thin wrapper that gives agents a search tool.
For programmatic use, use retrievers directly with utility functions.

Usage:
    uv run python examples/10_rag.py
"""

import asyncio

from config import get_embeddings, get_model

from agenticflow import Agent
from agenticflow.capabilities import RAG
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
    # Step 1: Prepare documents (OUTSIDE RAG)
    # =========================================================================
    print("=" * 60)
    print("Step 1: Prepare documents (outside RAG)")
    print("=" * 60)

    # Create document and split into chunks
    doc = Document(text=SAMPLE_TEXT, metadata={"source": "the_secret_garden.txt"})
    splitter = RecursiveCharacterSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents([doc])
    print(f"Created {len(chunks)} chunks")

    # Create vectorstore and index
    store = VectorStore(embeddings=embeddings)
    await store.add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks in vectorstore")

    # =========================================================================
    # Pattern 1: RAG Capability (for Agents)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Pattern 1: RAG Capability (for Agents)")
    print("=" * 60)

    # Create retriever and RAG capability
    dense = DenseRetriever(store)
    rag = RAG(dense)

    # Add to agent - agent gets search_documents tool
    agent = Agent(
        name="BookAssistant",
        model=model,
        capabilities=[rag],
    )

    # Agent uses search_documents tool automatically
    answer = await agent.run("What was Mary Lennox like when she arrived?")
    print(f"\n{answer}")

    # =========================================================================
    # Pattern 2: Direct Retriever + Utilities (Programmatic)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Pattern 2: Direct Retriever + Utilities")
    print("=" * 60)

    # Use retriever directly
    results = await dense.retrieve("Describe the moor", k=5, include_scores=True)
    
    # Filter low-quality results
    results = filter_by_score(results, min_score=0.3)
    
    # Add citation markers
    results = add_citations(results)
    
    print("Retrieved passages with citations:")
    for r in results:
        citation = r.metadata.get("citation", "")
        source = r.document.metadata.get("source", "unknown")
        print(f"  {citation} {source} (score: {r.score:.2f})")
        print(f"    {r.document.text[:80]}...")

    # =========================================================================
    # Pattern 3: Ensemble Retriever (Multiple Sources)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Pattern 3: Ensemble Retriever")
    print("=" * 60)

    # Create ensemble from multiple retrievers
    sparse = BM25Retriever(chunks)
    ensemble = EnsembleRetriever(
        retrievers=[dense, sparse],
        weights=[0.6, 0.4],
        fusion="rrf",
    )

    results = await ensemble.retrieve("Martha Yorkshire girl", k=3, include_scores=True)
    results = add_citations(results)
    
    print("Ensemble results (RRF fusion):")
    for r in results:
        source = r.document.metadata.get("source", "unknown")
        print(f"  {r.metadata['citation']} {source} (score: {r.score:.3f})")
        print(f"    {r.document.text[:70]}...")

    # =========================================================================
    # Formatting Context for LLM
    # =========================================================================
    print("\n" + "=" * 60)
    print("Formatting Context for LLM")
    print("=" * 60)

    results = await dense.retrieve("Mary Lennox appearance", k=3, include_scores=True)
    results = add_citations(results)
    
    # Format as context string
    context = format_context(results)
    print("Context for LLM prompt:")
    print("-" * 40)
    print(context[:500] + "..." if len(context) > 500 else context)
    
    # Format citations reference
    print("\n" + "-" * 40)
    reference = format_citations_reference(results)
    print(reference)

    # =========================================================================
    # Building a RAG Prompt
    # =========================================================================
    print("\n" + "=" * 60)
    print("Building a RAG Prompt")
    print("=" * 60)

    query = "What did Martha say about the moor air?"
    results = await dense.retrieve(query, k=3, include_scores=True)
    results = add_citations(results)
    context = format_context(results)

    # Build prompt
    prompt = f"""Based on the following context, answer the question.
Use citation markers like «1» to reference sources.

Context:
{context}

Question: {query}

Answer:"""

    print("Generated prompt (first 600 chars):")
    print(prompt[:600] + "...")

    print("\n✓ Done")


if __name__ == "__main__":
    asyncio.run(main())
