"""
Example 36: RAG with Citations

Unified `run()` API with customization options:

- `run(query)` → str (agent uses tools to search/answer)
- `run(query, citations=True)` → RAGResponse (structured with citations)
- `vectorstore.search(query)` → direct search (no LLM)

Usage:
    uv run python examples/36_rag_citations.py
"""

import asyncio
from pathlib import Path

from config import get_embeddings, get_model

from agenticflow.prebuilt import RAGAgent


async def main() -> None:
    data_dir = Path(__file__).parent / "data"
    text_path = data_dir / "the_secret_garden.txt"
    
    model = get_model()
    embeddings = get_embeddings()
    
    rag = RAGAgent(
        model=model,
        embeddings=embeddings,
        chunk_size=500,
        chunk_overlap=100,
        top_k=4,
        show_progress=False,
    )
    
    # Load document silently
    if text_path.exists():
        await rag.load(text_path)
    else:
        await rag.load_text(SAMPLE_TEXT, source="the_secret_garden.txt")
    
    # =========================================================================
    # 1. run() - Agent uses tools to search and answer
    # =========================================================================
    print("─" * 60)
    print("run() → Agent decides how to search")
    print("─" * 60)
    
    answer = await rag.run("Describe Mary Lennox when she arrived.")
    print(answer)
    
    # =========================================================================
    # 2. run(citations=True) - Structured response with bibliography
    # =========================================================================
    print("\n" + "─" * 60)
    print("run(citations=True) → RAGResponse with structured citations")
    print("─" * 60)
    
    response = await rag.run(
        "What was the moor like?",
        citations=True,
        k=3,  # limit passages
    )
    
    print(response.answer)
    print(response.format_bibliography())
    
    # =========================================================================
    # 3. vectorstore.search() - Direct search (no LLM)
    # =========================================================================
    print("\n" + "─" * 60)
    print("vectorstore.search() → Direct retrieval")
    print("─" * 60)
    
    results = await rag.vectorstore.search("Martha Yorkshire", k=2)
    for i, result in enumerate(results, 1):
        doc = result.document
        print(f"[{i}] {doc.metadata.get('source', 'unknown')} (score: {result.score:.2f})")
        print(f"    {doc.text[:80]}...")
    
    print("\n✓ Done")


SAMPLE_TEXT = """
The Secret Garden by Frances Hodgson Burnett

Chapter 1: There Is No One Left

When Mary Lennox was sent to Misselthwaite Manor to live with her uncle, 
everybody said she was the most disagreeable-looking child ever seen. 
It was true, too. She had a little thin face and a little thin body, 
thin light hair and a sour expression.

Chapter 3: Across the Moor

The moor was a vast stretch of wild land, covered with brown heather 
and gorse bushes. It stretched for miles in every direction.

"It's the moor," said Martha. "It's called the moor."

Chapter 4: Martha

Martha was a good-natured Yorkshire girl who had been hired to wait on Mary.
She was different from any servant Mary had ever known.

"Th' fresh air an' th' skippin' rope will make thee strong," Martha said.
"""


if __name__ == "__main__":
    asyncio.run(main())
