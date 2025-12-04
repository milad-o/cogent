"""
Example 36: RAG with Citations and References

Demonstrates the new citation and referencing functionality in RAGAgent:
- CitedPassage for structured citation data
- RAGResponse for structured query responses
- query_with_citations() for full citation tracking
- search_with_citations() for direct search with metadata

Usage:
    uv run python examples/36_rag_citations.py

Features demonstrated:
- Inline citations [1], [2], etc.
- Page number references [n, p.X] (when available from PDF)
- Source document tracking
- Bibliography generation
- Structured access to all citation metadata
"""

import asyncio
from pathlib import Path

from config import get_embeddings, get_model, settings

from agenticflow.prebuilt import RAGAgent, CitedPassage, RAGResponse


async def main() -> None:
    """Demonstrate RAG citation functionality."""
    
    # Use The Secret Garden text from examples/data
    data_dir = Path(__file__).parent / "data"
    text_path = data_dir / "the_secret_garden.txt"
    
    print("=" * 70)
    print("📚 RAG with Citations - Demonstrating Citation Functionality")
    print("=" * 70)
    print(f"LLM: {settings.llm_provider}")
    print(f"Embeddings: {settings.embedding_provider}")
    
    # Create model and embeddings
    model = get_model()
    embeddings = get_embeddings()
    
    # Create RAG Agent
    rag = RAGAgent(
        model=model,
        embeddings=embeddings,
        chunk_size=500,
        chunk_overlap=100,
        top_k=4,
        name="BookAnalyzer",
    )
    
    # Load document
    print("\n" + "-" * 70)
    print("📄 Loading Document...")
    print("-" * 70)
    
    if text_path.exists():
        await rag.load_documents([text_path], show_progress=True)
        source_name = text_path.name
    else:
        # Fallback: load sample text directly
        sample_text = """
        The Secret Garden by Frances Hodgson Burnett

        Chapter 1: There Is No One Left

        When Mary Lennox was sent to Misselthwaite Manor to live with her uncle, 
        everybody said she was the most disagreeable-looking child ever seen. 
        It was true, too. She had a little thin face and a little thin body, 
        thin light hair and a sour expression. Her hair was yellow, and her face 
        was yellow because she had been born in India and had always been ill 
        in one way or another.

        Her father had held a position under the English Government and had always 
        been busy and ill himself, and her mother had been a great beauty who cared 
        only to go to parties and amuse herself with gay people.

        Chapter 2: Mistress Mary Quite Contrary

        Mary had liked to look at her mother from a distance and she had thought 
        her very pretty, but as she knew very little of her she could scarcely 
        have been expected to love her or to miss her very much when she was gone.

        She did not miss her at all, in fact, and as she was a self-absorbed child 
        she gave her entire thought to herself, as she had always done.

        Chapter 3: Across the Moor

        The train had been going through a wild and desolate country. It was grey 
        and dark, and though it was winter the land was not covered with snow. 
        Mary looked out of the window with a peculiar interest.

        "What is it? Why does it look so different?" she asked Martha.
        "It's the moor," said Martha. "It's called the moor."
        
        The moor was a vast stretch of wild land, covered with brown heather 
        and gorse bushes. It stretched for miles in every direction, with no 
        trees or houses to break the monotony.

        Chapter 4: Martha

        Martha was a good-natured Yorkshire girl who had been hired to wait on Mary.
        She was different from any servant Mary had ever known. She talked to Mary 
        as if she were an ordinary person, not a little sahib who had always been 
        waited on hand and foot.

        "Th' fresh air an' th' skippin' rope will make thee strong," Martha said.
        "An' if tha' goes out o' doors an' runs about, tha'll get an appetite."
        """
        await rag.load_text(sample_text, source="the_secret_garden_excerpt.txt")
        source_name = "the_secret_garden_excerpt.txt"
    
    print(f"\n✓ Indexed {rag.document_count} chunks from {source_name}")
    
    # =========================================================================
    # Example 1: Standard query (shows inline citations in answer)
    # =========================================================================
    print("\n" + "=" * 70)
    print("📝 Example 1: Standard Query with Inline Citations")
    print("=" * 70)
    
    question1 = "What was Mary Lennox like when she first arrived at Misselthwaite Manor?"
    print(f"\nQuestion: {question1}")
    print("-" * 70)
    
    answer = await rag.query(question1)
    print(f"\nAnswer:\n{answer}")
    
    # Access citations from the last query
    print(f"\n📌 Citations from last query: {len(rag.citations)}")
    for cite in rag.citations:
        print(f"  {cite.format_reference()} - {cite.source} (score: {cite.score:.3f})")
    
    # =========================================================================
    # Example 2: Query with full citation response
    # =========================================================================
    print("\n" + "=" * 70)
    print("📝 Example 2: Query with Structured Citations (RAGResponse)")
    print("=" * 70)
    
    question2 = "What is the moor and how is it described?"
    print(f"\nQuestion: {question2}")
    print("-" * 70)
    
    # Use query_with_citations for structured response
    response: RAGResponse = await rag.query_with_citations(question2)
    
    print(f"\n📊 Response Metadata:")
    print(f"  - Query: {response.query}")
    print(f"  - Sources used: {response.sources_used}")
    print(f"  - Citation count: {response.citation_count}")
    print(f"  - Has citations: {response.has_citations}")
    
    print(f"\n📖 Answer:\n{response.answer}")
    
    print(f"\n📚 Full Response with Bibliography:")
    print("-" * 70)
    print(response.format_full_response())
    
    # =========================================================================
    # Example 3: Direct search with citations
    # =========================================================================
    print("\n" + "=" * 70)
    print("🔍 Example 3: Direct Search with Citations")
    print("=" * 70)
    
    search_query = "Martha Yorkshire servant"
    print(f"\nSearch: {search_query}")
    print("-" * 70)
    
    # Use search_with_citations for structured results
    citations: list[CitedPassage] = await rag.search_with_citations(search_query, k=3)
    
    print(f"\nFound {len(citations)} relevant passages:\n")
    
    for cite in citations:
        print(f"{cite.format_full()}")
        print(f"  Text excerpt: {cite.text[:150]}...")
        print()
    
    # =========================================================================
    # Example 4: Working with CitedPassage objects
    # =========================================================================
    print("\n" + "=" * 70)
    print("🔧 Example 4: Working with CitedPassage Objects")
    print("=" * 70)
    
    if citations:
        cite = citations[0]
        print(f"\nFirst citation details:")
        print(f"  citation_id: {cite.citation_id}")
        print(f"  source: {cite.source}")
        print(f"  page: {cite.page}")
        print(f"  chunk_index: {cite.chunk_index}")
        print(f"  score: {cite.score:.4f}")
        print(f"  text length: {len(cite.text)} chars")
        print(f"\n  format_reference(): {cite.format_reference()}")
        print(f"  format_full(): {cite.format_full()}")
    
    # =========================================================================
    # Example 5: Building a custom bibliography
    # =========================================================================
    print("\n" + "=" * 70)
    print("📑 Example 5: Custom Bibliography from Multiple Queries")
    print("=" * 70)
    
    all_citations: list[CitedPassage] = []
    questions = [
        "Who is Mary Lennox?",
        "What did Martha say about fresh air?",
    ]
    
    for q in questions:
        response = await rag.query_with_citations(q)
        all_citations.extend(response.citations)
    
    # Deduplicate by source + text (keep highest score)
    unique_sources = {}
    for cite in all_citations:
        key = (cite.source, cite.text[:100])
        if key not in unique_sources or cite.score > unique_sources[key].score:
            unique_sources[key] = cite
    
    print(f"\nUnique sources referenced across {len(questions)} queries:")
    for i, cite in enumerate(unique_sources.values(), 1):
        # Re-number citations for bibliography
        new_cite = CitedPassage(
            citation_id=i,
            source=cite.source,
            page=cite.page,
            chunk_index=cite.chunk_index,
            score=cite.score,
            text=cite.text,
        )
        print(f"  {new_cite.format_full()}")
    
    print("\n" + "=" * 70)
    print("✅ RAG Citations Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
