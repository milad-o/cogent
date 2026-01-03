"""Test ensemble retrieval (Dense + BM25) vs Dense-only for PDF tables.

Demonstrates how hybrid retrieval can improve ranking for exact term matches
like "average" vs "highest average".
"""

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_embeddings

from agenticflow.document.loaders.pdf import PDFVisionLoader
from agenticflow.models import ChatModel
from agenticflow.vectorstore import VectorStore
from agenticflow.retriever import DenseRetriever, BM25Retriever, EnsembleRetriever


async def main() -> None:
    """Compare dense vs hybrid retrieval for table queries."""
    
    # Load PDF
    print("📄 Loading PDF with HTML tables...")
    loader = PDFVisionLoader(model=ChatModel(model="gpt-4o"), output_format="html")
    docs = await loader.load(Path("examples/data/sample-tables.pdf"))
    print(f"✅ Loaded {len(docs)} pages\n")
    
    # Build vectorstore and retrievers
    print("🔍 Building retrievers...")
    vectorstore = VectorStore(embeddings=get_embeddings())
    await vectorstore.add_documents(docs)
    
    dense_retriever = DenseRetriever(vectorstore=vectorstore)
    bm25_retriever = BM25Retriever()
    bm25_retriever.add_documents(docs)
    
    # Create ensemble retriever with RRF fusion
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.5, 0.5],  # Equal weight
        fusion="rrf",  # Reciprocal Rank Fusion
    )
    
    print("✅ Created 3 retrievers: Dense, BM25, Ensemble\n")
    
    # Test query
    query = "What was the average rainfall in Asia in 2010?"
    
    print("=" * 80)
    print(f"Query: {query}")
    print("=" * 80)
    print()
    
    # Test each retriever
    for name, retriever in [
        ("Dense (Embeddings Only)", dense_retriever),
        ("BM25 (Lexical Only)", bm25_retriever),
        ("Ensemble (Dense + BM25 RRF)", ensemble_retriever),
    ]:
        print(f"{'=' * 80}")
        print(f"🔎 {name}")
        print(f"{'=' * 80}")
        
        results = await retriever.retrieve(query, k=5)
        
        for i, result in enumerate(results, 1):
            page_num = result.metadata.get("page", "?")
            
            # Check which table this is
            table_info = ""
            if "Table 6:" in result.content:
                table_info = "✅ Table 6 (CORRECT - 'Average' 201 inches)"
            elif "Table 29:" in result.content:
                table_info = "❌ Table 29 (WRONG - 'Highest average' 467.4)"
            elif "Table 9:" in result.content:
                table_info = "⚠️  Table 9 (2009 data only)"
            
            # Get score if available
            score_str = ""
            if hasattr(result, "score") and result.score is not None:
                score_str = f" (score: {result.score:.4f})"
            
            print(f"{i}. Page {page_num}{score_str}")
            if table_info:
                print(f"   {table_info}")
            
            # Show snippet
            snippet = result.content[:200].replace("\n", " ")
            print(f"   {snippet}...")
            print()
        
        print()
    
    # Summary
    print("=" * 80)
    print("📊 ANALYSIS")
    print("=" * 80)
    print()
    print("Expected outcome:")
    print("- Dense: May rank Table 29 higher (richer semantic context)")
    print("- BM25: Should rank Table 6 higher (exact 'average' + 'rainfall' match)")
    print("- Ensemble: Should balance both via RRF, ideally ranking Table 6 in top 3")


if __name__ == "__main__":
    asyncio.run(main())
