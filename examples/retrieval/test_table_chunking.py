"""Test table-aware splitting for improved retrieval."""

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_embeddings

from agenticflow.document.loaders.pdf import PDFVisionLoader
from agenticflow.models import ChatModel
from agenticflow.document.splitters import TableAwareSplitter
from agenticflow.vectorstore import VectorStore
from agenticflow.retriever import DenseRetriever, BM25Retriever, EnsembleRetriever


async def main() -> None:
    """Compare page-level vs table-level chunking."""
    
    # Load PDF
    print("📄 Loading PDF with HTML tables...")
    loader = PDFVisionLoader(model=ChatModel(model="gpt-4o"), output_format="html")
    page_docs = await loader.load(Path("examples/data/sample-tables.pdf"))
    print(f"✅ Loaded {len(page_docs)} pages\n")
    
    # Split by tables
    print("✂️  Splitting by tables...")
    splitter = TableAwareSplitter(
        context_before=1,
        context_after=1,
        min_text_chunk_size=50,
    )
    table_docs = splitter.split_documents(page_docs)
    print(f"✅ Created {len(table_docs)} table-aware chunks\n")
    
    # Show some examples
    print("=" * 80)
    print("SAMPLE CHUNKS")
    print("=" * 80)
    for i, doc in enumerate(table_docs[:5], 1):
        chunk_type = doc.metadata.get('chunk_type', 'unknown')
        page = doc.metadata.get('page', '?')
        table_num = doc.metadata.get('table_number', 'N/A')
        caption = doc.metadata.get('table_caption', '')
        
        print(f"\n{i}. Page {page}, Type: {chunk_type}, Table: {table_num}")
        if caption:
            print(f"   Caption: {caption[:100]}...")
        print(f"   Size: {len(doc.content)} chars")
        print(f"   Preview: {doc.content[:200].replace(chr(10), ' ')}...")
    print("\n")
    
    # Build retrievers for both approaches
    print("🔍 Building retrievers...")
    
    # Page-level vectorstore
    page_vectorstore = VectorStore(embeddings=get_embeddings())
    await page_vectorstore.add_documents(page_docs)
    page_dense = DenseRetriever(vectorstore=page_vectorstore)
    
    # Table-level vectorstore
    table_vectorstore = VectorStore(embeddings=get_embeddings())
    await table_vectorstore.add_documents(table_docs)
    table_dense = DenseRetriever(vectorstore=table_vectorstore)
    table_bm25 = BM25Retriever()
    table_bm25.add_documents(table_docs)
    table_ensemble = EnsembleRetriever(
        retrievers=[table_dense, table_bm25],
        weights=[0.5, 0.5],
        fusion="rrf",
    )
    
    print("✅ Created retrievers\n")
    
    # Test query
    query = "What was the average rainfall in Asia in 2010?"
    
    print("=" * 80)
    print(f"Query: {query}")
    print("=" * 80)
    print()
    
    # Compare approaches
    for name, retriever in [
        ("Page-Level (Dense)", page_dense),
        ("Table-Level (Dense)", table_dense),
        ("Table-Level (Ensemble)", table_ensemble),
    ]:
        print(f"{'=' * 80}")
        print(f"🔎 {name}")
        print(f"{'=' * 80}")
        
        results = await retriever.retrieve(query, k=5)
        
        for i, result in enumerate(results, 1):
            metadata = result.metadata
            page_num = metadata.get("page", "?")
            chunk_type = metadata.get("chunk_type", "page")
            table_num = metadata.get("table_number", "-")
            
            # Check which table this is
            table_info = ""
            if "Table 6:" in result.content:
                table_info = "✅ Table 6 (CORRECT - 'Average' 201)"
            elif "Table 29:" in result.content:
                table_info = "❌ Table 29 (WRONG - 'Highest average' 467.4)"
            elif "Table 9:" in result.content:
                table_info = "⚠️  Table 9 (2009 data)"
            
            print(f"{i}. Page {page_num}, Type: {chunk_type}, Table: {table_num}")
            if table_info:
                print(f"   {table_info}")
            
            # Show snippet
            snippet = result.content[:150].replace("\n", " ")
            print(f"   {snippet}...")
            print()
        
        print()
    
    # Summary
    print("=" * 80)
    print("📊 ANALYSIS")
    print("=" * 80)
    print()
    print("Expected improvements with table-level chunking:")
    print("- Isolates each table from noise of other tables on same page")
    print("- Table captions included in chunk metadata for filtering")
    print("- Smaller, focused chunks should improve ranking")
    print("- Ensemble should benefit from reduced noise in both dense and BM25")


if __name__ == "__main__":
    asyncio.run(main())
