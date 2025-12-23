"""RAG example with PDF HTML tables.

Demonstrates how LLMs can understand complex table structures (merged cells, 
hierarchical headers) when provided as HTML rather than markdown.
"""

import asyncio
from pathlib import Path
import sys

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_model, get_embeddings

from agenticflow import Agent
from agenticflow.document.loaders.handlers.pdf_html import PDFHTMLLoader
from agenticflow.vectorstore import VectorStore


async def main() -> None:
    """Run RAG example with HTML tables from PDF."""
    
    # Load PDF with complex tables as HTML
    print("📄 Loading PDF with HTML table extraction...")
    loader = PDFHTMLLoader()
    docs = await loader.load(Path("examples/data/sample-tables.pdf"))
    
    print(f"✅ Loaded {len(docs)} pages")
    print(f"📊 Total content: {sum(len(d.content) for d in docs):,} characters\n")
    
    # Create vector store and add documents
    print("🔍 Building vector store...")
    vectorstore = VectorStore(embeddings=get_embeddings())
    await vectorstore.add_documents(docs)
    print(f"✅ Indexed {len(docs)} documents\n")
    
    # Create search tool from vectorstore
    search_tool = vectorstore.as_retriever().as_tool(
        name="search_pdf_tables",
        description=(
            "Search the PDF document with complex HTML tables. "
            "Use this to find specific table data, financial information, "
            "or structured content. Tables include rowspan/colspan for merged cells."
        ),
        k_default=3,
        include_scores=False,
        include_metadata=True,
    )
    
    # Create RAG agent
    agent = Agent(
        name="TableAnalyst",
        model=get_model(),
        tools=[search_tool],
        system_prompt="""You are a table analysis expert. You receive HTML table content 
from PDFs and must interpret the structure accurately.

When analyzing tables:
- Pay attention to rowspan and colspan attributes for merged cells
- Understand hierarchical headers (colspan spanning multiple columns)
- Interpret nested row headers (rowspan spanning multiple rows)
- Provide precise answers based on the table structure

Always cite the specific table when providing answers.""",
    )
    
    # Test queries that require understanding complex table structure
    queries = [
        # Query 1: Simple lookup
        "What is the total expenditure for Banking in 2010/11?",
        
        # Query 2: Understanding merged cells (rowspan)
        "Which services fall under 'Remunerated functions' and what were their 2009/10 values?",
        
        # Query 3: Understanding hierarchical structure (colspan)
        "What was the average rainfall in Asia in 2010? Make sure you're reading the correct year.",
        
        # Query 4: Understanding table with merged column headers
        "In the financial statement table, what are the Property values for all three years?",
    ]
    
    print("=" * 80)
    print("🤖 Testing LLM's ability to interpret complex HTML tables")
    print("=" * 80)
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 80}")
        print(f"📝 Query {i}: {query}")
        print(f"{'=' * 80}\n")
        
        response = await agent.run(query)
        print(f"💡 Answer: {response}\n")
        
        # Show retrieved context for first query
        if i == 1:
            print("\n📚 Sample retrieved context:")
            retriever = vectorstore.as_retriever()
            results = await retriever.retrieve(query, k=1)
            if results:
                print(f"Page {results[0].metadata.get('page', '?')} content (first 800 chars):")
                print(results[0].content[:800])
                print("...\n")
    

if __name__ == "__main__":
    asyncio.run(main())
