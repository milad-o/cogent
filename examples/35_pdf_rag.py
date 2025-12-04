"""
Example: RAG with PDF Documents using PDFMarkdownLoader

Demonstrates building a searchable knowledge base from a complex PDF document
(IRS Publication 15 - Employer's Tax Guide) using the high-performance
PDFMarkdownLoader and RAGAgent.

Key features:
- PDF-to-Markdown conversion optimized for LLM/RAG
- Semantic search over 58 pages of tax documentation
- Question answering with source citations

Usage:
    uv run python examples/33_pdf_rag.py

Requires:
    uv add pymupdf4llm pymupdf
    Configure API keys in .env (see config.py)
"""

import asyncio
from pathlib import Path

from config import get_embeddings, get_model, settings

from agenticflow.document.loaders import PDFMarkdownLoader
from agenticflow.prebuilt import RAGAgent, DocumentPipeline


async def main() -> None:
    # Paths
    data_dir = Path(__file__).parent / "data"
    pdf_path = data_dir / "financial_report.pdf"

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        print("Download from: https://www.irs.gov/pub/irs-pdf/p15.pdf")
        return

    print("=" * 60)
    print("📚 PDF RAG - IRS Employer's Tax Guide")
    print("=" * 60)
    print(f"LLM: {settings.llm_provider}")
    print(f"Embeddings: {settings.embedding_provider}")
    print(f"PDF: {pdf_path.name}")

    # Create model and embeddings
    model = get_model()
    embeddings = get_embeddings()

    # Create RAG Agent with our PDFMarkdownLoader for PDFs
    rag = RAGAgent(
        model=model,
        embeddings=embeddings,
        chunk_size=800,
        chunk_overlap=150,
        top_k=5,
        name="TaxGuideAssistant",
        instructions="""You are an expert on US employer tax obligations.
Answer questions based ONLY on the IRS Publication 15 content provided.
Always cite the relevant section or page when answering.
If the information isn't in the document, say so clearly.""",
    )

    # Register our high-performance PDFMarkdownLoader for .pdf files
    rag.register_pipeline(
        ".pdf",
        DocumentPipeline(
            loader=PDFMarkdownLoader(max_workers=4, batch_size=10),
        ),
    )

    # Load the PDF (uses our PDFMarkdownLoader)
    print("\n" + "-" * 60)
    print("📄 Loading PDF with PDFMarkdownLoader...")
    print("-" * 60)

    await rag.load_documents([pdf_path], show_progress=True)

    print(f"\n  ✓ Indexed {rag.document_count} chunks")

    # Ask questions about tax topics
    print("\n" + "-" * 60)
    print("❓ Querying the Tax Guide")
    print("-" * 60)

    questions = [
        "What is the social security tax rate for 2025?",
        "What are the deposit rules for employment taxes?",
        "How do I handle tips for tax purposes?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {question}")
        print("=" * 60)

        answer = await rag.query(question)
        print(f"\n{answer}")

    # Direct semantic search
    print("\n" + "-" * 60)
    print("🔎 Direct Semantic Search")
    print("-" * 60)

    search_results = await rag.search("Form W-4 withholding requirements", k=3)

    print("\nTop 3 passages about Form W-4:")
    for i, result in enumerate(search_results, 1):
        preview = result.document.text[:200].replace("\n", " ")
        print(f"\n[{i}] (score: {result.score:.3f})")
        print(f"    {preview}...")

    print("\n" + "=" * 60)
    print("✅ PDF RAG Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
