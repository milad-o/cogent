"""
Example: PDF Summarization with Observability

Demonstrates summarizing a PDF document using PDFMarkdownLoader + Summarizer
with full progress tracking and timing metrics.

Usage:
    uv run python examples/35_pdf_summarizer.py

Requires:
    uv add pymupdf4llm pymupdf
"""

import asyncio
import time
from pathlib import Path

from config import get_model

from agenticflow.document.loaders import PDFMarkdownLoader
from agenticflow.document.summarizer import MapReduceSummarizer, RefineSummarizer


async def main() -> None:
    """Summarize a PDF with progress tracking."""
    
    model = get_model()
    data_dir = Path(__file__).parent / "data"
    pdf_path = data_dir / "wikipedia_water_cycle.pdf"
    
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        print("   Download with: curl -sL -o examples/data/wikipedia_water_cycle.pdf 'https://en.wikipedia.org/api/rest_v1/page/pdf/Water_cycle'")
        return
    
    print("=" * 70)
    print("PDF Summarization Demo")
    print("=" * 70)
    
    # =========================================================================
    # Step 1: Load PDF to Markdown
    # =========================================================================
    print("\n📄 Step 1: Loading PDF to Markdown\n")
    
    loader = PDFMarkdownLoader(
        show_progress=True,
        max_workers=4,
    )
    
    load_start = time.perf_counter()
    result = await loader.load(pdf_path, tracking=True)
    load_time = time.perf_counter() - load_start
    
    print(f"\n✅ PDF loaded in {load_time:.2f}s")
    print(f"   Pages: {result.total_pages}")
    print(f"   Success rate: {result.success_rate:.0%}")
    
    # Combine all page content into markdown
    markdown_content = "\n\n".join(
        page.content for page in result.page_results 
        if page.content.strip()
    )
    print(f"   Characters: {len(markdown_content):,}")
    
    # =========================================================================
    # Step 2: Summarize with Map-Reduce
    # =========================================================================
    print("\n" + "=" * 70)
    print("📝 Step 2: Summarizing with Map-Reduce Strategy")
    print("=" * 70 + "\n")
    
    summarizer = MapReduceSummarizer(
        model=model,
        chunk_size=4000,
        chunk_overlap=200,
        max_concurrent=3,  # Parallel API calls
        show_progress=True,
    )
    
    summary_result = await summarizer.summarize(
        markdown_content,
        context="Wikipedia article about the water cycle",
    )
    
    print(f"\n📊 Summary Statistics:")
    print(f"   Original: {summary_result.metadata.get('original_length', 0):,} chars")
    print(f"   Summary: {len(summary_result.summary):,} chars")
    print(f"   Reduction: {summary_result.reduction_ratio:.1f}x")
    print(f"   Chunks processed: {summary_result.chunks_processed}")
    print(f"   Total time: {summary_result.metadata.get('total_time_s', 0):.1f}s")
    print(f"   Map phase: {summary_result.metadata.get('map_time_s', 0):.1f}s")
    print(f"   Reduce phase: {summary_result.metadata.get('reduce_time_s', 0):.1f}s")
    
    print("\n" + "-" * 70)
    print("SUMMARY:")
    print("-" * 70)
    print(summary_result.summary)
    print("-" * 70)
    
    # =========================================================================
    # Step 3: Compare with Refine Strategy (optional - smaller text)
    # =========================================================================
    print("\n" + "=" * 70)
    print("🔄 Step 3: Compare with Refine Strategy (first 10K chars)")
    print("=" * 70 + "\n")
    
    refine_summarizer = RefineSummarizer(
        model=model,
        chunk_size=4000,
        show_progress=True,
    )
    
    # Use subset for speed comparison
    subset = markdown_content[:10000]
    
    refine_result = await refine_summarizer.summarize(
        subset,
        context="Wikipedia article excerpt about water cycle",
    )
    
    print(f"\n📊 Refine Summary (10K chars input):")
    print(f"   Chunks: {refine_result.chunks_processed}")
    print(f"   Summary: {len(refine_result.summary)} chars")
    print(f"\nPreview: {refine_result.summary[:300]}...")
    
    print("\n" + "=" * 70)
    print("✅ PDF Summarization complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
