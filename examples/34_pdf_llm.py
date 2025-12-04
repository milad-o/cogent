"""
Example: High-Performance PDF to Markdown Loader

Demonstrates PDFMarkdownLoader with CPU parallelization for converting
PDFs to Markdown format optimized for LLM/RAG applications.

Key features:
- True CPU parallelization with ProcessPoolExecutor
- Batch processing for memory efficiency
- Detailed processing metrics and tracking
- Markdown output saved for inspection

Usage:
    uv run python examples/32_pdf_llm.py

Requires: uv add pymupdf4llm pymupdf structlog
"""

import asyncio
from pathlib import Path

from agenticflow.document.loaders import PDFMarkdownLoader


async def main() -> None:
    # Example data directory
    data_dir = Path(__file__).parent / "data"
    output_dir = data_dir / "pdf_output"
    output_dir.mkdir(exist_ok=True)

    # Use the 58-page IRS Publication 15 (Employer's Tax Guide)
    pdf_path = data_dir / "financial_report.pdf"

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        print("Download from: https://www.irs.gov/pub/irs-pdf/p15.pdf")
        return

    # Configure loader with parallelization
    loader = PDFMarkdownLoader(
        max_workers=4,  # CPU workers for parallel processing
        batch_size=10,  # Pages per batch
    )

    print(f"Processing: {pdf_path.name}")
    print(f"Output dir: {output_dir}\n")

    # Load PDF - returns result with documents and metrics
    result = await loader.load(pdf_path)

    # Display processing metrics
    print("=" * 60)
    print("PROCESSING METRICS")
    print("=" * 60)
    print(f"Total pages:      {result.total_pages}")
    print(f"Successful:       {result.successful_pages}")
    print(f"Failed:           {result.failed_pages}")
    print(f"Empty pages:      {result.empty_pages}")
    print(f"Success rate:     {result.success_rate:.0%}")
    print(f"Total time:       {result.total_time_ms:.0f}ms")
    if result.total_pages > 0 and result.total_time_ms > 0:
        pages_per_sec = result.total_pages / (result.total_time_ms / 1000)
        print(f"Pages/second:     {pages_per_sec:.1f}")
    print("=" * 60)

    # Combine all page content into single Markdown
    combined_content = "\n\n---\n\n".join(
        f"## Page {pr.page_number}\n\n{pr.content}"
        for pr in result.page_results
        if pr.content.strip()
    )

    # Save combined Markdown output
    combined_md_path = output_dir / f"{pdf_path.stem}.md"
    combined_md_path.write_text(combined_content)
    print(f"\nSaved combined MD: {combined_md_path}")
    print(f"  Size: {combined_md_path.stat().st_size:,} bytes")

    # Save individual page Markdown files
    pages_dir = output_dir / f"{pdf_path.stem}_pages"
    pages_dir.mkdir(exist_ok=True)

    for page_result in result.page_results:
        if page_result.content.strip():
            page_md_path = pages_dir / f"page_{page_result.page_number:03d}.md"
            page_md_path.write_text(page_result.content)

    saved_pages = len(list(pages_dir.glob("*.md")))
    print(f"\nSaved individual pages: {pages_dir}/")
    print(f"  Pages: {saved_pages}")

    # Show preview of first page
    if result.page_results and result.page_results[0].content.strip():
        first_page = result.page_results[0].content
        preview_lines = first_page.split("\n")[:30]
        print("\n" + "=" * 60)
        print("FIRST PAGE PREVIEW (first 30 lines)")
        print("=" * 60)
        print("\n".join(preview_lines))
        if len(first_page.split("\n")) > 30:
            print("...")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
