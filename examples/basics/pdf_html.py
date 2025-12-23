"""Example demonstrating PDF HTML extraction for complex tables.

This example shows how PDFHTMLLoader preserves table structure by extracting
native HTML directly from PDFs, which is superior to markdown for complex tables.
"""

import asyncio
from pathlib import Path

from agenticflow.document.loaders.handlers import PDFHTMLLoader


async def main() -> None:
    """Demonstrate PDF HTML extraction."""
    
    # Create loader
    loader = PDFHTMLLoader(
        max_workers=4,
        batch_size=10,
        show_progress=True,
        verbose=False,
    )
    
    # Path to your PDF (using existing sample PDF)
    pdf_path = Path("examples/data/financial_report.pdf")
    
    if not pdf_path.exists():
        # Try alternative
        pdf_path = Path("examples/data/wikipedia_water_cycle.pdf")
    
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        print("Create a sample PDF or update the path in this script.")
        return
    
    print(f"📄 Loading PDF as HTML: {pdf_path.name}")
    print("=" * 60)
    
    # Load with tracking to see metrics
    result = await loader.load(pdf_path, tracking=True)
    
    print(f"\n📊 Processing Results:")
    print(f"   Total pages: {result.total_pages}")
    print(f"   Successful: {result.successful_pages}")
    print(f"   Failed: {result.failed_pages}")
    print(f"   Success rate: {result.success_rate:.1f}%")
    print(f"   Processing time: {result.total_time_ms:.0f}ms")
    
    # Show table statistics
    total_tables = sum(pr.tables_count for pr in result.page_results)
    print(f"   Tables found: {total_tables}")
    
    # Get documents
    docs = result.documents
    print(f"\n📄 Loaded {len(docs)} document(s)")
    
    # Show sample content from first page
    if docs:
        print(f"\n📝 First page preview (first 500 chars):")
        print("=" * 60)
        print(docs[0].text[:500])
        print("=" * 60)
        
        # Check for tables
        if "<table" in docs[0].text:
            print("\n✅ Tables detected in HTML output!")
            print("   Native HTML preserves complex table structures.")
        
    # Save options
    output_dir = Path("examples/data/pdf_output")
    output_dir.mkdir(exist_ok=True)
    
    # Save as single HTML file
    output_file = loader.save(
        output_dir / "output.html",
        mode="single",
        include_page_numbers=True,
    )
    print(f"\n💾 Saved to: {output_file}")
    
    # Save as JSON with metadata
    json_file = loader.save(
        output_dir / "output.json",
        mode="json",
    )
    print(f"💾 Saved JSON to: {json_file}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
