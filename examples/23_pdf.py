"""
Example: PDF Capability

Agent extracts and summarizes content from PDF documents.

Usage:
    uv run python examples/24_pdf.py

Requires: uv add pypdf reportlab
"""

import asyncio
import tempfile
from pathlib import Path

from config import get_model, settings

from agenticflow import Agent, Flow
from agenticflow.capabilities import PDF


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf = PDF(allowed_paths=[tmpdir])

        if not pdf._has_reportlab:
            print("Install reportlab to run this example: uv add reportlab pypdf")
            return

        # Create a sample PDF
        pdf_path = Path(tmpdir) / "report.pdf"
        content = """Q4 2024 Financial Summary

Revenue: $4.2M (+15% YoY)

Operating costs decreased by 8% due to automation.

Key growth drivers: Enterprise segment (+22%), API services (+18%).

Outlook: Projecting 20% growth in Q1 2025."""
        pdf._create_pdf(pdf_path, content, title="Q4 2024 Financial Summary")

        model = get_model()
        analyst = Agent(
            name="DocAnalyst",
            model=model,
            instructions="You analyze PDF documents and provide insights.",
            capabilities=[pdf],
        )

        flow = Flow(
            name="doc_analysis",
            agents=[analyst],
            verbose=settings.verbose_level,
        )

        result = await flow.run(
            f"Read the financial report at {pdf_path} and summarize the key metrics and outlook."
        )
        print(f"\n{result.output}")


if __name__ == "__main__":
    asyncio.run(main())
