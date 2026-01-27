"""
Technical Report Table Extraction with PDFVisionLoader

Uses PDFVisionLoader to extract tables and data from GPT-4's technical
report (real research paper with performance tables and charts).

Usage: uv run python examples/retrieval/finance_table_example.py
Prerequisites: uv add pymupdf
"""

import asyncio
from pathlib import Path


async def main() -> None:
    """Load GPT-4 technical report and query tables with PDFVisionLoader."""
    from cogent import Agent, create_chat
    from cogent.documents import PDFVisionLoader

    # Load GPT-4 technical report from examples/data
    report_path = Path(__file__).parent.parent / "data" / "gpt4_technical_report.pdf"

    if not report_path.exists():
        print(f"Error: {report_path} not found")
        print("Download it with: curl https://arxiv.org/pdf/2304.03442 -o examples/data/gpt4_technical_report.pdf")
        return

    print(f"Loading technical report: {report_path.name}")

    # Load with vision (first 10 pages contain key tables)
    print("\nLoading with PDFVisionLoader...")
    loader = PDFVisionLoader(model=create_chat("gpt-4o-mini"))
    documents = await loader.load(report_path, max_pages=10)
    print(f"✓ Loaded {len(documents)} pages")

    # Create agent
    agent = Agent(
        name="TechnicalAnalyst",
        model="gpt4",
        instructions="Extract exact data from technical reports. Be precise with numbers and benchmarks.",
    )

    # Sample queries about GPT-4's performance data
    print("\nQuerying data from GPT-4 Technical Report:")

    questions = [
        "What was GPT-4's score on the Uniform Bar Exam?",
        "What is GPT-4's MMLU (5-shot) accuracy?",
        "How does GPT-4 compare to GPT-3.5 on HumanEval coding tasks?",
        "What languages were tested in the multilingual benchmarks?",
    ]

    context = "\n\n".join(doc.text for doc in documents)

    for question in questions:
        prompt = f"Based on this technical report:\n\n{context}\n\nQuestion: {question}\nProvide a concise answer with exact figures or data points."

        response = await agent.run(prompt)
        answer = response.unwrap().strip()

        print(f"  • {question}")
        print(f"    → {answer}\n")

    print("✓ Analysis complete")


if __name__ == "__main__":
    asyncio.run(main())
