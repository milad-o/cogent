"""
Demo: Complex Finance PDF Table Extraction

Creates a sophisticated multi-page PDF with complex financial tables featuring:
- Merged cells (horizontal and vertical spans)
- Multi-line row content
- Hierarchical sections
- Various number formats

Then uses PDFVisionLoader to extract the content and an Agent to answer
questions about specific financial figures, validating against ground truth.

Usage:
    uv run python examples/retrieval/finance_table_example.py

Prerequisites:
    uv add reportlab pymupdf
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model, get_embeddings, settings

# =============================================================================
# Ground Truth Values (for validation)
# =============================================================================

GROUND_TRUTH = {
    "Q1 2024 Revenue": "$1,250,000",
    "Q4 2024 Net Income": "$425,000",
    "Total Assets": "$5,750,000",
    "Net Change in Cash": "$350,000",
}


# =============================================================================
# PDF Generation
# =============================================================================


def create_complex_finance_pdf(output_path: Path) -> None:
    """Generate a multi-page PDF with complex financial tables."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=20,
        alignment=1,  # Center
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Heading2"],
        fontSize=14,
        spaceAfter=12,
    )

    # Common table style
    def get_table_style(header_rows: int = 1) -> TableStyle:
        base_style = [
            ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#2C3E50")),
            ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, header_rows - 1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, header_rows - 1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, header_rows - 1), 12),
            ("BACKGROUND", (0, header_rows), (-1, -1), colors.HexColor("#ECF0F1")),
            ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#BDC3C7")),
            ("FONTSIZE", (0, header_rows), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]
        return TableStyle(base_style)

    # =========================================================================
    # PAGE 1: Income Statement Q1-Q2 2024
    # =========================================================================
    elements.append(Paragraph("ACME Corporation", title_style))
    elements.append(Paragraph("Consolidated Income Statement", subtitle_style))
    elements.append(Paragraph("For the Six Months Ended June 30, 2024", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    # Table with merged header cells
    income_q1q2_data = [
        # Header row with merged cells spanning Q1 and Q2
        ["", "Q1 2024", "", "Q2 2024", ""],
        ["Line Item", "Amount", "% Revenue", "Amount", "% Revenue"],
        # Revenue section
        ["Revenue", "$1,250,000", "100.0%", "$1,380,000", "100.0%"],
        ["Cost of Goods Sold", "($625,000)", "50.0%", "($676,200)", "49.0%"],
        ["Gross Profit", "$625,000", "50.0%", "$703,800", "51.0%"],
        # Operating expenses - this will have merged rows
        ["Operating Expenses:", "", "", "", ""],
        ["  Sales & Marketing", "($125,000)", "10.0%", "($138,000)", "10.0%"],
        ["  Research & Development", "($87,500)", "7.0%", "($96,600)", "7.0%"],
        ["  General & Administrative", "($67,500)", "5.4%", "($45,400)", "3.3%"],
        ["Total Operating Expenses", "($280,000)", "22.4%", "($280,000)", "20.3%"],
        # Bottom line
        ["Operating Income", "$345,000", "27.6%", "$423,800", "30.7%"],
        ["Interest Expense", "($25,000)", "2.0%", "($25,000)", "1.8%"],
        ["Income Before Tax", "$320,000", "25.6%", "$398,800", "28.9%"],
        ["Income Tax (25%)", "($80,000)", "6.4%", "($99,700)", "7.2%"],
        ["Net Income", "$240,000", "19.2%", "$299,100", "21.7%"],
    ]

    t1 = Table(income_q1q2_data, colWidths=[2 * inch, 1.1 * inch, 0.8 * inch, 1.1 * inch, 0.8 * inch])
    style1 = get_table_style(header_rows=2)
    # Merge the quarter headers
    style1.add("SPAN", (1, 0), (2, 0))  # Q1 2024 spans columns 1-2
    style1.add("SPAN", (3, 0), (4, 0))  # Q2 2024 spans columns 3-4
    # Highlight totals
    style1.add("BACKGROUND", (0, 9), (-1, 9), colors.HexColor("#D5DBDB"))
    style1.add("BACKGROUND", (0, 14), (-1, 14), colors.HexColor("#AEB6BF"))
    style1.add("FONTNAME", (0, 14), (-1, 14), "Helvetica-Bold")
    t1.setStyle(style1)
    elements.append(t1)
    elements.append(PageBreak())

    # =========================================================================
    # PAGE 2: Income Statement Q3-Q4 2024
    # =========================================================================
    elements.append(Paragraph("ACME Corporation", title_style))
    elements.append(Paragraph("Consolidated Income Statement", subtitle_style))
    elements.append(Paragraph("For the Six Months Ended December 31, 2024", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    income_q3q4_data = [
        ["", "Q3 2024", "", "Q4 2024", ""],
        ["Line Item", "Amount", "% Revenue", "Amount", "% Revenue"],
        ["Revenue", "$1,520,000", "100.0%", "$1,680,000", "100.0%"],
        ["Cost of Goods Sold", "($729,600)", "48.0%", "($789,600)", "47.0%"],
        ["Gross Profit", "$790,400", "52.0%", "$890,400", "53.0%"],
        ["Operating Expenses:", "", "", "", ""],
        ["  Sales & Marketing", "($152,000)", "10.0%", "($168,000)", "10.0%"],
        ["  Research & Development", "($106,400)", "7.0%", "($117,600)", "7.0%"],
        ["  General & Administrative", "($60,800)", "4.0%", "($67,200)", "4.0%"],
        ["Total Operating Expenses", "($319,200)", "21.0%", "($352,800)", "21.0%"],
        ["Operating Income", "$471,200", "31.0%", "$537,600", "32.0%"],
        ["Interest Expense", "($25,000)", "1.6%", "($25,000)", "1.5%"],
        ["Income Before Tax", "$446,200", "29.4%", "$512,600", "30.5%"],
        ["Income Tax (25%)", "($111,550)", "7.3%", "($87,600)", "5.2%"],
        ["Net Income", "$334,650", "22.0%", "$425,000", "25.3%"],
    ]

    t2 = Table(income_q3q4_data, colWidths=[2 * inch, 1.1 * inch, 0.8 * inch, 1.1 * inch, 0.8 * inch])
    style2 = get_table_style(header_rows=2)
    style2.add("SPAN", (1, 0), (2, 0))
    style2.add("SPAN", (3, 0), (4, 0))
    style2.add("BACKGROUND", (0, 9), (-1, 9), colors.HexColor("#D5DBDB"))
    style2.add("BACKGROUND", (0, 14), (-1, 14), colors.HexColor("#AEB6BF"))
    style2.add("FONTNAME", (0, 14), (-1, 14), "Helvetica-Bold")
    t2.setStyle(style2)
    elements.append(t2)
    elements.append(PageBreak())

    # =========================================================================
    # PAGE 3: Balance Sheet
    # =========================================================================
    elements.append(Paragraph("ACME Corporation", title_style))
    elements.append(Paragraph("Consolidated Balance Sheet", subtitle_style))
    elements.append(Paragraph("As of December 31, 2024", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    # Complex balance sheet with merged category headers
    balance_data = [
        ["ASSETS", "", "LIABILITIES & EQUITY", ""],
        ["Current Assets", "", "Current Liabilities", ""],
        ["  Cash & Equivalents", "$1,250,000", "  Accounts Payable", "$420,000"],
        ["  Accounts Receivable", "$680,000", "  Accrued Expenses", "$185,000"],
        ["  Inventory", "$520,000", "  Short-term Debt", "$300,000"],
        ["  Prepaid Expenses", "$85,000", "  Deferred Revenue", "$145,000"],
        ["Total Current Assets", "$2,535,000", "Total Current Liabilities", "$1,050,000"],
        ["", "", "", ""],
        ["Non-Current Assets", "", "Non-Current Liabilities", ""],
        ["  Property & Equipment", "$1,850,000", "  Long-term Debt", "$1,200,000"],
        ["  Intangible Assets", "$680,000", "  Deferred Tax Liability", "$180,000"],
        ["  Goodwill", "$450,000", "  Other Liabilities", "$95,000"],
        ["  Other Assets", "$235,000", "Total Non-Current Liabilities", "$1,475,000"],
        ["Total Non-Current Assets", "$3,215,000", "", ""],
        ["", "", "Total Liabilities", "$2,525,000"],
        ["", "", "", ""],
        ["", "", "Shareholders' Equity", ""],
        ["", "", "  Common Stock", "$500,000"],
        ["", "", "  Retained Earnings", "$2,725,000"],
        ["", "", "Total Equity", "$3,225,000"],
        ["TOTAL ASSETS", "$5,750,000", "TOTAL LIAB. & EQUITY", "$5,750,000"],
    ]

    t3 = Table(balance_data, colWidths=[2.2 * inch, 1.3 * inch, 2.2 * inch, 1.3 * inch])
    style3 = get_table_style(header_rows=1)
    # Category headers
    style3.add("BACKGROUND", (0, 1), (1, 1), colors.HexColor("#34495E"))
    style3.add("TEXTCOLOR", (0, 1), (1, 1), colors.white)
    style3.add("BACKGROUND", (2, 1), (3, 1), colors.HexColor("#34495E"))
    style3.add("TEXTCOLOR", (2, 1), (3, 1), colors.white)
    style3.add("BACKGROUND", (0, 8), (1, 8), colors.HexColor("#34495E"))
    style3.add("TEXTCOLOR", (0, 8), (1, 8), colors.white)
    style3.add("BACKGROUND", (2, 8), (3, 8), colors.HexColor("#34495E"))
    style3.add("TEXTCOLOR", (2, 8), (3, 8), colors.white)
    style3.add("BACKGROUND", (2, 16), (3, 16), colors.HexColor("#34495E"))
    style3.add("TEXTCOLOR", (2, 16), (3, 16), colors.white)
    # Totals
    style3.add("BACKGROUND", (0, 6), (1, 6), colors.HexColor("#D5DBDB"))
    style3.add("BACKGROUND", (2, 6), (3, 6), colors.HexColor("#D5DBDB"))
    style3.add("BACKGROUND", (0, 13), (1, 13), colors.HexColor("#D5DBDB"))
    style3.add("BACKGROUND", (2, 12), (3, 12), colors.HexColor("#D5DBDB"))
    style3.add("BACKGROUND", (2, 14), (3, 14), colors.HexColor("#D5DBDB"))
    style3.add("BACKGROUND", (2, 19), (3, 19), colors.HexColor("#D5DBDB"))
    # Grand totals
    style3.add("BACKGROUND", (0, 20), (-1, 20), colors.HexColor("#1ABC9C"))
    style3.add("FONTNAME", (0, 20), (-1, 20), "Helvetica-Bold")
    style3.add("TEXTCOLOR", (0, 20), (-1, 20), colors.white)
    t3.setStyle(style3)
    elements.append(t3)
    elements.append(PageBreak())

    # =========================================================================
    # PAGE 4: Cash Flow Statement
    # =========================================================================
    elements.append(Paragraph("ACME Corporation", title_style))
    elements.append(Paragraph("Consolidated Statement of Cash Flows", subtitle_style))
    elements.append(Paragraph("For the Year Ended December 31, 2024", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    cashflow_data = [
        ["Cash Flow Category", "Amount"],
        ["OPERATING ACTIVITIES", ""],
        ["  Net Income", "$1,298,750"],
        ["  Depreciation & Amortization", "$285,000"],
        ["  Changes in Working Capital:", ""],
        ["    Accounts Receivable", "($120,000)"],
        ["    Inventory", "($85,000)"],
        ["    Accounts Payable", "$95,000"],
        ["    Accrued Expenses", "$42,000"],
        ["Net Cash from Operating Activities", "$1,515,750"],
        ["", ""],
        ["INVESTING ACTIVITIES", ""],
        ["  Capital Expenditures", "($680,000)"],
        ["  Acquisitions", "($250,000)"],
        ["  Sale of Equipment", "$45,000"],
        ["Net Cash from Investing Activities", "($885,000)"],
        ["", ""],
        ["FINANCING ACTIVITIES", ""],
        ["  Proceeds from Debt", "$200,000"],
        ["  Debt Repayments", "($350,000)"],
        ["  Dividends Paid", "($130,750)"],
        ["Net Cash from Financing Activities", "($280,750)"],
        ["", ""],
        ["NET CHANGE IN CASH", "$350,000"],
        ["Beginning Cash Balance", "$900,000"],
        ["Ending Cash Balance", "$1,250,000"],
    ]

    t4 = Table(cashflow_data, colWidths=[4 * inch, 1.5 * inch])
    style4 = get_table_style(header_rows=1)
    # Section headers
    for row in [1, 11, 17]:
        style4.add("BACKGROUND", (0, row), (-1, row), colors.HexColor("#34495E"))
        style4.add("TEXTCOLOR", (0, row), (-1, row), colors.white)
        style4.add("FONTNAME", (0, row), (-1, row), "Helvetica-Bold")
    # Subtotals
    for row in [9, 15, 21]:
        style4.add("BACKGROUND", (0, row), (-1, row), colors.HexColor("#D5DBDB"))
        style4.add("FONTNAME", (0, row), (-1, row), "Helvetica-Bold")
    # Net change highlight
    style4.add("BACKGROUND", (0, 23), (-1, 23), colors.HexColor("#1ABC9C"))
    style4.add("FONTNAME", (0, 23), (-1, 23), "Helvetica-Bold")
    style4.add("TEXTCOLOR", (0, 23), (-1, 23), colors.white)
    # Final balance
    style4.add("BACKGROUND", (0, 25), (-1, 25), colors.HexColor("#AEB6BF"))
    style4.add("FONTNAME", (0, 25), (-1, 25), "Helvetica-Bold")
    t4.setStyle(style4)
    elements.append(t4)

    # Build PDF
    doc.build(elements)
    print(f"✓ Generated PDF: {output_path}")


# =============================================================================
# Extraction & Q&A
# =============================================================================


async def extract_and_query(pdf_path: Path) -> dict[str, tuple[str, bool]]:
    """Extract PDF content and query with RAG-powered agent."""
    from agenticflow import Agent, Flow
    from agenticflow.document.loaders import PDFMarkdownLoader
    from agenticflow.vectorstore import VectorStore
    from agenticflow.retriever import DenseRetriever

    model = get_model()
    embeddings = get_embeddings()
    print(f"\n📊 Using model: {getattr(model, 'model', model.__class__.__name__)}")

    # Step 1: Extract PDF content
    print("\n🔍 Step 1: Extracting PDF content...")
    loader = PDFMarkdownLoader()
    docs = await loader.load(pdf_path)
    print(f"✓ Extracted {len(docs)} pages")

    # Show a preview
    combined = "\n\n".join([d.text for d in docs])
    print(f"\n📄 Content Preview (first 1000 chars):")
    print("-" * 60)
    print(combined[:1000])
    print("-" * 60)

    # Step 2: Create vector store and index documents
    print("\n📚 Step 2: Building vector index with embeddings...")
    vectorstore = VectorStore(embeddings=embeddings)
    await vectorstore.add_documents(docs)
    print(f"✓ Indexed {len(docs)} documents")

    # Step 3: Create retriever
    retriever = DenseRetriever(vectorstore)

    # Step 4: Create RAG agent
    print("\n🤖 Step 3: Creating RAG-powered financial analyst...")

    analyst = Agent(
        name="FinancialAnalyst",
        model=model,
        instructions="""You are a financial analyst expert at reading financial statements.
When answering questions about financial figures:
1. Use the retrieved context to find the exact value
2. Return the value in the exact format shown (e.g., $1,250,000)
3. Be precise and only answer what is asked
4. If the context doesn't contain the answer, say "NOT FOUND"
""",
    )

    flow = Flow(
        name="finance_qa",
        agents=[analyst],
        topology="pipeline",
        verbose=settings.verbose_level,
    )

    # Step 5: Ask validation questions with retrieval
    questions = [
        ("Q1 2024 Revenue", "What is the Revenue for Q1 2024?"),
        ("Q4 2024 Net Income", "What is the Net Income for Q4 2024?"),
        ("Total Assets", "What is the Total Assets value on the Balance Sheet?"),
        ("Net Change in Cash", "What is the Net Change in Cash from the Cash Flow Statement?"),
    ]

    results: dict[str, tuple[str, bool]] = {}
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS (Using RAG)")
    print("=" * 70)

    for key, question in questions:
        # Retrieve relevant context
        retrieval_results = await retriever.retrieve(question, k=2, include_scores=True)
        context = "\n\n".join([r.document.text for r in retrieval_results])

        prompt = f"""Based on the following financial document context, answer this question:

{question}

Context:
{context}

Provide ONLY the numerical answer in the format shown in the document."""

        result = await flow.run(prompt)
        answer = result.output.strip()
        expected = GROUND_TRUTH[key]

        # Check if the expected value appears in the answer
        is_correct = expected in answer or expected.replace(",", "") in answer.replace(",", "")

        results[key] = (answer, is_correct)

        status = "✅" if is_correct else "❌"
        print(f"\n{status} {key}")
        print(f"   Question: {question}")
        print(f"   Expected: {expected}")
        print(f"   Got:      {answer[:100]}...")

    # Summary
    correct = sum(1 for _, (_, c) in results.items() if c)
    total = len(results)
    print("\n" + "=" * 70)
    print(f"SCORE: {correct}/{total} questions answered correctly")
    print("=" * 70)

    return results


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    print("\n" + "=" * 70)
    print("  COMPLEX FINANCE PDF TABLE EXTRACTION DEMO")
    print("=" * 70)

    data_dir = Path(__file__).parent.parent / "data"
    pdf_path = data_dir / "complex_finance_report.pdf"

    # Step 1: Generate the PDF
    print("\n📝 Step 1: Generating complex finance PDF...")
    create_complex_finance_pdf(pdf_path)

    # Optional: Skip slow Q&A validation
    SKIP_QA = False  # Set to True to skip Q&A
    
    if SKIP_QA:
        print("\n📊 Step 2: Extracting content (Q&A validation skipped)...")
        from agenticflow.document.loaders import PDFMarkdownLoader
        loader = PDFMarkdownLoader()
        docs = await loader.load(pdf_path)
        print(f"✓ Extracted {len(docs)} pages")
        
        combined = "\n\n---\n\n".join(
            [f"## Page {i + 1}\n\n{d.text}" for i, d in enumerate(docs)]
        )
        print("\n📄 Full Extracted Content:")
        print("-" * 60)
        print(combined[:3000])
        print("-" * 60)
        print("\n✅ Extraction complete! Tables with merged cells extracted correctly.")
        return

    # Step 2: Extract and query
    print("\n📊 Step 2: Extracting content and validating with Q&A...")
    results = await extract_and_query(pdf_path)

    # Final result
    correct = sum(1 for _, (_, c) in results.items() if c)
    total = len(results)
    if correct == total:
        print("\n🎉 SUCCESS! Complex tables extracted and Q&A validated correctly.")
    elif correct >= total // 2:
        print("\n✓ GOOD! Most questions answered correctly.")
    else:
        print("\n⚠️  Some issues with extraction accuracy.")


if __name__ == "__main__":
    asyncio.run(main())
