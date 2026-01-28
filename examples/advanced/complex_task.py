"""
Complex Task: Multi-Tool Workflow with Structured Output

Demonstrates a single agent using multiple tools to:
1. Gather data from different sources
2. Analyze the findings
3. Produce a structured report

This showcases:
- Tool composition (multiple tools working together)
- Structured output (Pydantic response) for the final result
- Detailed observability showing tool calls

Key insight: `output=Schema` only constrains the FINAL response.
The agent can use any tools freely in between.

Usage:
    uv run python examples/advanced/complex_task.py

Observer levels (from quiet to verbose):
    Observer(level="quiet")    - No output
    Observer(level="result")   - Final result only
    Observer(level="progress") - Key milestones (default)
    Observer(level="detailed") - Tool calls and timing
    Observer(level="debug")    - Everything
"""

import asyncio
from datetime import datetime

from pydantic import BaseModel, Field

from cogent import Agent, tool
from cogent.observability import Observer


# =============================================================================
# Structured Output Schema
# =============================================================================


class ResearchFinding(BaseModel):
    """A single research finding."""

    title: str = Field(description="Title of the finding")
    summary: str = Field(description="2-3 sentence summary")
    source: str = Field(description="Source or data origin")
    relevance: str = Field(description="Why this is relevant")


class ResearchReport(BaseModel):
    """Complete research report."""

    topic: str = Field(description="Research topic")
    executive_summary: str = Field(description="1 paragraph executive summary")
    key_findings: list[ResearchFinding] = Field(description="3-5 key findings")
    analysis: str = Field(description="Analysis of the findings")
    recommendations: list[str] = Field(description="2-3 actionable recommendations")
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# Research Tools (simulated for demo - no API keys needed)
# =============================================================================


@tool
def search_market_data(query: str) -> str:
    """Search for market data and industry statistics.

    Args:
        query: What market data to search for.

    Returns:
        Market data and statistics.
    """
    return f"""
    Market Data for: {query}
    
    - Global AI agent market size: $5.2B in 2025, projected $18.7B by 2028
    - Enterprise adoption rate: 45% of Fortune 500 using AI agents
    - Top use cases: customer service (34%), data analysis (28%), workflow automation (22%)
    - Average ROI: 3.2x within first year of deployment
    - Key players: Microsoft, Google, Anthropic, OpenAI, Salesforce
    """


@tool
def search_case_studies(industry: str) -> str:
    """Search for enterprise case studies and success stories.

    Args:
        industry: Industry sector to search case studies for.

    Returns:
        Relevant case studies.
    """
    return f"""
    Case Studies in {industry}:
    
    1. Global Bank Corp: Deployed AI agents for fraud detection
       - Result: 67% reduction in false positives, $12M annual savings
       
    2. TechManufacturing Inc: Automated supply chain monitoring
       - Result: 40% faster issue detection, 23% cost reduction
       
    3. HealthCare Systems: Patient scheduling AI agents
       - Result: 89% appointment fill rate, 50% less staff time
    """


# =============================================================================
# Main Demo
# =============================================================================


async def main():
    print("=" * 60)
    print("COMPLEX TASK: Research → Analysis → Report")
    print("=" * 60)

    # Use detailed observer to see tool calls
    observer = Observer(level="detailed")

    # Single agent with multiple tools - output schema only applies to final response
    agent = Agent(
        name="Researcher",
        model="gpt-4o-mini",
        instructions="""You are an expert research analyst. Your workflow:
        
        1. Use search_market_data to find market statistics
        2. Use search_case_studies to find real-world examples
        3. Return your findings as a structured ResearchReport
        
        Always use BOTH tools before writing your report.""",
        tools=[search_market_data, search_case_studies],
        output=ResearchReport,  # Only constrains the FINAL response
        observer=observer,
    )

    topic = "AI agents in enterprise automation"

    print(f"\n📋 Topic: {topic}")
    print("-" * 60)

    result = await agent.run(f"Create a research report on: {topic}")

    # Display the structured report
    structured = result.content
    if structured.valid:
        report = structured.data
        print("\n" + "=" * 60)
        print("📊 RESEARCH REPORT")
        print("=" * 60)

        print(f"\n📌 Topic: {report.topic}")
        print(f"\n📝 Executive Summary:\n{report.executive_summary}")

        print("\n🔍 Key Findings:")
        for i, finding in enumerate(report.key_findings, 1):
            print(f"\n  {i}. {finding.title}")
            print(f"     {finding.summary}")
            print(f"     Source: {finding.source}")
            print(f"     Relevance: {finding.relevance}")

        print(f"\n📈 Analysis:\n{report.analysis}")

        print("\n💡 Recommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")

        print(f"\n⏰ Generated: {report.generated_at}")
    else:
        print(f"\n❌ Parsing error: {structured.error}")
        print(f"Raw output: {result.output}")

    # Show metadata
    print("\n" + "-" * 60)
    print("📊 Execution Metadata:")
    tokens = result.metadata.tokens
    print(f"  • Total tokens: {tokens.total_tokens if tokens else 'N/A'}")
    print(f"  • Tool calls: {len(result.tool_calls)}")
    print(f"  • Duration: {result.metadata.duration:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
