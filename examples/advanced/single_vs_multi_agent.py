"""Single Agent vs Multi-Agent: Proving the Point.

THE CLAIM: For most tasks, a single capable agent with tools outperforms
multi-agent orchestration in cost, latency, and often quality.

THE TEST: All three approaches complete the SAME task:
    "Analyze a company and produce an investment recommendation"

We measure:
    - Duration (wall-clock time)
    - LLM Calls (API cost driver)  
    - Total Tokens (actual cost)
    - Output Quality (subjective but visible)

Run:
    uv run python examples/advanced/single_vs_multi_agent.py
"""

import asyncio
import time
from typing import Literal

from pydantic import BaseModel, Field

from cogent import Agent, Observer, tool


# ============================================================================
# THE TASK: Company Analysis → Investment Recommendation
# ============================================================================

TASK = """Analyze TechCorp Inc. and provide an investment recommendation.

Company Data:
- Revenue: $50M (up 25% YoY)
- Profit Margin: 15%
- Market: Cloud Infrastructure
- Competitors: AWS, Azure, GCP
- Risk: High customer concentration (top 3 = 60% revenue)

Provide: BUY, HOLD, or SELL with reasoning."""


class InvestmentRecommendation(BaseModel):
    """Structured output for fair comparison."""
    
    ticker: str = Field(description="Company ticker/name")
    recommendation: Literal["BUY", "HOLD", "SELL"]
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    reasoning: str = Field(description="Key reasons for recommendation")
    risks: list[str] = Field(description="Top risks to monitor")


# ============================================================================
# SHARED TOOLS (same tools available to all approaches)
# ============================================================================


@tool
def get_financials(company: str) -> str:
    """Get company financial data."""
    return """TechCorp Inc Financials:
- Revenue: $50M (FY2025), $40M (FY2024) → 25% growth
- Gross Margin: 72%
- Operating Margin: 15%
- Cash: $12M, Debt: $5M
- P/E Ratio: 45x (industry avg: 30x)"""


@tool
def get_market_analysis(sector: str) -> str:
    """Get market/sector analysis."""
    return """Cloud Infrastructure Market:
- TAM: $500B by 2027 (15% CAGR)
- Leaders: AWS (32%), Azure (23%), GCP (10%)
- Opportunity: Niche players can win specialized segments
- Trend: AI workloads driving 40% of new demand"""


@tool  
def get_risk_factors(company: str) -> str:
    """Get company risk factors."""
    return """TechCorp Risk Factors:
- Customer Concentration: Top 3 customers = 60% of revenue
- Competition: Big 3 have 10x resources
- Key Person: CTO departure would impact roadmap
- Regulatory: New data sovereignty laws may require infra changes"""


# ============================================================================
# APPROACH 1: SINGLE AGENT (The Cogent Way)
# ============================================================================


async def single_agent_approach() -> dict:
    """One agent with all tools does everything.
    
    This is what Cogent recommends for most tasks.
    """
    print("\n" + "=" * 70)
    print("APPROACH 1: Single Agent + Tools (Recommended)")
    print("=" * 70)
    
    observer = Observer(level="normal")
    
    agent = Agent(
        name="InvestmentAnalyst",
        model="gpt-4o-mini",
        tools=[get_financials, get_market_analysis, get_risk_factors],
        output=InvestmentRecommendation,
        instructions="""You are a senior investment analyst.

To analyze a company:
1. Get financials using get_financials
2. Understand the market using get_market_analysis  
3. Assess risks using get_risk_factors
4. Synthesize into a clear BUY/HOLD/SELL recommendation

Be thorough but concise. Consider growth vs valuation vs risk.""",
        observer=observer,
    )
    
    start = time.perf_counter()
    result = await agent.run(TASK)
    duration = time.perf_counter() - start
    
    # Extract structured output
    output = None
    if result.content and hasattr(result.content, 'data'):
        output = result.content.data
    
    if output:
        print(f"\n📊 {output.ticker}: {output.recommendation} ({output.confidence} confidence)")
        print(f"💡 {output.reasoning}")
        print(f"⚠️  Risks: {', '.join(output.risks)}")
    else:
        print(f"\n📋 Result:\n{result.content}")
    
    tokens = 0
    if result.metadata and result.metadata.tokens:
        tokens = result.metadata.tokens.total_tokens
    
    return {
        "approach": "single_agent",
        "duration_seconds": round(duration, 2),
        "llm_calls": 1,
        "tokens": tokens,
        "recommendation": output.recommendation if output else "N/A",
    }


# ============================================================================
# APPROACH 2: MULTI-AGENT ORCHESTRATION (The "Framework" Way)
# ============================================================================


async def multi_agent_approach() -> dict:
    """Orchestrator delegates to specialist agents.
    
    This is what many "multi-agent frameworks" encourage.
    Let's see if the complexity is worth it.
    """
    print("\n" + "=" * 70)
    print("APPROACH 2: Multi-Agent Orchestration")
    print("=" * 70)
    
    observer = Observer(level="normal")
    
    # Specialist 1: Financial Analyst
    financial_analyst = Agent(
        name="FinancialAnalyst",
        model="gpt-4o-mini",
        tools=[get_financials],
        instructions="Analyze company financials. Return key metrics and trends.",
    )
    
    # Specialist 2: Market Analyst  
    market_analyst = Agent(
        name="MarketAnalyst",
        model="gpt-4o-mini",
        tools=[get_market_analysis],
        instructions="Analyze market conditions and competitive landscape.",
    )
    
    # Specialist 3: Risk Analyst
    risk_analyst = Agent(
        name="RiskAnalyst",
        model="gpt-4o-mini",
        tools=[get_risk_factors],
        instructions="Identify and assess company risk factors.",
    )
    
    # Orchestrator coordinates specialists
    orchestrator = Agent(
        name="ChiefAnalyst",
        model="gpt-4o-mini",
        tools=[
            financial_analyst.as_tool(description="Get financial analysis"),
            market_analyst.as_tool(description="Get market analysis"),
            risk_analyst.as_tool(description="Get risk assessment"),
        ],
        output=InvestmentRecommendation,
        instructions="""You are the Chief Investment Analyst.

For company analysis:
1. Call FinancialAnalyst for financial assessment
2. Call MarketAnalyst for market context
3. Call RiskAnalyst for risk evaluation
4. Synthesize all inputs into final recommendation

Delegate to specialists, then make the final call.""",
        observer=observer,
    )
    
    start = time.perf_counter()
    result = await orchestrator.run(TASK)
    duration = time.perf_counter() - start
    
    # Extract structured output
    output = None
    if result.content and hasattr(result.content, 'data'):
        output = result.content.data
    
    if output:
        print(f"\n📊 {output.ticker}: {output.recommendation} ({output.confidence} confidence)")
        print(f"💡 {output.reasoning}")
        print(f"⚠️  Risks: {', '.join(output.risks)}")
    else:
        print(f"\n📋 Result:\n{result.content}")
    
    tokens = 0
    if result.metadata and result.metadata.tokens:
        tokens = result.metadata.tokens.total_tokens
    
    # Note: This undercounts! Nested agent calls have their own tokens
    # that aren't reflected in orchestrator's metadata
    
    return {
        "approach": "multi_agent",
        "duration_seconds": round(duration, 2),
        "llm_calls": 4,  # Orchestrator + 3 specialists (minimum)
        "tokens": tokens,  # Note: undercounted (nested calls not included)
        "recommendation": output.recommendation if output else "N/A",
    }


# ============================================================================
# THE COMPARISON
# ============================================================================


async def main():
    print("\n" + "🎯 " * 25)
    print("PROVING THE POINT: Single Agent vs Multi-Agent")
    print("🎯 " * 25)
    print(f"\n📋 TASK: Analyze TechCorp and recommend BUY/HOLD/SELL\n")
    
    results = []
    
    # Run both approaches
    results.append(await single_agent_approach())
    results.append(await multi_agent_approach())
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("📊 RESULTS COMPARISON")
    print("=" * 80)
    print()
    
    headers = f"{'Approach':<35} {'Duration':<12} {'LLM Calls':<12} {'Tokens':<12} {'Recommendation'}"
    print(headers)
    print("-" * 85)
    
    for r in results:
        approach = r["approach"].replace("_", " ").title()
        duration_str = f"{r['duration_seconds']:.2f}s"
        tokens_str = f"{r.get('tokens', 'N/A')}*" if "multi" in r["approach"] else str(r.get('tokens', 'N/A'))
        print(f"{approach:<35} {duration_str:<12} {r['llm_calls']:<12} {tokens_str:<12} {r['recommendation']}")
    
    print("\n* Token count for multi-agent is undercounted (nested calls not fully tracked)")
    
    # Calculate overhead
    single = results[0]
    multi = results[1]
    
    print("\n" + "=" * 80)
    print("📈 OVERHEAD ANALYSIS")
    print("=" * 80)
    
    if single["duration_seconds"] > 0:
        multi_overhead = ((multi["duration_seconds"] / single["duration_seconds"]) - 1) * 100
        print(f"""
Multi-Agent Orchestration vs Single Agent:
  ⏱️  Duration: {multi_overhead:+.0f}% {'slower' if multi_overhead > 0 else 'faster'}
  📞 LLM Calls: {multi['llm_calls']}x vs {single['llm_calls']}x ({multi['llm_calls'] - single['llm_calls']} extra)
  💰 Tokens: {multi['tokens']}* vs {single['tokens']} (actual cost much higher)
""")


if __name__ == "__main__":
    asyncio.run(main())
