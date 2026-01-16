"""
Example 28: Agent Reasoning - Extended Thinking

This example demonstrates the reasoning capability, which enables agents
to think through complex problems before acting - similar to how Claude
and other advanced AI systems approach challenging tasks.

Reasoning modes:
- ANALYTICAL: Step-by-step logical breakdown (default)
- EXPLORATORY: Consider multiple approaches
- CRITICAL: Question assumptions, find flaws
- CREATIVE: Generate novel solutions

Example output:
    💭 Analyst reasoning (analytical style)...
    💭 [Round 1] Analysis: I need to analyze the company's metrics...
    💭 Reasoning complete after 2 round(s)
    🔧 Analyst → get_financial_data
    🔧 Analyst → calculate_metrics
    ✅ Analyst: Based on the analysis...
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from agenticflow import Agent, Observer
from agenticflow.agent import ReasoningConfig, ReasoningStyle
from agenticflow.tools.base import tool

# Handle both direct execution and module import
try:
    from examples.config import get_model
except ImportError:
    from config import get_model


# =============================================================================
# Tools for demonstration
# =============================================================================


@tool
def get_company_data(company_name: str) -> dict:
    """Get company financial data."""
    data = {
        "ACME Corp": {
            "revenue": 10_500_000,
            "expenses": 8_200_000,
            "employees": 150,
            "growth_rate": 0.15,
            "market_share": 0.12,
        },
        "TechStart Inc": {
            "revenue": 2_300_000,
            "expenses": 2_800_000,
            "employees": 25,
            "growth_rate": 0.45,
            "market_share": 0.03,
        },
    }
    return data.get(company_name, {"error": "Company not found"})


@tool
def calculate_metrics(
    revenue: float,
    expenses: float,
    employees: int,
    growth_rate: float,
) -> dict:
    """Calculate business metrics from financial data."""
    profit = revenue - expenses
    profit_margin = profit / revenue if revenue > 0 else 0
    revenue_per_employee = revenue / employees if employees > 0 else 0
    
    return {
        "profit": profit,
        "profit_margin": f"{profit_margin:.1%}",
        "revenue_per_employee": f"${revenue_per_employee:,.0f}",
        "is_profitable": profit > 0,
        "health_score": min(100, max(0, (profit_margin * 100) + (growth_rate * 50))),
    }


@tool
def compare_companies(data_a: dict, data_b: dict) -> dict:
    """Compare two companies' metrics."""
    comparison = {
        "revenue_leader": "A" if data_a.get("revenue", 0) > data_b.get("revenue", 0) else "B",
        "growth_leader": "A" if data_a.get("growth_rate", 0) > data_b.get("growth_rate", 0) else "B",
        "efficiency_leader": "A" if (
            data_a.get("revenue", 0) / max(1, data_a.get("employees", 1))
            > data_b.get("revenue", 0) / max(1, data_b.get("employees", 1))
        ) else "B",
    }
    return comparison


# =============================================================================
# Example 1: Basic reasoning with default config
# =============================================================================


async def example_basic_reasoning():
    """Basic reasoning - agent thinks before acting."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Reasoning")
    print("=" * 60)
    
    model = get_model()
    
    # Enable reasoning with True (uses default config)
    agent = Agent(
        name="Analyst",
        model=model,
        tools=[get_company_data, calculate_metrics],
        reasoning=True,  # Enable default reasoning
        observer=Observer.trace(),  # Show all reasoning events
    )
    
    result = await agent.run(
        "Analyze ACME Corp's financial health and provide a recommendation."
    )
    
    print(f"\n📊 Result:\n{result}")


# =============================================================================
# Example 2: Custom reasoning styles
# =============================================================================


async def example_reasoning_styles():
    """Different reasoning styles for different tasks."""
    print("\n" + "=" * 60)
    print("Example 2: Reasoning Styles")
    print("=" * 60)
    
    model = get_model()
    
    # Critical reasoning - questions assumptions
    agent = Agent(
        name="CriticalAnalyst",
        model=model,
        tools=[get_company_data, calculate_metrics, compare_companies],
        reasoning=ReasoningConfig.critical(),
        observer=Observer.normal(),  # Show progress and reasoning
    )
    
    result = await agent.run(
        "Compare ACME Corp and TechStart Inc. Which is a better investment?"
    )
    
    print(f"\n📊 Critical Analysis:\n{result}")


# =============================================================================
# Example 3: Deep reasoning with confidence threshold
# =============================================================================


async def example_deep_reasoning():
    """Deep reasoning - more thinking rounds, higher confidence required."""
    print("\n" + "=" * 60)
    print("Example 3: Deep Reasoning")
    print("=" * 60)
    
    model = get_model()
    
    # Deep reasoning - requires 70% confidence to stop thinking
    agent = Agent(
        name="DeepAnalyst",
        model=model,
        tools=[get_company_data, calculate_metrics],
        reasoning=ReasoningConfig.deep(),
        observer=Observer.normal(),
    )
    
    result = await agent.run(
        "Given limited data, assess ACME Corp's long-term viability. "
        "Consider risks and uncertainty."
    )
    
    print(f"\n📊 Deep Analysis:\n{result}")


# =============================================================================
# Example 4: Quick reasoning for simpler tasks
# =============================================================================


async def example_quick_reasoning():
    """Quick reasoning - minimal overhead for simpler tasks."""
    print("\n" + "=" * 60)
    print("Example 4: Quick vs No Reasoning Comparison")
    print("=" * 60)
    
    model = get_model()
    
    # Agent with quick reasoning
    agent_reasoning = Agent(
        name="QuickAnalyst",
        model=model,
        tools=[get_company_data],
        reasoning=ReasoningConfig.quick(),
        observer=Observer.normal(),
    )
    
    # Agent without reasoning
    agent_no_reasoning = Agent(
        name="DirectAnalyst",
        model=model,
        tools=[get_company_data],
        reasoning=None,  # No reasoning
        observer=Observer.normal(),
    )
    
    task = "Get ACME Corp's employee count."
    
    print("With quick reasoning:")
    result1 = await agent_reasoning.run(task)
    print(f"Result: {result1[:100]}...")
    
    print("\nWithout reasoning:")
    result2 = await agent_no_reasoning.run(task)
    print(f"Result: {result2[:100]}...")


# =============================================================================
# Example 5: Custom reasoning configuration
# =============================================================================


async def example_custom_config():
    """Custom reasoning configuration for specific needs."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Reasoning Config")
    print("=" * 60)
    
    model = get_model()
    
    # Custom config: exploratory style, show thinking, 2 rounds
    custom_config = ReasoningConfig(
        enabled=True,
        max_thinking_rounds=2,
        style=ReasoningStyle.EXPLORATORY,
        show_thinking=True,  # Include <thinking> in output
        self_correct=True,
    )
    
    agent = Agent(
        name="Explorer",
        model=model,
        tools=[get_company_data, calculate_metrics, compare_companies],
        reasoning=custom_config,
        observer=Observer.normal(),
    )
    
    result = await agent.run(
        "Explore different ways to evaluate these two companies: "
        "ACME Corp and TechStart Inc."
    )
    
    print(f"\n📊 Exploratory Analysis:\n{result}")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all examples."""
    await example_basic_reasoning()
    await example_reasoning_styles()
    await example_deep_reasoning()
    await example_quick_reasoning()
    await example_custom_config()
    
    print("\n" + "=" * 60)
    print("All examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
