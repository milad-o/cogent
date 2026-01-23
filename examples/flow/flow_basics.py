"""Example: Reactive Multi-Agent Flow with Tools.

This example demonstrates the reactive orchestration model where
agents react to events rather than being called imperatively.

Each agent has specialized tools to accomplish their tasks.

API Levels:
- High-level: chain(), fanout(), route() - one-liner patterns
- Mid-level: Chain, FanOut, Router classes with .run()
- Low-level: Flow with manual trigger registration
"""

import asyncio
import random

from agenticflow import (
    Agent,
    Flow,
    FlowConfig,
    Observer,
    tool,
)
from agenticflow.flow.patterns import pipeline
from agenticflow.reactors import WaitAll

# ═══════════════════════════════════════════════════════════════════════════════
# TOOLS - Specialized functions for each agent type
# ═══════════════════════════════════════════════════════════════════════════════


@tool
def search_web(query: str) -> str:
    """Search the web for information on a topic."""
    # Simulated web search results
    results = {
        "renewable energy": [
            "Solar capacity grew 25% in 2024",
            "Wind power now cheapest energy source",
            "Battery storage costs dropped 40%",
        ],
        "ai trends": [
            "LLMs becoming multimodal",
            "AI agents gaining autonomy",
            "Edge AI deployment accelerating",
        ],
        "default": [
            f"Top result for '{query}'",
            f"Related article about {query}",
            f"Recent news on {query}",
        ],
    }
    key = query.lower()
    for k, v in results.items():
        if k in key:
            return "\n".join(f"- {r}" for r in v)
    return "\n".join(f"- {r}" for r in results["default"])


@tool
def search_database(table: str, filter_by: str | None = None) -> str:
    """Search internal database for records."""
    # Simulated database results
    data = {
        "customers": ["Enterprise Corp (500 users)", "Startup Inc (50 users)", "Tech LLC (200 users)"],
        "products": ["Pro Plan: $99/mo", "Team Plan: $49/mo", "Free Tier: $0"],
        "metrics": ["MRR: $125,000", "Churn: 2.5%", "NPS: 72"],
    }
    records = data.get(table.lower(), [f"Record from {table}"])
    if filter_by:
        records = [r for r in records if filter_by.lower() in r.lower()]
    return "\n".join(f"- {r}" for r in records) or "No matching records"


@tool
def analyze_data(data: str, analysis_type: str = "summary") -> str:
    """Analyze data and extract insights."""
    word_count = len(data.split())
    lines = len(data.strip().split("\n"))

    if analysis_type == "sentiment":
        sentiment = random.choice(["positive", "neutral", "mixed"])
        return f"Sentiment: {sentiment}\nConfidence: {random.randint(70, 95)}%"
    elif analysis_type == "trends":
        return f"Key trends identified:\n- Growth pattern detected\n- {lines} data points analyzed\n- Recommendation: investigate further"
    else:
        return f"Summary: {word_count} words, {lines} items\nData appears structured and valid."


@tool
def generate_chart(chart_type: str, title: str) -> str:
    """Generate a chart visualization (returns description)."""
    charts = {
        "bar": f"[Bar Chart: {title}]\n  ████ 45%\n  ██████ 65%\n  ████████ 85%",
        "line": f"[Line Chart: {title}]\n  ╱╲╱╲___╱╲",
        "pie": f"[Pie Chart: {title}]\n  ◐ 40% | ◑ 35% | ◒ 25%",
    }
    return charts.get(chart_type.lower(), f"[{chart_type} Chart: {title}]")


@tool
def write_document(content: str, format_as: str = "markdown") -> str:
    """Format content as a structured document."""
    if format_as == "markdown":
        return f"# Document\n\n{content}\n\n---\n*Generated document*"
    elif format_as == "html":
        return f"<article><h1>Document</h1><p>{content}</p></article>"
    else:
        return f"=== Document ===\n{content}\n================"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    # Simple safe eval for basic math
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: Only basic math operations allowed"
    try:
        result = eval(expression)  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def send_notification(message: str, channel: str = "slack") -> str:
    """Send a notification to a channel."""
    return f"✓ Notification sent to {channel}: '{message[:50]}...'"


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLES - Different reactive patterns with tool-equipped agents
# ═══════════════════════════════════════════════════════════════════════════════


async def research_and_write() -> None:
    """Chain pattern: Research → Write → Edit with tools."""
    print("\n" + "─" * 50)
    print("📝 CHAIN: Research → Write → Edit (Pipeline)")
    print("─" * 50)

    observer = Observer.trace()

    researcher = Agent(
        name="researcher",
        model="gpt4",
        system_prompt="You research topics using web search. Find 3 key facts. Be brief.",
        tools=[search_web],
    )
    writer = Agent(
        name="writer",
        model="gpt4",
        system_prompt="You write concise content based on research. Use the document tool to format output.",
        tools=[write_document],
    )
    editor = Agent(
        name="editor",
        model="gpt4",
        system_prompt="You polish content. Make it shorter and punchier. One paragraph max.",
    )

    # Use pipeline pattern
    result = await pipeline([researcher, writer, editor], observer=observer).run(
        "Write about renewable energy trends"
    )

    out = result.output
    preview = out.content if hasattr(out, "content") else out
    print(f"\n📄 Final Output:\n{str(preview)[:500]}")


async def parallel_search_and_aggregate() -> None:
    """Parallel/FanOut pattern: Parallel searches merged by aggregator."""
    print("\n" + "─" * 50)
    print("🔍 FANOUT: Parallel Search → WaitAll → Aggregate")
    print("─" * 50)

    observer = Observer.trace()

    web_searcher = Agent(
        name="web_searcher",
        model="gpt4",
        system_prompt="Search the web for the requested information. Report findings briefly.",
        tools=[search_web],
    )
    db_searcher = Agent(
        name="db_searcher",
        model="gpt4",
        system_prompt="Search internal databases for relevant records. Report findings briefly.",
        tools=[search_database],
    )
    aggregator = Agent(
        name="aggregator",
        model="gpt4",
        system_prompt="Combine search results into a unified brief report. Highlight key insights.",
        tools=[write_document],
    )

    # Manual flow construction for FanOut
    flow = Flow(observer=observer)

    # Register parallel agents trigger on same event
    flow.register(web_searcher, on="task.created", emits="search.web.done")
    flow.register(db_searcher, on="task.created", emits="search.db.done")

    # Wait for both to finish before triggering aggregator
    # We use a WaitAll reactor to synchronize
    waiter = WaitAll(
        collect=2,
        emit="all.search.done",
    )
    flow.register(waiter, on=["search.web.done", "search.db.done"])

    # Aggregator triggers when watcher completes
    flow.register(aggregator, on="all.search.done", emits="flow.done")

    result = await flow.run(
        "Find information about AI trends and our customer metrics",
        initial_event="task.created"
    )

    out = result.output
    preview = out.content if hasattr(out, "content") else out
    print(f"\n📄 Aggregated Report:\n{str(preview)[:500]}")


async def smart_routing() -> None:
    """Route pattern: Direct tasks to specialized agents using conditionals."""
    print("\n" + "─" * 50)
    print("🔀 ROUTE: Smart Task Routing (Conditional)")
    print("─" * 50)

    observer = Observer.trace()

    analyst = Agent(
        name="analyst",
        model="gpt4",
        system_prompt="You analyze data. Use analysis tools to extract insights.",
        tools=[analyze_data, generate_chart],
    )
    coder = Agent(
        name="coder",
        model="gpt4",
        system_prompt="You solve math and coding problems. Use the calculator for math.",
        tools=[calculate],
    )
    communicator = Agent(
        name="communicator",
        model="gpt4",
        system_prompt="You handle communications. Send notifications as needed.",
        tools=[send_notification],
    )

    flow = Flow(observer=observer)

    # Register agents with conditional triggers
    flow.register(
        analyst,
        on="task.created",
        when=lambda e: any(k in e.data.get("task", "").lower() for k in ["analyze", "data", "chart"])
    )
    flow.register(
        coder,
        on="task.created",
        when=lambda e: any(k in e.data.get("task", "").lower() for k in ["calculate", "math", "code"])
    )
    flow.register(
        communicator,
        on="task.created",
        when=lambda e: any(k in e.data.get("task", "").lower() for k in ["notify", "send", "alert"])
    )

    tasks = [
        "Analyze this data: sales up 20%, costs down 5%",
        "Calculate: (100 * 1.2) - (50 * 0.95)",
        "Notify the team that the report is ready",
    ]

    for task in tasks:
        print(f"\n📌 Task: {task}")
        result = await flow.run(task)
        # Output might be None if no agent matched or if observer captured it differently
        # But usually agent output is in result.output
        print(f"   Result: {str(result.output)[:150]}...")


async def event_driven_pipeline() -> None:
    """Flow: Custom event-driven orchestration with tools."""
    print("\n" + "─" * 50)
    print("⚡ EVENT FLOW: Custom Reactive Pipeline")
    print("─" * 50)

    observer = Observer.trace()

    # Data collector triggers on task.created
    collector = Agent(
        name="collector",
        model="gpt4",
        system_prompt="Collect data from web and database. Report raw findings.",
        tools=[search_web, search_database],
    )

    # Analyzer triggers when collector completes
    analyzer = Agent(
        name="analyzer",
        model="gpt4",
        system_prompt="Analyze collected data. Generate insights and a chart.",
        tools=[analyze_data, generate_chart],
    )

    # Reporter triggers when analyzer completes
    reporter = Agent(
        name="reporter",
        model="gpt4",
        system_prompt="Create a final report and notify stakeholders.",
        tools=[write_document, send_notification],
    )

    flow = Flow(observer=observer)

    # Register agents with their triggers and emissions
    flow.register(collector, on="task.created", emits="collector.completed")
    flow.register(analyzer, on="collector.completed", emits="analyzer.completed")
    flow.register(reporter, on="analyzer.completed", emits="flow.done")

    result = await flow.run(
        "Analyze our customer data and market trends, then report findings",
        initial_event="task.created",
    )

    out = result.output
    preview = out.content if hasattr(out, "content") else out
    print(f"\n📄 Final Report:\n{str(preview)[:500]}")


async def conditional_workflow() -> None:
    """Flow with conditions: Agents trigger based on event data."""
    print("\n" + "─" * 50)
    print("🎯 CONDITIONAL: Event-Based Agent Selection (Complexity)")
    print("─" * 50)

    observer = Observer.trace()

    # Quick responder for simple queries
    quick_responder = Agent(
        name="quick_responder",
        model="gpt4",
        system_prompt="Give quick, direct answers. One sentence max.",
        tools=[calculate],
    )

    # Deep researcher for complex queries
    deep_researcher = Agent(
        name="deep_researcher",
        model="gpt4",
        system_prompt="Do thorough research. Use all available tools.",
        tools=[search_web, search_database, analyze_data],
    )

    flow = Flow(
        config=FlowConfig(max_rounds=10),
        observer=observer,
    )

    # Route based on query complexity
    flow.register(
        quick_responder,
        on="task.created",
        when=lambda e: len(e.data.get("task", "")) < 50
    )
    flow.register(
        deep_researcher,
        on="task.created",
        when=lambda e: len(e.data.get("task", "")) >= 50
    )

    # Short query → quick responder
    print("\n📌 Short Query:")
    result1 = await flow.run("What is 15% of 200?", initial_event="task.created")
    out1 = result1.output
    preview1 = out1.content if hasattr(out1, "content") else out1
    print(f"   → {str(preview1)[:100]}")

    # Long query → deep researcher
    print("\n📌 Complex Query:")
    result2 = await flow.run(
        "Research AI trends in the market, analyze our customer database, and identify growth opportunities",
        initial_event="task.created",
    )
    print(f"   → {str(result2.output)[:200]}...")


async def main() -> None:
    """Run all reactive flow examples."""
    print("\n" + "═" * 60)
    print("REACTIVE MULTI-AGENT FLOWS WITH TOOLS")
    print("═" * 60)

    await research_and_write()
    await parallel_search_and_aggregate()
    await smart_routing()
    await event_driven_pipeline()
    await conditional_workflow()

    print("\n" + "═" * 60)
    print("✅ All examples completed!")
    print("═" * 60)


if __name__ == "__main__":
    asyncio.run(main())
