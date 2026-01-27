"""
Tool Composition Patterns - Leveraging NativeExecutor and SequentialExecutor.

This example demonstrates:
1. Parallel execution (default) - Tools run concurrently
2. Sequential execution - Tools run one at a time  
3. Conditional logic - If/else based on tool results
4. Error recovery - Fallback patterns

Research basis: arXiv:2601.11327 - Tools are the primary value driver.
Composition makes tools more powerful than isolated execution.
"""

import asyncio

from cogent import Agent
from cogent.capabilities import WebSearch
from cogent.executors import NativeExecutor, SequentialExecutor
from cogent.tools import tool


# ============================================================================
# PATTERN 1: Parallel Execution (Default - NativeExecutor)
# ============================================================================
# Use when: Tools are independent and can run concurrently
# Performance: Faster - runs all tools at once with asyncio.gather


@tool
async def fetch_weather(city: str) -> str:
    """Fetch weather for a city (simulated)."""
    await asyncio.sleep(0.5)  # Simulate API delay
    return f"Weather in {city}: Sunny, 72°F"


@tool
async def fetch_news(topic: str) -> str:
    """Fetch news about a topic (simulated)."""
    await asyncio.sleep(0.5)  # Simulate API delay
    return f"Latest {topic} news: AI research breakthroughs announced"


@tool
async def fetch_stock(symbol: str) -> str:
    """Fetch stock price (simulated)."""
    await asyncio.sleep(0.5)  # Simulate API delay
    return f"Stock {symbol}: $150.25 (+2.3%)"


async def parallel_execution_demo():
    """Parallel execution - all tools run concurrently."""
    print("=" * 80)
    print("PATTERN 1: Parallel Execution (Default)")
    print("=" * 80)

    # NativeExecutor is the DEFAULT - enables parallel tool execution
    agent = Agent(
        name="parallel_demo",
        model="gpt-4o-mini",
        tools=[fetch_weather, fetch_news, fetch_stock],
    )

    # This task will trigger multiple independent tools
    task = """
    Get me:
    1. Weather in San Francisco
    2. Latest AI news
    3. Stock price for AAPL
    
    All three are independent - run them in parallel.
    """

    print(f"\nTask: {task.strip()}\n")
    print("🚀 Running with NativeExecutor (parallel)...")

    import time

    start = time.time()
    result = await agent.run(task)
    elapsed = time.time() - start

    print(f"\n✅ Result:\n{result}")
    print(f"\n⏱️  Time: {elapsed:.2f}s (expect ~0.5s with parallel execution)")
    print(
        "   If sequential: would take ~1.5s (3 tools × 0.5s each)\n"
    )  # Sequential would be 3×0.5s = 1.5s


# ============================================================================
# PATTERN 2: Sequential Execution (Two Approaches)
# ============================================================================
# Use when: Tools depend on previous results or order matters
# 
# Approach 1: NativeExecutor + Clear Prompting (Recommended)
#   - LLM naturally calls tools one at a time based on prompt
#   - More flexible - can parallelize when appropriate
#   - Trust the model to decide
#
# Approach 2: SequentialExecutor (Strict Ordering)
#   - Forces sequential execution even if LLM batches tools
#   - Use only when you need guaranteed sequential order
#   - Less flexible but more predictable


@tool
async def search_company(name: str) -> str:
    """Search for company information."""
    return f"Found: {name} - Tech company founded in 2020, specializes in AI"


@tool
async def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of text."""
    if "AI" in text and "founded" in text:
        return "Sentiment: POSITIVE (innovative company with strong fundamentals)"
    return "Sentiment: NEUTRAL"


@tool
async def generate_report(company_info: str, sentiment: str) -> str:
    """Generate investment report."""
    return f"""
Investment Report:
- Company: {company_info[:50]}...
- Market Sentiment: {sentiment}
- Recommendation: BUY (positive outlook, growing sector)
"""


async def sequential_execution_demo():
    """Sequential execution - showing both approaches."""
    print("=" * 80)
    print("PATTERN 2: Sequential Execution (Two Approaches)")
    print("=" * 80)

    # Approach 1: NativeExecutor with clear prompting (RECOMMENDED)
    print("\n--- Approach 1: NativeExecutor + Clear Prompting ---\n")
    
    agent_native = Agent(
        name="sequential_native",
        model="gpt-4o-mini",
        tools=[search_company, analyze_sentiment, generate_report],
    )

    task_native = """
    Research ACME Corp step by step:
    1. FIRST search for company info
    2. THEN analyze the sentiment of what you found
    3. FINALLY generate a report using both pieces
    
    Do these ONE AT A TIME, in order. Each step depends on the previous one.
    """

    print(f"Task: {task_native.strip()}\n")
    print("🔄 Running with NativeExecutor (LLM decides to go sequential)...")

    result_native = await agent_native.run(task_native)

    print(f"\n✅ Result:\n{result_native.content}\n")
    print("💡 LLM naturally called tools sequentially based on prompt\n")

    # Approach 2: SequentialExecutor (forces sequential even if LLM batches)
    print("\n--- Approach 2: SequentialExecutor (Forced Sequential) ---\n")
    
    agent_forced = Agent(
        name="sequential_forced",
        model="gpt-4o-mini",
        tools=[search_company, analyze_sentiment, generate_report],
    )

    executor = SequentialExecutor(agent_forced)

    task_forced = """
    Research ACME Corp:
    1. Search for company info
    2. Analyze sentiment
    3. Generate report
    """

    print(f"Task: {task_forced.strip()}\n")
    print("🔒 Running with SequentialExecutor (forces sequential order)...")

    result_forced = await executor.execute(task_forced)

    print(f"\n✅ Result:\n{result_forced}\n")
    print("💡 SequentialExecutor forces one-at-a-time execution\n")
    print("=" * 80)
    print("Recommendation: Use Approach 1 (NativeExecutor + prompting) unless you")
    print("need guaranteed sequential order regardless of what the LLM decides.")
    print("=" * 80 + "\n")


# ============================================================================
# PATTERN 3: Conditional Logic (If/Else based on tool results)
# ============================================================================
# Use when: Next action depends on tool output


@tool
async def check_inventory(product: str) -> str:
    """Check product inventory."""
    # Simulate: laptop is out of stock, phone is in stock
    if "laptop" in product.lower():
        return f"Product '{product}': OUT OF STOCK (restock date: 2024-03-15)"
    return f"Product '{product}': IN STOCK (23 units available)"


@tool
async def backorder(product: str) -> str:
    """Place item on backorder."""
    return f"Backorder created for '{product}' - you'll be notified when available"


@tool
async def complete_purchase(product: str, quantity: int = 1) -> str:
    """Complete purchase for in-stock item."""
    total = quantity * 999  # Mock price
    return f"Purchase complete: {quantity}x {product} - Total: ${total}"


async def conditional_execution_demo():
    """Conditional logic - agent decides next action based on tool results."""
    print("=" * 80)
    print("PATTERN 3: Conditional Logic")
    print("=" * 80)

    agent = Agent(
        name="conditional_demo",
        model="gpt-4o-mini",
        tools=[check_inventory, backorder, complete_purchase],
    )

    # The agent will check inventory, then either backorder OR purchase
    task = """
    Try to purchase a 'Gaming Laptop'.
    
    First check inventory. Then:
    - If IN STOCK: complete the purchase
    - If OUT OF STOCK: place a backorder
    
    Make the decision based on what you find.
    """

    print(f"\nTask: {task.strip()}\n")
    print("🧠 Running with conditional logic (agent decides)...")

    result = await agent.run(task)

    print(f"\n✅ Result:\n{result}")
    print("\n💡 Agent used if/else logic based on check_inventory result\n")


# ============================================================================
# PATTERN 4: Error Recovery (Fallback chains)
# ============================================================================
# Use when: Primary tool might fail, need backup plan


@tool
async def premium_api(query: str) -> str:
    """Premium API (sometimes fails)."""
    # Simulate failure
    raise RuntimeError("Premium API quota exceeded")


@tool
async def fallback_api(query: str) -> str:
    """Fallback API (more reliable)."""
    return f"Fallback result for '{query}': Basic data available"


@tool
async def cached_data(query: str) -> str:
    """Get cached data (always works)."""
    return f"Cached result for '{query}': Older data from last week"


async def error_recovery_demo():
    """Error recovery - try premium, fall back to alternatives."""
    print("=" * 80)
    print("PATTERN 4: Error Recovery")
    print("=" * 80)

    agent = Agent(
        name="error_recovery_demo",
        model="gpt-4o-mini",
        tools=[premium_api, fallback_api, cached_data],
        system_prompt="""
You have three data sources with different reliability:
1. premium_api - Best quality but might fail
2. fallback_api - Good quality, more reliable
3. cached_data - Always works but data is older

Strategy: Try premium_api first. If it fails, use fallback_api.
If that fails too, use cached_data as last resort.
""",
    )

    task = "Get data about 'quantum computing'"

    print(f"\nTask: {task}\n")
    print("🔄 Running with error recovery strategy...")

    result = await agent.run(task)

    print(f"\n✅ Result:\n{result}")
    print("\n💡 Agent handled premium_api failure and used fallback\n")


# ============================================================================
# PATTERN 5: Mixed Strategy (Parallel + Sequential)
# ============================================================================
# Use when: Some tools can run parallel, but final step needs all results


async def mixed_strategy_demo():
    """Mixed strategy - parallel data gathering + sequential report."""
    print("=" * 80)
    print("PATTERN 5: Mixed Strategy (Parallel + Sequential)")
    print("=" * 80)

    agent = Agent(
        name="mixed_strategy_demo",
        model="gpt-4o-mini",
        tools=[
            fetch_weather,  # Can run parallel
            fetch_news,  # Can run parallel
            fetch_stock,  # Can run parallel
            generate_report,  # Must run after gathering data
        ],
    )

    task = """
    Create a market summary report:
    
    1. Gather data in parallel:
       - Weather in NYC (affects consumer behavior)
       - Latest tech news
       - TSLA stock price
    
    2. Once you have all three, generate a summary report
    
    Use parallel execution for data gathering, sequential for the final report.
    """

    print(f"\nTask: {task.strip()}\n")
    print("🚀 Running with mixed strategy...")

    result = await agent.run(task)

    print(f"\n✅ Result:\n{result}")
    print("\n💡 Agent parallelized independent data fetching, then generated report\n")


# ============================================================================
# PATTERN 6: Real-World Example (WebSearch with Composition)
# ============================================================================


async def real_world_demo():
    """Real-world example using WebSearch capability."""
    print("=" * 80)
    print("PATTERN 6: Real-World Composition (WebSearch)")
    print("=" * 80)

    @tool
    async def summarize_findings(research: str) -> str:
        """Create executive summary from research."""
        return f"""
EXECUTIVE SUMMARY:
- Research completed: {len(research)} chars of data gathered
- Key topics: AI, productivity, automation
- Recommendation: Implement AI tools to improve workflow
"""

    agent = Agent(
        name="websearch_demo",
        model="gpt-4o-mini",
        capabilities=[WebSearch()],
        tools=[summarize_findings],
    )

    task = """
    Research the top 3 AI productivity tools of 2024.
    
    Then create an executive summary of your findings.
    
    Use parallel search if possible, then summarize sequentially.
    """

    print(f"\nTask: {task.strip()}\n")
    print("🔍 Running real-world composition...")

    result = await agent.run(task)

    print(f"\n✅ Result:\n{result}\n")


# ============================================================================
# Main
# ============================================================================


async def main():
    """Run all composition pattern demos."""
    print("\n" + "=" * 80)
    print(" Tool Composition Patterns - cogent Framework")
    print("=" * 80 + "\n")

    demos = [
        ("Parallel Execution", parallel_execution_demo),
        ("Sequential Execution", sequential_execution_demo),
        ("Conditional Logic", conditional_execution_demo),
        ("Error Recovery", error_recovery_demo),
        ("Mixed Strategy", mixed_strategy_demo),
        ("Real-World Example", real_world_demo),
    ]

    for name, demo_func in demos:
        try:
            await demo_func()
        except Exception as e:
            print(f"\n❌ {name} failed: {e}\n")

    print("=" * 80)
    print("✅ All composition patterns demonstrated!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
