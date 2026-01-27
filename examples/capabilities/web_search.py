"""
Example: WebSearch Capability

Professional example demonstrating web search and page fetching capabilities.
Uses DuckDuckGo (free, no API key required).

Features:
- Web search with DuckDuckGo
- News search
- Page content fetching
- Result caching
- Agent-based research workflows

Usage:
    uv run python examples/capabilities/web_search.py
"""

import asyncio
from pathlib import Path

from cogent import Agent, Observer
from cogent.capabilities import FileSystem, KnowledgeGraph, WebSearch

# ============================================================
# API-Level Usage
# ============================================================

def api_usage_example():
    """Demonstrate direct WebSearch API usage."""
    print("\n" + "=" * 60)
    print("API Usage Example: Direct WebSearch Operations")
    print("=" * 60)

    ws = WebSearch(max_results=5)

    print("\n✓ WebSearch initialized")
    print(f"  Provider: {ws.provider.name}")
    print(f"  Tools: {[t.name for t in ws.tools]}")

    # Web search
    print("\n→ Searching: 'photosynthesis process'")
    results = ws.search("photosynthesis process", max_results=5)

    if results:
        print(f"  ✓ Found {len(results)} results")
        print(f"  Top result: {results[0].title}")
    else:
        print("  ⚠ No results found")

    # News search
    print("\n→ News search: 'renewable energy'")
    news = ws.search_news("renewable energy", max_results=3)
    print(f"  ✓ Found {len(news)} news articles")

    # Fetch a webpage
    print("\n→ Fetching: http://example.com")
    page = ws.fetch("http://example.com")

    if page.error:
        print(f"  ✗ Error: {page.error}")
    else:
        print(f"  ✓ Title: {page.title}")
        print(f"  ✓ Content: {len(page.content)} chars")

    # Cache demonstration
    print("\n→ Testing cache")
    ws.fetch("http://example.com")
    ws.fetch("http://example.com")  # Should hit cache
    cleared = ws.clear_cache()
    print(f"  ✓ Cache cleared: {cleared} pages")


# ============================================================
# Agent-Based Research
# ============================================================

async def agent_research_example():
    """Demonstrate agent-based research with WebSearch and KnowledgeGraph."""
    print("\n" + "=" * 60)
    print("Agent Research Example: AI-Powered Web Research")
    print("=" * 60)


    # Combine WebSearch with KnowledgeGraph for research + memory
    ws = WebSearch(max_results=5)
    kg = KnowledgeGraph()

    # Create agent with verbose observer to see its work
    agent = Agent(
        name="Researcher",
        model="gpt4",
        instructions="Search the web for facts and remember important information using the available tools.",
        capabilities=[ws, kg],
        verbosity="debug",  # Built-in observability
    )

    print(f"\n✓ Agent '{agent.name}' initialized")
    print(f"  Tools: {len(ws.tools + kg.tools)} available")

    # Research queries
    queries = [
        "Search the web for solar system planets and remember the 3 largest ones",
        "What did you remember about the largest planets?",
    ]

    for query in queries:
        print(f"\n→ Query: {query}")
        print("-" * 60)
        response = await agent.run(query)
        print(f"\n✓ Response received ({len(response)} chars)")



# ============================================================
# Multi-Capability Research Workflow
# ============================================================

async def multi_capability_example():
    """Demonstrate multi-capability research workflow."""
    import tempfile

    print("\n" + "=" * 60)
    print("Multi-Capability Example: Web + Memory + Files")
    print("=" * 60)


    with tempfile.TemporaryDirectory() as workspace:
        workspace_path = Path(workspace).resolve()

        # Initialize three complementary capabilities
        ws = WebSearch(max_results=3)
        kg = KnowledgeGraph()
        fs = FileSystem(allowed_paths=[str(workspace_path)], allow_write=True)

        # Create observer for detailed execution tracing
        observer = Observer.trace()

        agent = Agent(
            name="ResearchAssistant",
            model="gpt4",
            instructions=(
                f"You are a research assistant with access to web search, knowledge storage, and file writing tools. "
                f"Always complete all parts of user requests. "
                f"Working directory: {workspace_path}"
            ),
            capabilities=[ws, kg, fs],
            observer=observer,
        )

        all_tools = ws.tools + kg.tools + fs.tools
        print(f"\n✓ Agent '{agent.name}' configured")
        print(f"  Capabilities: WebSearch ({len(ws.tools)}), KnowledgeGraph ({len(kg.tools)}), FileSystem ({len(fs.tools)})")
        print(f"  Total tools: {len(all_tools)}")

        # Multi-step research task
        query = (
            "Please do these 3 steps: "
            "1) Search the web for Egyptian pyramids "
            "2) Remember 2 key facts "
            "3) Write a brief summary to research.txt"
        )

        print("\n→ Task: Multi-step research and file creation")
        print("-" * 60)
        await agent.run(query)

        # Verify results
        research_file = workspace_path / "research.txt"
        if research_file.exists():
            content = research_file.read_text()
            print(f"\n✓ Research file created: {research_file.name}")
            print(f"  Size: {len(content)} chars")
            print("\n  Preview:")
            print(f"  {content[:200]}...")
        else:
            print("\n⚠ Research file was not created")
            print("  Agent may have failed to execute the write_file tool")

        # Show execution summary
        print("\n" + "=" * 60)
        print("Execution Summary")
        print("=" * 60)
        print(observer.summary())

        # Show execution graph
        print("\n" + "=" * 60)
# Main Entry Point
# ============================================================

async def main() -> None:
    """Run all WebSearch capability examples."""
    print("\n" + "=" * 70)
    print(" " * 15 + "WEBSEARCH CAPABILITY EXAMPLES")
    print("=" * 70)

    # 1. API-level usage (direct interaction)
    api_usage_example()

    # 2. Agent-based research
    await agent_research_example()

    # 3. Multi-capability workflow
    await multi_capability_example()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("✓ Provider: DuckDuckGo (free, no API key required)")
    print("✓ Features: web_search, news_search, fetch_webpage")
    print("✓ Built-in: page caching, result filtering")
    print("✓ Integrations: works seamlessly with KnowledgeGraph, FileSystem")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

