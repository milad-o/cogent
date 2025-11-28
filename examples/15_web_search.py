"""
Example 15: WebSearch Capability

Demonstrates the WebSearch capability for searching the web and fetching pages.
Uses DuckDuckGo (free, no API key required).

Features:
- Web search
- News search  
- Page content fetching
- Result caching
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Programmatic Demo (no LLM needed)
# ============================================================

def programmatic_demo():
    """Demonstrate WebSearch capability without an agent."""
    print("=" * 60)
    print("🔍 WebSearch Programmatic Demo")
    print("=" * 60)
    
    from agenticflow.capabilities import WebSearch
    
    # Initialize WebSearch (uses DuckDuckGo by default - free!)
    ws = WebSearch(max_results=5)
    
    print(f"\n✓ Created WebSearch capability")
    print(f"  Provider: {ws.provider.name}")
    print(f"  Tools: {[t.name for t in ws.tools]}")
    
    # Web search
    print("\n" + "-" * 40)
    print("🔍 Searching for 'Python 3.13 new features'...")
    
    results = ws.search("Python 3.13 new features", max_results=5)
    
    if results:
        print(f"\nFound {len(results)} results:")
        for r in results:
            print(f"\n  {r.position}. {r.title}")
            print(f"     URL: {r.url}")
            print(f"     {r.snippet[:100]}...")
    else:
        print("  No results found (rate limited or network issue)")
    
    # News search
    print("\n" + "-" * 40)
    print("📰 Searching news for 'artificial intelligence'...")
    
    news = ws.search_news("artificial intelligence", max_results=3)
    
    if news:
        print(f"\nFound {len(news)} news articles:")
        for r in news:
            print(f"\n  {r.position}. {r.title}")
            print(f"     URL: {r.url}")
    else:
        print("  No news found")
    
    # Fetch a webpage
    print("\n" + "-" * 40)
    print("📄 Fetching content from example.com...")
    
    page = ws.fetch("https://example.com")
    
    if page.error:
        print(f"  Error: {page.error}")
    else:
        print(f"  Title: {page.title}")
        print(f"  Content length: {len(page.content)} chars")
        print(f"  Preview: {page.content[:200]}...")
    
    # Cache demonstration
    print("\n" + "-" * 40)
    print("💾 Cache demonstration...")
    
    # First fetch (cached)
    page1 = ws.fetch("https://example.com")
    print(f"  First fetch: {page1.title}")
    
    # Second fetch (from cache)
    page2 = ws.fetch("https://example.com")
    print(f"  Second fetch (cached): {page2.title}")
    
    # Clear cache
    cleared = ws.clear_cache()
    print(f"  Cleared {cleared} cached pages")
    
    print("\n" + "=" * 60)


# ============================================================
# Agent Demo (requires LLM)
# ============================================================

async def agent_demo():
    """Demonstrate WebSearch capability with an agent."""
    print("🤖 Agent with WebSearch Demo")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set - skipping agent demo")
        return
    
    from agenticflow.models import ChatModel
    from agenticflow import Agent
    from agenticflow.capabilities import WebSearch, KnowledgeGraph
    
    model = ChatModel(model="gpt-4o-mini", temperature=0)
    
    # Combine WebSearch with KnowledgeGraph for research + memory
    ws = WebSearch(max_results=5)
    kg = KnowledgeGraph()
    
    agent = Agent(
        name="Researcher",
        model=model,
        instructions=(
            "You are a research assistant. "
            "Use web_search to find information, then remember key facts. "
            "Be concise and cite your sources."
        ),
        capabilities=[ws, kg],
    )
    
    print(f"\nAgent tools: {[t.name for t in ws.tools + kg.tools]}")
    
    # Research queries
    queries = [
        "Search for Python 3.13 features and remember the top 3 new features",
        "What new features did you remember about Python 3.13?",
    ]
    
    for query in queries:
        print(f"\n❓ {query}")
        print("-" * 40)
        response = await agent.run(query, strategy="dag")
        answer = response.replace("FINAL ANSWER:", "").strip()
        print(f"💡 {answer[:600]}")


# ============================================================
# Combined Capabilities Demo
# ============================================================

async def combined_demo():
    """Show WebSearch combined with other capabilities."""
    print("\n" + "=" * 60)
    print("🔗 Combined Capabilities Demo")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set - skipping")
        return
    
    from agenticflow.models import ChatModel
    from agenticflow import Agent
    from agenticflow.capabilities import WebSearch, KnowledgeGraph, FileSystem
    import tempfile
    from pathlib import Path
    
    model = ChatModel(model="gpt-4o-mini", temperature=0)
    
    with tempfile.TemporaryDirectory() as workspace:
        workspace_path = Path(workspace).resolve()
        
        # All three capabilities
        ws = WebSearch(max_results=3)
        kg = KnowledgeGraph()
        fs = FileSystem(allowed_paths=[str(workspace_path)], allow_write=True)
        
        agent = Agent(
            name="ResearchAssistant",
            model=model,
            instructions=(
                f"You are a research assistant that can search the web, "
                f"remember facts, and save files. "
                f"Workspace: {workspace_path}"
            ),
            capabilities=[ws, kg, fs],
        )
        
        all_tools = ws.tools + kg.tools + fs.tools
        print(f"\nAgent has {len(all_tools)} tools from 3 capabilities:")
        print(f"  WebSearch: {[t.name for t in ws.tools]}")
        print(f"  KnowledgeGraph: {[t.name for t in kg.tools]}")
        print(f"  FileSystem: {[t.name for t in fs.tools]}")
        
        # Multi-step research task
        query = (
            "Search for 'DuckDuckGo search API', remember the key facts, "
            f"and save a brief summary to {workspace_path}/research.txt"
        )
        
        print(f"\n❓ {query}")
        print("-" * 40)
        
        response = await agent.run(query, strategy="dag")
        answer = response.replace("FINAL ANSWER:", "").strip()
        print(f"💡 {answer[:500]}")
        
        # Check if file was created
        research_file = workspace_path / "research.txt"
        if research_file.exists():
            print(f"\n📄 Created {research_file.name}:")
            print(research_file.read_text()[:300])


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    programmatic_demo()
    
    print("\n")
    asyncio.run(agent_demo())
    asyncio.run(combined_demo())
    
    print("\n" + "=" * 60)
    print("✅ Summary:")
    print("   - WebSearch uses DuckDuckGo (free, no API key)")
    print("   - Tools: web_search, news_search, fetch_webpage")
    print("   - Built-in page caching for efficiency")
    print("   - Combines well with KnowledgeGraph for research")
    print("=" * 60)
