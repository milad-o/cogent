#!/usr/bin/env python3
"""
Example: MCP (Model Context Protocol) Integration.

Demonstrates AI agents using MCP server tools with different transports:
- STDIO: Local subprocess (default, managed by AgenticFlow)
- HTTP/SSE: Remote web server
- WebSocket: Bidirectional real-time

The agent discovers available tools automatically from the MCP server
and uses them based on their descriptions - no hardcoded tool names
in agent instructions!

Included MCP Server: examples/data/mcp_server/search_server.py
Provides: web search, news search, image search, instant answers

Requirements:
    uv add mcp ddgs starlette uvicorn

Transports:
    # STDIO (this example uses this - subprocess managed automatically)
    MCP.stdio(command="uv", args=["run", "python", "server.py", "stdio"])
    
    # HTTP/SSE (connect to running server)
    # First: uv run python examples/data/mcp_server/search_server.py http --port 8765
    MCP.sse(url="http://127.0.0.1:8765/sse")
    
    # HTTP Streamable (newer protocol)
    MCP.http(url="http://127.0.0.1:8765/mcp")
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from pathlib import Path

from agenticflow import Agent, Observer
from agenticflow.capabilities import MCP

from config import get_model


async def demo_research_agent() -> None:
    """Agent using MCP search capability to research topics."""
    print("\n" + "=" * 60)
    print("  Demo 1: Research Agent")
    print("=" * 60)

    server_path = Path(__file__).parent.parent / "data" / "mcp_server" / "search_server.py"

    # Create MCP capability - tools are discovered automatically
    mcp = MCP.stdio(
        command="uv",
        args=["run", "python", str(server_path), "stdio"],
        name="search",
    )

    # Agent instructions are generic - no tool names mentioned!
    # The LLM will discover available tools and use them appropriately
    agent = Agent(
        name="ResearchAgent",
        model=get_model(),
        instructions="""You are a research assistant. 
Use your available tools to find current, accurate information.
Make ONE search, then respond with a concise answer citing sources.""",
        capabilities=[mcp],
        observer=Observer.debug(),
    )

    try:
        query = "What are the latest developments in AI agents? Brief summary with sources."
        print(f"\n  Query: {query}\n")
        print("-" * 60)

        response = await agent.run(query, max_iterations=3)
        print(f"\n  Response:\n{response}")

    finally:
        await mcp.shutdown()


async def demo_news_analyst() -> None:
    """Agent analyzing recent news."""
    print("\n" + "=" * 60)
    print("  Demo 2: News Analyst Agent")
    print("=" * 60)

    server_path = Path(__file__).parent.parent / "data" / "mcp_server" / "search_server.py"

    mcp = MCP.stdio(
        command="uv",
        args=["run", "python", str(server_path), "stdio"],
        name="search",
    )

    # Generic instructions - agent discovers news search capability
    agent = Agent(
        name="NewsAnalyst",
        model=get_model(),
        instructions="""You are a news analyst.
Find recent news on the topic and provide a brief summary.
Make ONE search, then respond with key headlines and sources.""",
        capabilities=[mcp],
        observer=Observer.debug(),
    )

    try:
        query = "What's the latest news about Python programming?"
        print(f"\n  Query: {query}\n")
        print("-" * 60)

        response = await agent.run(query, max_iterations=3)
        print(f"\n  Response:\n{response}")

    finally:
        await mcp.shutdown()


async def demo_fact_checker() -> None:
    """Agent using search to verify facts."""
    print("\n" + "=" * 60)
    print("  Demo 3: Fact-Checking Agent")
    print("=" * 60)

    server_path = Path(__file__).parent.parent / "data" / "mcp_server" / "search_server.py"

    mcp = MCP.stdio(
        command="uv",
        args=["run", "python", str(server_path), "stdio"],
        name="search",
    )

    # Generic instructions - agent uses tools based on descriptions
    agent = Agent(
        name="FactChecker",
        model=get_model(),
        instructions="""You are a fact-checker.
Make ONE search to find the answer.
Once you find the fact, IMMEDIATELY respond with the answer and source.
Do NOT keep searching for more sources.""",
        capabilities=[mcp],
        observer=Observer.debug(),
    )

    try:
        query = "Who is the CEO of OpenAI?"
        print(f"\n  Query: {query}\n")
        print("-" * 60)

        response = await agent.run(query, max_iterations=4)
        print(f"\n  Response:\n{response}")

    finally:
        await mcp.shutdown()


async def demo_http_transport() -> None:
    """Demo using HTTP/SSE transport (requires running server separately)."""
    print("\n" + "=" * 60)
    print("  Demo 4: HTTP/SSE Transport")
    print("=" * 60)
    
    # Connect to MCP server via HTTP/SSE
    mcp = MCP.sse(
        url="http://127.0.0.1:8765/sse",
        name="search-http",
    )
    
    agent = Agent(
        name="HTTPAgent",
        model=get_model(),
        instructions="You are a research assistant. Use your tools to search. Be concise.",
        capabilities=[mcp],
        observer=Observer.debug(),
    )
    
    try:
        query = "What is the Model Context Protocol?"
        print(f"\n  Query: {query}")
        print("-" * 60)
        
        response = await agent.run(query, max_iterations=3)
        print(f"\n  Response:\n{response}")
    finally:
        await mcp.shutdown()


async def demo_stdio_subprocess() -> None:
    """Demo showing stdio with subprocess management."""
    print("\n" + "=" * 60)
    print("  Demo 5: Stdio Transport (Subprocess)")
    print("=" * 60)
    
    server_path = Path(__file__).parent.parent / "data" / "mcp_server" / "search_server.py"
    
    # Stdio transport - AgenticFlow manages the subprocess
    mcp = MCP.stdio(
        command="uv",
        args=["run", "python", str(server_path), "stdio"],
        name="search-subprocess",
    )
    
    agent = Agent(
        name="SubprocessAgent",
        model=get_model(),
        instructions="Use your search tool to answer briefly. Make ONE search only.",
        capabilities=[mcp],
    )
    
    try:
        query = "What is the Model Context Protocol?"
        print(f"\n  Query: {query}")
        print("-" * 60)
        
        response = await agent.run(query, max_iterations=3)
        print(f"\n  Response:\n{response}")
        
    finally:
        await mcp.shutdown()


async def main() -> None:
    """Run MCP demos."""
    print("\n" + "=" * 60)
    print("  MCP (Model Context Protocol) - AI Agent Integration")
    print("=" * 60)
    print("""
    This example demonstrates MCP capability integration:
    
    • Agent connects to MCP server (search_server.py)
    • Tools are discovered automatically from the server
    • Agent uses tools based on their descriptions
    • No hardcoded tool names in agent instructions!
    
    Transports demonstrated:
    • STDIO - Local subprocess (Demos 1-3, 5)
    • HTTP/SSE - Remote server (Demo 4 - requires manual server start)
    """)

    # Stdio demos (subprocess managed by AgenticFlow)
    await demo_research_agent()
    await demo_news_analyst()
    await demo_fact_checker()
    
    # HTTP demo (shows how to connect to remote server)
    await demo_http_transport()
    
    # Explicit subprocess demo
    await demo_stdio_subprocess()

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print("""
    MCP enables agents to connect to external tool servers:
    
    ✅ Auto tool discovery - Tools found on connection
    ✅ Description-based usage - Agent reads tool descriptions
    ✅ Multiple transports:
       • STDIO - Local subprocess (most common)
       • HTTP/SSE - Remote web server
       • WebSocket - Bidirectional real-time
    ✅ Seamless integration - Works like any other capability
    
    To test HTTP/SSE transport manually:
    
    Terminal 1 (start server):
        uv run python examples/data/mcp_server/search_server.py http --port 8765
    
    Terminal 2 (Python):
        from agenticflow.capabilities import MCP
        mcp = MCP.sse(url="http://127.0.0.1:8765/sse")
        # ... use with agent
    """)


if __name__ == "__main__":
    asyncio.run(main())
