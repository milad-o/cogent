#!/usr/bin/env python3
"""
Example 22: MCP (Model Context Protocol) Integration.

Shows how to connect agents to local and remote MCP servers
and use their tools seamlessly.

Requirements:
    uv add mcp  # or: pip install mcp
"""

import asyncio

from agenticflow.capabilities.mcp import MCP, MCPServerConfig, MCPTransport


async def demo_mcp_configuration() -> None:
    """Demonstrate MCP configuration options."""
    print("\n" + "=" * 60)
    print("  MCP Configuration Demo")
    print("=" * 60)

    # Method 1: Factory method for stdio (local servers)
    print("\n--- Method 1: Stdio Factory ---")
    mcp_stdio = MCP.stdio(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "."],
        name="filesystem",
    )
    print(f"  Created: {mcp_stdio}")
    print(f"  Transport: {mcp_stdio.servers[0].transport.value}")

    # Method 2: Factory method for HTTP (remote servers)
    print("\n--- Method 2: HTTP Factory ---")
    mcp_http = MCP.http(
        url="https://api.example.com/mcp",
        headers={"Authorization": "Bearer token123"},
        timeout=60.0,
        name="remote-api",
    )
    print(f"  Created: {mcp_http}")
    print(f"  URL: {mcp_http.servers[0].url}")

    # Method 3: Factory method for SSE
    print("\n--- Method 3: SSE Factory ---")
    mcp_sse = MCP.sse(
        url="https://sse.example.com/events",
        name="sse-server",
    )
    print(f"  Created: {mcp_sse}")

    # Method 4: Multiple servers
    print("\n--- Method 4: Multiple Servers ---")
    mcp_multi = MCP(
        servers=[
            MCPServerConfig(
                transport=MCPTransport.STDIO,
                command="uv",
                args=["run", "my-local-server"],
                name="local",
            ),
            MCPServerConfig(
                transport=MCPTransport.HTTP,
                url="https://weather.example.com/mcp",
                name="weather",
            ),
            MCPServerConfig(
                transport=MCPTransport.HTTP,
                url="https://search.example.com/mcp",
                headers={"X-API-Key": "key123"},
                name="search",
            ),
        ],
        tool_name_prefix="mcp_",  # All tools prefixed with "mcp_"
    )
    print(f"  Created: {mcp_multi}")
    print(f"  Server count: {len(mcp_multi.servers)}")

    # Method 5: Dynamic server addition
    print("\n--- Method 5: Dynamic Server Addition ---")
    mcp_dynamic = MCP()
    mcp_dynamic.add_server(
        MCPServerConfig(
            transport=MCPTransport.STDIO,
            command="python",
            args=["-m", "my_mcp_server"],
        )
    )
    print(f"  Added server dynamically: {len(mcp_dynamic.servers)} servers")


async def demo_agent_integration() -> None:
    """Demonstrate how MCP integrates with agents."""
    print("\n" + "=" * 60)
    print("  Agent Integration (Conceptual)")
    print("=" * 60)

    print("""
    # Full agent integration example:
    
    from agenticflow import Agent
    from agenticflow.capabilities import MCP
    from langchain_openai import ChatOpenAI
    
    # Create agent with MCP capability
    agent = Agent(
        name="Assistant",
        model=ChatOpenAI(model="gpt-4"),
        capabilities=[
            # Connect to a local filesystem server
            MCP.stdio(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "."],
            ),
            # Connect to a remote weather API
            MCP.http("https://weather-api.example.com/mcp"),
        ],
    )
    
    # Agent now has access to all MCP server tools!
    # The tools are automatically discovered on connection.
    
    # Use the agent - it can now:
    # - Read/write files (from filesystem server)
    # - Get weather data (from weather API)
    result = await agent.run(
        "Read the README.md file and summarize it, "
        "then check the weather in San Francisco"
    )
    """)


async def demo_tool_naming() -> None:
    """Demonstrate tool naming conventions."""
    print("\n" + "=" * 60)
    print("  Tool Naming")
    print("=" * 60)

    # Single server - tools keep original names
    print("\n--- Single Server ---")
    single = MCP.stdio(command="server")
    print("  Tool name: get_weather -> get_weather")

    # With prefix
    print("\n--- With Prefix ---")
    prefixed = MCP.stdio(command="server", tool_name_prefix="mcp_")
    print("  Tool name: get_weather -> mcp_get_weather")

    # Multiple servers - server suffix added
    print("\n--- Multiple Servers ---")
    multi = MCP(
        servers=[
            MCPServerConfig(transport=MCPTransport.STDIO, command="server1"),
            MCPServerConfig(transport=MCPTransport.STDIO, command="server2"),
        ]
    )
    print("  Tool names are suffixed with server identifier:")
    print("    server1: get_weather -> get_weather_server1")
    print("    server2: get_weather -> get_weather_server2")


async def demo_serialization() -> None:
    """Demonstrate MCP serialization."""
    print("\n" + "=" * 60)
    print("  Serialization")
    print("=" * 60)

    mcp = MCP(
        servers=[
            MCPServerConfig(
                transport=MCPTransport.STDIO,
                command="local-server",
                name="local",
            ),
            MCPServerConfig(
                transport=MCPTransport.HTTP,
                url="https://api.example.com/mcp",
                name="remote",
            ),
        ]
    )

    data = mcp.to_dict()
    print(f"\n  MCP State:")
    print(f"    Name: {data['name']}")
    print(f"    Initialized: {data['initialized']}")
    print(f"    Tool count: {data['tool_count']}")
    print(f"    Servers:")
    for server in data["servers"]:
        print(f"      - {server['name']} ({server['transport']})")


async def demo_common_servers() -> None:
    """Show common MCP server configurations."""
    print("\n" + "=" * 60)
    print("  Common MCP Server Configurations")
    print("=" * 60)

    print("""
    # Filesystem server (local file access)
    MCP.stdio(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"],
    )
    
    # GitHub server
    MCP.stdio(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_TOKEN": "your-token"},
    )
    
    # SQLite server
    MCP.stdio(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-sqlite", "/path/to/database.db"],
    )
    
    # Brave Search server
    MCP.stdio(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-brave-search"],
        env={"BRAVE_API_KEY": "your-key"},
    )
    
    # Custom Python server
    MCP.stdio(
        command="uv",
        args=["run", "my-custom-mcp-server"],
    )
    
    # Remote server with auth
    MCP.http(
        url="https://api.company.com/mcp",
        headers={
            "Authorization": "Bearer token",
            "X-Tenant-ID": "tenant123",
        },
    )
    """)


async def main() -> None:
    """Run all MCP demos."""
    print("\n" + "=" * 60)
    print("  MCP (Model Context Protocol) Integration")
    print("=" * 60)

    await demo_mcp_configuration()
    await demo_agent_integration()
    await demo_tool_naming()
    await demo_serialization()
    await demo_common_servers()

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print("""
    MCP Capability Features:
    
    ✅ Local servers (stdio transport)
       - Run any command as an MCP server
       - Environment variables and working directory support
    
    ✅ Remote servers (HTTP/SSE transport)
       - Connect to hosted MCP endpoints
       - Authentication headers support
    
    ✅ Multiple servers
       - Connect to many servers simultaneously
       - Tools automatically namespaced to avoid conflicts
    
    ✅ Automatic tool discovery
       - Tools discovered on connection
       - Refresh when servers notify of changes
    
    ✅ Seamless integration
       - MCP tools work like any other agent tool
       - Full LangChain tool compatibility
    
    Install MCP support:
        uv add agenticflow[mcp]
    """)


if __name__ == "__main__":
    asyncio.run(main())
