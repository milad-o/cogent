#!/usr/bin/env python3
"""
Sample MCP Server - Web Search with DuckDuckGo

A useful MCP server that provides web search capabilities using DuckDuckGo.
Can be run with different transports:

    # Stdio (for local testing)
    uv run python examples/mcp_server/search_server.py stdio
    
    # HTTP with SSE (for remote/web)
    uv run python examples/mcp_server/search_server.py http --port 8000
    
    # WebSocket (for real-time bidirectional)
    uv run python examples/mcp_server/search_server.py websocket --port 8001

Requires: uv add ddgs
"""

import argparse
import asyncio
import json
from datetime import datetime

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
)

# DuckDuckGo search
from ddgs import DDGS


# Create the MCP server
server = Server("search-server")


# =============================================================================
# Tool Implementations
# =============================================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="web_search",
            description="Search the web using DuckDuckGo. Returns relevant web pages with titles, URLs, and snippets.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5, max: 20)",
                        "default": 5,
                    },
                    "region": {
                        "type": "string",
                        "description": "Region for results (e.g., 'us-en', 'uk-en', 'de-de')",
                        "default": "wt-wt",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="news_search",
            description="Search for recent news articles using DuckDuckGo News.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "News search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5, max: 20)",
                        "default": 5,
                    },
                    "timelimit": {
                        "type": "string",
                        "description": "Time limit: 'd' (day), 'w' (week), 'm' (month)",
                        "enum": ["d", "w", "m"],
                        "default": "w",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="image_search",
            description="Search for images using DuckDuckGo Images.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Image search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5, max: 20)",
                        "default": 5,
                    },
                    "size": {
                        "type": "string",
                        "description": "Image size filter",
                        "enum": ["Small", "Medium", "Large", "Wallpaper"],
                    },
                    "type_image": {
                        "type": "string",
                        "description": "Image type filter",
                        "enum": ["photo", "clipart", "gif", "transparent", "line"],
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="instant_answer",
            description="Get an instant answer from DuckDuckGo (definitions, calculations, quick facts).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Question or query for instant answer"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_suggestions",
            description="Get search suggestions/autocomplete for a query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Partial query to get suggestions for"},
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    
    try:
        if name == "web_search":
            return await _web_search(arguments)
        elif name == "news_search":
            return await _news_search(arguments)
        elif name == "image_search":
            return await _image_search(arguments)
        elif name == "instant_answer":
            return await _instant_answer(arguments)
        elif name == "get_suggestions":
            return await _get_suggestions(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


async def _web_search(arguments: dict) -> list[TextContent]:
    """Perform web search."""
    query = arguments["query"]
    max_results = min(arguments.get("max_results", 5), 20)
    region = arguments.get("region", "wt-wt")
    
    with DDGS() as ddgs:
        results = list(ddgs.text(query, region=region, max_results=max_results))
    
    if not results:
        return [TextContent(type="text", text=f"No results found for: {query}")]
    
    output = [f"Web search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        output.append(f"\n{i}. {r.get('title', 'No title')}")
        output.append(f"   URL: {r.get('href', 'N/A')}")
        # Include full body for better context
        body = r.get('body', 'No description')
        output.append(f"   {body}")
    
    return [TextContent(type="text", text="\n".join(output))]


async def _news_search(arguments: dict) -> list[TextContent]:
    """Perform news search."""
    query = arguments["query"]
    max_results = min(arguments.get("max_results", 5), 20)
    timelimit = arguments.get("timelimit", "w")
    
    with DDGS() as ddgs:
        results = list(ddgs.news(query, timelimit=timelimit, max_results=max_results))
    
    if not results:
        return [TextContent(type="text", text=f"No news found for: {query}")]
    
    output = [f"News results for: {query}\n"]
    for i, r in enumerate(results, 1):
        output.append(f"\n{i}. {r.get('title', 'No title')}")
        output.append(f"   Source: {r.get('source', 'Unknown')} | {r.get('date', 'N/A')}")
        output.append(f"   URL: {r.get('url', 'N/A')}")
        output.append(f"   {r.get('body', 'No description')[:150]}...")
    
    return [TextContent(type="text", text="\n".join(output))]


async def _image_search(arguments: dict) -> list[TextContent]:
    """Perform image search."""
    query = arguments["query"]
    max_results = min(arguments.get("max_results", 5), 20)
    size = arguments.get("size")
    type_image = arguments.get("type_image")
    
    with DDGS() as ddgs:
        results = list(ddgs.images(
            query,
            max_results=max_results,
            size=size,
            type_image=type_image,
        ))
    
    if not results:
        return [TextContent(type="text", text=f"No images found for: {query}")]
    
    output = [f"Image results for: {query}\n"]
    for i, r in enumerate(results, 1):
        output.append(f"\n{i}. {r.get('title', 'No title')}")
        output.append(f"   Image: {r.get('image', 'N/A')}")
        output.append(f"   Source: {r.get('source', 'N/A')}")
        output.append(f"   Size: {r.get('width', '?')}x{r.get('height', '?')}")
    
    return [TextContent(type="text", text="\n".join(output))]


async def _instant_answer(arguments: dict) -> list[TextContent]:
    """Get instant answer via focused web search."""
    query = arguments["query"]
    
    # The ddgs package no longer has answers() method
    # Use a focused text search instead
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
    
    if not results:
        return [TextContent(type="text", text=f"No instant answer for: {query}")]
    
    output = [f"Quick facts for: {query}\n"]
    for r in results:
        output.append(f"\n• {r.get('title', 'No title')}")
        output.append(f"  {r.get('body', 'No content')}")
        if r.get("href"):
            output.append(f"  Source: {r.get('href')}")
    
    return [TextContent(type="text", text="\n".join(output))]


async def _get_suggestions(arguments: dict) -> list[TextContent]:
    """Get search suggestions."""
    query = arguments["query"]
    
    with DDGS() as ddgs:
        results = list(ddgs.suggestions(query))
    
    if not results:
        return [TextContent(type="text", text=f"No suggestions for: {query}")]
    
    output = [f"Search suggestions for: {query}\n"]
    for r in results[:10]:
        output.append(f"  • {r.get('phrase', r)}")
    
    return [TextContent(type="text", text="\n".join(output))]


# =============================================================================
# Transport Runners
# =============================================================================

async def run_stdio():
    """Run server with stdio transport (for local use)."""
    # IMPORTANT: Don't print to stdout - it's used for JSON-RPC!
    # Use stderr for any logging
    import sys
    print("Starting MCP Search Server (stdio)...", file=sys.stderr, flush=True)
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


async def run_http(host: str, port: int):
    """Run server with SSE transport (Server-Sent Events over HTTP)."""
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Route, Mount
    from starlette.responses import JSONResponse, Response
    import uvicorn
    
    print(f"Starting MCP Search Server (SSE) on http://{host}:{port}")
    print(f"  - SSE endpoint: http://{host}:{port}/sse")
    print(f"  - Message endpoint: http://{host}:{port}/messages/")
    
    # Create SSE transport
    sse = SseServerTransport("/messages/")
    
    async def handle_sse(request):
        """Handle SSE connection."""
        async with sse.connect_sse(
            request.scope, request.receive, request._send  # type: ignore[reportPrivateUsage]
        ) as streams:
            await server.run(
                streams[0], streams[1], server.create_initialization_options()
            )
        # Return empty response to avoid NoneType error when client disconnects
        return Response()
    
    async def health(request):
        return JSONResponse({"status": "ok", "server": "search-server"})
    
    app = Starlette(
        routes=[
            Route("/health", health),
            Route("/sse", handle_sse, methods=["GET"]),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )
    
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server_instance = uvicorn.Server(config)
    await server_instance.serve()


async def run_websocket(host: str, port: int):
    """Run server with WebSocket transport."""
    import websockets
    from websockets.server import serve
    
    print(f"Starting MCP Search Server (WebSocket) on ws://{host}:{port}")
    
    async def handle_connection(websocket):
        """Handle a WebSocket connection."""
        print(f"Client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                
                # Handle MCP JSON-RPC messages
                if data.get("method") == "tools/list":
                    tools = await list_tools()
                    response = {
                        "jsonrpc": "2.0",
                        "id": data.get("id"),
                        "result": {"tools": [t.model_dump() for t in tools]},
                    }
                    await websocket.send(json.dumps(response))
                
                elif data.get("method") == "tools/call":
                    params = data.get("params", {})
                    result = await call_tool(params.get("name"), params.get("arguments", {}))
                    response = {
                        "jsonrpc": "2.0",
                        "id": data.get("id"),
                        "result": {"content": [c.model_dump() for c in result]},
                    }
                    await websocket.send(json.dumps(response))
                
                elif data.get("method") == "initialize":
                    response = {
                        "jsonrpc": "2.0",
                        "id": data.get("id"),
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "serverInfo": {"name": "search-server", "version": "1.0.0"},
                            "capabilities": {"tools": {}},
                        },
                    }
                    await websocket.send(json.dumps(response))
                
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": data.get("id"),
                        "error": {"code": -32601, "message": f"Unknown method: {data.get('method')}"},
                    }
                    await websocket.send(json.dumps(response))
        
        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {websocket.remote_address}")
    
    async with serve(handle_connection, host, port):
        print(f"WebSocket server listening on ws://{host}:{port}")
        await asyncio.Future()  # Run forever


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MCP Search Server (DuckDuckGo)")
    parser.add_argument(
        "transport",
        choices=["stdio", "http", "websocket"],
        default="stdio",
        nargs="?",
        help="Transport type (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    
    args = parser.parse_args()
    
    if args.transport == "stdio":
        asyncio.run(run_stdio())
    elif args.transport == "http":
        asyncio.run(run_http(args.host, args.port))
    elif args.transport == "websocket":
        asyncio.run(run_websocket(args.host, args.port))


if __name__ == "__main__":
    main()
