"""
MCP (Model Context Protocol) Capability.

Connect to local and remote MCP servers and use their tools seamlessly.
Supports stdio (local), HTTP/SSE, and WebSocket transports.

Example:
    ```python
    from agenticflow import Agent
    from agenticflow.capabilities import MCP

    # Connect to a local MCP server (stdio)
    agent = Agent(
        name="Assistant",
        model=model,
        capabilities=[
            MCP.stdio(
                command="uv",
                args=["run", "my-mcp-server"],
            ),
        ],
    )

    # Connect to a remote MCP server (HTTP/Streamable HTTP)
    agent = Agent(
        name="Assistant",
        model=model,
        capabilities=[
            MCP.http("https://api.example.com/mcp"),
        ],
    )

    # Connect via WebSocket
    agent = Agent(
        name="Assistant",
        model=model,
        capabilities=[
            MCP.websocket("ws://localhost:8766"),
        ],
    )

    # Connect to multiple servers
    agent = Agent(
        name="Assistant",
        model=model,
        capabilities=[
            MCP.stdio(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "."]),
            MCP.http("https://weather-api.example.com/mcp"),
        ],
    )
    ```
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from agenticflow.capabilities.base import BaseCapability
from agenticflow.tools.base import BaseTool

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.types import Tool as MCPTool

    from agenticflow.agent.base import Agent

logger = logging.getLogger(__name__)


class MCPTransport(str, Enum):
    """MCP transport types."""

    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"
    WEBSOCKET = "websocket"


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""

    transport: MCPTransport

    # Stdio transport options
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None

    # HTTP/SSE transport options
    url: str | None = None
    headers: dict[str, str] | None = None
    timeout: float = 30.0

    # Common options
    name: str | None = None  # Optional name for this server connection

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.transport == MCPTransport.STDIO:
            if not self.command:
                msg = "command is required for stdio transport"
                raise ValueError(msg)
        elif self.transport in (MCPTransport.HTTP, MCPTransport.SSE, MCPTransport.WEBSOCKET):
            if not self.url:
                msg = "url is required for HTTP/SSE/WebSocket transport"
                raise ValueError(msg)


@dataclass
class MCPToolInfo:
    """Information about a discovered MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str


class MCP(BaseCapability):
    """
    MCP (Model Context Protocol) capability.

    Connects to MCP servers and exposes their tools to agents.
    Supports local (stdio) and remote (HTTP/SSE) servers.

    Features:
    - Auto-discovery of server tools on connection
    - Dynamic tool refresh when servers notify of changes
    - Support for multiple concurrent server connections
    - Graceful connection management and cleanup

    Example:
        ```python
        # Local server via stdio
        mcp = MCP.stdio(command="uv", args=["run", "my-server"])

        # Remote server via HTTP
        mcp = MCP.http("https://api.example.com/mcp")

        # Multiple servers
        mcp = MCP(servers=[
            MCPServerConfig(transport=MCPTransport.STDIO, command="npx", args=["server"]),
            MCPServerConfig(transport=MCPTransport.HTTP, url="https://api.example.com/mcp"),
        ])
        ```
    """

    def __init__(
        self,
        servers: list[MCPServerConfig] | None = None,
        *,
        auto_refresh: bool = True,
        tool_name_prefix: str | None = None,
    ) -> None:
        """
        Initialize MCP capability.

        Args:
            servers: List of MCP server configurations.
            auto_refresh: Whether to auto-refresh tools when servers notify of changes.
            tool_name_prefix: Optional prefix for tool names (e.g., "mcp_").
        """
        self._servers = servers or []
        self._auto_refresh = auto_refresh
        self._tool_name_prefix = tool_name_prefix

        # Runtime state
        self._sessions: dict[str, ClientSession] = {}
        self._contexts: list[Any] = []  # Context managers for cleanup
        self._discovered_tools: dict[str, MCPToolInfo] = {}
        self._native_tools: list[BaseTool] = []
        self._initialized = False

    async def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event through the agent's event bus if available."""
        if self._agent and hasattr(self._agent, "event_bus") and self._agent.event_bus:
            from agenticflow.observability.trace_record import Trace, TraceType
            try:
                agent_id = self._agent.id if hasattr(self._agent, "id") else None
                event = Trace(
                    type=TraceType(event_type),
                    data={"capability": "mcp", "agent_id": agent_id, **data},
                    source=f"capability:mcp:{agent_id or 'unknown'}",
                )
                await self._agent.event_bus.publish(event)
            except ValueError:
                # Unknown event type, log and continue
                logger.debug("Unknown event type: %s", event_type)

    @classmethod
    def stdio(
        cls,
        command: str,
        args: list[str] | None = None,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        name: str | None = None,
        auto_refresh: bool = True,
        tool_name_prefix: str | None = None,
    ) -> MCP:
        """
        Create MCP capability with a local stdio server.

        Args:
            command: Command to run the server (e.g., "uv", "npx", "python").
            args: Arguments to pass to the command.
            env: Environment variables for the server process.
            cwd: Working directory for the server process.
            name: Optional name for this server connection.
            auto_refresh: Whether to auto-refresh tools on changes.
            tool_name_prefix: Optional prefix for tool names.

        Returns:
            MCP capability configured for stdio transport.

        Example:
            ```python
            # Run a Python MCP server
            mcp = MCP.stdio(command="uv", args=["run", "my-mcp-server"])

            # Run an npm MCP server
            mcp = MCP.stdio(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "."])
            ```
        """
        config = MCPServerConfig(
            transport=MCPTransport.STDIO,
            command=command,
            args=args or [],
            env=env,
            cwd=cwd,
            name=name,
        )
        return cls(servers=[config], auto_refresh=auto_refresh, tool_name_prefix=tool_name_prefix)

    @classmethod
    def http(
        cls,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        name: str | None = None,
        auto_refresh: bool = True,
        tool_name_prefix: str | None = None,
    ) -> MCP:
        """
        Create MCP capability with a remote HTTP server.

        Args:
            url: URL of the MCP server (e.g., "https://api.example.com/mcp").
            headers: HTTP headers for authentication/authorization.
            timeout: Connection timeout in seconds.
            name: Optional name for this server connection.
            auto_refresh: Whether to auto-refresh tools on changes.
            tool_name_prefix: Optional prefix for tool names.

        Returns:
            MCP capability configured for HTTP transport.

        Example:
            ```python
            # Public MCP server
            mcp = MCP.http("https://weather-api.example.com/mcp")

            # Authenticated server
            mcp = MCP.http(
                "https://api.example.com/mcp",
                headers={"Authorization": "Bearer token123"},
            )
            ```
        """
        config = MCPServerConfig(
            transport=MCPTransport.HTTP,
            url=url,
            headers=headers,
            timeout=timeout,
            name=name,
        )
        return cls(servers=[config], auto_refresh=auto_refresh, tool_name_prefix=tool_name_prefix)

    @classmethod
    def sse(
        cls,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        name: str | None = None,
        auto_refresh: bool = True,
        tool_name_prefix: str | None = None,
    ) -> MCP:
        """
        Create MCP capability with a remote SSE server.

        Args:
            url: URL of the MCP SSE endpoint.
            headers: HTTP headers for authentication/authorization.
            timeout: Connection timeout in seconds.
            name: Optional name for this server connection.
            auto_refresh: Whether to auto-refresh tools on changes.
            tool_name_prefix: Optional prefix for tool names.

        Returns:
            MCP capability configured for SSE transport.

        Note:
            SSE transport is being superseded by Streamable HTTP transport.
            Consider using `MCP.http()` for new integrations.
        """
        config = MCPServerConfig(
            transport=MCPTransport.SSE,
            url=url,
            headers=headers,
            timeout=timeout,
            name=name,
        )
        return cls(servers=[config], auto_refresh=auto_refresh, tool_name_prefix=tool_name_prefix)

    @classmethod
    def websocket(
        cls,
        url: str,
        *,
        name: str | None = None,
        auto_refresh: bool = True,
        tool_name_prefix: str | None = None,
    ) -> MCP:
        """
        Create MCP capability with a WebSocket server.

        Args:
            url: WebSocket URL (ws:// or wss://).
            name: Optional name for this server connection.
            auto_refresh: Whether to auto-refresh tools on changes.
            tool_name_prefix: Optional prefix for tool names.

        Returns:
            MCP capability configured for WebSocket transport.

        Example:
            ```python
            mcp = MCP.websocket("ws://localhost:8766")
            ```
        """
        config = MCPServerConfig(
            transport=MCPTransport.WEBSOCKET,
            url=url,
            name=name,
        )
        return cls(servers=[config], auto_refresh=auto_refresh, tool_name_prefix=tool_name_prefix)

    @property
    def name(self) -> str:
        """Unique name for this capability."""
        return "mcp"

    @property
    def description(self) -> str:
        """Human-readable description."""
        server_count = len(self._servers)
        tool_count = len(self._discovered_tools)
        return f"MCP capability ({server_count} servers, {tool_count} tools)"

    @property
    def tools(self) -> list[BaseTool]:
        """Tools discovered from MCP servers."""
        return self._native_tools

    @property
    def servers(self) -> list[MCPServerConfig]:
        """Configured MCP servers."""
        return self._servers

    @property
    def discovered_tools(self) -> dict[str, MCPToolInfo]:
        """Tools discovered from connected servers."""
        return self._discovered_tools

    def add_server(self, config: MCPServerConfig) -> None:
        """
        Add an MCP server configuration.

        Args:
            config: Server configuration to add.

        Note:
            If already initialized, you'll need to call `refresh_tools()` to connect.
        """
        self._servers.append(config)

    async def initialize(self, agent: Agent) -> None:
        """
        Initialize capability and connect to MCP servers.

        Args:
            agent: The agent this capability is attached to.
        """
        await super().initialize(agent)

        if not self._servers:
            logger.warning("MCP capability initialized with no servers configured")
            return

        # Connect to all configured servers
        await self._connect_all()
        self._initialized = True

    async def shutdown(self) -> None:
        """Disconnect from all MCP servers and cleanup."""
        await self._disconnect_all()
        await super().shutdown()

    async def refresh_tools(self) -> None:
        """
        Refresh tools from all connected servers.

        Call this to manually refresh the tool list, or when a server
        sends a tools/list_changed notification.
        """
        self._discovered_tools.clear()
        self._native_tools.clear()

        for server_name, session in self._sessions.items():
            try:
                await self._discover_tools(session, server_name)
            except Exception as e:
                logger.error("Failed to discover tools from %s: %s", server_name, e)

        logger.info("Discovered %d MCP tools total", len(self._discovered_tools))

    async def _connect_all(self) -> None:
        """Connect to all configured MCP servers."""
        tasks = [self._connect_server(config) for config in self._servers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for config, result in zip(self._servers, results, strict=False):
            if isinstance(result, Exception):
                server_name = config.name or config.command or config.url
                logger.error("Failed to connect to MCP server %s: %s", server_name, result)

    async def _disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        server_count = len(self._sessions)
        tool_count = len(self._discovered_tools)

        # Emit disconnection event
        await self._emit("mcp.server.disconnected", {
            "server_count": server_count,
            "tool_count": tool_count,
        })

        # Close all context managers
        # Note: MCP uses anyio internally, which requires contexts to be
        # entered/exited in the same task. When cleanup happens from a
        # different task (common in agent shutdown), we may get cancel scope
        # errors. These are harmless - the underlying resources are still freed.
        for ctx in reversed(self._contexts):
            try:
                await ctx.__aexit__(None, None, None)
            except RuntimeError as e:
                # Suppress anyio cancel scope errors - resources are still freed
                if "cancel scope" in str(e).lower():
                    logger.debug("MCP context cleanup from different task (harmless): %s", e)
                else:
                    logger.error("Error closing MCP context: %s", e)
            except Exception as e:
                logger.error("Error closing MCP context: %s", e)

        self._contexts.clear()
        self._sessions.clear()
        self._discovered_tools.clear()
        self._native_tools.clear()
        self._initialized = False

    async def _connect_server(self, config: MCPServerConfig) -> None:
        """Connect to a single MCP server."""
        server_name = config.name or config.command or config.url or "unknown"

        # Emit connecting event
        await self._emit("mcp.server.connecting", {
            "server_name": server_name,
            "transport": config.transport.value,
            "url": config.url,
            "command": config.command,
        })

        try:
            if config.transport == MCPTransport.STDIO:
                await self._connect_stdio(config, server_name)
            elif config.transport == MCPTransport.HTTP:
                await self._connect_http(config, server_name)
            elif config.transport == MCPTransport.SSE:
                await self._connect_sse(config, server_name)
            elif config.transport == MCPTransport.WEBSOCKET:
                await self._connect_websocket(config, server_name)
            else:
                msg = f"Unsupported transport: {config.transport}"
                raise ValueError(msg)

            # Emit connected event
            await self._emit("mcp.server.connected", {
                "server_name": server_name,
                "transport": config.transport.value,
                "tool_count": len([t for t in self._discovered_tools.values() if t.server_name == server_name]),
            })
            logger.info("Connected to MCP server: %s", server_name)

        except ImportError as e:
            logger.error(
                "MCP SDK not installed. Install with: uv add mcp\nError: %s",
                e,
            )
            raise
        except Exception as e:
            # Emit error event
            await self._emit("mcp.server.error", {
                "server_name": server_name,
                "transport": config.transport.value,
                "error": str(e),
            })
            logger.error("Failed to connect to MCP server %s: %s", server_name, e)
            raise

    async def _connect_stdio(self, config: MCPServerConfig, server_name: str) -> None:
        """Connect to an MCP server via stdio transport."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env,
            cwd=config.cwd,
        )

        # Enter the stdio_client context
        client_ctx = stdio_client(server_params)
        read_stream, write_stream = await client_ctx.__aenter__()
        self._contexts.append(client_ctx)

        # Create and initialize session
        session_ctx = ClientSession(read_stream, write_stream)
        session = await session_ctx.__aenter__()
        self._contexts.append(session_ctx)

        await session.initialize()
        self._sessions[server_name] = session

        # Discover tools
        await self._discover_tools(session, server_name)

    async def _connect_http(self, config: MCPServerConfig, server_name: str) -> None:
        """Connect to an MCP server via HTTP transport."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        # Enter the HTTP client context
        client_ctx = streamablehttp_client(
            config.url,
            headers=config.headers,
            timeout=config.timeout,
        )
        read_stream, write_stream, _ = await client_ctx.__aenter__()
        self._contexts.append(client_ctx)

        # Create and initialize session
        session_ctx = ClientSession(read_stream, write_stream)
        session = await session_ctx.__aenter__()
        self._contexts.append(session_ctx)

        await session.initialize()
        self._sessions[server_name] = session

        # Discover tools
        await self._discover_tools(session, server_name)

    async def _connect_sse(self, config: MCPServerConfig, server_name: str) -> None:
        """Connect to an MCP server via SSE transport."""
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        # Enter the SSE client context
        client_ctx = sse_client(
            config.url,
            headers=config.headers,
            timeout=config.timeout,
        )
        read_stream, write_stream = await client_ctx.__aenter__()
        self._contexts.append(client_ctx)

        # Create and initialize session
        session_ctx = ClientSession(read_stream, write_stream)
        session = await session_ctx.__aenter__()
        self._contexts.append(session_ctx)

        await session.initialize()
        self._sessions[server_name] = session

        # Discover tools
        await self._discover_tools(session, server_name)

    async def _connect_websocket(self, config: MCPServerConfig, server_name: str) -> None:
        """Connect to an MCP server via WebSocket transport."""
        from mcp import ClientSession
        from mcp.client.websocket import websocket_client

        # Enter the WebSocket client context
        client_ctx = websocket_client(config.url)
        read_stream, write_stream = await client_ctx.__aenter__()
        self._contexts.append(client_ctx)

        # Create and initialize session
        session_ctx = ClientSession(read_stream, write_stream)
        session = await session_ctx.__aenter__()
        self._contexts.append(session_ctx)

        await session.initialize()
        self._sessions[server_name] = session

        # Discover tools
        await self._discover_tools(session, server_name)

    async def _discover_tools(self, session: ClientSession, server_name: str) -> None:
        """Discover tools from an MCP server."""

        tools_response = await session.list_tools()
        discovered_tool_names = []

        for tool in tools_response.tools:
            tool_name = self._make_tool_name(tool.name, server_name)

            # Store tool info
            self._discovered_tools[tool_name] = MCPToolInfo(
                name=tool.name,
                description=tool.description or f"MCP tool: {tool.name}",
                input_schema=tool.inputSchema if isinstance(tool.inputSchema, dict) else {},
                server_name=server_name,
            )

            # Create native tool
            native_tool = self._create_native_tool(tool, server_name, tool_name)
            self._native_tools.append(native_tool)
            discovered_tool_names.append(tool_name)

            logger.debug("Discovered tool: %s from %s", tool_name, server_name)

        # Emit tools discovered event
        await self._emit("mcp.tools.discovered", {
            "server_name": server_name,
            "tool_count": len(discovered_tool_names),
            "tools": discovered_tool_names,
        })

    def _make_tool_name(self, original_name: str, server_name: str) -> str:
        """Create a unique tool name."""
        # Add prefix if configured
        if self._tool_name_prefix:
            name = f"{self._tool_name_prefix}{original_name}"
        else:
            name = original_name

        # If multiple servers, add server suffix for uniqueness
        if len(self._servers) > 1:
            # Clean server name for use in tool name
            clean_server = server_name.replace(".", "_").replace("/", "_").replace("-", "_")
            name = f"{name}_{clean_server}"

        return name

    def _create_native_tool(
        self,
        mcp_tool: MCPTool,
        server_name: str,
        tool_name: str,
    ) -> BaseTool:
        """Create a native tool wrapper for an MCP tool."""
        # Capture self for event emission in closure
        capability = self

        async def call_mcp_tool(**kwargs: Any) -> str:
            """Call the MCP tool."""
            session = capability._sessions.get(server_name)
            if not session:
                return f"Error: Not connected to MCP server {server_name}"

            # Emit tool called event
            await capability._emit("mcp.tool.called", {
                "tool_name": tool_name,
                "mcp_tool_name": mcp_tool.name,
                "server_name": server_name,
                "arguments": kwargs,
            })

            try:
                result = await session.call_tool(mcp_tool.name, arguments=kwargs)

                # Parse result content
                output_parts = []
                for content in result.content:
                    if hasattr(content, "text"):
                        output_parts.append(content.text)
                    elif hasattr(content, "data"):
                        output_parts.append(f"[Binary data: {len(content.data)} bytes]")
                    else:
                        output_parts.append(str(content))

                # Include structured content if available
                if result.structuredContent:
                    output_parts.append(f"\nStructured: {json.dumps(result.structuredContent)}")

                result_text = "\n".join(output_parts) if output_parts else "Tool executed successfully"

                # Emit tool result event
                await capability._emit("mcp.tool.result", {
                    "tool_name": tool_name,
                    "mcp_tool_name": mcp_tool.name,
                    "server_name": server_name,
                    "result_length": len(result_text),
                })

                return result_text

            except Exception as e:
                # Emit tool error event
                await capability._emit("mcp.tool.error", {
                    "tool_name": tool_name,
                    "mcp_tool_name": mcp_tool.name,
                    "server_name": server_name,
                    "error": str(e),
                })
                logger.error("Error calling MCP tool %s: %s", mcp_tool.name, e)
                return f"Error calling tool: {e}"

        # Build args schema from input schema
        input_schema = mcp_tool.inputSchema if isinstance(mcp_tool.inputSchema, dict) else {}
        args_schema = self._build_args_schema_dict(input_schema)

        return BaseTool(
            name=tool_name,
            description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
            func=call_mcp_tool,
            args_schema=args_schema,
        )

    def _build_args_schema_dict(self, input_schema: dict[str, Any]) -> dict[str, Any]:
        """Build args schema dict from JSON schema for native BaseTool."""
        properties = input_schema.get("properties", {})
        args_schema: dict[str, Any] = {}

        for prop_name, prop_schema in properties.items():
            args_schema[prop_name] = {
                "type": prop_schema.get("type", "string"),
            }
            if "description" in prop_schema:
                args_schema[prop_name]["description"] = prop_schema["description"]

        return args_schema

    def _build_args_schema(self, tool_name: str, input_schema: dict[str, Any]) -> type | None:
        """Build a Pydantic model for tool arguments from JSON schema."""

        from pydantic import Field, create_model

        properties = input_schema.get("properties", {})
        required = set(input_schema.get("required", []))

        if not properties:
            return None

        # Build field definitions
        fields: dict[str, Any] = {}
        for prop_name, prop_schema in properties.items():
            prop_type = self._json_type_to_python(prop_schema.get("type", "string"))
            prop_description = prop_schema.get("description", "")
            prop_default = prop_schema.get("default", ...)

            if prop_name in required:
                fields[prop_name] = (prop_type, Field(description=prop_description))
            else:
                default = prop_default if prop_default != ... else None
                fields[prop_name] = (
                    prop_type | None,
                    Field(default=default, description=prop_description),
                )

        # Create dynamic model
        model_name = f"{tool_name.title().replace('_', '')}Args"
        return create_model(model_name, **fields)

    def _json_type_to_python(self, json_type: str) -> type:
        """Convert JSON schema type to Python type."""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        return type_mapping.get(json_type, str)

    def to_dict(self) -> dict[str, Any]:
        """Convert capability info to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "servers": [
                {
                    "transport": s.transport.value,
                    "name": s.name or s.command or s.url,
                }
                for s in self._servers
            ],
            "tool_count": len(self._discovered_tools),
            "tools": list(self._discovered_tools.keys()),
            "initialized": self._initialized,
        }

    def __repr__(self) -> str:
        return (
            f"MCP(servers={len(self._servers)}, "
            f"tools={len(self._discovered_tools)}, "
            f"initialized={self._initialized})"
        )
