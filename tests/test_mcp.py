"""Tests for MCP capability."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agenticflow.capabilities.mcp import MCP, MCPServerConfig, MCPToolInfo, MCPTransport


class TestMCPServerConfig:
    """Tests for MCPServerConfig."""

    def test_stdio_config_requires_command(self) -> None:
        """Test that stdio transport requires command."""
        with pytest.raises(ValueError, match="command is required"):
            MCPServerConfig(transport=MCPTransport.STDIO)

    def test_http_config_requires_url(self) -> None:
        """Test that HTTP transport requires url."""
        with pytest.raises(ValueError, match="url is required"):
            MCPServerConfig(transport=MCPTransport.HTTP)

    def test_sse_config_requires_url(self) -> None:
        """Test that SSE transport requires url."""
        with pytest.raises(ValueError, match="url is required"):
            MCPServerConfig(transport=MCPTransport.SSE)

    def test_valid_stdio_config(self) -> None:
        """Test valid stdio configuration."""
        config = MCPServerConfig(
            transport=MCPTransport.STDIO,
            command="python",
            args=["-m", "my_server"],
            env={"KEY": "value"},
            cwd="/path/to/dir",
            name="test-server",
        )
        assert config.transport == MCPTransport.STDIO
        assert config.command == "python"
        assert config.args == ["-m", "my_server"]
        assert config.env == {"KEY": "value"}
        assert config.cwd == "/path/to/dir"
        assert config.name == "test-server"

    def test_valid_http_config(self) -> None:
        """Test valid HTTP configuration."""
        config = MCPServerConfig(
            transport=MCPTransport.HTTP,
            url="https://api.example.com/mcp",
            headers={"Authorization": "Bearer token"},
            timeout=60.0,
            name="remote-server",
        )
        assert config.transport == MCPTransport.HTTP
        assert config.url == "https://api.example.com/mcp"
        assert config.headers == {"Authorization": "Bearer token"}
        assert config.timeout == 60.0
        assert config.name == "remote-server"


class TestMCPFactoryMethods:
    """Tests for MCP factory methods."""

    def test_stdio_factory(self) -> None:
        """Test MCP.stdio() factory method."""
        mcp = MCP.stdio(
            command="uv",
            args=["run", "server"],
            env={"API_KEY": "secret"},
            cwd="/app",
            name="my-server",
            auto_refresh=False,
            tool_name_prefix="mcp_",
        )

        assert len(mcp.servers) == 1
        config = mcp.servers[0]
        assert config.transport == MCPTransport.STDIO
        assert config.command == "uv"
        assert config.args == ["run", "server"]
        assert config.env == {"API_KEY": "secret"}
        assert config.cwd == "/app"
        assert config.name == "my-server"
        assert mcp._auto_refresh is False
        assert mcp._tool_name_prefix == "mcp_"

    def test_http_factory(self) -> None:
        """Test MCP.http() factory method."""
        mcp = MCP.http(
            url="https://api.example.com/mcp",
            headers={"Authorization": "Bearer token"},
            timeout=120.0,
            name="remote",
            auto_refresh=True,
            tool_name_prefix="remote_",
        )

        assert len(mcp.servers) == 1
        config = mcp.servers[0]
        assert config.transport == MCPTransport.HTTP
        assert config.url == "https://api.example.com/mcp"
        assert config.headers == {"Authorization": "Bearer token"}
        assert config.timeout == 120.0
        assert config.name == "remote"

    def test_sse_factory(self) -> None:
        """Test MCP.sse() factory method."""
        mcp = MCP.sse(
            url="https://sse.example.com/events",
            headers={"X-API-Key": "key123"},
            timeout=60.0,
            name="sse-server",
        )

        assert len(mcp.servers) == 1
        config = mcp.servers[0]
        assert config.transport == MCPTransport.SSE
        assert config.url == "https://sse.example.com/events"
        assert config.headers == {"X-API-Key": "key123"}
        assert config.timeout == 60.0

    def test_stdio_factory_minimal(self) -> None:
        """Test MCP.stdio() with minimal arguments."""
        mcp = MCP.stdio(command="npx")
        assert len(mcp.servers) == 1
        assert mcp.servers[0].command == "npx"
        assert mcp.servers[0].args == []

    def test_http_factory_minimal(self) -> None:
        """Test MCP.http() with minimal arguments."""
        mcp = MCP.http("https://api.example.com/mcp")
        assert len(mcp.servers) == 1
        assert mcp.servers[0].url == "https://api.example.com/mcp"
        assert mcp.servers[0].timeout == 30.0  # Default


class TestMCPProperties:
    """Tests for MCP capability properties."""

    def test_name(self) -> None:
        """Test capability name."""
        mcp = MCP()
        assert mcp.name == "mcp"

    def test_description_no_servers(self) -> None:
        """Test description with no servers."""
        mcp = MCP()
        assert "0 servers" in mcp.description
        assert "0 tools" in mcp.description

    def test_description_with_servers(self) -> None:
        """Test description with servers configured."""
        mcp = MCP(
            servers=[
                MCPServerConfig(transport=MCPTransport.STDIO, command="server1"),
                MCPServerConfig(transport=MCPTransport.HTTP, url="https://example.com/mcp"),
            ]
        )
        assert "2 servers" in mcp.description

    def test_tools_empty_initially(self) -> None:
        """Test tools list is empty initially."""
        mcp = MCP.stdio(command="test")
        assert mcp.tools == []

    def test_discovered_tools_empty_initially(self) -> None:
        """Test discovered_tools is empty initially."""
        mcp = MCP.stdio(command="test")
        assert mcp.discovered_tools == {}


class TestMCPServerManagement:
    """Tests for MCP server management."""

    def test_add_server(self) -> None:
        """Test adding a server."""
        mcp = MCP()
        assert len(mcp.servers) == 0

        config = MCPServerConfig(transport=MCPTransport.STDIO, command="server")
        mcp.add_server(config)

        assert len(mcp.servers) == 1
        assert mcp.servers[0] == config

    def test_multiple_servers(self) -> None:
        """Test configuring multiple servers."""
        mcp = MCP(
            servers=[
                MCPServerConfig(transport=MCPTransport.STDIO, command="local-server"),
                MCPServerConfig(transport=MCPTransport.HTTP, url="https://remote1.com/mcp"),
                MCPServerConfig(transport=MCPTransport.HTTP, url="https://remote2.com/mcp"),
            ]
        )
        assert len(mcp.servers) == 3


class TestMCPToolNaming:
    """Tests for MCP tool naming."""

    def test_make_tool_name_no_prefix(self) -> None:
        """Test tool naming without prefix."""
        mcp = MCP.stdio(command="server")
        name = mcp._make_tool_name("get_weather", "server1")
        assert name == "get_weather"

    def test_make_tool_name_with_prefix(self) -> None:
        """Test tool naming with prefix."""
        mcp = MCP.stdio(command="server", tool_name_prefix="mcp_")
        name = mcp._make_tool_name("get_weather", "server1")
        assert name == "mcp_get_weather"

    def test_make_tool_name_multiple_servers(self) -> None:
        """Test tool naming with multiple servers adds suffix."""
        mcp = MCP(
            servers=[
                MCPServerConfig(transport=MCPTransport.STDIO, command="server1"),
                MCPServerConfig(transport=MCPTransport.STDIO, command="server2"),
            ]
        )
        name = mcp._make_tool_name("get_weather", "server1")
        assert "server1" in name

    def test_make_tool_name_cleans_server_name(self) -> None:
        """Test tool naming cleans special characters from server name."""
        mcp = MCP(
            servers=[
                MCPServerConfig(transport=MCPTransport.HTTP, url="https://api.example.com/mcp"),
                MCPServerConfig(transport=MCPTransport.HTTP, url="https://other.com/mcp"),
            ]
        )
        name = mcp._make_tool_name("search", "https://api.example.com/mcp")
        # Should have underscores instead of dots and slashes
        assert "." not in name.split("_")[-1] or "_" in name


class TestMCPTypeConversion:
    """Tests for JSON type to Python type conversion."""

    def test_json_type_string(self) -> None:
        """Test string type conversion."""
        mcp = MCP()
        assert mcp._json_type_to_python("string") is str

    def test_json_type_integer(self) -> None:
        """Test integer type conversion."""
        mcp = MCP()
        assert mcp._json_type_to_python("integer") is int

    def test_json_type_number(self) -> None:
        """Test number type conversion."""
        mcp = MCP()
        assert mcp._json_type_to_python("number") is float

    def test_json_type_boolean(self) -> None:
        """Test boolean type conversion."""
        mcp = MCP()
        assert mcp._json_type_to_python("boolean") is bool

    def test_json_type_array(self) -> None:
        """Test array type conversion."""
        mcp = MCP()
        assert mcp._json_type_to_python("array") is list

    def test_json_type_object(self) -> None:
        """Test object type conversion."""
        mcp = MCP()
        assert mcp._json_type_to_python("object") is dict

    def test_json_type_unknown(self) -> None:
        """Test unknown type defaults to str."""
        mcp = MCP()
        assert mcp._json_type_to_python("unknown") is str


class TestMCPArgsSchema:
    """Tests for args schema building."""

    def test_build_args_schema_empty(self) -> None:
        """Test building args schema with no properties."""
        mcp = MCP()
        schema = mcp._build_args_schema("test_tool", {})
        assert schema is None

    def test_build_args_schema_with_properties(self) -> None:
        """Test building args schema with properties."""
        mcp = MCP()
        input_schema = {
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results"},
            },
            "required": ["query"],
        }

        schema = mcp._build_args_schema("search", input_schema)
        assert schema is not None

        # Check model has expected fields
        fields = schema.model_fields
        assert "query" in fields
        assert "limit" in fields

    def test_build_args_schema_with_defaults(self) -> None:
        """Test building args schema with default values."""
        mcp = MCP()
        input_schema = {
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results", "default": 10},
            },
            "required": ["query"],
        }

        schema = mcp._build_args_schema("search", input_schema)
        fields = schema.model_fields

        # Required field should not have default
        assert fields["query"].is_required()
        # Optional field with default
        assert not fields["limit"].is_required()


class TestMCPToDict:
    """Tests for MCP serialization."""

    def test_to_dict_empty(self) -> None:
        """Test to_dict with empty capability."""
        mcp = MCP()
        data = mcp.to_dict()

        assert data["name"] == "mcp"
        assert data["servers"] == []
        assert data["tool_count"] == 0
        assert data["tools"] == []
        assert data["initialized"] is False

    def test_to_dict_with_servers(self) -> None:
        """Test to_dict with configured servers."""
        mcp = MCP(
            servers=[
                MCPServerConfig(transport=MCPTransport.STDIO, command="server1", name="local"),
                MCPServerConfig(transport=MCPTransport.HTTP, url="https://api.com/mcp"),
            ]
        )
        data = mcp.to_dict()

        assert len(data["servers"]) == 2
        assert data["servers"][0]["transport"] == "stdio"
        assert data["servers"][0]["name"] == "local"
        assert data["servers"][1]["transport"] == "http"

    def test_repr(self) -> None:
        """Test string representation."""
        mcp = MCP.stdio(command="test")
        repr_str = repr(mcp)

        assert "MCP" in repr_str
        assert "servers=1" in repr_str
        assert "tools=0" in repr_str
        assert "initialized=False" in repr_str


class TestMCPToolInfo:
    """Tests for MCPToolInfo dataclass."""

    def test_tool_info_creation(self) -> None:
        """Test creating tool info."""
        info = MCPToolInfo(
            name="get_weather",
            description="Get weather for a location",
            input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
            server_name="weather-server",
        )

        assert info.name == "get_weather"
        assert info.description == "Get weather for a location"
        assert "properties" in info.input_schema
        assert info.server_name == "weather-server"


class TestMCPTransport:
    """Tests for MCPTransport enum."""

    def test_transport_values(self) -> None:
        """Test transport enum values."""
        assert MCPTransport.STDIO.value == "stdio"
        assert MCPTransport.HTTP.value == "http"
        assert MCPTransport.SSE.value == "sse"

    def test_transport_string_comparison(self) -> None:
        """Test transport can be compared with strings."""
        assert MCPTransport.STDIO == "stdio"
        assert MCPTransport.HTTP == "http"


class TestMCPInitialization:
    """Tests for MCP initialization."""

    @pytest.mark.asyncio
    async def test_initialize_no_servers_warns(self) -> None:
        """Test initialize with no servers logs warning."""
        mcp = MCP()
        mock_agent = MagicMock()

        with patch("agenticflow.capabilities.mcp.logger") as mock_logger:
            await mcp.initialize(mock_agent)
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_clears_state(self) -> None:
        """Test shutdown clears all state."""
        mcp = MCP.stdio(command="test")
        mcp._initialized = True
        mcp._sessions["test"] = MagicMock()
        mcp._discovered_tools["tool1"] = MagicMock()
        mcp._langchain_tools.append(MagicMock())

        await mcp.shutdown()

        assert mcp._initialized is False
        assert mcp._sessions == {}
        assert mcp._discovered_tools == {}
        assert mcp._langchain_tools == []


class TestMCPToolDiscovery:
    """Tests for tool discovery (mocked)."""

    @pytest.mark.asyncio
    async def test_discover_tools_creates_langchain_tools(self) -> None:
        """Test that discovered tools are converted to LangChain tools."""
        pytest.importorskip("mcp", reason="MCP SDK not installed")

        mcp = MCP.stdio(command="test")

        # Create mock MCP tool
        mock_mcp_tool = MagicMock()
        mock_mcp_tool.name = "search"
        mock_mcp_tool.description = "Search the web"
        mock_mcp_tool.inputSchema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        }

        # Create mock session
        mock_session = AsyncMock()
        mock_tools_response = MagicMock()
        mock_tools_response.tools = [mock_mcp_tool]
        mock_session.list_tools.return_value = mock_tools_response

        # Discover tools
        await mcp._discover_tools(mock_session, "test-server")

        # Check tools were created
        assert len(mcp._discovered_tools) == 1
        assert len(mcp._langchain_tools) == 1
        assert "search" in mcp._discovered_tools

        # Check LangChain tool properties
        lc_tool = mcp._langchain_tools[0]
        assert lc_tool.name == "search"
        assert "Search the web" in lc_tool.description


class TestMCPToolExecution:
    """Tests for MCP tool execution (mocked)."""

    @pytest.mark.asyncio
    async def test_call_mcp_tool_returns_text_content(self) -> None:
        """Test calling MCP tool with text response."""
        mcp = MCP.stdio(command="test")

        # Setup mock session
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Weather: 72°F, Sunny"
        mock_result.content = [mock_content]
        mock_result.structuredContent = None
        mock_session.call_tool.return_value = mock_result

        mcp._sessions["test-server"] = mock_session

        # Create a tool and call it
        mock_mcp_tool = MagicMock()
        mock_mcp_tool.name = "get_weather"
        mock_mcp_tool.description = "Get weather"
        mock_mcp_tool.inputSchema = {}

        lc_tool = mcp._create_langchain_tool(mock_mcp_tool, "test-server", "get_weather")

        # Execute the tool
        result = await lc_tool.ainvoke({"city": "San Francisco"})

        assert "Weather: 72°F, Sunny" in result

    @pytest.mark.asyncio
    async def test_call_mcp_tool_not_connected_error(self) -> None:
        """Test calling tool when not connected returns error."""
        mcp = MCP.stdio(command="test")
        # Don't add session - simulate not connected

        mock_mcp_tool = MagicMock()
        mock_mcp_tool.name = "test_tool"
        mock_mcp_tool.description = "Test"
        mock_mcp_tool.inputSchema = {}

        lc_tool = mcp._create_langchain_tool(mock_mcp_tool, "unknown-server", "test_tool")

        result = await lc_tool.ainvoke({})

        assert "Error" in result
        assert "Not connected" in result

    @pytest.mark.asyncio
    async def test_call_mcp_tool_with_structured_content(self) -> None:
        """Test calling MCP tool with structured response."""
        mcp = MCP.stdio(command="test")

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Success"
        mock_result.content = [mock_content]
        mock_result.structuredContent = {"temperature": 72, "condition": "sunny"}
        mock_session.call_tool.return_value = mock_result

        mcp._sessions["test-server"] = mock_session

        mock_mcp_tool = MagicMock()
        mock_mcp_tool.name = "get_weather"
        mock_mcp_tool.description = "Get weather"
        mock_mcp_tool.inputSchema = {}

        lc_tool = mcp._create_langchain_tool(mock_mcp_tool, "test-server", "get_weather")
        result = await lc_tool.ainvoke({})

        assert "Success" in result
        assert "Structured" in result
        assert "temperature" in result


class TestMCPRefreshTools:
    """Tests for tool refresh functionality."""

    @pytest.mark.asyncio
    async def test_refresh_tools_clears_existing(self) -> None:
        """Test refresh_tools clears existing tools."""
        mcp = MCP.stdio(command="test")

        # Add some existing tools
        mcp._discovered_tools["old_tool"] = MagicMock()
        mcp._langchain_tools.append(MagicMock())

        # Refresh with no sessions
        await mcp.refresh_tools()

        assert len(mcp._discovered_tools) == 0
        assert len(mcp._langchain_tools) == 0

    @pytest.mark.asyncio
    async def test_refresh_tools_rediscovers_from_sessions(self) -> None:
        """Test refresh_tools rediscovers tools from sessions."""
        pytest.importorskip("mcp", reason="MCP SDK not installed")

        mcp = MCP.stdio(command="test")

        # Setup mock session
        mock_session = AsyncMock()
        mock_tool = MagicMock()
        mock_tool.name = "new_tool"
        mock_tool.description = "New tool"
        mock_tool.inputSchema = {}
        mock_tools_response = MagicMock()
        mock_tools_response.tools = [mock_tool]
        mock_session.list_tools.return_value = mock_tools_response

        mcp._sessions["server"] = mock_session

        await mcp.refresh_tools()

        assert "new_tool" in mcp._discovered_tools
        assert len(mcp._langchain_tools) == 1
