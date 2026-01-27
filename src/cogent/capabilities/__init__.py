"""
Capabilities - composable tools that plug into any agent.

Capabilities are reusable building blocks that add tools to agents:
- KnowledgeGraph: Entity/relationship memory with multi-hop reasoning
- CodebaseAnalyzer: Python AST parsing and code exploration
- FileSystem: Sandboxed file operations with security controls
- WebSearch: Web search and page fetching (DuckDuckGo, free)
- CodeSandbox: Safe Python code execution
- SSISAnalyzer: SSIS package analysis and data lineage tracing
- MCP: Connect to local/remote MCP servers and use their tools
- Spreadsheet: Excel/CSV reading, writing, and analysis
- Browser: Headless browser automation with Playwright
- Shell: Sandboxed terminal command execution
- Summarizer: Document summarization (map-reduce, refine, hierarchical)

Example:
    ```python
    from cogent import Agent
    from cogent.capabilities import (
        KnowledgeGraph, FileSystem, WebSearch, CodeSandbox, MCP,
        Spreadsheet, Browser, Shell, Summarizer,
    )

    agent = Agent(
        name="Assistant",
        model=model,
        capabilities=[
            KnowledgeGraph(),                     # Adds remember, recall, query tools
            FileSystem(allowed_paths=["./data"]), # Adds read, write, search tools
            WebSearch(),                          # Adds web_search, news_search, fetch tools
            CodeSandbox(),                        # Adds execute_python, run_function tools
            Spreadsheet(),                        # Adds read/write spreadsheet tools
            Browser(headless=True),               # Adds browser automation tools
            Shell(allowed_commands=["ls", "cat"]), # Adds shell command tools
            Summarizer(),                         # Adds summarize_text, summarize_file tools
            MCP.stdio(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "."]),
        ],
    )

    # Agent automatically has all capability tools
    await agent.run("Write a function to calculate fibonacci and test it")
    ```
"""

from cogent.capabilities.base import BaseCapability
from cogent.capabilities.browser import Browser
from cogent.capabilities.code_sandbox import CodeSandbox
from cogent.capabilities.codebase import CodebaseAnalyzer
from cogent.capabilities.filesystem import FileSystem
from cogent.capabilities.knowledge_graph import KnowledgeGraph
from cogent.capabilities.mcp import MCP, MCPServerConfig, MCPTransport
from cogent.capabilities.shell import Shell
from cogent.capabilities.spreadsheet import Spreadsheet
from cogent.capabilities.summarizer import Summarizer, SummarizerConfig
from cogent.capabilities.web_search import WebSearch

__all__ = [
    "BaseCapability",
    "Browser",
    "CodebaseAnalyzer",
    "CodeSandbox",
    "FileSystem",
    "KnowledgeGraph",
    "MCP",
    "MCPServerConfig",
    "MCPTransport",
    "Shell",
    "Spreadsheet",
    "Summarizer",
    "SummarizerConfig",
    "WebSearch",
]
