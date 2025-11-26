"""
Capabilities - composable tools that plug into any agent.

Capabilities are reusable building blocks that add tools to agents:
- KnowledgeGraph: Entity/relationship memory with multi-hop reasoning
- CodebaseAnalyzer: Python AST parsing and code exploration
- FileSystem: Sandboxed file operations with security controls
- WebSearch: Web search and page fetching (DuckDuckGo, free)
- CodeSandbox: Safe Python code execution

Example:
    ```python
    from agenticflow import Agent
    from agenticflow.capabilities import KnowledgeGraph, FileSystem, WebSearch, CodeSandbox
    
    agent = Agent(
        name="Assistant",
        model=model,
        capabilities=[
            KnowledgeGraph(),                    # Adds remember, recall, query tools
            FileSystem(allowed_paths=["./data"]), # Adds read, write, search tools
            WebSearch(),                          # Adds web_search, news_search, fetch tools
            CodeSandbox(),                        # Adds execute_python, run_function tools
        ],
    )
    
    # Agent automatically has all capability tools
    await agent.run("Write a function to calculate fibonacci and test it")
    ```
"""

from agenticflow.capabilities.base import BaseCapability
from agenticflow.capabilities.codebase import CodebaseAnalyzer
from agenticflow.capabilities.code_sandbox import CodeSandbox
from agenticflow.capabilities.filesystem import FileSystem
from agenticflow.capabilities.knowledge_graph import KnowledgeGraph
from agenticflow.capabilities.web_search import WebSearch

__all__ = [
    "BaseCapability",
    "CodebaseAnalyzer",
    "CodeSandbox",
    "FileSystem",
    "KnowledgeGraph",
    "WebSearch",
]
