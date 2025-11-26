"""
Capabilities - composable tools that plug into any agent.

Capabilities are reusable building blocks that add tools to agents:
- KnowledgeGraph: Entity/relationship memory with multi-hop reasoning
- CodebaseAnalyzer: Python AST parsing and code exploration
- FileSystem: Sandboxed file operations with security controls
- WebSearch: Multi-provider web search and fetch (coming soon)
- CodeSandbox: Safe code execution (coming soon)

Example:
    ```python
    from agenticflow import Agent
    from agenticflow.capabilities import KnowledgeGraph, FileSystem
    
    agent = Agent(
        name="Assistant",
        model=model,
        capabilities=[
            KnowledgeGraph(),                    # Adds remember, recall, query tools
            FileSystem(allowed_paths=["./data"]), # Adds read, write, search tools
        ],
    )
    
    # Agent automatically has all capability tools
    await agent.run("Read config.json and remember key settings")
    ```
"""

from agenticflow.capabilities.base import BaseCapability
from agenticflow.capabilities.codebase import CodebaseAnalyzer
from agenticflow.capabilities.filesystem import FileSystem
from agenticflow.capabilities.knowledge_graph import KnowledgeGraph

__all__ = [
    "BaseCapability",
    "CodebaseAnalyzer",
    "FileSystem",
    "KnowledgeGraph",
]
