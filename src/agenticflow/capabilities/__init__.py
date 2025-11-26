"""
Capabilities - composable tools that plug into any agent.

Capabilities are reusable building blocks that add tools to agents:
- KnowledgeGraph: Entity/relationship memory with multi-hop reasoning
- WebSearch: Multi-provider web search and fetch (coming soon)
- CodeSandbox: Safe code execution (coming soon)

Example:
    ```python
    from agenticflow import Agent
    from agenticflow.capabilities import KnowledgeGraph, WebSearch
    
    agent = Agent(
        name="ResearchAssistant",
        model=model,
        capabilities=[
            KnowledgeGraph(),      # Adds remember, recall, query tools
            WebSearch(),           # Adds search, fetch tools
        ],
    )
    
    # Agent automatically has all capability tools
    await agent.run("Research AI trends and remember key facts")
    ```
"""

from agenticflow.capabilities.base import BaseCapability
from agenticflow.capabilities.codebase import CodebaseAnalyzer
from agenticflow.capabilities.knowledge_graph import KnowledgeGraph

__all__ = [
    "BaseCapability",
    "CodebaseAnalyzer",
    "KnowledgeGraph",
]
