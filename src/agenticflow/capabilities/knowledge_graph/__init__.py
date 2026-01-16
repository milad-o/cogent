"""
KnowledgeGraph capability for persistent entity/relationship memory.

This module provides a knowledge graph capability with multiple storage backends:
- In-memory graph (uses networkx if available)
- SQLite database for persistence and large graphs
- JSON file for simple persistence with auto-save

Example:
    ```python
    from agenticflow.capabilities import KnowledgeGraph

    # In-memory (default)
    kg = KnowledgeGraph()

    # SQLite for persistence
    kg = KnowledgeGraph(backend="sqlite", path="knowledge.db")

    # Or load from file (auto-detects backend)
    kg = KnowledgeGraph.from_file("knowledge.db")
    ```
"""

from agenticflow.capabilities.knowledge_graph.backends import (
    GraphBackend,
    InMemoryGraph,
    JSONFileGraph,
    SQLiteGraph,
)
from agenticflow.capabilities.knowledge_graph.capability import KnowledgeGraph
from agenticflow.capabilities.knowledge_graph.models import Entity, Relationship

__all__ = [
    # Main capability
    "KnowledgeGraph",
    # Models
    "Entity",
    "Relationship",
    # Backends
    "GraphBackend",
    "InMemoryGraph",
    "SQLiteGraph",
    "JSONFileGraph",
]
