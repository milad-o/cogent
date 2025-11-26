"""Graph backend implementations.

This module provides different storage backends for the KnowledgeGraph:
- InMemoryGraph: Fast in-memory storage using networkx (if available)
- SQLiteGraph: Persistent storage using SQLite for large graphs
- JSONFileGraph: Simple JSON file persistence with auto-save
"""

from agenticflow.capabilities.knowledge_graph.backends.base import GraphBackend
from agenticflow.capabilities.knowledge_graph.backends.memory import InMemoryGraph
from agenticflow.capabilities.knowledge_graph.backends.sqlite import SQLiteGraph
from agenticflow.capabilities.knowledge_graph.backends.json_file import JSONFileGraph

__all__ = [
    "GraphBackend",
    "InMemoryGraph",
    "SQLiteGraph",
    "JSONFileGraph",
]
