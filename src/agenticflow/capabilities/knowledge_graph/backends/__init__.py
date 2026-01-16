"""Graph backend implementations.

This module provides different storage backends for the KnowledgeGraph:
- InMemoryGraph: Fast in-memory storage using networkx (if available)
- SQLiteGraph: Persistent storage using SQLite for large graphs
- JSONFileGraph: Simple JSON file persistence with auto-save
- Neo4jGraph: Production graph database with Cypher queries
"""

from agenticflow.capabilities.knowledge_graph.backends.base import GraphBackend
from agenticflow.capabilities.knowledge_graph.backends.json_file import JSONFileGraph
from agenticflow.capabilities.knowledge_graph.backends.memory import InMemoryGraph
from agenticflow.capabilities.knowledge_graph.backends.sqlite import SQLiteGraph

# Neo4j is optional - only import if neo4j package is installed
try:
    from agenticflow.capabilities.knowledge_graph.backends.neo4j import Neo4jGraph
    _HAS_NEO4J = True
except ImportError:
    Neo4jGraph = None  # type: ignore
    _HAS_NEO4J = False

__all__ = [
    "GraphBackend",
    "InMemoryGraph",
    "SQLiteGraph",
    "JSONFileGraph",
    "Neo4jGraph",
]
