"""Knowledge Graph module for Cogent.

This module provides a clean, modern API for building and querying knowledge graphs.
"""

from cogent.graph import visualization
from cogent.graph.engines import Engine, NativeEngine
from cogent.graph.graph import Graph
from cogent.graph.models import Entity, Relationship
from cogent.graph.query import QueryPattern, QueryResult, match, parse_pattern
from cogent.graph.storage import FileStorage, MemoryStorage, SQLStorage, Storage

# Conditionally import NetworkXEngine
try:
    from cogent.graph.engines import NetworkXEngine
    __all__ = [
        "Graph",
        "Entity",
        "Relationship",
        "Engine",
        "NativeEngine",
        "NetworkXEngine",
        "Storage",
        "MemoryStorage",
        "FileStorage",
        "SQLStorage",
        "QueryPattern",
        "QueryResult",
        "parse_pattern",
        "match",
        "visualization",
    ]
except ImportError:
    __all__ = [
        "Graph",
        "Entity",
        "Relationship",
        "Engine",
        "NativeEngine",
        "Storage",
        "MemoryStorage",
        "FileStorage",
        "SQLStorage",
        "QueryPattern",
        "QueryResult",
        "parse_pattern",
        "match",
        "visualization",
    ]
