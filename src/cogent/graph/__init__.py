"""Knowledge Graph module for Cogent.

This module provides a clean, modern API for building and querying knowledge graphs.
"""

from cogent.graph.models import Entity, Relationship
from cogent.graph.storage import Storage, MemoryStorage, FileStorage, SQLStorage
from cogent.graph.engines import Engine, NativeEngine
from cogent.graph.graph import Graph

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
    ]
