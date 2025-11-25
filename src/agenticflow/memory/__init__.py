"""
Memory module - agent memory management.

This module provides memory capabilities for agents:
- Short-term memory (thread-scoped, conversation history)
- Long-term memory (cross-thread, persistent)
- Shared memory (multi-agent collaboration)
- Semantic memory (vector-based retrieval)
"""

from agenticflow.memory.base import (
    Memory as BaseMemory,
    MemoryConfig,
    MemoryEntry,
    MemoryType,
)
from agenticflow.memory.short_term import ShortTermMemory
from agenticflow.memory.long_term import LongTermMemory
from agenticflow.memory.shared import SharedMemory
from agenticflow.memory.manager import MemoryManager

__all__ = [
    # Base classes
    "BaseMemory",
    "MemoryConfig",
    "MemoryEntry",
    "MemoryType",
    # Memory implementations
    "ShortTermMemory",
    "LongTermMemory",
    "SharedMemory",
    # Manager
    "MemoryManager",
]
