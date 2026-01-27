"""
Execution strategies for agents.

This module provides execution strategies that define HOW agents process tasks.

Module Structure:
    base.py        - BaseExecutor, ExecutionStrategy enum
    models.py      - ToolCall, ExecutionPlan data classes
    native.py      - NativeExecutor (default), SequentialExecutor, run()
    factory.py     - create_executor() helper

Execution Strategies:
    NativeExecutor: High-performance parallel execution (DEFAULT)
    SequentialExecutor: Sequential tool execution for ordered tasks

Standalone Execution:
    run(): Execute tasks without creating an Agent (fastest path)

Usage:
    # Standalone execution (no Agent needed):
    from cogent.executors import run
    from cogent.tools import tool

    @tool
    def search(query: str) -> str:
        '''Search the web.'''
        return f"Results for {query}"

    result = await run("Search for Python tutorials", tools=[search])

    # With Agent:
    from cogent.executors import NativeExecutor

    executor = NativeExecutor(agent)
    result = await executor.execute("Research and calculate metrics")
"""

# Base classes and enums
from cogent.executors.base import BaseExecutor, CompletionCheck, ExecutionStrategy

# Factory
from cogent.executors.factory import create_executor

# Data models
from cogent.executors.models import ExecutionPlan, ToolCall

# Native executors (default)
from cogent.executors.native import NativeExecutor, SequentialExecutor, run

# Tree search executor - REMOVED (was part of multi-agent orchestration)
# from cogent.executors.tree_search import (
#     NodeState,
#     SearchNode,
#     TreeSearchExecutor,
#     TreeSearchResult,
# )

__all__ = [
    # Strategy enum
    "ExecutionStrategy",
    # Data classes
    "ToolCall",
    "ExecutionPlan",
    "CompletionCheck",
    # Tree search classes - REMOVED
    # "SearchNode",
    # "NodeState",
    # "TreeSearchResult",
    # Base
    "BaseExecutor",
    # Executors
    "NativeExecutor",
    "SequentialExecutor",
    # "TreeSearchExecutor",  # REMOVED
    # Standalone execution
    "run",
    # Factory
    "create_executor",
]
