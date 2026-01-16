"""
Execution strategies for agents.

This module provides execution strategies that define HOW agents process tasks.

Module Structure:
    base.py        - BaseExecutor, ExecutionStrategy enum
    models.py      - ToolCall, ExecutionPlan data classes
    native.py      - NativeExecutor (default), SequentialExecutor, run()
    tree_search.py - TreeSearchExecutor (LATS Monte Carlo tree search)
    factory.py     - create_executor() helper

Execution Strategies:
    NativeExecutor: High-performance parallel execution (DEFAULT)
    SequentialExecutor: Sequential tool execution for ordered tasks
    TreeSearchExecutor: LATS-style MCTS with backtracking (BEST ACCURACY)

Standalone Execution:
    run(): Execute tasks without creating an Agent (fastest path)

Usage:
    # Standalone execution (no Agent needed):
    from agenticflow.executors import run
    from agenticflow.tools import tool

    @tool
    def search(query: str) -> str:
        '''Search the web.'''
        return f"Results for {query}"

    result = await run("Search for Python tutorials", tools=[search])

    # With Agent:
    from agenticflow.executors import NativeExecutor

    executor = NativeExecutor(agent)
    result = await executor.execute("Research and calculate metrics")
"""

# Base classes and enums
from agenticflow.executors.base import BaseExecutor, CompletionCheck, ExecutionStrategy

# Factory
from agenticflow.executors.factory import create_executor

# Data models
from agenticflow.executors.models import ExecutionPlan, ToolCall

# Native executors (default)
from agenticflow.executors.native import NativeExecutor, SequentialExecutor, run

# Tree search executor
from agenticflow.executors.tree_search import (
    NodeState,
    SearchNode,
    TreeSearchExecutor,
    TreeSearchResult,
)

__all__ = [
    # Strategy enum
    "ExecutionStrategy",
    # Data classes
    "ToolCall",
    "ExecutionPlan",
    "CompletionCheck",
    # Tree search classes
    "SearchNode",
    "NodeState",
    "TreeSearchResult",
    # Base
    "BaseExecutor",
    # Executors
    "NativeExecutor",
    "SequentialExecutor",
    "TreeSearchExecutor",
    # Standalone execution
    "run",
    # Factory
    "create_executor",
]
