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
    from agenticflow.graphs import run
    from agenticflow.tools import tool
    
    @tool
    def search(query: str) -> str:
        '''Search the web.'''
        return f"Results for {query}"
    
    result = await run("Search for Python tutorials", tools=[search])
    
    # With Agent:
    from agenticflow.graphs import NativeExecutor
    
    executor = NativeExecutor(agent)
    result = await executor.execute("Research and calculate metrics")
"""

# Base classes and enums
from agenticflow.graphs.base import BaseExecutor, CompletionCheck, ExecutionStrategy

# Data models
from agenticflow.graphs.models import ExecutionPlan, ToolCall

# Native executors (default)
from agenticflow.graphs.native import NativeExecutor, SequentialExecutor, run

# Tree search executor
from agenticflow.graphs.tree_search import TreeSearchExecutor, SearchNode, NodeState, TreeSearchResult

# Factory
from agenticflow.graphs.factory import create_executor

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
