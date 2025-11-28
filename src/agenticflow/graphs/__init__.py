"""
Graph-based execution patterns for agents.

This module provides execution strategies that define HOW agents
process tasks. Our DAG-based executor is superior to simple ReAct
because it identifies dependencies and runs independent steps in parallel.

Module Structure:
    base.py        - BaseExecutor, ExecutionStrategy enum
    models.py      - ToolCall, ExecutionPlan data classes
    react.py       - ReActExecutor (think-act-observe loop)
    plan.py        - PlanExecutor (plan then execute)
    dag.py         - DAGExecutor (parallel waves), StreamingDAGExecutor (LLMCompiler-style)
    tree_search.py - TreeSearchExecutor (LATS Monte Carlo tree search)
    adaptive.py    - AdaptiveExecutor (auto-select strategy)
    factory.py     - create_executor() helper

Execution Strategies:
    TreeSearchExecutor: LATS-style MCTS with backtracking (BEST ACCURACY)
    StreamingDAGExecutor: Stream execution as calls become ready (FASTEST, LLMCompiler-style)
    DAGExecutor: Build dependency DAG, execute in parallel waves
    PlanExecutor: Plan upfront, execute sequentially
    ReActExecutor: Think-Act-Observe loop (baseline)
    AdaptiveExecutor: Auto-select based on task complexity

Choosing an Executor:
    - TreeSearchExecutor: Best for complex tasks where accuracy matters more than speed
    - StreamingDAGExecutor: Best for speed when tasks have clear parallelism
    - DAGExecutor: Good default for most multi-step tasks
    - PlanExecutor: When you need predictable sequential execution
    - ReActExecutor: Simple tasks, debugging, or learning

Usage:
    from agenticflow.graphs import TreeSearchExecutor, StreamingDAGExecutor, create_executor
    
    # Use LATS tree search (best accuracy for complex tasks)
    executor = TreeSearchExecutor(agent)
    executor.max_iterations = 10
    executor.num_candidates = 3
    result = await executor.execute("Solve this complex problem")
    
    # Use streaming DAG (fastest for parallel tasks)
    executor = StreamingDAGExecutor(agent)
    result = await executor.execute("Search A and B, then combine")
"""

# Base classes and enums
from agenticflow.graphs.base import BaseExecutor, CompletionCheck, ExecutionStrategy

# Data models
from agenticflow.graphs.models import ExecutionPlan, ToolCall

# Concrete executors
from agenticflow.graphs.dag import DAGExecutor, StreamingDAGExecutor
from agenticflow.graphs.tree_search import TreeSearchExecutor, SearchNode, NodeState, TreeSearchResult
from agenticflow.graphs.plan import PlanExecutor
from agenticflow.graphs.react import ReActExecutor
from agenticflow.graphs.adaptive import AdaptiveExecutor

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
    # Executors (TreeSearch first - best accuracy, then StreamingDAG - fastest)
    "TreeSearchExecutor",
    "StreamingDAGExecutor",
    "DAGExecutor",
    "PlanExecutor",
    "ReActExecutor",
    "AdaptiveExecutor",
    # Factory
    "create_executor",
]
