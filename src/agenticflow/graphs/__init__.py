"""
Graph-based execution patterns for agents.

This module provides execution strategies that define HOW agents
process tasks. Our DAG-based executor is superior to simple ReAct
because it identifies dependencies and runs independent steps in parallel.

Module Structure:
    base.py     - BaseExecutor, ExecutionStrategy enum
    models.py   - ToolCall, ExecutionPlan data classes
    react.py    - ReActExecutor (think-act-observe loop)
    plan.py     - PlanExecutor (plan then execute)
    dag.py      - DAGExecutor (parallel waves) - RECOMMENDED
    adaptive.py - AdaptiveExecutor (auto-select strategy)
    factory.py  - create_executor() helper

Execution Strategies:
    DAGExecutor: Build dependency DAG, execute in parallel waves (FASTEST)
    PlanExecutor: Plan upfront, execute sequentially
    ReActExecutor: Think-Act-Observe loop (baseline)
    AdaptiveExecutor: Auto-select based on task complexity

Example DAG execution:
    search_A ─┐
              ├─► combine ─► final
    search_B ─┘
    
    search_A and search_B run in PARALLEL, then combine runs.

Usage:
    from agenticflow.graphs import DAGExecutor, create_executor
    
    # Use DAG execution (recommended for complex tasks)
    executor = DAGExecutor(agent)
    result = await executor.execute("Search A and B, then combine")
    
    # Or use factory with strategy enum
    executor = create_executor(agent, ExecutionStrategy.ADAPTIVE)
"""

# Base classes and enums
from agenticflow.graphs.base import BaseExecutor, ExecutionStrategy

# Data models
from agenticflow.graphs.models import ExecutionPlan, ToolCall

# Concrete executors
from agenticflow.graphs.dag import DAGExecutor
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
    # Base
    "BaseExecutor",
    # Executors (DAG first - it's our recommended approach)
    "DAGExecutor",
    "PlanExecutor",
    "ReActExecutor",
    "AdaptiveExecutor",
    # Factory
    "create_executor",
]
