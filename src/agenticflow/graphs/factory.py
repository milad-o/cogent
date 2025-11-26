"""
Factory function for creating executors.

Provides a simple way to create the right executor for a strategy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agenticflow.graphs.base import BaseExecutor, ExecutionStrategy

if TYPE_CHECKING:
    from agenticflow.agent.base import Agent


def create_executor(
    agent: Agent,
    strategy: ExecutionStrategy = ExecutionStrategy.DAG,
) -> BaseExecutor:
    """Create an executor with the specified strategy.
    
    This is the recommended way to create executors.
    DAG is the default because it provides the best performance
    for complex tasks with parallelizable steps.
    
    Args:
        agent: The agent to execute with.
        strategy: Execution strategy to use. Defaults to DAG.
        
    Returns:
        Configured executor instance.
        
    Example:
        # Use DAG executor (recommended for complex tasks)
        executor = create_executor(agent, ExecutionStrategy.DAG)
        result = await executor.execute("Search for X and Y, then combine")
        
        # Use ReAct for simple tasks
        executor = create_executor(agent, ExecutionStrategy.REACT)
        result = await executor.execute("What is 2+2?")
        
        # Let the system choose
        executor = create_executor(agent, ExecutionStrategy.ADAPTIVE)
        result = await executor.execute(task)
    """
    # Import here to avoid circular imports at module load time
    from agenticflow.graphs.adaptive import AdaptiveExecutor
    from agenticflow.graphs.dag import DAGExecutor
    from agenticflow.graphs.plan import PlanExecutor
    from agenticflow.graphs.react import ReActExecutor
    
    executors = {
        ExecutionStrategy.REACT: ReActExecutor,
        ExecutionStrategy.PLAN_EXECUTE: PlanExecutor,
        ExecutionStrategy.DAG: DAGExecutor,
        ExecutionStrategy.ADAPTIVE: AdaptiveExecutor,
    }
    
    executor_class = executors.get(strategy)
    if executor_class is None:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return executor_class(agent)
