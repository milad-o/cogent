"""
Factory function for creating executors.

Provides a simple way to create the right executor for a strategy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agenticflow.executors.base import BaseExecutor, ExecutionStrategy

if TYPE_CHECKING:
    from agenticflow.agent.base import Agent


def create_executor(
    agent: Agent,
    strategy: ExecutionStrategy = ExecutionStrategy.NATIVE,
) -> BaseExecutor:
    """Create an executor with the specified strategy.
    
    This is the recommended way to create executors.
    NATIVE is the default for best performance.
    
    Args:
        agent: The agent to execute with.
        strategy: Execution strategy to use. Defaults to NATIVE.
        
    Returns:
        Configured executor instance.
        
    Example:
        # Use Native executor (recommended for most tasks)
        executor = create_executor(agent, ExecutionStrategy.NATIVE)
        result = await executor.execute("Do something")
        
        # Use TreeSearch for complex tasks requiring exploration
        executor = create_executor(agent, ExecutionStrategy.TREE_SEARCH)
        result = await executor.execute("Solve this complex problem")
        
        # Use Sequential for ordered tool execution
        executor = create_executor(agent, ExecutionStrategy.SEQUENTIAL)
        result = await executor.execute(task)
    """
    # Import here to avoid circular imports at module load time
    from agenticflow.executors.native import NativeExecutor, SequentialExecutor
    from agenticflow.executors.tree_search import TreeSearchExecutor
    
    executors = {
        ExecutionStrategy.NATIVE: NativeExecutor,
        ExecutionStrategy.SEQUENTIAL: SequentialExecutor,
        ExecutionStrategy.TREE_SEARCH: TreeSearchExecutor,
    }
    
    executor_class = executors.get(strategy)
    if executor_class is None:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return executor_class(agent)
