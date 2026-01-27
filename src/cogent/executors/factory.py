"""
Factory function for creating executors.

Provides a simple way to create the right executor for a strategy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cogent.executors.base import BaseExecutor, ExecutionStrategy

if TYPE_CHECKING:
    from cogent.agent.base import Agent


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

        # Use Sequential for ordered tool execution
        executor = create_executor(agent, ExecutionStrategy.SEQUENTIAL)
        result = await executor.execute(task)
    """
    # Import here to avoid circular imports at module load time
    from cogent.executors.native import NativeExecutor, SequentialExecutor

    executors = {
        ExecutionStrategy.NATIVE: NativeExecutor,
        ExecutionStrategy.SEQUENTIAL: SequentialExecutor,
    }

    executor_class = executors.get(strategy)
    if executor_class is None:
        raise ValueError(f"Unknown strategy: {strategy}. Available: NATIVE, SEQUENTIAL")

    return executor_class(agent)
