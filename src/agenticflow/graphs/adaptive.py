"""
Adaptive execution strategy.

Automatically chooses the best strategy based on task complexity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agenticflow.graphs.base import BaseExecutor, ExecutionStrategy

if TYPE_CHECKING:
    from agenticflow.agent.base import Agent


class AdaptiveExecutor(BaseExecutor):
    """
    Adaptive execution that automatically chooses the best strategy.
    
    Analyzes the task to determine:
    - Simple task (no tools) → Direct thinking
    - Single tool → ReAct
    - Multiple independent tools → DAG (parallel)
    - Complex dependencies → DAG
    
    Pros:
        - No need to choose strategy manually
        - Adapts to task complexity
        
    Cons:
        - Extra LLM call to analyze task
        - May not always choose optimally
        
    Example:
        executor = AdaptiveExecutor(agent)
        result = await executor.execute("Do something complex")
        # Strategy is chosen automatically
    """
    
    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute using automatically selected strategy.
        
        Args:
            task: The task to execute.
            context: Optional context dictionary.
            
        Returns:
            The result from the chosen strategy.
        """
        # Import here to avoid circular imports
        from agenticflow.graphs.dag import DAGExecutor
        from agenticflow.graphs.plan import PlanExecutor
        from agenticflow.graphs.react import ReActExecutor
        
        # Analyze task to choose strategy
        strategy = await self._choose_strategy(task, context)
        self._emit_step("strategy_selected", {"strategy": strategy.value})
        
        if strategy == ExecutionStrategy.REACT:
            executor = ReActExecutor(self.agent)
        elif strategy == ExecutionStrategy.PLAN_EXECUTE:
            executor = PlanExecutor(self.agent)
        else:  # DAG is default for complex
            executor = DAGExecutor(self.agent)
        
        executor.max_iterations = self.max_iterations
        executor.on_step = self.on_step
        executor.tracker = self.tracker
        
        return await executor.execute(task, context)
    
    async def _choose_strategy(
        self,
        task: str,
        context: dict[str, Any] | None,
    ) -> ExecutionStrategy:
        """Analyze task to choose best strategy.
        
        Args:
            task: The task to analyze.
            context: Optional context.
            
        Returns:
            The recommended ExecutionStrategy.
        """
        if not self.agent.all_tools:
            return ExecutionStrategy.REACT  # No tools, just think
        
        # Ask agent to analyze complexity
        prompt = f"""Analyze this task's complexity:

Task: {task}

Respond with ONE word:
- SIMPLE: No tools needed, or single straightforward tool call
- SEQUENTIAL: Multiple tools, each depends on previous result  
- PARALLEL: Multiple independent tool calls possible
"""
        
        response = await self.agent.think(prompt)
        response_lower = response.lower()
        
        if "parallel" in response_lower:
            return ExecutionStrategy.DAG
        elif "sequential" in response_lower:
            return ExecutionStrategy.PLAN_EXECUTE
        else:
            return ExecutionStrategy.REACT
