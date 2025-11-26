"""
Plan-and-Execute execution strategy.

Plans all steps upfront, then executes sequentially.
Faster than ReAct because planning is done once.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agenticflow.core.utils import now_utc
from agenticflow.graphs.base import BaseExecutor
from agenticflow.graphs.models import ExecutionPlan


class PlanExecutor(BaseExecutor):
    """
    Plan-and-Execute strategy.
    
    Pattern: Plan all steps → Execute sequentially
    
    The agent first creates a complete plan of what tools to call,
    then executes them one by one.
    
    Pros:
        - Planning is done once upfront
        - Faster than ReAct for multi-step tasks
        - Clear execution structure
        
    Cons:
        - Still sequential execution
        - Plan may need revision if steps fail
        - No parallelism
        
    Example:
        executor = PlanExecutor(agent)
        result = await executor.execute("Research topic X and summarize")
    """
    
    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute using plan-and-execute pattern.
        
        Args:
            task: The task to execute.
            context: Optional context dictionary.
            
        Returns:
            The synthesized final answer.
        """
        # Phase 1: Create plan
        self._emit_step("planning", {"task": task})
        plan = await self._create_plan(task, context)
        
        if not plan.calls:
            # No tools needed, just think
            return await self.agent.think(task)
        
        # Phase 2: Execute plan sequentially
        results: dict[str, Any] = {}
        for call in plan.calls:
            self._emit_step("executing", {"tool": call.tool_name, "id": call.id})
            
            # Substitute results from previous calls
            args = self._substitute_args(call.args, results)
            
            # Track the tool call
            self._track_tool_call(call.tool_name, args)
            
            try:
                start = now_utc()
                result = await self.agent.act(call.tool_name, args)
                call.duration_ms = (now_utc() - start).total_seconds() * 1000
                call.result = result
                call.status = "completed"
                results[call.id] = result
                
                # Track the result
                self._track_tool_result(call.tool_name, result, call.duration_ms)
            except Exception as e:
                call.error = str(e)
                call.status = "failed"
                results[call.id] = f"ERROR: {e}"
                
                # Track the error
                self._track_tool_error(call.tool_name, str(e))
        
        # Phase 3: Synthesize final answer
        return await self._synthesize(task, plan, results)
    
    async def _create_plan(
        self,
        task: str,
        context: dict[str, Any] | None,
    ) -> ExecutionPlan:
        """Ask agent to create an execution plan.
        
        Args:
            task: The task to plan for.
            context: Optional context.
            
        Returns:
            ExecutionPlan with tool calls.
        """
        tools_desc = self.agent.get_tool_descriptions()
        
        prompt = f"""Create a step-by-step plan to accomplish this task:

Task: {task}
{f"Context: {json.dumps(context)}" if context else ""}

Available tools:
{tools_desc}

Respond with a JSON plan:
{{
  "steps": [
    {{"tool": "tool_name", "args": {{}}, "depends_on": []}},
    ...
  ]
}}

If no tools are needed, respond with: {{"steps": []}}
"""
        
        response = await self.agent.think(prompt)
        return self._parse_plan(response)
    
    def _parse_plan(self, response: str) -> ExecutionPlan:
        """Parse plan from agent response.
        
        Args:
            response: Raw response containing JSON plan.
            
        Returns:
            Parsed ExecutionPlan.
        """
        plan = ExecutionPlan()
        
        # Try to extract JSON
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                for step in data.get("steps", []):
                    plan.add_call(
                        tool_name=step["tool"],
                        args=step.get("args", {}),
                        depends_on=step.get("depends_on", []),
                    )
            except (json.JSONDecodeError, KeyError):
                pass
        
        return plan
    
    def _substitute_args(
        self,
        args: dict[str, Any],
        results: dict[str, Any],
    ) -> dict[str, Any]:
        """Substitute $call_N references with actual results.
        
        Args:
            args: Arguments with potential $references.
            results: Map of call_id -> result.
            
        Returns:
            Args with references resolved.
        """
        substituted = {}
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("$"):
                ref = value[1:]  # Remove $
                if ref in results:
                    substituted[key] = results[ref]
                else:
                    substituted[key] = value
            else:
                substituted[key] = value
        return substituted
    
    async def _synthesize(
        self,
        task: str,
        plan: ExecutionPlan,
        results: dict[str, Any],
    ) -> str:
        """Synthesize final answer from results.
        
        Args:
            task: Original task.
            plan: The execution plan.
            results: Map of call_id -> result.
            
        Returns:
            Synthesized answer string.
        """
        prompt = f"""Task: {task}

Tool results:
{json.dumps(results, indent=2, default=str)}

Based on these results, provide a final answer."""
        
        return await self.agent.think(prompt)
