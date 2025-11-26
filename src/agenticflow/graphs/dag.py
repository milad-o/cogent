"""
DAG-based execution with maximum parallelism.

This is our RECOMMENDED execution strategy for complex tasks.
It builds a dependency graph and executes independent steps in parallel.

LLMCompiler-style approach:
1. Agent creates a DAG plan identifying dependencies
2. Execute independent steps in parallel "waves"
3. Synthesize final answer from all results
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from agenticflow.core.utils import now_utc
from agenticflow.graphs.base import BaseExecutor
from agenticflow.graphs.models import ExecutionPlan, ToolCall


class DAGExecutor(BaseExecutor):
    """
    DAG-based execution with maximum parallelism (LLMCompiler-style).
    
    Pattern: Plan DAG → Execute in parallel waves → Synthesize
    
    This is the FASTEST approach for complex tasks. It identifies 
    dependencies between tool calls and executes independent steps 
    in parallel.
    
    Example DAG:
        search_A ─┐
                  ├─► combine ─► final
        search_B ─┘
        
    search_A and search_B run in PARALLEL (wave 1), 
    then combine runs (wave 2).
    
    Pros:
        - Maximum parallelism
        - Fastest for complex multi-tool tasks
        - Efficient use of async
        
    Cons:
        - Requires good dependency planning from LLM
        - More complex error handling
        
    Example:
        executor = DAGExecutor(agent)
        result = await executor.execute(
            "Search for topic A and topic B, then combine findings"
        )
    """
    
    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute using DAG with parallel execution.
        
        Args:
            task: The task to execute.
            context: Optional context dictionary.
            
        Returns:
            The synthesized final answer.
        """
        # Phase 1: Create DAG plan
        self._emit_step("planning_dag", {"task": task})
        plan = await self._create_dag_plan(task, context)
        
        if not plan.calls:
            return await self.agent.think(task)
        
        # Phase 2: Execute DAG with parallelism
        waves = plan.get_execution_order()
        self._emit_step("dag_waves", {"waves": len(waves), "total_calls": len(plan.calls)})
        
        results: dict[str, Any] = {}
        
        for wave_idx, wave in enumerate(waves):
            wave_calls = [c for c in plan.calls if c.id in wave]
            self._emit_step("executing_wave", {
                "wave": wave_idx + 1,
                "total_waves": len(waves),
                "parallel_calls": len(wave_calls),
            })
            
            # Execute wave in parallel
            wave_results = await self._execute_wave(wave_calls, results)
            results.update(wave_results)
        
        # Phase 3: Synthesize
        return await self._synthesize(task, plan, results)
    
    async def _create_dag_plan(
        self,
        task: str,
        context: dict[str, Any] | None,
    ) -> ExecutionPlan:
        """Create a DAG plan identifying dependencies.
        
        Asks the agent to create a plan with explicit dependencies,
        so we know which steps can run in parallel.
        
        Args:
            task: The task to plan for.
            context: Optional context.
            
        Returns:
            ExecutionPlan with dependency information.
        """
        tools_desc = self.agent.tool_registry.get_tool_descriptions() if self.agent.tool_registry else "No tools available"
        
        prompt = f"""Create an execution plan as a Directed Acyclic Graph (DAG).
Identify which steps can run IN PARALLEL (no dependencies between them).

Task: {task}
{f"Context: {json.dumps(context)}" if context else ""}

Available tools:
{tools_desc}

Respond with a JSON DAG plan:
{{
  "steps": [
    {{"id": "call_0", "tool": "tool_name", "args": {{}}, "depends_on": []}},
    {{"id": "call_1", "tool": "other_tool", "args": {{}}, "depends_on": []}},
    {{"id": "call_2", "tool": "combine", "args": {{"a": "$call_0", "b": "$call_1"}}, "depends_on": ["call_0", "call_1"]}}
  ]
}}

Rules:
- Steps with NO dependencies can run in parallel
- Use $call_N to reference results from previous steps
- List all dependencies in depends_on array
"""
        
        response = await self.agent.think(prompt)
        return self._parse_dag_plan(response)
    
    def _parse_dag_plan(self, response: str) -> ExecutionPlan:
        """Parse DAG plan from agent response.
        
        Args:
            response: Raw response containing JSON plan.
            
        Returns:
            Parsed ExecutionPlan with dependencies.
        """
        plan = ExecutionPlan()
        
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                for step in data.get("steps", []):
                    call = ToolCall(
                        id=step.get("id", f"call_{len(plan.calls)}"),
                        tool_name=step["tool"],
                        args=step.get("args", {}),
                        depends_on=step.get("depends_on", []),
                    )
                    plan.calls.append(call)
            except (json.JSONDecodeError, KeyError):
                pass
        
        return plan
    
    async def _execute_wave(
        self,
        calls: list[ToolCall],
        prior_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a wave of calls in parallel.
        
        All calls in a wave have their dependencies satisfied,
        so they can run concurrently.
        
        Args:
            calls: List of ToolCalls to execute.
            prior_results: Results from previous waves.
            
        Returns:
            Dict mapping call_id to result.
        """
        
        async def execute_one(call: ToolCall) -> tuple[str, Any]:
            """Execute a single call."""
            args = self._substitute_args(call.args, prior_results)
            
            call.status = "running"
            start = now_utc()
            
            # Track the tool call
            self._track_tool_call(call.tool_name, args)
            
            try:
                result = await self.agent.act(call.tool_name, args)
                call.duration_ms = (now_utc() - start).total_seconds() * 1000
                call.result = result
                call.status = "completed"
                
                # Track the result
                self._track_tool_result(call.tool_name, result, call.duration_ms)
                
                return (call.id, result)
            except Exception as e:
                call.duration_ms = (now_utc() - start).total_seconds() * 1000
                call.error = str(e)
                call.status = "failed"
                
                # Track the error
                self._track_tool_error(call.tool_name, str(e))
                
                return (call.id, f"ERROR: {e}")
        
        # Execute all calls in parallel using asyncio.gather
        results = await asyncio.gather(*[execute_one(c) for c in calls])
        return dict(results)
    
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
                ref = value[1:]
                substituted[key] = results.get(ref, value)
            else:
                substituted[key] = value
        return substituted
    
    async def _synthesize(
        self,
        task: str,
        plan: ExecutionPlan,
        results: dict[str, Any],
    ) -> str:
        """Synthesize final answer from DAG results.
        
        Args:
            task: Original task.
            plan: The executed plan.
            results: All results from execution.
            
        Returns:
            Synthesized final answer.
        """
        # Show execution summary
        successful = [c for c in plan.calls if c.status == "completed"]
        failed = [c for c in plan.calls if c.status == "failed"]
        
        prompt = f"""Task: {task}

Execution completed:
- Successful: {len(successful)} calls
- Failed: {len(failed)} calls

Results:
{json.dumps(results, indent=2, default=str)}

Provide the final answer based on these results."""
        
        return await self.agent.think(prompt)
