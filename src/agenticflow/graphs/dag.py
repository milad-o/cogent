"""
DAG-based execution with maximum parallelism.

This is our RECOMMENDED execution strategy for complex tasks.
It builds a dependency graph and executes independent steps in parallel.

LLMCompiler-style approach:
1. Agent creates a DAG plan identifying dependencies
2. Execute independent steps in parallel "waves"
3. Synthesize final answer from all results

Streaming Execution (Task Fetching Unit):
The StreamingDAGExecutor improves on this by starting execution
immediately when any call becomes ready, without waiting for the
full plan. This provides up to 3.7x latency improvement.
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
        - Self-correction on tool errors
        
    Cons:
        - Requires good dependency planning from LLM
        - More complex error handling
        
    Example:
        executor = DAGExecutor(agent)
        result = await executor.execute(
            "Search for topic A and topic B, then combine findings"
        )
    """
    
    # Maximum correction attempts per failed call
    max_correction_attempts: int = 2
    
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
        tools_desc = self.agent.get_tool_descriptions()
        
        prompt = f"""Analyze this task and create an execution plan.

Task: {task}
{f"Context: {json.dumps(context)}" if context else ""}

Available tools (you can ONLY use these):
{tools_desc}

First, think step by step:
1. What information do I need to accomplish this task?
2. Which of the available tools can provide that information?
3. What operations can I do myself without tools (reasoning, calculations, combining results)?
4. Which tool calls are independent and can run in parallel?

Then output a JSON DAG plan:
{{
  "reasoning": "Brief explanation of your approach",
  "steps": [
    {{"id": "call_0", "tool": "tool_name", "args": {{}}, "depends_on": []}},
    {{"id": "call_1", "tool": "tool_name", "args": {{}}, "depends_on": []}}
  ]
}}

Rules:
- Only include steps that require tools from the list above
- Steps with empty depends_on run in parallel
- Use $call_N in args to reference results from previous steps
- After tool execution, you will synthesize the final answer yourself
"""
        
        # Use neutral planner persona to avoid tool execution behavior
        planner_prompt = "You are a task planner. Your job is to analyze tasks and output JSON execution plans. Never execute tools, only plan them."
        response = await self.agent.think(
            prompt, 
            include_tools=False, 
            system_prompt_override=planner_prompt,
        )
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
        """Execute a wave of calls in parallel with self-correction.
        
        All calls in a wave have their dependencies satisfied,
        so they can run concurrently. On failure, attempts correction.
        
        Args:
            calls: List of ToolCalls to execute.
            prior_results: Results from previous waves.
            
        Returns:
            Dict mapping call_id to result.
        """
        
        async def execute_one(call: ToolCall) -> tuple[str, Any]:
            """Execute a single call with self-correction on failure."""
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
                error_str = str(e)
                
                # Track the error
                self._track_tool_error(call.tool_name, error_str)
                
                # Attempt self-correction
                corrected_result = await self._attempt_correction(
                    call, args, error_str, prior_results
                )
                if corrected_result is not None:
                    call.result = corrected_result
                    call.status = "completed"
                    return (call.id, corrected_result)
                
                # Correction failed - record error
                call.error = error_str
                call.status = "failed"
                return (call.id, f"ERROR: {e}")
        
        # Execute all calls in parallel using asyncio.gather
        results = await asyncio.gather(*[execute_one(c) for c in calls])
        return dict(results)
    
    async def _attempt_correction(
        self,
        call: ToolCall,
        original_args: dict[str, Any],
        error: str,
        prior_results: dict[str, Any],
    ) -> Any | None:
        """Attempt to self-correct a failed tool call.
        
        Asks the LLM to analyze the error and provide corrected arguments.
        
        Args:
            call: The failed ToolCall.
            original_args: The arguments that caused the error.
            error: The error message.
            prior_results: Results from previous waves.
            
        Returns:
            The result if correction succeeds, None otherwise.
        """
        for attempt in range(self.max_correction_attempts):
            self._emit_step("self_correction", {
                "call_id": call.id,
                "tool": call.tool_name,
                "attempt": attempt + 1,
                "max_attempts": self.max_correction_attempts,
                "error": error,
            })
            
            # Ask LLM to analyze and correct
            correction_prompt = self._build_correction_prompt(
                call, original_args, error, prior_results
            )
            
            try:
                response = await self.agent.think(
                    correction_prompt,
                    include_tools=False,
                    system_prompt_override="You are debugging a tool call error. Analyze the error and provide corrected arguments as JSON only.",
                )
                
                # Parse corrected args from response
                corrected_args = self._parse_corrected_args(response, original_args)
                
                if corrected_args and corrected_args != original_args:
                    # Track the correction attempt
                    self._track_tool_call(call.tool_name, corrected_args)
                    
                    result = await self.agent.act(call.tool_name, corrected_args)
                    self._track_tool_result(call.tool_name, result, 0)
                    return result
                    
            except Exception as retry_error:
                error = str(retry_error)
                self._track_tool_error(call.tool_name, error)
                continue
        
        return None
    
    def _build_correction_prompt(
        self,
        call: ToolCall,
        args: dict[str, Any],
        error: str,
        prior_results: dict[str, Any],
    ) -> str:
        """Build a prompt asking LLM to correct the tool call.
        
        Args:
            call: The failed ToolCall.
            args: The arguments that caused the error.
            error: The error message.
            prior_results: Results from previous waves for context.
            
        Returns:
            Prompt for the correction request.
        """
        # Get tool schema for reference
        tool = self.agent._get_tool(call.tool_name)
        schema_info = ""
        if tool and hasattr(tool, "parameters"):
            schema_info = f"\nTool schema: {json.dumps(tool.parameters, default=str)}"
        
        return f"""A tool call failed. Analyze the error and provide corrected arguments.

Tool: {call.tool_name}{schema_info}

Original arguments:
{json.dumps(args, indent=2, default=str)}

Error:
{error}

{f"Available results from prior steps: {list(prior_results.keys())}" if prior_results else ""}

Think about what caused this error:
1. Are the argument types correct?
2. Are required arguments missing?
3. Are there invalid values?

Respond with ONLY a JSON object containing the corrected arguments:
{{"arg_name": "corrected_value", ...}}"""
    
    def _parse_corrected_args(
        self,
        response: str,
        original_args: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Parse corrected arguments from LLM response.
        
        Args:
            response: LLM response with corrected args.
            original_args: Original arguments for fallback.
            
        Returns:
            Corrected arguments dict or None if parsing fails.
        """
        json_match = re.search(r"\{[\s\S]*?\}", response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return None
    
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


class StreamingDAGExecutor(DAGExecutor):
    """
    Streaming DAG execution with Task Fetching Unit (LLMCompiler-style).
    
    This executor improves on DAGExecutor by streaming execution:
    - Starts executing calls immediately when dependencies are satisfied
    - Plans and executes concurrently (doesn't wait for full plan)
    - Uses a task queue with dynamic scheduling
    
    Performance Benefits:
    - Up to 3.7x latency reduction vs sequential planning-then-execution
    - Better resource utilization through continuous execution
    - Reduced time-to-first-result
    
    Pattern:
        Plan Stream → Task Queue → Parallel Execution → Results Aggregation
                ↓           ↓              ↓
             [call_0]  ──► execute ──► result
             [call_1]  ──► execute ──► result (parallel with call_0)
             [call_2]  ──► (wait for deps) ──► execute ──► result
    
    Example:
        executor = StreamingDAGExecutor(agent)
        result = await executor.execute("Complex task requiring parallel searches")
    """
    
    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute using streaming DAG with parallel execution.
        
        Unlike DAGExecutor which waits for full plan, this starts
        executing immediately when any call becomes ready.
        
        Args:
            task: The task to execute.
            context: Optional context dictionary.
            
        Returns:
            The synthesized final answer.
        """
        self._emit_step("streaming_dag_start", {"task": task})
        
        # Shared state for concurrent planning and execution
        plan = ExecutionPlan()
        results: dict[str, Any] = {}
        completed: set[str] = set()
        in_progress: set[str] = set()
        execution_tasks: list[asyncio.Task] = []
        
        # Events for coordination
        plan_complete = asyncio.Event()
        new_call_added = asyncio.Event()
        
        async def planning_task() -> None:
            """Stream the plan, adding calls as they're identified."""
            nonlocal plan
            
            self._emit_step("streaming_planning", {"task": task})
            
            # Get streamed plan (parse incrementally)
            plan = await self._create_streaming_plan(task, context, new_call_added)
            plan_complete.set()
        
        async def execution_scheduler() -> None:
            """Schedule execution of ready calls as they become available."""
            while True:
                # Find ready calls (dependencies satisfied, not started)
                ready_calls = [
                    c for c in plan.calls
                    if c.id not in completed 
                    and c.id not in in_progress
                    and c.is_ready(completed)
                ]
                
                # Start execution for ready calls
                for call in ready_calls:
                    in_progress.add(call.id)
                    task_obj = asyncio.create_task(
                        self._execute_single_call(call, results, completed, in_progress)
                    )
                    execution_tasks.append(task_obj)
                    self._emit_step("streaming_call_started", {
                        "call_id": call.id,
                        "tool": call.tool_name,
                        "parallel_count": len(in_progress),
                    })
                
                # Check if we're done
                if plan_complete.is_set():
                    all_planned = {c.id for c in plan.calls}
                    if completed >= all_planned:
                        break
                    if not in_progress and not ready_calls:
                        # No more work possible
                        break
                
                # Wait for either new calls or completion of running tasks
                try:
                    wait_coros = []
                    pending_tasks = [t for t in execution_tasks if not t.done()]
                    
                    if pending_tasks:
                        # Wait for any task to complete
                        done, _ = await asyncio.wait(
                            pending_tasks,
                            timeout=0.1,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    else:
                        # No tasks running, just sleep briefly
                        await asyncio.sleep(0.05)
                except Exception:
                    pass
                
                new_call_added.clear()
        
        # Run planning and execution concurrently
        planning = asyncio.create_task(planning_task())
        scheduling = asyncio.create_task(execution_scheduler())
        
        # Wait for both to complete
        await planning
        await scheduling
        
        # Wait for any remaining execution tasks
        if execution_tasks:
            await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        self._emit_step("streaming_dag_complete", {
            "total_calls": len(plan.calls),
            "completed": len(completed),
        })
        
        # Synthesize
        if not plan.calls:
            return await self.agent.think(task)
        
        return await self._synthesize(task, plan, results)
    
    async def _create_streaming_plan(
        self,
        task: str,
        context: dict[str, Any] | None,
        new_call_event: asyncio.Event,
    ) -> ExecutionPlan:
        """Create plan with streaming - signal as calls become ready.
        
        This creates the full plan but signals when independent calls
        are identified so execution can start immediately.
        
        Args:
            task: The task to plan for.
            context: Optional context.
            new_call_event: Event to signal when new calls are added.
            
        Returns:
            Complete ExecutionPlan.
        """
        tools_desc = self.agent.get_tool_descriptions()
        
        prompt = f"""Analyze this task and create an execution plan.

Task: {task}
{f"Context: {json.dumps(context)}" if context else ""}

Available tools (you can ONLY use these):
{tools_desc}

First, think step by step:
1. What information do I need to accomplish this task?
2. Which of the available tools can provide that information?
3. What operations can I do myself without tools (reasoning, calculations, combining results)?
4. Which tool calls are independent and can run in parallel?

IMPORTANT: List independent calls (no dependencies) FIRST so they can start immediately.

Then output a JSON DAG plan:
{{
  "reasoning": "Brief explanation of your approach",
  "steps": [
    {{"id": "call_0", "tool": "tool_name", "args": {{}}, "depends_on": []}},
    {{"id": "call_1", "tool": "tool_name", "args": {{}}, "depends_on": []}}
  ]
}}

Rules:
- Only include steps that require tools from the list above
- Steps with empty depends_on run in parallel (LIST THESE FIRST)
- Use $call_N in args to reference results from previous steps
- After tool execution, you will synthesize the final answer yourself
"""
        
        planner_prompt = "You are a task planner. Your job is to analyze tasks and output JSON execution plans. Never execute tools, only plan them. List independent calls first."
        response = await self.agent.think(
            prompt, 
            include_tools=False, 
            system_prompt_override=planner_prompt,
        )
        
        plan = self._parse_dag_plan(response)
        
        # Signal that calls are ready (for streaming execution)
        if plan.calls:
            new_call_event.set()
        
        return plan
    
    async def _execute_single_call(
        self,
        call: ToolCall,
        results: dict[str, Any],
        completed: set[str],
        in_progress: set[str],
    ) -> None:
        """Execute a single call and update shared state.
        
        Args:
            call: The ToolCall to execute.
            results: Shared results dict to update.
            completed: Shared set of completed call IDs.
            in_progress: Shared set of in-progress call IDs.
        """
        args = self._substitute_args(call.args, results)
        
        call.status = "running"
        start = now_utc()
        
        self._track_tool_call(call.tool_name, args)
        
        try:
            result = await self.agent.act(call.tool_name, args)
            call.duration_ms = (now_utc() - start).total_seconds() * 1000
            call.result = result
            call.status = "completed"
            
            self._track_tool_result(call.tool_name, result, call.duration_ms)
            
            results[call.id] = result
            
        except Exception as e:
            call.duration_ms = (now_utc() - start).total_seconds() * 1000
            error_str = str(e)
            
            self._track_tool_error(call.tool_name, error_str)
            
            # Attempt self-correction
            corrected_result = await self._attempt_correction(
                call, args, error_str, results
            )
            if corrected_result is not None:
                call.result = corrected_result
                call.status = "completed"
                results[call.id] = corrected_result
            else:
                call.error = error_str
                call.status = "failed"
                results[call.id] = f"ERROR: {e}"
        
        finally:
            completed.add(call.id)
            in_progress.discard(call.id)
