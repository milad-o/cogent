"""
Multi-Agent System: EXTENSIBLE Parallel Agent with DAG Execution
=================================================================

Features:
1. EXTENSIBLE: Register any tools dynamically (no hardcoded names)
2. DAG EXECUTION: Supports complex plans with dependencies
   - Parallel execution for independent tasks
   - Sequential execution when tasks depend on each other
3. AUTOMATIC DEPENDENCY RESOLUTION: LLM plans with dependencies

Example DAG Plan:
┌─────────────────────────────────────────────────────────────────────┐
│  "Write haiku, write story, translate both, then critique all"      │
│                                                                     │
│  Step 1 (parallel):  [write_haiku] ─┐     [write_story] ─┐          │
│                                     │                    │          │
│  Step 2 (parallel):  [translate] ◄──┘     [translate] ◄──┘          │
│                           │                    │                    │
│  Step 3 (sequential):     └──────► [critique] ◄┘                    │
│                                         │                           │
│                                    Final Result                     │
└─────────────────────────────────────────────────────────────────────┘

The LLM outputs a plan like:
{
  "steps": [
    {"id": "haiku", "tool": "write_haiku", "args": {...}, "depends_on": []},
    {"id": "story", "tool": "write_story", "args": {...}, "depends_on": []},
    {"id": "trans1", "tool": "translate_text", "args": {"text": "$ref:haiku"}, "depends_on": ["haiku"]},
    {"id": "trans2", "tool": "translate_text", "args": {"text": "$ref:story"}, "depends_on": ["story"]},
    {"id": "crit", "tool": "critique_writing", "args": {"text": "$ref:trans1 $ref:trans2"}, "depends_on": ["trans1", "trans2"]}
  ]
}
"""

import os
import time
import asyncio
import operator
import json
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool, BaseTool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END


# =============================================================================
# Performance Tracking
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Track detailed performance metrics."""
    
    start_time: float = 0
    end_time: float = 0
    llm_calls: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    execution_steps: list = field(default_factory=list)
    
    def start(self):
        self.start_time = time.time()
        self.llm_calls = []
        self.tool_calls = []
        self.execution_steps = []
    
    def stop(self):
        self.end_time = time.time()
    
    def record_llm_call(self, name: str, duration: float):
        self.llm_calls.append((name, duration))
    
    def record_tool_call(self, name: str, duration: float, step: int, parallel: bool = False):
        self.tool_calls.append((name, duration, step, parallel))
    
    def record_step(self, step_num: int, task_count: int, duration: float):
        self.execution_steps.append((step_num, task_count, duration))
    
    @property
    def total_duration(self) -> float:
        return self.end_time - self.start_time
    
    def print_report(self):
        print(f"\n{'─' * 60}")
        print(f"📊 PERFORMANCE REPORT")
        print(f"{'─' * 60}")
        print(f"⏱️  Total: {self.total_duration:.2f}s")
        print(f"   ├── LLM calls: {len(self.llm_calls)}")
        print(f"   └── Tool calls: {len(self.tool_calls)}")
        
        if self.execution_steps:
            print(f"\n📋 DAG Execution Steps:")
            sequential_time = sum(d for _, d, _, _ in self.tool_calls)
            actual_time = sum(d for _, _, d in self.execution_steps)
            for step_num, task_count, duration in self.execution_steps:
                parallel_marker = " ⚡[PARALLEL]" if task_count > 1 else ""
                print(f"   Step {step_num}: {task_count} task(s) in {duration:.2f}s{parallel_marker}")
            
            if sequential_time > 0 and actual_time > 0:
                print(f"\n   📈 Tool execution: {actual_time:.2f}s actual vs {sequential_time:.2f}s sequential")
                print(f"      Parallelization saved: {sequential_time - actual_time:.2f}s")
        
        if self.tool_calls:
            print(f"\n🔧 Tool Calls:")
            for name, duration, step, parallel in self.tool_calls:
                marker = f" [Step {step}]" + (" ⚡" if parallel else "")
                print(f"   • {name}: {duration:.2f}s{marker}")


metrics = PerformanceMetrics()


# =============================================================================
# Tool Registry - EXTENSIBLE Design
# =============================================================================

class ToolRegistry:
    """
    Extensible tool registry. Register any tool and it's automatically
    available to the planner.
    """
    
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
    
    def register(self, tool_instance: BaseTool) -> "ToolRegistry":
        """Register a tool. Returns self for chaining."""
        self._tools[tool_instance.name] = tool_instance
        return self
    
    def register_many(self, tools: list[BaseTool]) -> "ToolRegistry":
        """Register multiple tools at once."""
        for t in tools:
            self.register(t)
        return self
    
    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_tool_descriptions(self) -> str:
        """Generate descriptions for the planner prompt."""
        descriptions = []
        for name, tool_obj in self._tools.items():
            # Get the tool's schema for args (Pydantic v2 compatible)
            if tool_obj.args_schema:
                schema = tool_obj.args_schema.model_json_schema()
            else:
                schema = {}
            props = schema.get("properties", {})
            args_desc = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in props.items())
            descriptions.append(f"- {name}({args_desc}): {tool_obj.description}")
        return "\n".join(descriptions)
    
    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())
    
    def __len__(self) -> int:
        return len(self._tools)


# =============================================================================
# DAG Plan Data Structures
# =============================================================================

@dataclass
class TaskNode:
    """A single task in the execution DAG."""
    id: str
    tool: str
    args: dict
    depends_on: list[str] = field(default_factory=list)
    result: str | None = None
    completed: bool = False


@dataclass  
class ExecutionPlan:
    """A DAG of tasks to execute."""
    tasks: dict[str, TaskNode] = field(default_factory=dict)
    
    @classmethod
    def from_json(cls, plan_json: dict) -> "ExecutionPlan":
        """Parse a plan from JSON."""
        plan = cls()
        for step in plan_json.get("steps", []):
            task = TaskNode(
                id=step["id"],
                tool=step["tool"],
                args=step.get("args", {}),
                depends_on=step.get("depends_on", [])
            )
            plan.tasks[task.id] = task
        return plan
    
    def get_ready_tasks(self) -> list[TaskNode]:
        """Get tasks whose dependencies are all completed."""
        ready = []
        for task in self.tasks.values():
            if task.completed:
                continue
            # Check if all dependencies are completed
            deps_completed = all(
                self.tasks[dep].completed 
                for dep in task.depends_on 
                if dep in self.tasks
            )
            if deps_completed:
                ready.append(task)
        return ready
    
    def is_complete(self) -> bool:
        """Check if all tasks are completed."""
        return all(task.completed for task in self.tasks.values())
    
    def get_result(self, task_id: str) -> str | None:
        """Get the result of a completed task."""
        task = self.tasks.get(task_id)
        return task.result if task else None
    
    def resolve_args(self, task: TaskNode) -> dict:
        """
        Resolve task arguments, replacing $ref:task_id with actual results.
        This allows tasks to reference results from dependencies.
        """
        resolved = {}
        for key, value in task.args.items():
            if isinstance(value, str):
                # Replace all $ref:xxx patterns
                import re
                def replacer(match):
                    ref_id = match.group(1)
                    return self.get_result(ref_id) or match.group(0)
                resolved[key] = re.sub(r'\$ref:(\w+)', replacer, value)
            else:
                resolved[key] = value
        return resolved


# =============================================================================
# Agent State
# =============================================================================

class DAGAgentState(TypedDict):
    """State for DAG-based parallel agent."""
    messages: Annotated[list[BaseMessage], operator.add]
    plan: ExecutionPlan | None
    step_number: int
    all_results: dict[str, str]
    phase: str


# =============================================================================
# DAG Agent Builder - EXTENSIBLE
# =============================================================================

class DAGAgentBuilder:
    """
    Builder for creating an extensible DAG-based agent.
    
    Usage:
        agent = (DAGAgentBuilder(model)
            .register_tools([tool1, tool2, tool3])
            .with_system_prompt("You are a helpful assistant")
            .build())
        
        result = await agent.ainvoke({"messages": [...]})
    """
    
    def __init__(self, model):
        self.model = model
        self.registry = ToolRegistry()
        self.system_prompt = ""
        self._max_steps = 10
    
    def register_tool(self, tool_instance: BaseTool) -> "DAGAgentBuilder":
        """Register a single tool."""
        self.registry.register(tool_instance)
        return self
    
    def register_tools(self, tools: list[BaseTool]) -> "DAGAgentBuilder":
        """Register multiple tools."""
        self.registry.register_many(tools)
        return self
    
    def with_system_prompt(self, prompt: str) -> "DAGAgentBuilder":
        """Set a custom system prompt prefix."""
        self.system_prompt = prompt
        return self
    
    def with_max_steps(self, max_steps: int) -> "DAGAgentBuilder":
        """Set maximum execution steps."""
        self._max_steps = max_steps
        return self
    
    def build(self) -> StateGraph:
        """Build and compile the agent graph."""
        
        # Capture in closure
        model = self.model
        registry = self.registry
        system_prompt = self.system_prompt
        max_steps = self._max_steps
        
        # ----- Planning Node -----
        async def planning_node(state: DAGAgentState) -> dict:
            """Plan the execution DAG."""
            start = time.time()
            
            tool_descriptions = registry.get_tool_descriptions()
            
            planning_prompt = f"""{system_prompt}

You are a planning agent. Create an execution plan as a DAG (directed acyclic graph).

AVAILABLE TOOLS:
{tool_descriptions}

OUTPUT FORMAT (JSON only, no other text):
{{
  "steps": [
    {{"id": "unique_id", "tool": "tool_name", "args": {{"arg1": "value"}}, "depends_on": []}},
    {{"id": "step2", "tool": "tool_name", "args": {{"text": "$ref:unique_id"}}, "depends_on": ["unique_id"]}}
  ]
}}

RULES:
1. Each step needs a unique "id" 
2. "depends_on" = IDs of steps that MUST complete first
3. Use "$ref:step_id" in args to use another step's output
4. Steps with NO dependencies run in PARALLEL (faster!)
5. Only add dependencies when truly needed

EXAMPLE - "Write haiku and story, translate both":
{{
  "steps": [
    {{"id": "haiku", "tool": "write_haiku", "args": {{"subject": "mountains"}}, "depends_on": []}},
    {{"id": "story", "tool": "write_story", "args": {{"subject": "ocean"}}, "depends_on": []}},
    {{"id": "t1", "tool": "translate_text", "args": {{"text": "$ref:haiku", "language": "French"}}, "depends_on": ["haiku"]}},
    {{"id": "t2", "tool": "translate_text", "args": {{"text": "$ref:story", "language": "French"}}, "depends_on": ["story"]}}
  ]
}}

User request: {state["messages"][-1].content}

JSON plan:"""

            response = await model.ainvoke([HumanMessage(content=planning_prompt)])
            
            duration = time.time() - start
            metrics.record_llm_call("Planning", duration)
            
            # Parse the plan
            try:
                content = response.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                plan_json = json.loads(content.strip())
                plan = ExecutionPlan.from_json(plan_json)
                
                print(f"\n  📋 Plan created ({duration:.2f}s): {len(plan.tasks)} tasks")
                for task in plan.tasks.values():
                    deps = f" → depends on {task.depends_on}" if task.depends_on else " (can run immediately)"
                    print(f"      • {task.id}: {task.tool}{deps}")
                
            except Exception as e:
                print(f"  ⚠️ Plan parsing failed: {e}")
                plan = ExecutionPlan()
            
            return {
                "plan": plan,
                "phase": "execute",
                "step_number": 0,
                "all_results": {},
                "messages": [AIMessage(content=f"Plan: {len(plan.tasks)} tasks")]
            }
        
        # ----- Execution Node -----
        async def execution_node(state: DAGAgentState) -> dict:
            """Execute ready tasks in parallel."""
            plan = state["plan"]
            step_num = state["step_number"] + 1
            all_results = dict(state.get("all_results", {}))
            
            if plan is None or plan.is_complete():
                return {"phase": "aggregate"}
            
            if step_num > max_steps:
                print(f"  ⚠️ Max steps ({max_steps}) reached")
                return {"phase": "aggregate"}
            
            ready_tasks = plan.get_ready_tasks()
            
            if not ready_tasks:
                return {"phase": "aggregate"}
            
            is_parallel = len(ready_tasks) > 1
            print(f"\n  {'⚡' if is_parallel else '📝'} Step {step_num}: {len(ready_tasks)} task(s){' IN PARALLEL' if is_parallel else ''}")
            
            step_start = time.time()
            
            async def execute_single_task(task: TaskNode) -> tuple[str, str]:
                """Execute a single task."""
                tool_obj = registry.get(task.tool)
                if not tool_obj:
                    return task.id, f"Error: Unknown tool '{task.tool}'"
                
                resolved_args = plan.resolve_args(task)
                
                task_start = time.time()
                
                loop = asyncio.get_event_loop()
                try:
                    result = await loop.run_in_executor(
                        None,
                        lambda: tool_obj.invoke(resolved_args)
                    )
                except Exception as e:
                    result = f"Error: {str(e)}"
                
                duration = time.time() - task_start
                metrics.record_tool_call(task.tool, duration, step_num, is_parallel)
                print(f"      ✓ {task.id} ({task.tool}): {duration:.2f}s")
                
                return task.id, str(result)
            
            results = await asyncio.gather(*[execute_single_task(t) for t in ready_tasks])
            
            step_duration = time.time() - step_start
            metrics.record_step(step_num, len(ready_tasks), step_duration)
            
            for task_id, result in results:
                plan.tasks[task_id].result = result
                plan.tasks[task_id].completed = True
                all_results[task_id] = result
            
            next_phase = "aggregate" if plan.is_complete() else "execute"
            
            return {
                "plan": plan,
                "step_number": step_num,
                "all_results": all_results,
                "phase": next_phase
            }
        
        # ----- Routing Function -----
        def should_continue(state: DAGAgentState) -> str:
            """Route based on phase."""
            phase = state.get("phase", "aggregate")
            if phase == "execute":
                plan = state.get("plan")
                if plan and not plan.is_complete():
                    return "execute"
            return "aggregate"
        
        # ----- Aggregation Node -----
        async def aggregation_node(state: DAGAgentState) -> dict:
            """Combine results into final response."""
            start = time.time()
            
            all_results = state.get("all_results", {})
            original_request = state["messages"][0].content if state["messages"] else ""
            
            results_text = "\n".join(f"[{k}]: {v}" for k, v in all_results.items())
            
            prompt = f"""User asked: {original_request}

Results:
{results_text}

Provide a well-formatted response. Be concise."""
            
            response = await model.ainvoke([HumanMessage(content=prompt)])
            
            duration = time.time() - start
            metrics.record_llm_call("Aggregation", duration)
            print(f"\n  📝 Aggregation: {duration:.2f}s")
            
            return {
                "messages": [AIMessage(content=response.content)],
                "phase": "done"
            }
        
        # ----- Build Graph -----
        builder = StateGraph(DAGAgentState)
        
        builder.add_node("plan", planning_node)
        builder.add_node("execute", execution_node)
        builder.add_node("aggregate", aggregation_node)
        
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "execute")
        builder.add_conditional_edges("execute", should_continue, ["execute", "aggregate"])
        builder.add_edge("aggregate", END)
        
        return builder.compile()


# =============================================================================
# Example Tools (Register your own!)
# =============================================================================

@tool
def write_haiku(subject: str) -> str:
    """Write a haiku (5-7-5 syllable poem) about a subject."""
    time.sleep(1.5)
    return f"Gentle {subject} calls\nThrough the misty morning air\nPeace settles within"


@tool
def write_story(subject: str) -> str:
    """Write a short story about a subject."""
    time.sleep(2.0)
    return f"In a land where {subject} ruled, ancient secrets awaited discovery..."


@tool
def translate_text(text: str, language: str) -> str:
    """Translate text to a target language."""
    time.sleep(1.0)
    return f"[{language}]: {text[:80]}..."


@tool
def critique_writing(text: str) -> str:
    """Provide constructive feedback on writing."""
    time.sleep(0.8)
    return f"Strong imagery and evocative language. Consider expanding sensory details."


@tool
def summarize_text(text: str) -> str:
    """Summarize text into a brief overview."""
    time.sleep(0.5)
    return f"Summary: {text[:40]}... (key themes extracted)"


# =============================================================================
# Initialize Model
# =============================================================================

model = init_chat_model(
    os.getenv("OPENAI_MODEL", "gpt-4o"),
    temperature=0.7,
)


# =============================================================================
# Demos
# =============================================================================

async def run_complex_demo():
    """Demo with complex DAG (parallel + sequential)."""
    
    print("\n" + "🔷" * 35)
    print("EXTENSIBLE DAG AGENT")
    print("Complex parallel + sequential execution")
    print("🔷" * 35)
    
    # Build agent - tools are registered dynamically!
    agent = (DAGAgentBuilder(model)
        .register_tools([
            write_haiku,
            write_story,
            translate_text,
            critique_writing,
            summarize_text,
        ])
        .with_system_prompt("You are a creative writing assistant.")
        .with_max_steps(10)
        .build())
    
    # Complex request: mixed parallel/sequential
    request = """
    Write a haiku about mountains AND write a short story about the ocean.
    Then translate BOTH to French.
    Finally, critique all the writing together.
    """
    
    print(f"\n📝 Request: {request.strip()}")
    print("\n" + "─" * 60)
    
    metrics.start()
    
    result = await agent.ainvoke({
        "messages": [HumanMessage(content=request)],
        "plan": None,
        "step_number": 0,
        "all_results": {},
        "phase": "plan"
    })
    
    metrics.stop()
    
    print("\n" + "=" * 60)
    print("📚 FINAL RESPONSE")
    print("=" * 60)
    print(result["messages"][-1].content)
    
    metrics.print_report()
    
    print("\n" + "=" * 60)
    print("🎓 DAG EXECUTION PATTERN")
    print("=" * 60)
    print("""
    What the LLM planned:
    
    Step 1 (PARALLEL - no deps):
        [write_haiku] ─────┐
        [write_story] ─────┤
                           │
    Step 2 (PARALLEL - each depends on one):
        [translate haiku] ─┐
        [translate story] ─┤
                           │
    Step 3 (SEQUENTIAL - depends on both translations):
        [critique all] ────► Final
    
    This is a DAG with 3 levels of execution!
    """)


async def run_simple_demo():
    """Simple demo with 2 parallel tasks."""
    
    print("\n" + "⭐" * 35)
    print("SIMPLE DEMO: Two Independent Tasks")
    print("⭐" * 35)
    
    agent = (DAGAgentBuilder(model)
        .register_tools([write_haiku, write_story])
        .build())
    
    request = "Write a haiku about rain AND write a story about sunshine"
    
    print(f"\n📝 Request: {request}")
    print("─" * 60)
    
    metrics.start()
    
    result = await agent.ainvoke({
        "messages": [HumanMessage(content=request)],
        "plan": None,
        "step_number": 0,
        "all_results": {},
        "phase": "plan"
    })
    
    metrics.stop()
    
    print("\n📚 Response:")
    print(result["messages"][-1].content[:500])
    
    metrics.print_report()


# =============================================================================
# Main
# =============================================================================

def main():
    """Run demos."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "simple":
        asyncio.run(run_simple_demo())
    else:
        asyncio.run(run_complex_demo())


if __name__ == "__main__":
    main()
