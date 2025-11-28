"""
Empirical Performance Benchmark for AgenticFlow

Measures:
- LLM call counts
- Latency (wall clock time)
- Token usage (if available)
- Tool execution time vs LLM time

Run with: uv run python scripts/benchmark.py
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any

# Patch LLM to count calls
_llm_call_count = 0
_llm_total_tokens = 0
_llm_call_times: list[float] = []


def patch_langchain_for_metrics():
    """Monkey-patch LangChain to count LLM calls and tokens."""
    global _llm_call_count, _llm_total_tokens, _llm_call_times
    
    from langchain_core.language_models.chat_models import BaseChatModel
    
    original_ainvoke = BaseChatModel.ainvoke
    
    async def patched_ainvoke(self, input, config=None, **kwargs):
        global _llm_call_count, _llm_total_tokens, _llm_call_times
        start = time.perf_counter()
        result = await original_ainvoke(self, input, config, **kwargs)
        elapsed = time.perf_counter() - start
        
        _llm_call_count += 1
        _llm_call_times.append(elapsed)
        
        # Try to get token usage
        if hasattr(result, 'usage_metadata') and result.usage_metadata:
            _llm_total_tokens += result.usage_metadata.get('total_tokens', 0)
        elif hasattr(result, 'response_metadata'):
            usage = result.response_metadata.get('usage', {})
            _llm_total_tokens += usage.get('total_tokens', 0)
        
        return result
    
    BaseChatModel.ainvoke = patched_ainvoke


def reset_metrics():
    """Reset all metrics."""
    global _llm_call_count, _llm_total_tokens, _llm_call_times
    _llm_call_count = 0
    _llm_total_tokens = 0
    _llm_call_times = []


def get_metrics() -> dict[str, Any]:
    """Get current metrics."""
    return {
        "llm_calls": _llm_call_count,
        "total_tokens": _llm_total_tokens,
        "llm_times": _llm_call_times.copy(),
        "total_llm_time": sum(_llm_call_times),
        "avg_llm_time": sum(_llm_call_times) / len(_llm_call_times) if _llm_call_times else 0,
    }


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    wall_time_ms: float
    llm_calls: int
    llm_time_ms: float
    tokens: int
    tool_calls: int = 0
    error: str | None = None
    
    @property
    def overhead_pct(self) -> float:
        """Percentage of time NOT spent in LLM calls."""
        if self.wall_time_ms == 0:
            return 0
        return ((self.wall_time_ms - self.llm_time_ms) / self.wall_time_ms) * 100
    
    def __str__(self) -> str:
        if self.error:
            return f"❌ {self.name}: ERROR - {self.error}"
        return (
            f"✓ {self.name}\n"
            f"  Wall time: {self.wall_time_ms:.0f}ms | "
            f"LLM time: {self.llm_time_ms:.0f}ms ({100-self.overhead_pct:.1f}%)\n"
            f"  LLM calls: {self.llm_calls} | "
            f"Tokens: {self.tokens} | "
            f"Overhead: {self.overhead_pct:.1f}%"
        )


@dataclass 
class BenchmarkSuite:
    """Collection of benchmark results."""
    results: list[BenchmarkResult] = field(default_factory=list)
    
    def add(self, result: BenchmarkResult):
        self.results.append(result)
    
    def summary(self) -> str:
        lines = [
            "",
            "=" * 70,
            "AGENTICFLOW PERFORMANCE BENCHMARK RESULTS",
            "=" * 70,
            "",
        ]
        
        for r in self.results:
            lines.append(str(r))
            lines.append("-" * 50)
        
        # Aggregate stats
        successful = [r for r in self.results if not r.error]
        if successful:
            total_wall = sum(r.wall_time_ms for r in successful)
            total_llm = sum(r.llm_time_ms for r in successful)
            total_calls = sum(r.llm_calls for r in successful)
            total_tokens = sum(r.tokens for r in successful)
            
            # Calculate sequential vs parallel efficiency
            parallel_result = next((r for r in successful if "Parallel" in r.name), None)
            
            lines.append("")
            lines.append("=" * 70)
            lines.append("ANALYSIS")
            lines.append("=" * 70)
            lines.append("")
            lines.append("📊 TOTALS:")
            lines.append(f"   Total wall time: {total_wall/1000:.1f}s")
            lines.append(f"   Total LLM time:  {total_llm/1000:.1f}s")
            lines.append(f"   Total LLM calls: {total_calls}")
            lines.append(f"   Total tokens:    {total_tokens:,}")
            lines.append("")
            
            # Key insights
            lines.append("🔍 KEY INSIGHTS:")
            
            # Single agent comparison
            think_only = next((r for r in successful if "think()" in r.name), None)
            dag_tools = next((r for r in successful if "dag" in r.name.lower() and "Agent" in r.name), None)
            react_tools = next((r for r in successful if "react" in r.name.lower()), None)
            
            if think_only and dag_tools:
                overhead = dag_tools.wall_time_ms - think_only.wall_time_ms
                lines.append(f"   • DAG strategy adds {overhead:.0f}ms ({dag_tools.llm_calls-think_only.llm_calls} extra LLM calls) for tool planning")
            
            if react_tools and dag_tools:
                diff = dag_tools.wall_time_ms - react_tools.wall_time_ms
                if diff > 0:
                    lines.append(f"   • ReAct is {diff:.0f}ms faster than DAG for simple tasks")
                else:
                    lines.append(f"   • DAG is {-diff:.0f}ms faster than ReAct for simple tasks")
            
            if parallel_result:
                speedup = parallel_result.llm_time_ms / parallel_result.wall_time_ms
                lines.append(f"   • Parallel execution achieved {speedup:.1f}x speedup (3 agents, {parallel_result.wall_time_ms:.0f}ms wall)")
            
            # Pipeline analysis
            pipeline2 = next((r for r in successful if "Pipeline (2 agents)" in r.name), None)
            pipeline3 = next((r for r in successful if "Pipeline (3 agents)" in r.name), None)
            if pipeline2 and pipeline3:
                per_agent_2 = pipeline2.wall_time_ms / 2
                per_agent_3 = pipeline3.wall_time_ms / 3
                lines.append(f"   • Pipeline avg per agent: 2-agent={per_agent_2:.0f}ms, 3-agent={per_agent_3:.0f}ms")
            
            # Supervisor analysis
            supervisor = next((r for r in successful if "Supervisor" in r.name), None)
            if supervisor:
                lines.append(f"   • Supervisor pattern: {supervisor.llm_calls} LLM calls (routing overhead)")
            
            lines.append("")
            lines.append("💡 RECOMMENDATIONS:")
            lines.append("   • Use think() for simple no-tool tasks (1 LLM call)")
            lines.append("   • Use ReAct strategy for simple tool tasks")
            lines.append("   • Use DAG strategy for complex multi-tool parallel tasks")
            lines.append("   • Use run_parallel() for embarrassingly parallel workloads")
            lines.append("   • Supervisor pattern has high LLM overhead - use sparingly")
            lines.append("")
        
        return "\n".join(lines)


async def benchmark_single_agent_think():
    """Benchmark: Single agent, think() only (no tools)."""
    from langchain_openai import ChatOpenAI
    from agenticflow import Agent
    
    reset_metrics()
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = Agent(name="thinker", model=model)
    
    start = time.perf_counter()
    result = await agent.think("What is 2 + 2? Answer briefly.")
    wall_time = (time.perf_counter() - start) * 1000
    
    metrics = get_metrics()
    
    return BenchmarkResult(
        name="Single Agent - think()",
        wall_time_ms=wall_time,
        llm_calls=metrics["llm_calls"],
        llm_time_ms=metrics["total_llm_time"] * 1000,
        tokens=metrics["total_tokens"],
    )


async def benchmark_single_agent_with_tools_dag():
    """Benchmark: Single agent with tools, DAG strategy."""
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from agenticflow import Agent
    
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b
    
    reset_metrics()
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = Agent(name="calculator", model=model, tools=[add, multiply])
    
    start = time.perf_counter()
    result = await agent.run("What is 5 + 3?", strategy="dag")
    wall_time = (time.perf_counter() - start) * 1000
    
    metrics = get_metrics()
    
    return BenchmarkResult(
        name="Single Agent - run(strategy='dag') with tools",
        wall_time_ms=wall_time,
        llm_calls=metrics["llm_calls"],
        llm_time_ms=metrics["total_llm_time"] * 1000,
        tokens=metrics["total_tokens"],
    )


async def benchmark_single_agent_with_tools_react():
    """Benchmark: Single agent with tools, ReAct strategy."""
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from agenticflow import Agent
    
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    reset_metrics()
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = Agent(name="calculator", model=model, tools=[add])
    
    start = time.perf_counter()
    result = await agent.run("What is 5 + 3?", strategy="react")
    wall_time = (time.perf_counter() - start) * 1000
    
    metrics = get_metrics()
    
    return BenchmarkResult(
        name="Single Agent - run(strategy='react') with tools",
        wall_time_ms=wall_time,
        llm_calls=metrics["llm_calls"],
        llm_time_ms=metrics["total_llm_time"] * 1000,
        tokens=metrics["total_tokens"],
    )


async def benchmark_flow_pipeline_2_agents():
    """Benchmark: Flow with 2 agents in pipeline."""
    from langchain_openai import ChatOpenAI
    from agenticflow import Agent, Flow
    
    reset_metrics()
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    researcher = Agent(
        name="researcher",
        model=model,
        system_prompt="You research topics. Be brief.",
    )
    writer = Agent(
        name="writer", 
        model=model,
        system_prompt="You write based on research. Be brief.",
    )
    
    flow = Flow(
        name="bench-pipeline-2",
        agents=[researcher, writer],
        topology="pipeline",
    )
    
    start = time.perf_counter()
    result = await flow.run("Tell me about Python in one sentence.")
    wall_time = (time.perf_counter() - start) * 1000
    
    metrics = get_metrics()
    
    return BenchmarkResult(
        name="Flow - Pipeline (2 agents)",
        wall_time_ms=wall_time,
        llm_calls=metrics["llm_calls"],
        llm_time_ms=metrics["total_llm_time"] * 1000,
        tokens=metrics["total_tokens"],
    )


async def benchmark_flow_pipeline_3_agents():
    """Benchmark: Flow with 3 agents in pipeline."""
    from langchain_openai import ChatOpenAI
    from agenticflow import Agent, Flow
    
    reset_metrics()
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    researcher = Agent(name="researcher", model=model, system_prompt="Research briefly.")
    writer = Agent(name="writer", model=model, system_prompt="Write briefly.")
    reviewer = Agent(name="reviewer", model=model, system_prompt="Review briefly. Say FINAL ANSWER to finish.")
    
    flow = Flow(
        name="bench-pipeline-3",
        agents=[researcher, writer, reviewer],
        topology="pipeline",
    )
    
    start = time.perf_counter()
    result = await flow.run("Write about async Python.")
    wall_time = (time.perf_counter() - start) * 1000
    
    metrics = get_metrics()
    
    return BenchmarkResult(
        name="Flow - Pipeline (3 agents)",
        wall_time_ms=wall_time,
        llm_calls=metrics["llm_calls"],
        llm_time_ms=metrics["total_llm_time"] * 1000,
        tokens=metrics["total_tokens"],
    )


async def benchmark_flow_supervisor():
    """Benchmark: Flow with supervisor pattern."""
    from langchain_openai import ChatOpenAI
    from agenticflow import Agent, Flow
    from agenticflow.flow import FlowConfig
    
    reset_metrics()
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    supervisor = Agent(
        name="supervisor",
        model=model,
        role="supervisor",
        system_prompt="You supervise. Ask 'worker' to help. After worker responds, say FINAL ANSWER: with the result.",
    )
    worker = Agent(
        name="worker",
        model=model, 
        role="worker",
        system_prompt="You do work. Be brief - one sentence max.",
    )
    
    flow = Flow(
        name="bench-supervisor",
        agents=[supervisor, worker],
        topology="supervisor",
        config=FlowConfig(max_iterations=10),
    )
    
    start = time.perf_counter()
    result = await flow.run("Summarize what Python is in one sentence.")
    wall_time = (time.perf_counter() - start) * 1000
    
    metrics = get_metrics()
    
    return BenchmarkResult(
        name="Flow - Supervisor (2 agents)",
        wall_time_ms=wall_time,
        llm_calls=metrics["llm_calls"],
        llm_time_ms=metrics["total_llm_time"] * 1000,
        tokens=metrics["total_tokens"],
    )


async def benchmark_flow_parallel():
    """Benchmark: Flow with parallel execution."""
    from langchain_openai import ChatOpenAI
    from agenticflow import Agent, Flow
    
    reset_metrics()
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    analyst1 = Agent(name="analyst1", model=model, system_prompt="Analyze briefly.")
    analyst2 = Agent(name="analyst2", model=model, system_prompt="Analyze briefly.")
    analyst3 = Agent(name="analyst3", model=model, system_prompt="Analyze briefly.")
    
    flow = Flow(
        name="bench-parallel",
        agents=[analyst1, analyst2, analyst3],
        topology="mesh",
    )
    
    start = time.perf_counter()
    result = await flow.run_parallel(
        "What makes Python popular?",
        merge_strategy="combine",
    )
    wall_time = (time.perf_counter() - start) * 1000
    
    metrics = get_metrics()
    
    return BenchmarkResult(
        name="Flow - Parallel (3 agents)",
        wall_time_ms=wall_time,
        llm_calls=metrics["llm_calls"],
        llm_time_ms=metrics["total_llm_time"] * 1000,
        tokens=metrics["total_tokens"],
    )


async def benchmark_agent_with_multiple_tools():
    """Benchmark: Agent with multiple tools, complex task."""
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from agenticflow import Agent
    
    @tool
    def search_web(query: str) -> str:
        """Search the web for information."""
        time.sleep(0.1)  # Simulate network latency
        return f"Search results for '{query}': Python is a popular programming language."
    
    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        return f"Contents of {path}: Sample file content."
    
    @tool
    def calculate(expression: str) -> str:
        """Evaluate a math expression."""
        return str(eval(expression))
    
    reset_metrics()
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = Agent(
        name="assistant",
        model=model,
        tools=[search_web, read_file, calculate],
        system_prompt="You are a helpful assistant. Use tools when needed.",
    )
    
    start = time.perf_counter()
    result = await agent.run(
        "Search for Python info and tell me what you find.",
        strategy="dag",
    )
    wall_time = (time.perf_counter() - start) * 1000
    
    metrics = get_metrics()
    
    return BenchmarkResult(
        name="Agent - Multiple tools (DAG)",
        wall_time_ms=wall_time,
        llm_calls=metrics["llm_calls"],
        llm_time_ms=metrics["total_llm_time"] * 1000,
        tokens=metrics["total_tokens"],
    )


async def run_benchmarks():
    """Run all benchmarks."""
    print("AgenticFlow Performance Benchmark")
    print("=" * 60)
    print(f"Model: gpt-4o-mini")
    print("=" * 60)
    print()
    
    suite = BenchmarkSuite()
    
    benchmarks = [
        ("1/8", benchmark_single_agent_think),
        ("2/8", benchmark_single_agent_with_tools_dag),
        ("3/8", benchmark_single_agent_with_tools_react),
        ("4/8", benchmark_flow_pipeline_2_agents),
        ("5/8", benchmark_flow_pipeline_3_agents),
        ("6/8", benchmark_flow_supervisor),
        ("7/8", benchmark_flow_parallel),
        ("8/8", benchmark_agent_with_multiple_tools),
    ]
    
    for progress, bench_fn in benchmarks:
        print(f"Running [{progress}] {bench_fn.__name__}...")
        try:
            result = await bench_fn()
            suite.add(result)
            print(f"  Done: {result.llm_calls} LLM calls, {result.wall_time_ms:.0f}ms")
        except Exception as e:
            suite.add(BenchmarkResult(
                name=bench_fn.__name__,
                wall_time_ms=0,
                llm_calls=0,
                llm_time_ms=0,
                tokens=0,
                error=str(e),
            ))
            print(f"  ERROR: {e}")
    
    print()
    print(suite.summary())


def main():
    # Load .env file if present
    from pathlib import Path
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Run: export OPENAI_API_KEY=your-key")
        return
    
    # Patch LangChain before importing agenticflow
    patch_langchain_for_metrics()
    
    asyncio.run(run_benchmarks())


if __name__ == "__main__":
    main()
