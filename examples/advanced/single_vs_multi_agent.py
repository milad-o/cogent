"""Single Agent vs Multi-Agent: A Practical Comparison.

This example demonstrates why Cogent favors single agent + tools over
multi-agent orchestration, while showing when agent.as_tool() adds value.

Key Findings (from practical experience):
    1. Single agent + tools is simpler, faster, and cheaper
    2. Multi-agent adds overhead (coordination, context passing, token cost)
    3. Use agent.as_tool() ONLY for verified benefits:
       - Generator + Verifier pattern (model diversity improves accuracy)
       - Truly parallel independent subtasks (embarrassingly parallel work)
       - Specialized domain experts that benefit from isolation

Why This Matters:
    Multi-agent systems have coordination overhead:
    - Each agent needs its own LLM call(s)
    - Context must be serialized and passed between agents
    - Information is lost in summarization between agents
    - Debugging becomes exponentially harder
    
    Single agent with tools:
    - One context window = full information preservation
    - Tools are function calls, not LLM calls
    - Clear execution trace
    - Lower cost and latency

When to Consider Multi-Agent (agent.as_tool):
    1. Generator + Verifier: Different model can catch blind spots
    2. Parallel Analysis: Multiple independent analyses can run concurrently
    3. Model Diversity: Different models excel at different tasks
    
    If you're chaining agents linearly, you probably just need one agent
    with better instructions.

Run:
    uv run python examples/advanced/single_vs_multi_agent.py
"""

import asyncio
import time

from cogent import Agent, Observer, tool


# ============================================================================
# SHARED TOOLS
# ============================================================================


@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Simulated search
    results = {
        "python async": "Python asyncio provides concurrent execution for I/O-bound tasks using async/await syntax.",
        "rust memory": "Rust uses ownership and borrowing for memory safety without garbage collection.",
        "go concurrency": "Go uses goroutines and channels for lightweight concurrent programming.",
    }
    for key, value in results.items():
        if key in query.lower():
            return value
    return f"Search results for: {query}"


@tool
def analyze_code(code: str) -> str:
    """Analyze code for issues."""
    issues = []
    if "def " in code and "->" not in code:
        issues.append("Missing return type hints")
    if "def " in code and '"""' not in code:
        issues.append("Missing docstrings")
    if not issues:
        return "Code looks good! No issues found."
    return f"Issues found: {', '.join(issues)}"


@tool
def format_output(content: str, style: str = "markdown") -> str:
    """Format content in specified style."""
    if style == "markdown":
        return f"## Summary\n\n{content}"
    elif style == "json":
        return f'{{"summary": "{content}"}}'
    return content


# ============================================================================
# APPROACH 1: SINGLE AGENT + TOOLS (Recommended)
# ============================================================================


async def single_agent_approach() -> dict:
    """Single agent with direct tool access.
    
    Pros:
        - Simple, no coordination overhead
        - Direct tool access = fewer LLM calls
        - Easy to debug (one agent, clear flow)
        - Lower latency and cost
    
    Cons:
        - One model's perspective
        - May miss diverse viewpoints
    """
    print("\n" + "=" * 60)
    print("APPROACH 1: Single Agent + Tools")
    print("=" * 60)
    
    observer = Observer(level="normal")
    
    agent = Agent(
        name="Researcher",
        model="gpt-4o-mini",
        tools=[search_web, analyze_code, format_output],
        instructions="""You are a technical researcher.

For research tasks:
1. Use search_web to find information
2. Synthesize findings into clear summary
3. Use format_output to structure the response

Be concise and factual.""",
        observer=observer,
    )
    
    start = time.perf_counter()
    result = await agent.run(
        "Research Python async programming and provide a formatted summary"
    )
    duration = time.perf_counter() - start
    
    print(f"\n📋 Result:\n{result.content}\n")
    
    # Collect metrics from response metadata
    tokens = 0
    if result.metadata and result.metadata.tokens:
        tokens = result.metadata.tokens.total_tokens
    
    return {
        "approach": "single_agent",
        "duration_seconds": round(duration, 2),
        "llm_calls": 1,  # Single agent, single run
        "tokens": tokens,
        "content_preview": str(result.content)[:100] if result.content else "",
    }


# ============================================================================
# APPROACH 2: MULTI-AGENT (Peer Coordination)
# ============================================================================


async def multi_agent_approach() -> dict:
    """Multiple specialized agents coordinating.
    
    Pros:
        - Separation of concerns
        - Each agent focused on one task
    
    Cons:
        - Coordination overhead (extra LLM calls)
        - Context must be passed between agents
        - More complex debugging
        - Higher latency and cost
    """
    print("\n" + "=" * 60)
    print("APPROACH 2: Multi-Agent (Peer Coordination)")
    print("=" * 60)
    
    observer = Observer(level="normal")
    
    # Agent 1: Searcher
    searcher = Agent(
        name="Searcher",
        model="gpt-4o-mini",
        tools=[search_web],
        instructions="Search for information and return raw findings. Be concise.",
        observer=observer,
    )
    
    # Agent 2: Analyst
    analyst = Agent(
        name="Analyst",
        model="gpt-4o-mini",
        tools=[analyze_code],
        instructions="Analyze and synthesize information. Create clear summaries.",
        observer=observer,
    )
    
    # Agent 3: Formatter  
    formatter = Agent(
        name="Formatter",
        model="gpt-4o-mini",
        tools=[format_output],
        instructions="Format content professionally using the format_output tool.",
        observer=observer,
    )
    
    start = time.perf_counter()
    
    # Step 1: Search
    search_result = await searcher.run(
        "Search for Python async programming"
    )
    print(f"\n📎 Searcher output: {str(search_result.content)[:80]}...")
    
    # Step 2: Analyze (passing context from previous agent)
    analysis_result = await analyst.run(
        f"Analyze and summarize this information:\n\n{search_result.content}"
    )
    print(f"📎 Analyst output: {str(analysis_result.content)[:80]}...")
    
    # Step 3: Format (passing context from previous agent)
    format_result = await formatter.run(
        f"Format this summary in markdown:\n\n{analysis_result.content}"
    )
    print(f"📎 Formatter output: {str(format_result.content)[:80]}...")
    
    duration = time.perf_counter() - start
    
    print(f"\n📋 Final Result:\n{format_result.content}\n")
    
    # Collect tokens from all agents
    total_tokens = 0
    for r in [search_result, analysis_result, format_result]:
        if r.metadata and r.metadata.tokens:
            total_tokens += r.metadata.tokens.total_tokens
    
    return {
        "approach": "multi_agent_peer",
        "duration_seconds": round(duration, 2),
        "llm_calls": 3,  # 3 agents = 3 LLM calls minimum
        "tokens": total_tokens,
        "content_preview": str(format_result.content)[:100] if format_result.content else "",
    }


# ============================================================================
# APPROACH 3: SINGLE ORCHESTRATOR + AGENT TOOLS (When Beneficial)
# ============================================================================


async def orchestrator_with_agent_tools() -> dict:
    """Orchestrator that strategically uses agents as tools.
    
    USE THIS WHEN:
        - You need model diversity (different perspectives)
        - Verification pattern (generate + check)
        - Truly parallel independent subtasks
    
    DON'T USE FOR:
        - Linear pipelines (just use tools!)
        - Simple tasks (overkill)
        - When single agent suffices
    """
    print("\n" + "=" * 60)
    print("APPROACH 3: Orchestrator + Strategic Agent Tools")
    print("=" * 60)
    
    observer = Observer(level="normal")
    
    # Specialist: Deep researcher (could use different model)
    researcher = Agent(
        name="DeepResearcher",
        model="gpt-4o-mini",
        tools=[search_web],
        instructions="Conduct thorough research. Return detailed findings.",
    )
    
    # Specialist: Quality checker (verification pattern)
    quality_checker = Agent(
        name="QualityChecker",
        model="claude",  # Different model provides diverse perspective
        instructions="""Check content for:
- Accuracy and factual correctness
- Completeness (covers key points)
- Clarity (easy to understand)

Return: PASS or NEEDS_IMPROVEMENT with specific feedback.""",
    )
    
    # Orchestrator controls the flow
    orchestrator = Agent(
        name="ResearchOrchestrator",
        model="gpt-4o-mini",
        tools=[
            researcher.as_tool(description="Conduct deep research on a topic"),
            quality_checker.as_tool(description="Verify content quality and accuracy"),
            format_output,  # Direct tool - no agent overhead needed
        ],
        instructions="""For research tasks:
1. Use DeepResearcher for thorough research
2. Use QualityChecker to verify accuracy
3. Use format_output to structure the final response

Only call quality checker if research returns substantial content.""",
        observer=observer,
    )
    
    start = time.perf_counter()
    result = await orchestrator.run(
        "Research Python async programming and provide a verified, formatted summary"
    )
    duration = time.perf_counter() - start
    
    print(f"\n📋 Result:\n{result.content}\n")
    
    # Tokens from orchestrator (includes nested agent-tool calls)
    tokens = 0
    if result.metadata and result.metadata.tokens:
        tokens = result.metadata.tokens.total_tokens
    
    return {
        "approach": "orchestrator_with_agent_tools",
        "duration_seconds": round(duration, 2),
        "llm_calls": 3,  # Orchestrator + 2 agent-tools (at minimum)
        "tokens": tokens,
        "content_preview": str(result.content)[:100] if result.content else "",
    }


# ============================================================================
# COMPARISON
# ============================================================================


async def main():
    print("\n" + "🔬 " * 20)
    print("SINGLE AGENT vs MULTI-AGENT COMPARISON")
    print("🔬 " * 20)
    
    results = []
    
    # Run all approaches
    results.append(await single_agent_approach())
    results.append(await multi_agent_approach())
    results.append(await orchestrator_with_agent_tools())
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("📊 COMPARISON SUMMARY")
    print("=" * 70)
    print()
    
    print(f"{'Approach':<40} {'Duration':<12} {'LLM Calls':<12} {'Tokens':<10}")
    print("-" * 74)
    
    for r in results:
        approach = r["approach"].replace("_", " ").title()
        tokens_str = str(r.get("tokens", "N/A"))
        duration_str = f"{r['duration_seconds']:.2f}s"
        print(f"{approach:<40} {duration_str:<12} {r['llm_calls']:<12} {tokens_str:<10}")
    
    print()
    print("=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. SINGLE AGENT + TOOLS (Default Choice)
   ✅ Simplest, fastest, cheapest
   ✅ One context window = no information loss
   ✅ Easy to debug and understand
   
2. MULTI-AGENT PEER (Avoid Unless Necessary)
   ❌ N agents = N+ LLM calls = higher cost
   ❌ Context passing loses information
   ❌ Coordination complexity
   
3. ORCHESTRATOR + AGENT TOOLS (Strategic Use)
   ✅ Model diversity for verification
   ✅ Parallel independent tasks
   ⚠️  Still more expensive than single agent
   
RULE OF THUMB:
    Start with single agent + tools.
    Add agent.as_tool() only for:
    - Generator + Verifier (different model perspectives)
    - Truly parallel independent subtasks
    - Specialized domain experts that benefit from isolation
""")


if __name__ == "__main__":
    asyncio.run(main())
