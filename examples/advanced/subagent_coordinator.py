"""Advanced subagent coordinator example.

Demonstrates a full-featured coordinator with multiple specialists,
metadata aggregation, and delegation chain tracking.

Run:
    uv run python examples/advanced/subagent_coordinator.py
"""

import asyncio

from cogent import Agent, Observer


async def main():
    """Research coordinator with multiple specialist subagents."""
    
    observer = Observer(level="progress")
    
    # Create specialist agents
    data_analyst = Agent(
        name="data_analyst",
        model="gemini:gemini-2.5-flash",
        instructions="You analyze data and provide statistical insights. Be thorough and precise.",
    )
    
    market_researcher = Agent(
        name="market_researcher",
        model="gemini:gemini-2.5-flash",
        instructions="You research market trends, competitive landscape, and industry dynamics.",
    )
    
    technical_writer = Agent(
        name="technical_writer",
        model="gemini:gemini-2.5-flash",
        instructions="You write clear, concise technical documentation and reports.",
    )
    
    # Create coordinator with multiple subagents
    coordinator = Agent(
        name="research_coordinator",
        model="gemini:gemini-2.5-pro",
        instructions="""You coordinate complex research tasks by delegating to specialists:

- data_analyst: For statistical analysis, data interpretation, trends
- market_researcher: For market analysis, competitive intelligence, industry research
- technical_writer: For creating polished reports and documentation

Workflow:
1. Delegate to appropriate specialists based on the task
2. You may delegate to multiple specialists for comprehensive analysis
3. Synthesize their findings into a cohesive final response

Be strategic about delegation - use specialists when their expertise adds value.""",
        # Simply pass the agents - uses their names automatically
        subagents=[data_analyst, market_researcher, technical_writer],
        observer=observer,
    )
    
    # Example task
    print("=" * 70)
    print("TASK: Comprehensive Market Analysis")
    print("=" * 70)
    
    task = """Analyze the global AI chip market for Q4 2025:

Data points:
- Market size: $28B (up 45% YoY)
- NVIDIA market share: 82%
- AMD market share: 12%
- Others: 6%
- Average selling price: +18% vs Q4 2024
- Data center demand: 3x consumer demand

Deliverable: Executive summary with key insights and implications."""
    
    response = await coordinator.run(task)
    
    # Display results
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"\n{response.content}\n")
    
    # Show metadata aggregation
    print("=" * 70)
    print("METADATA & DELEGATION CHAIN")
    print("=" * 70)
    
    # Token usage (coordinator + all subagents)
    tokens = response.metadata.tokens
    print(f"\n💰 TOTAL TOKEN USAGE: {tokens.total_tokens}")
    print(f"   └─ Prompt: {tokens.prompt_tokens}, Completion: {tokens.completion_tokens}")
    
    # Subagent breakdown
    if response.subagent_responses:
        print(f"\n🤝 SUBAGENT CALLS: {len(response.subagent_responses)}")
        for i, sub_resp in enumerate(response.subagent_responses, 1):
            print(f"\n   {i}. {sub_resp.metadata.agent}")
            print(f"      ├─ Tokens: {sub_resp.metadata.tokens.total_tokens}")
            print(f"      ├─ Duration: {sub_resp.metadata.duration:.2f}s")
            print(f"      └─ Model: {sub_resp.metadata.model}")
    
    # Delegation chain
    if response.metadata.delegation_chain:
        print(f"\n📋 DELEGATION CHAIN:")
        total_subagent_tokens = 0
        total_subagent_duration = 0
        
        for i, delegation in enumerate(response.metadata.delegation_chain, 1):
            print(f"\n   {i}. {delegation['agent']}")
            print(f"      ├─ Tokens: {delegation['tokens']}")
            print(f"      ├─ Duration: {delegation['duration']:.2f}s")
            print(f"      └─ Model: {delegation['model']}")
            
            total_subagent_tokens += delegation['tokens']
            total_subagent_duration += delegation['duration']
        
        print(f"\n   Summary:")
        print(f"   ├─ Coordinator tokens: {tokens.total_tokens - total_subagent_tokens}")
        print(f"   ├─ Subagent tokens: {total_subagent_tokens}")
        print(f"   └─ Subagent time: {total_subagent_duration:.2f}s")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
