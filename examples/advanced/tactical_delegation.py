"""Tactical Multi-Agent Delegation (Research-Backed Pattern).

This example demonstrates when to use Agent.as_tool() based on:
arXiv:2601.14652 - Multi-agent only helps for verification + parallel tasks

VALID Use Cases:
1. Generator + Verifier (different models for diversity)
2. Parallel independent tasks (orchestrator calls multiple agent-tools concurrently)

Usage:
    # Ensure OPENAI_API_KEY and ANTHROPIC_API_KEY are set in .env
    uv run python examples/advanced/tactical_delegation.py
"""

import asyncio

from cogent import Agent, Observer


async def demo_generator_verifier() -> None:
    """Generator + Verifier pattern."""
    print("=" * 60)
    print("Generator + Verifier Pattern")
    print("=" * 60)
    print()
    
    observer = Observer(level="trace")
    
    # Generator agent
    generator = Agent(
        name="CodeGenerator",
        model="gpt4",  # Auto-loads API key from .env
        instructions="""You are a code generator.
        
Generate clean, working Python code for the given task.
Follow best practices:
- Type hints
- Docstrings
- Simple, readable code
- Handle edge cases

Return ONLY the code, no explanations.""",
    )
    
    # Verifier agent (different model for diversity)
    verifier = Agent(
        name="CodeVerifier",
        model="claude",  # Different model for diversity
        instructions="""You are a code reviewer.

Review code for:
1. Correctness (logic, edge cases)
2. Type safety (proper hints)
3. Style (readability, idioms)
4. Bugs (potential errors)

Return feedback in this format:
- Issues: [list any problems]
- Suggestions: [improvements]
- Verdict: PASS or REVISE

If REVISE, explain what needs fixing.""",
    )
    
    # Orchestrator coordinates generator + verifier
    orchestrator = Agent(
        name="CodeOrchestrator",
        model="gpt4",
        tools=[
            generator.as_tool(
                description="Generate Python code for a task",
            ),
            verifier.as_tool(
                description="Verify code correctness and quality",
            ),
        ],
        instructions="""Generate code using CodeGenerator, verify with CodeVerifier, return final code.""",
        observer=observer,
    )
    
    # Execute
    task = "Create a simple Python function that reverses a string"
    
    print(f"Task: {task}")
    print()
    result = await orchestrator.run(task)
    
    print()
    print("Final Result:")
    if result.content:
        print(result.content)
    else:
        print("(No response - check API keys are set)")
    print()
    
    if result.metadata and result.metadata.tokens:
        print(f"Tokens used: {result.metadata.tokens.total_tokens}")
    print()
    
    print("\n" + "=" * 60)
    print("EXECUTION TRACE")
    print("=" * 60)
    print(observer.summary())
    print("\n" + observer.timeline(detailed=True))


async def demo_parallel_analysis() -> None:
    """Orchestrator using multiple agent-tools in parallel."""
    print("\n" + "=" * 60)
    print("Parallel Agent-Tool Execution")
    print("=" * 60)
    print()
    
    observer = Observer(level="trace")

    # Three specialist agents
    sentiment_agent = Agent(
        name="SentimentAnalyzer",
        model="gpt4",
        instructions="Analyze sentiment. Return ONLY: Positive, Negative, or Neutral with brief explanation.",
    )

    entity_agent = Agent(
        name="EntityExtractor",
        model="gpt4",
        instructions="Extract key entities (products, metrics, organizations). Return concise list.",
    )

    theme_agent = Agent(
        name="ThemeIdentifier",
        model="claude",
        instructions="Identify 2-3 main themes. Be concise.",
    )

    # Orchestrator with all three as tools
    orchestrator = Agent(
        name="TextAnalyzer",
        model="gpt4",
        tools=[
            sentiment_agent.as_tool(description="Analyze text sentiment"),
            entity_agent.as_tool(description="Extract entities from text"),
            theme_agent.as_tool(description="Identify themes in text"),
        ],
        instructions="""Analyze the text using all three tools:
1. SentimentAnalyzer for sentiment
2. EntityExtractor for entities  
3. ThemeIdentifier for themes

Call all tools in one response to execute them concurrently.
Format results clearly.""",
        observer=observer,
    )

    text = """The new AI framework launched today shows impressive performance gains.
Users reported 40% faster processing times. However, some developers 
expressed concerns about the learning curve. Overall, response has been positive."""

    print(f"Text: {text[:100]}...")
    print()
    result = await orchestrator.run(f"Analyze this text:\n\n{text}")
    
    print("Analysis Results:")
    if result.content:
        print(result.content)
    else:
        print("(No response)")
    print()
    
    print("\n" + "=" * 60)
    print("EXECUTION TRACE")
    print("=" * 60)
    print(observer.summary())
    print("\n" + observer.timeline(detailed=True))


async def main() -> None:
    """Run demonstrations."""
    await demo_generator_verifier()
    await demo_parallel_analysis()


if __name__ == "__main__":
    asyncio.run(main())
