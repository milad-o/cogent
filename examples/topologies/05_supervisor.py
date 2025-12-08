#!/usr/bin/env python3
"""
Demo: Supervisor Chatbot Pattern (no prebuilt needed!)

Shows how easy it is to create a conversational supervisor
that delegates to workers using just Agent + tools.
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model, settings


async def demo():
    from agenticflow.tools.base import tool
    from agenticflow import Agent, AgentRole
    
    print("=" * 60)
    print("👔 Supervisor Chatbot Pattern")
    print("=" * 60)
    
    model = get_model()
    
    # === Step 1: Create worker agents ===
    
    @tool
    def search_web(query: str) -> str:
        """Search the web for information."""
        results = {
            "ai trends": "AI trends 2024: LLMs, agents, multimodal AI, RAG systems.",
            "python": "Python remains #1 language for AI/ML.",
        }
        for key, val in results.items():
            if key in query.lower():
                return val
        return f'Results for "{query}": Various articles found.'
    
    researcher = Agent(
        name="Researcher",
        role=AgentRole.WORKER,
        model=model,
        tools=[search_web],
    )
    
    writer = Agent(
        name="Writer",
        role=AgentRole.WORKER,
        model=model,
        instructions="You write clear, concise summaries.",
    )
    
    # === Step 2: Create delegation tools ===
    
    @tool
    async def ask_researcher(task: str) -> str:
        """Delegate a research task to the Researcher."""
        print(f"  📤 → Researcher: {task[:50]}...")
        result = await researcher.run(task, strategy="dag")
        print(f"  📥 ← Researcher done")
        return result
    
    @tool
    async def ask_writer(task: str) -> str:
        """Delegate a writing task to the Writer."""
        print(f"  📤 → Writer: {task[:50]}...")
        result = await writer.think(task)
        print(f"  📥 ← Writer done")
        return result
    
    # === Step 3: Create supervisor with delegation tools ===
    
    supervisor = Agent(
        name="TeamLead",
        role=AgentRole.SUPERVISOR,
        model=model,
        instructions="""You are a helpful team lead. You have two team members:
- Researcher: Can search the web for information
- Writer: Can write clear summaries

For simple questions, answer directly.
For research tasks, use ask_researcher.
For writing tasks, use ask_writer.
Be conversational and friendly.""",
        tools=[ask_researcher, ask_writer],
        memory=True,
    )
    
    print(f"\n📌 Created: {supervisor.name}")
    print(f"👥 Workers: Researcher, Writer")
    
    # === Step 4: Chat! ===
    
    print("\n" + "-" * 40)
    print("👤 User: Hi! What can you help me with?")
    response = await supervisor.run("Hi! What can you help me with?", strategy="dag")
    print(f"👔 TeamLead: {response}")
    
    print("\n" + "-" * 40)
    print("👤 User: Can you research AI trends for me?")
    response = await supervisor.run("Can you research AI trends for me?", strategy="dag")
    print(f"\n👔 TeamLead: {response}")
    
    print("\n" + "-" * 40)
    print("👤 User: Now have the writer summarize that in one sentence.")
    response = await supervisor.run(
        "Now have the writer summarize that in one sentence.",
        strategy="dag"
    )
    print(f"\n👔 TeamLead: {response}")
    
    print("\n" + "=" * 60)
    print("✅ That's it! No prebuilt needed - just Agent + tools.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())
