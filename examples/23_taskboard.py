#!/usr/bin/env python3
"""
Demo: TaskBoard - Human-Like Task Tracking

Shows how agents can manage their own work like humans do:
- Breaking tasks into a checklist
- Tracking progress step by step  
- Verifying completion before finishing
- Recording notes and observations

This creates more reliable, thorough agents that don't skip steps.

Usage:
    # Just enable taskboard - that's it!
    agent = Agent(
        name="Researcher",
        model=model,
        tools=[search, summarize],
        taskboard=True,  # Adds task management + instructions
    )
"""

import asyncio

from config import get_model


async def demo():
    from agenticflow import Agent
    from agenticflow.tools.base import tool
    
    print("=" * 70)
    print("📋 TaskBoard: Human-Like Task Tracking")
    print("=" * 70)
    
    model = get_model()
    
    # === Step 1: Create work tools ===
    
    @tool
    def search_web(query: str) -> str:
        """Search the web for information."""
        results = {
            "python async": "Python async uses async/await syntax. Key concepts: coroutines, event loop, asyncio library.",
            "python generators": "Generators use yield to produce values lazily. Memory efficient for large sequences.",
            "python decorators": "Decorators wrap functions to extend behavior. Use @decorator syntax.",
        }
        for key, val in results.items():
            if key in query.lower():
                return val
        return f"Found general information about: {query}"
    
    @tool
    def write_summary(topic: str, content: str) -> str:
        """Write a summary of content about a topic."""
        return f"**{topic.title()} Summary**\n{content[:100]}... [summarized]"
    
    # === Step 2: Create agent with taskboard enabled ===
    
    agent = Agent(
        name="Researcher",
        model=model,
        tools=[search_web, write_summary],
        taskboard=True,  # 🎯 That's it! Agent now has task management
        instructions="""You are a thorough researcher who works systematically.
        
For multi-step tasks, break them down and track your progress.
Verify your work is complete before giving your final answer.""",
        verbose="debug",
    )
    
    # Note: Default is 10 iterations. For complex tasks needing more,
    # the executor's max_iterations can be configured.
    
    print(f"\n📌 Created: {agent.name}")
    print("🛠️  Work tools: search_web, write_summary")
    print("📋 TaskBoard: enabled (adds plan_tasks, update_task, add_note, check_progress, verify_done)")
    
    # === Step 3: Give a task that requires multiple steps ===
    
    task = """
    Research Python async programming:
    1. Search for information about it
    2. Write a brief summary
    
    Make sure to track your progress and verify completion.
    """
    
    print("\n" + "-" * 70)
    print("📝 Task:")
    print(task)
    print("-" * 70)
    
    print("\n🔄 Agent working (watch for task management)...\n")
    
    # Use max_iterations=20 for multi-step tasks with taskboard
    result = await agent.run(task, strategy="dag", max_iterations=20)
    
    print("\n" + "-" * 70)
    print("📊 Final Result:")
    print("-" * 70)
    print(result)
    
    # === Step 4: Show taskboard state ===
    
    print("\n" + "=" * 70)
    print("📋 Agent's TaskBoard After Work:")
    print("=" * 70)
    print(agent.taskboard.summary())
    
    print("\n" + "=" * 70)
    print("✅ Demo complete!")
    print("   Notice how the agent used taskboard tools to:")
    print("   1. Break down the task into steps")
    print("   2. Track progress through each step")
    print("   3. Take notes on findings")
    print("   4. Verify completion")
    print("=" * 70)


async def demo_simple():
    """Even simpler demo."""
    from agenticflow import Agent
    from agenticflow.tools.base import tool
    
    model = get_model()
    
    @tool
    def calculate(expression: str) -> str:
        """Calculate a math expression."""
        try:
            return f"{expression} = {eval(expression)}"
        except Exception as e:
            return f"Error: {e}"
    
    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Found information about: {query}"
    
    # Just pass taskboard=True - super simple!
    agent = Agent(
        name="Assistant",
        model=model,
        tools=[calculate, search],
        taskboard=True,
    )
    
    print("\n🤖 Simple TaskBoard Demo\n")
    
    task = "Calculate 15% tip on $85.50, then search for tipping etiquette"
    print(f"Task: {task}\n")
    
    result = await agent.run(task, strategy="dag")
    print(f"\nResult: {result}")
    
    # Show what got tracked
    print(f"\n📋 Tasks tracked:")
    for t in agent.taskboard.get_all_tasks():
        print(f"  {t}")


if __name__ == "__main__":
    asyncio.run(demo())
    # asyncio.run(demo_simple())
