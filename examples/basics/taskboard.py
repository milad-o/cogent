"""
TaskBoard Example - Agent working memory and task tracking.

Demonstrates how agents can manage their own todo lists and track progress
on complex multi-step work.

TaskBoard provides:
- Task planning and tracking (pending → in_progress → done/failed)
- Note-taking for observations and insights
- Error tracking and pattern learning
- Self-verification before completion

Run:
    uv run python basics/taskboard.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model

from agenticflow import Agent
from agenticflow.agent.taskboard import TaskBoardConfig
from agenticflow.tools.base import tool


@tool
def search_docs(query: str) -> str:
    """Search documentation (mock).
    
    Args:
        query: Search query
        
    Returns:
        Mock search results
    """
    results = {
        "async": "Python async uses event loop, async/await syntax, asyncio library",
        "await": "await pauses execution until coroutine completes",
        "asyncio": "asyncio is the standard library for async programming",
        "patterns": "Common patterns: gather, create_task, TaskGroup",
    }
    
    for key, value in results.items():
        if key in query.lower():
            return f"📚 Found: {value}"
    
    return "No results found."


@tool
def run_code(code: str) -> str:
    """Run Python code (mock).
    
    Args:
        code: Python code to execute
        
    Returns:
        Execution result
    """
    # Mock execution
    if "async" in code and "await" in code:
        return "✅ Code runs successfully\nOutput: Task completed"
    elif "syntax error" in code.lower():
        return "❌ SyntaxError: invalid syntax"
    else:
        return "✅ Execution successful"


async def demo_basic_taskboard():
    """Demo 1: Basic TaskBoard usage."""
    print("=" * 70)
    print("Demo 1: Basic TaskBoard - Agent Plans and Tracks Work")
    print("=" * 70)
    print()
    
    model = get_model()
    
    # Enable taskboard with default configuration
    agent = Agent(
        name="Researcher",
        model=model,
        tools=[search_docs, run_code],
        taskboard=True,  # ← Adds task management tools
    )
    
    print(f"✅ Agent '{agent.name}' created with TaskBoard enabled")
    print(f"   Tools available: {[t.name for t in agent.all_tools]}")
    print()
    
    # The agent will naturally use taskboard tools to:
    # 1. plan_tasks() - Break down work
    # 2. update_task() - Track progress
    # 3. check_progress() - Monitor status
    # 4. verify_done() - Confirm completion
    
    task = "Research Python async/await patterns and provide 3 key concepts with examples"
    
    print(f"Task: {task}")
    print()
    print("Agent working...")
    print("-" * 70)
    
    result = await agent.run(task)
    
    print("-" * 70)
    print()
    print("Final Answer:")
    print(result)
    print()
    
    # Inspect what the agent tracked
    print("=" * 70)
    print("TaskBoard Summary:")
    print("=" * 70)
    print(agent.taskboard.summary())
    print()


async def demo_taskboard_with_verification():
    """Demo 2: TaskBoard with auto-verification."""
    print("=" * 70)
    print("Demo 2: TaskBoard with Auto-Verification")
    print("=" * 70)
    print()
    
    model = get_model()
    
    # Configure taskboard to require verification
    agent = Agent(
        name="CodeReviewer",
        model=model,
        tools=[search_docs, run_code],
        taskboard=TaskBoardConfig(
            auto_verify=True,      # Agent must verify completion
            max_tasks=20,
            track_errors=True,
        ),
    )
    
    print(f"✅ Agent '{agent.name}' created with auto-verification")
    print()
    
    task = "Write and test async function to fetch 3 URLs concurrently"
    
    print(f"Task: {task}")
    print()
    print("Agent working (will verify before finishing)...")
    print("-" * 70)
    
    result = await agent.run(task)
    
    print("-" * 70)
    print()
    print("Final Answer:")
    print(result)
    print()
    print("TaskBoard:")
    print(agent.taskboard.summary())
    print()


async def demo_taskboard_inspection():
    """Demo 3: Inspecting TaskBoard state."""
    print("=" * 70)
    print("Demo 3: TaskBoard Inspection - Direct Access")
    print("=" * 70)
    print()
    
    model = get_model()
    
    agent = Agent(
        name="Analyst",
        model=model,
        tools=[search_docs],
        taskboard=True,
    )
    
    task = "Compare async vs threading for I/O-bound tasks"
    
    print(f"Task: {task}")
    print()
    
    result = await agent.run(task)
    
    print("Result:", result)
    print()
    
    # Direct inspection of taskboard state
    print("=" * 70)
    print("TaskBoard Detailed Inspection:")
    print("=" * 70)
    print()
    
    # Goal
    goal = agent.taskboard.get_goal()
    if goal:
        print(f"🎯 Goal: {goal}")
        print()
    
    # Tasks breakdown
    all_tasks = agent.taskboard.get_all_tasks()
    done = agent.taskboard.get_done()
    pending = agent.taskboard.get_pending()
    failed = agent.taskboard.get_failed()
    
    print(f"📋 Tasks: {len(all_tasks)} total")
    print(f"   ✓ Done: {len(done)}")
    print(f"   ○ Pending: {len(pending)}")
    print(f"   ✗ Failed: {len(failed)}")
    print()
    
    if all_tasks:
        print("Task Details:")
        for task in all_tasks:
            print(f"  {task}")
            if task.result:
                print(f"    → Result: {task.result}")
        print()
    
    # Notes
    notes = agent.taskboard.get_notes()
    if notes:
        print(f"📝 Notes: {len(notes)}")
        for note in notes:
            icon = {"observation": "•", "insight": "💡", "question": "❓"}.get(note.category, "•")
            print(f"  {icon} [{note.category}] {note.content}")
        print()
    
    # Verification
    verification = agent.taskboard.verify_completion()
    print("✓ Verification:")
    print(f"  Complete: {verification['complete']}")
    if verification['issues']:
        print("  Issues:")
        for issue in verification['issues']:
            print(f"    - {issue}")
    else:
        print("  No issues found")
    print()


async def demo_taskboard_serialization():
    """Demo 4: TaskBoard serialization."""
    print("=" * 70)
    print("Demo 4: TaskBoard Serialization - Export State")
    print("=" * 70)
    print()
    
    model = get_model()
    
    agent = Agent(
        name="Worker",
        model=model,
        tools=[search_docs],
        taskboard=True,
    )
    
    task = "Find 2 use cases for Python asyncio"
    
    print(f"Task: {task}")
    print()
    
    await agent.run(task)
    
    # Export taskboard state as dict
    state = agent.taskboard.to_dict()
    
    print("=" * 70)
    print("TaskBoard State (JSON-serializable):")
    print("=" * 70)
    print()
    
    import json
    print(json.dumps(state, indent=2, default=str))
    print()
    
    print(f"✅ TaskBoard state can be saved/restored")
    print(f"   - Goal: {state.get('goal')}")
    print(f"   - Tasks: {len(state.get('tasks', []))}")
    print(f"   - Notes: {len(state.get('notes', []))}")
    print(f"   - Reflections: {len(state.get('reflections', []))}")
    print()


async def main():
    """Run all TaskBoard demos."""
    print()
    print("🎯 TaskBoard Examples - Agent Working Memory")
    print()
    
    # Demo 1: Basic usage
    await demo_basic_taskboard()
    
    input("Press Enter to continue to Demo 2...")
    print()
    
    # Demo 2: With verification
    await demo_taskboard_with_verification()
    
    input("Press Enter to continue to Demo 3...")
    print()
    
    # Demo 3: Detailed inspection
    await demo_taskboard_inspection()
    
    input("Press Enter to continue to Demo 4...")
    print()
    
    # Demo 4: Serialization
    await demo_taskboard_serialization()
    
    print("=" * 70)
    print("✅ All TaskBoard demos completed!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  1. TaskBoard gives agents structured working memory")
    print("  2. Agents naturally plan, track, and verify work")
    print("  3. TaskBoard state is inspectable and serializable")
    print("  4. Useful for complex multi-step tasks")
    print()


if __name__ == "__main__":
    asyncio.run(main())
