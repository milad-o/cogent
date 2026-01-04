"""
Agent + TaskBoard Observability - Complete Event Tracking

This example demonstrates comprehensive observability of both:
1. **Agent lifecycle events** (invoked, thinking, LLM calls, tools, responses)
2. **TaskBoard events** (tasks added, started, completed, notes added)

Shows how the standard Observer formats all events consistently, including
the agent's internal task planning and progress.

## Run

    uv run python examples/observability/agent_and_taskboard_events.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model

from agenticflow import Agent
from agenticflow.observability import Observer, ObservabilityLevel


async def main():
    """Run comprehensive observability demo."""
    print("="*80)
    print("COMPREHENSIVE AGENT + TASKBOARD OBSERVABILITY")
    print("="*80)
    
    # Create model
    model = get_model()
    
    # Use standard observer to show consistent formatting
    observer = Observer(level=ObservabilityLevel.DEBUG)
    
    # Create agent with taskboard and observability
    print("\n✓ Creating agent with taskboard and event tracking...")
    agent = Agent(
        name="ResearchAssistant",
        model=model,
        taskboard=True,  # Enable taskboard with tools
        observer=observer,
    )
    
    # Define a multi-step task
    task = """
    Research Python async programming and create a summary:
    1. First, understand what async/await is
    2. Then, identify 3 key benefits
    3. Finally, provide a simple example
    
    Track your progress using tasks and notes.
    """
    
    print(f"\n📝 Task:\n{task}\n")
    print("="*80)
    print("EXECUTION LOG (Real-time Events)")
    print("="*80 + "\n")
    
    # Run the agent
    result = await agent.run(task)
    
    # Print result
    print("\n" + "="*80)
    print("AGENT RESULT")
    print("="*80)
    print(result)
    
    # Show taskboard state
    print("\n" + "="*80)
    print("TASKBOARD STATE")
    print("="*80)
    print(agent.taskboard.summary())


if __name__ == "__main__":
    asyncio.run(main())
