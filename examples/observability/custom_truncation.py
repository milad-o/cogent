"""
Custom Observer Truncation Settings

This example demonstrates how to customize truncation limits for different
types of content in the observer output.

## Truncation Parameters

- `truncate`: General content (default: 500)
- `truncate_tool_args`: Tool arguments (default: 300)
- `truncate_tool_results`: Tool results (default: 400)
- `truncate_messages`: Message content (default: 500)

## Run

    uv run python examples/observability/custom_truncation.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model

from agenticflow import Agent
from agenticflow.observability import Observer, ObservabilityLevel


async def main():
    """Demonstrate custom truncation settings."""
    print("="*80)
    print("CUSTOM TRUNCATION SETTINGS")
    print("="*80)
    
    # Create observer with custom truncation for each content type
    observer = Observer(
        level=ObservabilityLevel.DEBUG,
        truncate=800,              # General content: 800 chars
        truncate_tool_args=500,    # Tool arguments: 500 chars (more detail)
        truncate_tool_results=600, # Tool results: 600 chars (more detail)
        truncate_messages=400,     # Messages: 400 chars
    )
    
    print("\n✓ Observer configured with custom truncation:")
    print(f"  • General content: {observer.config.truncate} chars")
    print(f"  • Tool arguments: {observer.config.truncate_tool_args} chars")
    print(f"  • Tool results: {observer.config.truncate_tool_results} chars")
    print(f"  • Messages: {observer.config.truncate_messages} chars")
    
    # Create agent with taskboard
    model = get_model()
    agent = Agent(
        name="DetailedAgent",
        model=model,
        taskboard=True,
        observer=observer,
    )
    
    task = """
    Create a detailed plan for building a REST API:
    1. Define the core endpoints (list at least 5)
    2. Identify authentication requirements
    3. Describe error handling strategy
    
    Track your work with tasks and add detailed notes.
    """
    
    print(f"\n📝 Task:\n{task}\n")
    print("="*80)
    print("EXECUTION LOG (with custom truncation)")
    print("="*80 + "\n")
    
    result = await agent.run(task)
    
    print("\n" + "="*80)
    print("RESULT")
    print("="*80)
    print(result)
    
    print("\n" + "="*80)
    print("TASKBOARD")
    print("="*80)
    print(agent.taskboard.summary())


if __name__ == "__main__":
    asyncio.run(main())
