"""
Observer Output Truncation Settings

This example demonstrates the tiered API for controlling output truncation.

## API Levels

### High Level (Presets)
```python
observer = Observer.minimal()   # Minimal output
observer = Observer.verbose()   # Verbose output
observer = Observer.debug()     # Debug with no truncation
observer = Observer.trace()     # Full tracing, no truncation
```

### Mid Level (Simple)
```python
observer = Observer(max_output=500)   # Limit ALL content to 500 chars
observer = Observer(max_output=None)  # No limit (default)
```

### Low Level (Advanced)
```python
observer = Observer(
    truncate=500,              # General content
    truncate_tool_args=300,    # Tool arguments
    truncate_tool_results=400, # Tool results
    truncate_messages=500,     # Messages
)
```

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
    """Demonstrate the tiered truncation API."""
    print("="*80)
    print("OBSERVER OUTPUT TRUNCATION - TIERED API")
    print("="*80)
    
    # =========================================================================
    # HIGH LEVEL: Use presets (easiest)
    # =========================================================================
    # observer = Observer.minimal()   # Minimal output
    # observer = Observer.verbose()   # Verbose output  
    # observer = Observer.debug()     # Debug, no truncation
    # observer = Observer.trace()     # Full trace, no truncation
    
    # =========================================================================
    # MID LEVEL: Single max_output param (simple)
    # =========================================================================
    # observer = Observer(max_output=500)   # Limit all content to 500 chars
    # observer = Observer(max_output=None)  # No limit (default)
    
    # =========================================================================
    # LOW LEVEL: Fine-grained control (advanced)
    # =========================================================================
    observer = Observer(
        level=ObservabilityLevel.DEBUG,
        max_output=600,            # Base limit for all content
        truncate_tool_args=400,    # Override: tool args get less space
        truncate_messages=800,     # Override: messages get more space
    )
    
    print("\n✓ Observer configured:")
    print(f"  • Level: {observer.config.level.name}")
    print(f"  • General content: {observer.config.truncate} chars")
    print(f"  • Tool arguments: {observer.config.truncate_tool_args} chars")
    print(f"  • Tool results: {observer.config.truncate_tool_results} chars")
    print(f"  • Messages: {observer.config.truncate_messages} chars")
    print(f"\n💡 Tip: Use max_output=500 to limit all, or None for no limit")
    
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
