"""
Demo: Simple Model Usage - 3 Tiers

Shows the 3 ways to create models in AgenticFlow:
1. High-level: Just model name (easiest)
2. Medium-level: Provider + model (more control)
3. Low-level: Full configuration (most control)

Usage:
    uv run python examples/basics/simple_models.py
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from agenticflow import Agent


async def tier_1_high_level():
    """Tier 1: High-level - Just model name (easiest)."""
    print("\n=== Tier 1: High-Level API (Model Strings) ===\n")
    
    # Just pass a string - framework auto-detects provider
    agent1 = Agent(name="Helper1", model="gemini")
    response1 = await agent1.run("What is 2+2? Answer with just the number.")
    print(f"gemini: {response1.content}")
    print(f"  [Debug] Type: {type(response1.content)}, Length: {len(response1.content) if response1.content else 0}")
    print(f"  [Debug] Messages: {len(response1.messages)}, Model: {response1.metadata.model}")
    
    # Model aliases
    agent2 = Agent(name="Helper2", model="gemini-flash")
    response2 = await agent2.run("Count to 5")
    print(f"\ngemini-flash: {response2.content}")
    
    # Provider prefix for explicit control
    agent3 = Agent(name="Helper3", model="gemini:gemini-2.5-pro")
    response3 = await agent3.run("Name a color")
    print(f"\ngemini:gemini-2.5-pro: {response3.content}")


async def tier_2_medium_level():
    """Tier 2: Medium-level - Provider + model (more control)."""
    print("\n=== Tier 2: Medium-Level API (Factory Function) ===\n")
    
    from agenticflow.models import create_chat
    
    # Provider + model name, auto-loads API key from env
    model1 = create_chat("gemini", "gemini-2.5-pro")
    agent1 = Agent(name="Helper1", model=model1)
    response1 = await agent1.run("What is 10+10?")
    print(f"Gemini (explicit): {response1.content or '(empty)'}")
    
    # Or use the shorthand (single argument)
    model2 = create_chat("gemini")
    agent2 = Agent(name="Helper2", model=model2)
    response2 = await agent2.run("What is 20+20?")
    print(f"Gemini (shorthand): {response2.content[:50] if response2.content else '(empty)'}...")


async def tier_3_low_level():
    """Tier 3: Low-level - Full configuration (most control)."""
    print("\n=== Tier 3: Low-Level API (Direct Model Classes) ===\n")
    
    from agenticflow.models import GeminiChat
    
    # Full control over all parameters
    model = GeminiChat(
        model="gemini-2.5-pro",
        temperature=0.3,
    )
    
    agent = Agent(name="Helper", model=model)
    response = await agent.run("What is 100+100?")
    print(f"Gemini (custom config): {response.content or '(empty)'}")


async def comparison():
    """All 3 approaches side by side."""
    print("\n=== Comparison: All 3 Tiers ===\n")
    
    from agenticflow.models import create_chat, GeminiChat
    
    task = "Say hello in 3 words"
    
    # Tier 1: String (easiest - recommended for most cases)
    agent1 = Agent(name="Tier1", model="gemini")
    r1 = await agent1.run(task)
    print(f"Tier 1 (string):  {r1.content or '(empty)'}")
    
    # Tier 2: Factory (good balance of ease and control)
    agent2 = Agent(name="Tier2", model=create_chat("gemini", "gemini-2.5-pro"))
    r2 = await agent2.run(task)
    print(f"Tier 2 (factory): {r2.content or '(empty)'}")
    
    # Tier 3: Direct (full control)
    agent3 = Agent(name="Tier3", model=GeminiChat(model="gemini-2.5-pro", temperature=0.1))
    r3 = await agent3.run(task)
    print(f"Tier 3 (direct):  {r3.content or '(empty)'}")


async def main():
    """Run all demos."""
    await tier_1_high_level()
    await tier_2_medium_level()
    await tier_3_low_level()
    await comparison()
    
    print("\n" + "="*60)
    print("RECOMMENDATION: Use Tier 1 (strings) for simplicity.")
    print("Only use Tier 2/3 when you need fine-grained control.")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
