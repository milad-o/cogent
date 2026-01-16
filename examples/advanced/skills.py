"""Reactive Skills Demo.

This example demonstrates how Skills inject specialized prompts and tools
into agents based on event patterns. Skills are event-triggered behavioral
specializations that dynamically modify agent behavior.

Run with:
    uv run python examples/reactive/skills_demo.py
"""

import asyncio
import sys
from pathlib import Path

# Add examples to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_model

from agenticflow import Agent, tool
from agenticflow.reactive import (
    ReactiveFlow,
    Observer,
    react_to,
    skill,
)


# =============================================================================
# Define tools that will be available to skills
# =============================================================================


@tool
def run_python(code: str) -> str:
    """Execute Python code and return output (simulated)."""
    return f"✓ Executed Python:\n```python\n{code[:100]}...\n```\nOutput: <execution result>"


@tool
def lint_code(code: str) -> str:
    """Lint Python code and report issues."""
    return "✓ Linting passed: No PEP 8 violations found."


@tool
def read_logs(path: str) -> str:
    """Read log files for debugging."""
    return f"📋 Logs from {path}:\n[2024-01-10 10:15:32] ERROR: Connection timeout\n[2024-01-10 10:15:33] RETRY: Attempting reconnect..."


@tool
def inspect_vars(var_name: str) -> str:
    """Inspect runtime variable values."""
    return f"🔍 {var_name} = <Response status=500, body='Connection refused'>"


# =============================================================================
# Define skills with the clean kwargs API
# =============================================================================

# Python expert skill: activates on code.write events for Python
python_skill = skill(
    "python_expert",
    on="code.write",
    when=lambda e: e.data.get("language") == "python",
    prompt="""## Python Expert Mode

You are now operating as a Python expert. Follow these principles:
- Write type-annotated, PEP 8 compliant code
- Prefer composition over inheritance
- Use dataclasses for simple data containers
- Handle errors explicitly with custom exceptions
- Use modern Python 3.13+ features
""",
    tools=[run_python, lint_code],
    priority=10,
)

# Debugging skill: activates on any error.* event
debug_skill = skill(
    "debugger",
    on="error.*",
    prompt="""## Debugging Mode

You are now in systematic debugging mode:
1. Read the logs to understand what happened
2. Inspect relevant variables
3. Form a hypothesis about the root cause
4. Propose a fix with confidence level
""",
    tools=[read_logs, inspect_vars],
    priority=20,
)


# =============================================================================
# Demo: ReactiveFlow with Skills and Real LLM
# =============================================================================


async def main() -> None:
    """Demonstrate skills in action with a real LLM."""
    print("=" * 60)
    print("Reactive Skills Demo (with LLM)")
    print("=" * 60)

    # Create flow with observer
    observer = Observer.progress()
    flow = ReactiveFlow(observer=observer)

    # Register skills
    flow.register_skill(python_skill)
    flow.register_skill(debug_skill)
    print(f"\n✓ Registered skills: {flow.skills}")

    # Create a coder agent that reacts to code events
    coder = Agent(
        name="coder",
        model=get_model(),
        system_prompt="You are a helpful coding assistant. When asked to write code, produce clean, working solutions.",
    )
    flow.register(coder, [react_to("code.write")])

    # Create an error handler agent that reacts to errors
    debugger_agent = Agent(
        name="debugger_agent",
        model=get_model(),
        system_prompt="You are a debugging expert. Analyze errors systematically and propose solutions.",
    )
    flow.register(debugger_agent, [react_to("error.*")])

    print(f"✓ Registered agents: {flow.agents}")

    # Run 1: Python code write event → triggers python_skill + coder agent
    print("\n" + "-" * 60)
    print("Test 1: code.write event (language=python)")
    print("-" * 60)

    result1 = await flow.run(
        "Write a function to calculate fibonacci numbers",
        initial_event="code.write",
        initial_data={"language": "python"},
    )
    print(f"\n📄 Output:\n{result1.output[:500]}...")

    # Run 2: Error event → triggers debug_skill + debugger agent
    print("\n" + "-" * 60)
    print("Test 2: error.network event")
    print("-" * 60)

    result2 = await flow.run(
        "Investigate why the API connection is failing",
        initial_event="error.network",
        initial_data={"message": "Connection refused", "endpoint": "/api/v1/users"},
    )
    print(f"\n📄 Output:\n{result2.output[:500]}...")

    print("\n" + "=" * 60)
    print("✅ Skills demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
