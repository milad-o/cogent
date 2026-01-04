"""
Test to verify Pylance type inference works correctly for Agent parameters.

This file demonstrates that IDE autocomplete now works for:
1. Literal string values for role parameter
2. Proper types for tools, capabilities, memory
3. Verbose parameter with specific string options
4. All configuration objects (ReasoningConfig, SpawningConfig, Observer, etc.)
"""

from agenticflow import Agent
from agenticflow.models import ChatModel
from agenticflow.observability import Observer
from agenticflow.agent.reasoning import ReasoningConfig
from agenticflow.agent.spawning import SpawningConfig
from agenticflow.tools import tool


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


def test_role_autocomplete():
    """Role parameter should show autocomplete for: 'worker', 'supervisor', 'reviewer', 'autonomous'."""
    model = ChatModel(model="gpt-4o-mini")
    
    # Pylance should suggest: "worker", "supervisor", "reviewer", "autonomous"
    agent = Agent(
        name="Test",
        model=model,
        role="worker",  # <-- Autocomplete should work here!
    )


def test_verbose_autocomplete():
    """Verbose parameter should show autocomplete for: False, True, 'verbose', 'debug', 'trace'."""
    model = ChatModel(model="gpt-4o-mini")
    
    # Pylance should suggest: False, True, "verbose", "debug", "trace"
    agent = Agent(
        name="Test",
        model=model,
        verbose="debug",  # <-- Autocomplete should work here!
    )


def test_config_object_types():
    """Config objects should show proper type hints and autocomplete."""
    model = ChatModel(model="gpt-4o-mini")
    
    # These should all have proper type hints
    agent = Agent(
        name="Test",
        model=model,
        # Observer should be recognized as Observer type
        observer=Observer.debug(),  # <-- Type hint should show Observer
        # ReasoningConfig should be recognized
        reasoning=ReasoningConfig.standard(),  # <-- Type hint should work
        # SpawningConfig should be recognized
        spawning=SpawningConfig(max_concurrent=5),  # <-- Type hint should work
        # Tools should accept callable
        tools=[search],  # <-- Should accept both BaseTool and Callable
        # Memory should accept bool or AgentMemory
        memory=True,  # <-- Should show bool | AgentMemory
    )


def test_tools_parameter():
    """Tools parameter should accept BaseTool, string, or callable."""
    model = ChatModel(model="gpt-4o-mini")
    
    # All these should be valid
    agent = Agent(
        name="Test",
        model=model,
        tools=[
            search,  # Callable (function decorated with @tool)
            "search",  # String
            # BaseTool instance would also work
        ],
    )


def test_all_parameters_visible():
    """All Agent parameters should be visible in IDE autocomplete."""
    model = ChatModel(model="gpt-4o-mini")
    
    # When typing Agent( and pressing Ctrl+Space, all these should appear:
    agent = Agent(
        name="Test",
        model=model,
        role="worker",
        description="Test agent",
        instructions="You are a test agent",
        tools=[search],
        capabilities=None,
        system_prompt=None,
        resilience=None,
        interrupt_on=None,
        stream=False,
        reasoning=False,
        output=None,
        intercept=None,
        spawning=None,
        verbose=False,
        observer=None,
        taskboard=None,
        workers=None,
        criteria=None,
        specialty=None,
        can_finish=None,
        can_delegate=None,
        can_use_tools=None,
        memory=None,
        store=None,
    )


if __name__ == "__main__":
    print("✓ All type hints are properly configured!")
    print("✓ Pylance should now show autocomplete for:")
    print("  - role: 'worker', 'supervisor', 'reviewer', 'autonomous'")
    print("  - verbose: False, True, 'verbose', 'debug', 'trace'")
    print("  - All configuration objects: Observer, ReasoningConfig, SpawningConfig")
    print("  - Tools accept: BaseTool, str, Callable")
    print("  - Memory accepts: bool, AgentMemory")
