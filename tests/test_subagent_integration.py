"""Integration tests for subagent functionality."""

from cogent.agent.base import Agent
from cogent.models.openai import OpenAIChat


def test_agent_with_subagents_parameter():
    """Test that Agent accepts subagents parameter and initializes registry."""
    model = OpenAIChat(model="gpt-4o-mini")

    # Create subagents
    analyst = Agent(
        name="analyst",
        model=model,
        description="Data analysis specialist",
    )

    writer = Agent(
        name="writer",
        model=model,
        description="Report writing specialist",
    )

    # Create coordinator with subagents
    coordinator = Agent(
        name="coordinator",
        model=model,
        subagents={
            "analyst": analyst,
            "writer": writer,
        },
    )

    # Verify subagent registry exists
    assert coordinator._subagent_registry is not None
    assert coordinator._subagent_registry.count == 2
    assert coordinator._subagent_registry.has_subagent("analyst")
    assert coordinator._subagent_registry.has_subagent("writer")


def test_agent_without_subagents():
    """Test that Agent works without subagents (backward compatibility)."""
    model = OpenAIChat(model="gpt-4o-mini")

    agent = Agent(
        name="simple",
        model=model,
        description="Simple agent",
    )

    # Registry should exist but be empty
    assert agent._subagent_registry is not None
    assert agent._subagent_registry.count == 0


def test_subagents_registered_as_tools():
    """Test that subagents are automatically registered as tools."""
    model = OpenAIChat(model="gpt-4o-mini")

    analyst = Agent(
        name="analyst",
        model=model,
        description="Data analysis specialist",
    )

    coordinator = Agent(
        name="coordinator",
        model=model,
        subagents={"analyst": analyst},
    )

    # Check that analyst was added as a tool
    tool_names = [t.name for t in coordinator.all_tools]
    assert "analyst" in tool_names

    # Get the tool
    analyst_tool = next(t for t in coordinator.all_tools if t.name == "analyst")
    assert analyst_tool is not None
    assert "Data analysis specialist" in analyst_tool.description


def test_system_prompt_includes_subagent_docs():
    """Test that system prompt includes subagent documentation."""
    model = OpenAIChat(model="gpt-4o-mini")

    analyst = Agent(
        name="analyst",
        model=model,
        description="Financial data analysis specialist",
    )

    writer = Agent(
        name="writer",
        model=model,
        description="Report writing specialist",
    )

    coordinator = Agent(
        name="coordinator",
        model=model,
        instructions="You are a coordinator.",
        subagents={
            "analyst": analyst,
            "writer": writer,
        },
    )

    system_prompt = coordinator.get_effective_system_prompt()

    assert system_prompt is not None
    assert "Specialist Agents" in system_prompt
    assert "analyst" in system_prompt
    assert "Financial data analysis specialist" in system_prompt
    assert "writer" in system_prompt
    assert "Report writing specialist" in system_prompt


def test_multiple_subagents():
    """Test agent with multiple subagents."""
    model = OpenAIChat(model="gpt-4o-mini")

    subagents_dict = {
        f"agent{i}": Agent(
            name=f"agent{i}",
            model=model,
            description=f"Specialist {i}",
        )
        for i in range(5)
    }

    coordinator = Agent(
        name="coordinator",
        model=model,
        subagents=subagents_dict,
    )

    assert coordinator._subagent_registry.count == 5
    for i in range(5):
        assert coordinator._subagent_registry.has_subagent(f"agent{i}")


def test_subagent_tool_has_task_parameter():
    """Test that generated subagent tools have task parameter."""
    model = OpenAIChat(model="gpt-4o-mini")

    analyst = Agent(
        name="analyst",
        model=model,
        description="Data specialist",
    )

    coordinator = Agent(
        name="coordinator",
        model=model,
        subagents={"analyst": analyst},
    )

    analyst_tool = next(t for t in coordinator.all_tools if t.name == "analyst")

    # Check args schema
    assert analyst_tool.args_schema is not None
    schema_fields = analyst_tool.args_schema.model_fields
    assert "task" in schema_fields

    # Check description mentions task
    task_field = schema_fields["task"]
    assert (
        "task" in task_field.description.lower()
        or "question" in task_field.description.lower()
    )
