"""Test model_kwargs parameter for passing model-specific configuration."""

from cogent import Agent


def test_model_kwargs_with_string_model():
    """Test that model_kwargs works with string model specification."""
    # Create agent with string model + model_kwargs
    agent = Agent(
        name="tester",
        model="gemini-2.5-flash",
        model_kwargs={"thinking_budget": 8192},
        instructions="Test agent",
    )

    # Agent should be created successfully
    assert agent.name == "tester"

    # Model should be converted to BaseChatModel instance
    from cogent.models.base import BaseChatModel

    assert isinstance(agent.model, BaseChatModel)


def test_model_kwargs_empty():
    """Test that empty model_kwargs works."""
    agent = Agent(
        name="tester",
        model="gemini-2.5-flash",
        model_kwargs={},
        instructions="Test agent",
    )

    assert agent.name is not None


def test_model_kwargs_none():
    """Test that None model_kwargs works."""
    agent = Agent(
        name="tester",
        model="gemini-2.5-flash",
        model_kwargs=None,
        instructions="Test agent",
    )

    assert agent.name is not None


def test_model_kwargs_with_chat_model_instance():
    """Test that model_kwargs is ignored when model is already a ChatModel instance."""
    from cogent.models.gemini import GeminiChat

    model = GeminiChat(model="gemini-2.5-flash")

    # model_kwargs should be ignored when model is already an instance
    agent = Agent(
        name="tester",
        model=model,
        model_kwargs={"thinking_budget": 8192},  # This gets ignored
        instructions="Test agent",
    )

    assert agent.model is model  # Same instance
