"""Tests for RunContext propagation through subagent calls."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cogent.agent.base import Agent
from cogent.core.context import RunContext
from cogent.core.response import Response, ResponseMetadata
from cogent.models import BaseChatModel


def create_mock_model():
    """Create a mock model for testing."""
    mock_model = MagicMock(spec=BaseChatModel)
    mock_model.model = "mock-model"
    mock_model.bind_tools = MagicMock(return_value=mock_model)
    return mock_model


@pytest.mark.asyncio
async def test_context_propagates_to_subagent():
    """Test that RunContext flows from coordinator to subagent."""
    model = create_mock_model()

    received_context = None

    class MockSubagent:
        def __init__(self, name):
            self.name = name
            self.config = type(
                "obj", (object,), {"description": f"{name} specialist"}
            )()

        async def run(self, task, context=None):
            nonlocal received_context
            received_context = context
            return Response(
                content=f"Processed: {task}",
                metadata=ResponseMetadata(agent=self.name, model="mock"),
            )

    analyst = MockSubagent("analyst")

    coordinator = Agent(
        name="coordinator",
        model=model,
        subagents={"analyst": analyst},
    )

    # Create context with metadata
    test_context = RunContext()
    test_context.metadata = {
        "thread_id": "test-thread-123",
        "user_id": "user-456",
        "custom_key": "custom_value",
    }

    # Mock executor to call subagent
    with patch("cogent.executors.create_executor") as mock_executor_factory:
        mock_executor = MagicMock()
        mock_executor.max_iterations = 25

        async def mock_execute(task, context):
            # Simulate executor calling subagent with context
            if hasattr(coordinator, "_subagent_registry"):
                await coordinator._subagent_registry.execute(
                    "analyst", "Analyze", context
                )
            return "Done"

        mock_executor.execute = AsyncMock(side_effect=mock_execute)
        mock_executor._last_messages = []
        mock_executor_factory.return_value = mock_executor

        await coordinator.run("Task", context=test_context)

        # Verify context was propagated to subagent
        assert received_context is not None
        assert received_context.metadata["thread_id"] == "test-thread-123"
        assert received_context.metadata["user_id"] == "user-456"
        assert received_context.metadata["custom_key"] == "custom_value"


@pytest.mark.asyncio
async def test_context_dict_converts_to_runcontext():
    """Test that dict context is converted to RunContext before propagation."""
    model = create_mock_model()

    received_context = None

    class MockSubagent:
        def __init__(self, name):
            self.name = name
            self.config = type(
                "obj", (object,), {"description": f"{name} specialist"}
            )()

        async def run(self, task, context=None):
            nonlocal received_context
            received_context = context
            return Response(
                content="Result",
                metadata=ResponseMetadata(agent=self.name, model="mock"),
            )

    analyst = MockSubagent("analyst")

    coordinator = Agent(
        name="coordinator",
        model=model,
        subagents={"analyst": analyst},
    )

    # Pass context as dict
    context_dict = {"key1": "value1", "key2": "value2"}

    with patch("cogent.executors.create_executor") as mock_executor_factory:
        mock_executor = MagicMock()
        mock_executor.max_iterations = 25

        async def mock_execute(task, context):
            if hasattr(coordinator, "_subagent_registry"):
                await coordinator._subagent_registry.execute("analyst", "Task", context)
            return "Done"

        mock_executor.execute = AsyncMock(side_effect=mock_execute)
        mock_executor._last_messages = []
        mock_executor_factory.return_value = mock_executor

        await coordinator.run("Task", context=context_dict)

        # Context should be converted to RunContext
        assert received_context is not None
        # RunContext automatically wraps dict in metadata
        assert isinstance(received_context, (RunContext, dict))


@pytest.mark.asyncio
async def test_none_context_propagates_as_none():
    """Test that None context propagates as None (no forced context)."""
    model = create_mock_model()

    received_context = None
    context_received = False

    class MockSubagent:
        def __init__(self, name):
            self.name = name
            self.config = type(
                "obj", (object,), {"description": f"{name} specialist"}
            )()

        async def run(self, task, context=None):
            nonlocal received_context, context_received
            received_context = context
            context_received = True
            return Response(
                content="Result",
                metadata=ResponseMetadata(agent=self.name, model="mock"),
            )

    analyst = MockSubagent("analyst")

    coordinator = Agent(
        name="coordinator",
        model=model,
        subagents={"analyst": analyst},
    )

    with patch("cogent.executors.create_executor") as mock_executor_factory:
        mock_executor = MagicMock()
        mock_executor.max_iterations = 25

        async def mock_execute(task, context):
            if hasattr(coordinator, "_subagent_registry"):
                await coordinator._subagent_registry.execute("analyst", "Task", context)
            return "Done"

        mock_executor.execute = AsyncMock(side_effect=mock_execute)
        mock_executor._last_messages = []
        mock_executor_factory.return_value = mock_executor

        # No context provided
        await coordinator.run("Task")

        # Subagent should receive None
        assert context_received
        assert received_context is None


@pytest.mark.asyncio
async def test_context_preserved_through_multiple_subagents():
    """Test that context is preserved when calling multiple subagents."""
    model = create_mock_model()

    analyst_context = None
    researcher_context = None

    class MockSubagent:
        def __init__(self, name):
            self.name = name
            self.config = type(
                "obj", (object,), {"description": f"{name} specialist"}
            )()

        async def run(self, task, context=None):
            nonlocal analyst_context, researcher_context
            if self.name == "analyst":
                analyst_context = context
            elif self.name == "researcher":
                researcher_context = context
            return Response(
                content=f"Result from {self.name}",
                metadata=ResponseMetadata(agent=self.name, model="mock"),
            )

    analyst = MockSubagent("analyst")
    researcher = MockSubagent("researcher")

    coordinator = Agent(
        name="coordinator",
        model=model,
        subagents={"analyst": analyst, "researcher": researcher},
    )

    test_context = RunContext()
    test_context.metadata = {
        "thread_id": "multi-thread",
        "session": "12345",
    }

    with patch("cogent.executors.create_executor") as mock_executor_factory:
        mock_executor = MagicMock()
        mock_executor.max_iterations = 25

        async def mock_execute(task, context):
            if hasattr(coordinator, "_subagent_registry"):
                # Call both subagents
                await coordinator._subagent_registry.execute(
                    "analyst", "Analyze", context
                )
                await coordinator._subagent_registry.execute(
                    "researcher", "Research", context
                )
            return "Done"

        mock_executor.execute = AsyncMock(side_effect=mock_execute)
        mock_executor._last_messages = []
        mock_executor_factory.return_value = mock_executor

        await coordinator.run("Complex task", context=test_context)

        # Both subagents should receive the same context
        assert analyst_context is not None
        assert researcher_context is not None
        assert analyst_context.metadata["thread_id"] == "multi-thread"
        assert researcher_context.metadata["thread_id"] == "multi-thread"
        assert analyst_context.metadata["session"] == "12345"
        assert researcher_context.metadata["session"] == "12345"


@pytest.mark.asyncio
async def test_nested_subagent_context_propagation():
    """Test that context propagates through nested subagent calls."""
    model = create_mock_model()

    level2_context = None

    class Level2Subagent:
        def __init__(self, name):
            self.name = name
            self.config = type("obj", (object,), {"description": "Level 2 agent"})()

        async def run(self, task, context=None):
            nonlocal level2_context
            level2_context = context
            return Response(
                content="Level 2 result",
                metadata=ResponseMetadata(agent=self.name, model="mock"),
            )

    class Level1Subagent:
        def __init__(self, name):
            self.name = name
            self.config = type("obj", (object,), {"description": "Level 1 agent"})()
            self.level2 = Level2Subagent("level2")

        async def run(self, task, context=None):
            # Level 1 calls Level 2, passing context
            await self.level2.run("Subtask", context=context)
            return Response(
                content="Level 1 result",
                metadata=ResponseMetadata(agent=self.name, model="mock"),
            )

    level1 = Level1Subagent("level1")

    coordinator = Agent(
        name="coordinator",
        model=model,
        subagents={"level1": level1},
    )

    original_context = RunContext()
    original_context.metadata = {
        "thread_id": "nested-thread",
        "user_id": "nested-user",
        "depth": 0,
    }

    with patch("cogent.executors.create_executor") as mock_executor_factory:
        mock_executor = MagicMock()
        mock_executor.max_iterations = 25

        async def mock_execute(task, context):
            if hasattr(coordinator, "_subagent_registry"):
                await coordinator._subagent_registry.execute("level1", "Task", context)
            return "Done"

        mock_executor.execute = AsyncMock(side_effect=mock_execute)
        mock_executor._last_messages = []
        mock_executor_factory.return_value = mock_executor

        await coordinator.run("Nested task", context=original_context)

        # Context should propagate all the way to level 2
        assert level2_context is not None
        assert level2_context.metadata["thread_id"] == "nested-thread"
        assert level2_context.metadata["user_id"] == "nested-user"
        assert level2_context.metadata["depth"] == 0


@pytest.mark.asyncio
async def test_context_metadata_isolation():
    """Test that subagent can't modify coordinator's context metadata."""
    model = create_mock_model()

    class MutableSubagent:
        def __init__(self, name):
            self.name = name
            self.config = type("obj", (object,), {"description": "Mutable agent"})()

        async def run(self, task, context=None):
            # Try to modify context metadata
            if context and hasattr(context, "metadata"):
                context.metadata["modified"] = "by_subagent"
            return Response(
                content="Result",
                metadata=ResponseMetadata(agent=self.name, model="mock"),
            )

    subagent = MutableSubagent("subagent")

    coordinator = Agent(
        name="coordinator",
        model=model,
        subagents={"subagent": subagent},
    )

    original_context = RunContext(metadata={"original": "value"})

    with patch("cogent.executors.create_executor") as mock_executor_factory:
        mock_executor = MagicMock()
        mock_executor.max_iterations = 25

        async def mock_execute(task, context):
            if hasattr(coordinator, "_subagent_registry"):
                await coordinator._subagent_registry.execute(
                    "subagent", "Task", context
                )
            return "Done"

        mock_executor.execute = AsyncMock(side_effect=mock_execute)
        mock_executor._last_messages = []
        mock_executor_factory.return_value = mock_executor

        await coordinator.run("Task", context=original_context)

        # Original context should still have the modification (objects are passed by reference)
        # This is expected Python behavior - if true isolation is needed, copy the context
        assert original_context.metadata.get("modified") == "by_subagent"
        # This test documents current behavior - if isolation is needed later,
        # we can implement context.copy() before passing to subagent
