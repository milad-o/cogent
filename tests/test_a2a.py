"""Tests for Agent-to-Agent (A2A) communication."""

import pytest

from agenticflow.flow.a2a import AgentRequest, AgentResponse, create_request, create_response
from agenticflow.core.response import Response, ResponseMetadata, TokenUsage, ErrorInfo


class TestAgentRequest:
    """Tests for AgentRequest."""

    def test_create_request(self) -> None:
        """Test creating an agent request."""
        request = AgentRequest(
            from_agent="coordinator",
            to_agent="specialist",
            task="analyze data",
            data={"dataset": "users.csv"},
        )

        assert request.from_agent == "coordinator"
        assert request.to_agent == "specialist"
        assert request.task == "analyze data"
        assert request.data == {"dataset": "users.csv"}
        assert request.correlation_id is not None
        assert request.reply_to == "agent.response"

    def test_request_to_event(self) -> None:
        """Test converting request to event."""
        request = AgentRequest(
            from_agent="coordinator",
            to_agent="specialist",
            task="analyze",
            correlation_id="corr-123",
        )

        event = request.to_event()

        assert event.name == "agent.request"
        assert event.data["from_agent"] == "coordinator"
        assert event.data["to_agent"] == "specialist"
        assert event.data["task"] == "analyze"
        assert event.correlation_id == "corr-123"

    def test_create_request_helper(self) -> None:
        """Test create_request helper function."""
        request = create_request(
            from_agent="agent1",
            to_agent="agent2",
            task="test task",
            data={"key": "value"},
        )

        assert isinstance(request, AgentRequest)
        assert request.from_agent == "agent1"
        assert request.to_agent == "agent2"


class TestAgentResponse:
    """Tests for AgentResponse."""

    def test_create_response(self) -> None:
        """Test creating an agent response."""
        response = AgentResponse(
            from_agent="specialist",
            to_agent="coordinator",
            result={"analysis": "complete"},
            correlation_id="corr-123",
        )

        assert response.from_agent == "specialist"
        assert response.to_agent == "coordinator"
        assert response.result == {"analysis": "complete"}
        assert response.correlation_id == "corr-123"
        assert response.success is True
        assert response.error is None

    def test_response_creates_internal_response_object(self) -> None:
        """Test that AgentResponse creates internal Response object."""
        response = AgentResponse(
            from_agent="agent1",
            to_agent="agent2",
            result="test result",
            correlation_id="corr-123",
        )

        assert response.response is not None
        assert isinstance(response.response, Response)
        assert response.response.content == "test result"
        assert response.response.success is True
        assert response.response.metadata.agent == "agent1"
        assert response.response.metadata.correlation_id == "corr-123"

    def test_response_with_error(self) -> None:
        """Test creating error response."""
        response = AgentResponse(
            from_agent="agent1",
            to_agent="agent2",
            result=None,
            correlation_id="corr-123",
            success=False,
            error="Task failed",
        )

        assert response.success is False
        assert response.error == "Task failed"
        assert response.response is not None
        assert response.response.success is False
        assert response.response.error is not None
        assert response.response.error.message == "Task failed"
        assert response.response.error.type == "AgentError"

    def test_from_response_success(self) -> None:
        """Test creating AgentResponse from Response object."""
        core_response = Response(
            content="analysis complete",
            metadata=ResponseMetadata(
                agent="analyst",
                correlation_id="corr-456",
                tokens=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            ),
            messages=[],
        )

        agent_response = AgentResponse.from_response(
            core_response,
            from_agent="analyst",
            to_agent="coordinator",
        )

        assert agent_response.from_agent == "analyst"
        assert agent_response.to_agent == "coordinator"
        assert agent_response.result == "analysis complete"
        assert agent_response.correlation_id == "corr-456"
        assert agent_response.success is True
        assert agent_response.error is None
        assert agent_response.response == core_response

    def test_from_response_error(self) -> None:
        """Test creating AgentResponse from error Response."""
        core_response = Response(
            content=None,
            metadata=ResponseMetadata(
                agent="worker",
                correlation_id="corr-789",
            ),
            error=ErrorInfo(
                type="ValidationError",
                message="Invalid input data",
            ),
            messages=[],
        )

        agent_response = AgentResponse.from_response(
            core_response,
            from_agent="worker",
            to_agent="supervisor",
        )

        assert agent_response.from_agent == "worker"
        assert agent_response.to_agent == "supervisor"
        assert agent_response.success is False
        assert agent_response.error == "Invalid input data"
        assert agent_response.response == core_response

    def test_response_to_event(self) -> None:
        """Test converting response to event."""
        response = AgentResponse(
            from_agent="specialist",
            to_agent="coordinator",
            result="done",
            correlation_id="corr-123",
        )

        event = response.to_event()

        assert event.name == "agent.response"
        assert event.data["from_agent"] == "specialist"
        assert event.data["to_agent"] == "coordinator"
        assert event.data["result"] == "done"
        assert event.data["success"] is True
        assert event.correlation_id == "corr-123"

    def test_response_to_event_custom_name(self) -> None:
        """Test converting response to event with custom name."""
        response = AgentResponse(
            from_agent="agent1",
            to_agent="agent2",
            result="test",
            correlation_id="corr-123",
        )

        event = response.to_event(event_name="custom.response")

        assert event.name == "custom.response"

    def test_unwrap_response(self) -> None:
        """Test unwrapping to get Response object."""
        agent_response = AgentResponse(
            from_agent="agent1",
            to_agent="agent2",
            result="test",
            correlation_id="corr-123",
        )

        core_response = agent_response.unwrap()

        assert isinstance(core_response, Response)
        assert core_response.content == "test"
        assert core_response.metadata.agent == "agent1"
        assert core_response.metadata.correlation_id == "corr-123"

    def test_create_response_helper(self) -> None:
        """Test create_response helper function."""
        response = create_response(
            from_agent="agent1",
            to_agent="agent2",
            correlation_id="corr-123",
            result="test result",
        )

        assert isinstance(response, AgentResponse)
        assert response.from_agent == "agent1"
        assert response.to_agent == "agent2"
        assert response.result == "test result"

    def test_create_response_with_error(self) -> None:
        """Test create_response with error."""
        response = create_response(
            from_agent="agent1",
            to_agent="agent2",
            correlation_id="corr-123",
            result=None,
            success=False,
            error="Failed to process",
        )

        assert response.success is False
        assert response.error == "Failed to process"


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing A2A code."""

    def test_old_style_response_creation_still_works(self) -> None:
        """Test that old code creating AgentResponse still works."""
        # This is how old code created responses
        response = AgentResponse(
            from_agent="agent1",
            to_agent="agent2",
            result={"data": "value"},
            correlation_id="corr-123",
            success=True,
        )

        # Should still work as before
        assert response.result == {"data": "value"}
        assert response.success is True

        # But now also has Response object
        assert response.response is not None
        assert response.response.content == {"data": "value"}

    def test_response_fields_match_between_wrapper_and_internal(self) -> None:
        """Test that AgentResponse fields match internal Response."""
        agent_resp = AgentResponse(
            from_agent="test",
            to_agent="other",
            result="result",
            correlation_id="corr-123",
            success=True,
        )

        # Fields should match
        assert agent_resp.result == agent_resp.response.content
        assert agent_resp.success == agent_resp.response.success
        assert agent_resp.correlation_id == agent_resp.response.metadata.correlation_id
