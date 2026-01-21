"""Tests for unified response protocol (Phase 6)."""

import time

import pytest

from agenticflow.core import (
    ErrorInfo,
    Response,
    ResponseError,
    ResponseMetadata,
    TokenUsage,
    ToolCall,
)


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_token_usage_creation(self):
        """Test creating TokenUsage with all fields."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_token_usage_defaults(self):
        """Test TokenUsage defaults to zero."""
        usage = TokenUsage()

        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_token_usage_to_dict(self):
        """Test TokenUsage serialization."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        result = usage.to_dict()

        assert result == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_tool_call_creation(self):
        """Test creating ToolCall with all fields."""
        call = ToolCall(
            tool_name="calculator",
            arguments={"operation": "add", "x": 1, "y": 2},
            result=3,
            duration=0.5,
            success=True,
            error=None,
        )

        assert call.tool_name == "calculator"
        assert call.arguments == {"operation": "add", "x": 1, "y": 2}
        assert call.result == 3
        assert call.duration == 0.5
        assert call.success is True
        assert call.error is None

    def test_tool_call_with_error(self):
        """Test ToolCall with error information."""
        call = ToolCall(
            tool_name="web_search",
            arguments={"query": "test"},
            result=None,
            duration=0.2,
            success=False,
            error="Connection timeout",
        )

        assert call.success is False
        assert call.error == "Connection timeout"

    def test_tool_call_to_dict(self):
        """Test ToolCall serialization."""
        call = ToolCall(
            tool_name="calculator",
            arguments={"x": 1},
            result=42,
            duration=0.1,
            success=True,
        )

        result = call.to_dict()

        assert result["tool_name"] == "calculator"
        assert result["arguments"] == {"x": 1}
        assert result["result"] == 42
        assert result["duration"] == 0.1
        assert result["success"] is True
        assert result["error"] is None


class TestErrorInfo:
    """Tests for ErrorInfo dataclass."""

    def test_error_info_creation(self):
        """Test creating ErrorInfo."""
        error = ErrorInfo(
            message="Something went wrong",
            type="ValueError",
            traceback="Traceback (most recent call last):\n  ...",
        )

        assert error.message == "Something went wrong"
        assert error.type == "ValueError"
        assert error.traceback is not None

    def test_error_info_without_traceback(self):
        """Test ErrorInfo without traceback."""
        error = ErrorInfo(message="Error", type="RuntimeError")

        assert error.message == "Error"
        assert error.type == "RuntimeError"
        assert error.traceback is None

    def test_error_info_to_dict(self):
        """Test ErrorInfo serialization."""
        error = ErrorInfo(message="Test error", type="TestError", traceback="trace")

        result = error.to_dict()

        assert result == {
            "message": "Test error",
            "type": "TestError",
            "traceback": "trace",
        }


class TestResponseMetadata:
    """Tests for ResponseMetadata dataclass."""

    def test_metadata_creation(self):
        """Test creating ResponseMetadata."""
        tokens = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        metadata = ResponseMetadata(
            agent="test_agent",
            model="gpt-4",
            tokens=tokens,
            duration=1.5,
            correlation_id="corr-123",
            trace_id="trace-456",
        )

        assert metadata.agent == "test_agent"
        assert metadata.model == "gpt-4"
        assert metadata.tokens == tokens
        assert metadata.duration == 1.5
        assert metadata.correlation_id == "corr-123"
        assert metadata.trace_id == "trace-456"

    def test_metadata_timestamp_auto_generated(self):
        """Test that timestamp is auto-generated."""
        before = time.time()
        metadata = ResponseMetadata(agent="test")
        after = time.time()

        assert before <= metadata.timestamp <= after

    def test_metadata_to_dict(self):
        """Test ResponseMetadata serialization."""
        tokens = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        metadata = ResponseMetadata(
            agent="agent1",
            model="model1",
            tokens=tokens,
            duration=0.5,
            timestamp=1234567890.0,
            correlation_id="corr",
            trace_id="trace",
        )

        result = metadata.to_dict()

        assert result["agent"] == "agent1"
        assert result["model"] == "model1"
        assert result["tokens"] == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        assert result["duration"] == 0.5
        assert result["timestamp"] == 1234567890.0
        assert result["correlation_id"] == "corr"
        assert result["trace_id"] == "trace"

    def test_metadata_to_dict_with_none_tokens(self):
        """Test ResponseMetadata serialization with None tokens."""
        metadata = ResponseMetadata(agent="test", tokens=None)

        result = metadata.to_dict()

        assert result["tokens"] is None


class TestResponse:
    """Tests for Response[T] dataclass."""

    def test_response_creation(self):
        """Test creating Response with all fields."""
        metadata = ResponseMetadata(agent="test")
        tool_call = ToolCall(
            tool_name="test",
            arguments={},
            result=None,
            duration=0,
            success=True,
        )

        response = Response(
            content="Hello",
            metadata=metadata,
            tool_calls=[tool_call],
            events=[],
            error=None,
        )

        assert response.content == "Hello"
        assert response.metadata == metadata
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0] == tool_call
        assert response.events == []
        assert response.error is None

    def test_response_success_property_true(self):
        """Test success property when no error."""
        metadata = ResponseMetadata(agent="test")
        response = Response(content="ok", metadata=metadata, error=None)

        assert response.success is True

    def test_response_success_property_false(self):
        """Test success property when error exists."""
        metadata = ResponseMetadata(agent="test")
        error = ErrorInfo(message="Failed", type="Error")
        response = Response(content=None, metadata=metadata, error=error)

        assert response.success is False

    def test_response_unwrap_success(self):
        """Test unwrap() returns content when successful."""
        metadata = ResponseMetadata(agent="test")
        response = Response(content=42, metadata=metadata)

        result = response.unwrap()

        assert result == 42

    def test_response_unwrap_error_raises(self):
        """Test unwrap() raises ResponseError when failed."""
        metadata = ResponseMetadata(agent="test")
        error = ErrorInfo(message="Something failed", type="RuntimeError")
        response = Response(content=None, metadata=metadata, error=error)

        with pytest.raises(ResponseError) as exc_info:
            response.unwrap()

        assert "Something failed" in str(exc_info.value)
        assert exc_info.value.response == response

    def test_response_to_dict(self):
        """Test Response serialization."""
        tokens = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        metadata = ResponseMetadata(agent="test", tokens=tokens, duration=0.5)
        tool_call = ToolCall(
            tool_name="calc",
            arguments={"x": 1},
            result=2,
            duration=0.1,
            success=True,
        )

        response = Response(
            content="result",
            metadata=metadata,
            tool_calls=[tool_call],
            events=[],
            error=None,
        )

        result = response.to_dict()

        assert result["content"] == "result"
        assert result["metadata"]["agent"] == "test"
        assert result["metadata"]["tokens"]["total_tokens"] == 15
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["tool_name"] == "calc"
        assert result["events"] == []
        assert result["error"] is None
        assert result["success"] is True

    def test_response_to_dict_with_error(self):
        """Test Response serialization with error."""
        metadata = ResponseMetadata(agent="test")
        error = ErrorInfo(message="Failed", type="Error")
        response = Response(content=None, metadata=metadata, error=error)

        result = response.to_dict()

        assert result["error"]["message"] == "Failed"
        assert result["error"]["type"] == "Error"
        assert result["success"] is False

    def test_response_to_event(self):
        """Test converting Response to Event."""
        tokens = TokenUsage(total_tokens=100)
        metadata = ResponseMetadata(
            agent="agent1", tokens=tokens, correlation_id="corr-123"
        )
        response = Response(content="done", metadata=metadata)

        event = response.to_event(name="task.done", source="agent1")

        assert event.name == "task.done"
        assert event.source == "agent1"
        assert event.data["content"] == "done"
        assert event.data["success"] is True
        assert event.correlation_id == "corr-123"
        assert event.metadata["agent"] == "agent1"

    def test_response_generic_type_preservation(self):
        """Test that Response preserves generic type."""
        metadata = ResponseMetadata(agent="test")

        # String response
        str_response: Response[str] = Response(content="text", metadata=metadata)
        assert isinstance(str_response.content, str)

        # Int response
        int_response: Response[int] = Response(content=42, metadata=metadata)
        assert isinstance(int_response.content, int)

        # Dict response
        dict_response: Response[dict] = Response(content={"key": "value"}, metadata=metadata)
        assert isinstance(dict_response.content, dict)


class TestResponseError:
    """Tests for ResponseError exception."""

    def test_response_error_creation(self):
        """Test creating ResponseError."""
        metadata = ResponseMetadata(agent="test")
        error_info = ErrorInfo(message="Test error", type="TestError")
        response = Response(content=None, metadata=metadata, error=error_info)

        error = ResponseError("Test error", response=response)

        assert str(error) == "Test error"
        assert error.response == response

    def test_response_error_preserves_response(self):
        """Test that ResponseError preserves original Response."""
        metadata = ResponseMetadata(agent="agent1", duration=1.5)
        error_info = ErrorInfo(message="Failed", type="ValueError")
        response = Response(content=None, metadata=metadata, error=error_info)

        try:
            response.unwrap()
        except ResponseError as e:
            # Can access original response
            assert e.response.metadata.agent == "agent1"
            assert e.response.metadata.duration == 1.5
            assert e.response.error.type == "ValueError"
