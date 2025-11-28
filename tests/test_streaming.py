"""
Tests for streaming functionality.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from agenticflow.agent.streaming import (
    StreamChunk,
    StreamEvent,
    StreamEventType,
    StreamConfig,
    ToolCallChunk,
    PrintStreamCallback,
    CollectorStreamCallback,
    chunk_from_langchain,
    extract_tool_calls,
    collect_stream,
    print_stream,
)


# =============================================================================
# StreamChunk Tests
# =============================================================================

class TestStreamChunk:
    """Tests for StreamChunk dataclass."""
    
    def test_stream_chunk_basic(self):
        """Test basic StreamChunk creation."""
        chunk = StreamChunk(content="Hello")
        assert chunk.content == "Hello"
        assert chunk.finish_reason is None
        assert chunk.model is None
        assert chunk.index == 0
        assert chunk.token_count is None
    
    def test_stream_chunk_with_metadata(self):
        """Test StreamChunk with all metadata."""
        chunk = StreamChunk(
            content="World",
            finish_reason="stop",
            model="gpt-4",
            index=5,
            token_count=1,
        )
        assert chunk.content == "World"
        assert chunk.finish_reason == "stop"
        assert chunk.model == "gpt-4"
        assert chunk.index == 5
        assert chunk.token_count == 1
    
    def test_stream_chunk_str(self):
        """Test StreamChunk string representation."""
        chunk = StreamChunk(content="test")
        assert str(chunk) == "test"
    
    def test_stream_chunk_repr(self):
        """Test StreamChunk repr."""
        chunk = StreamChunk(content="test")
        assert repr(chunk) == "StreamChunk('test')"
    
    def test_stream_chunk_addition(self):
        """Test StreamChunk concatenation."""
        chunk1 = StreamChunk(content="Hello ", index=0, token_count=1)
        chunk2 = StreamChunk(content="World", index=1, finish_reason="stop", token_count=1)
        
        combined = chunk1 + chunk2
        assert combined.content == "Hello World"
        assert combined.finish_reason == "stop"
        assert combined.index == 1
        assert combined.token_count == 2


# =============================================================================
# StreamEvent Tests
# =============================================================================

class TestStreamEvent:
    """Tests for StreamEvent dataclass."""
    
    def test_stream_event_basic(self):
        """Test basic StreamEvent creation."""
        event = StreamEvent(type=StreamEventType.TOKEN, content="Hello")
        assert event.type == StreamEventType.TOKEN
        assert event.content == "Hello"
        assert event.accumulated == ""
    
    def test_stream_event_is_final(self):
        """Test is_final property."""
        token_event = StreamEvent(type=StreamEventType.TOKEN)
        end_event = StreamEvent(type=StreamEventType.STREAM_END)
        error_event = StreamEvent(type=StreamEventType.ERROR, error="fail")
        
        assert not token_event.is_final
        assert end_event.is_final
        assert error_event.is_final
    
    def test_stream_event_has_content(self):
        """Test has_content property."""
        with_content = StreamEvent(type=StreamEventType.TOKEN, content="hi")
        without_content = StreamEvent(type=StreamEventType.STREAM_START)
        
        assert with_content.has_content
        assert not without_content.has_content
    
    def test_stream_event_is_tool_related(self):
        """Test is_tool_related property."""
        token = StreamEvent(type=StreamEventType.TOKEN)
        tool_start = StreamEvent(type=StreamEventType.TOOL_CALL_START)
        tool_args = StreamEvent(type=StreamEventType.TOOL_CALL_ARGS)
        tool_end = StreamEvent(type=StreamEventType.TOOL_CALL_END)
        tool_result = StreamEvent(type=StreamEventType.TOOL_RESULT)
        
        assert not token.is_tool_related
        assert tool_start.is_tool_related
        assert tool_args.is_tool_related
        assert tool_end.is_tool_related
        assert tool_result.is_tool_related
    
    def test_stream_event_with_tool_call(self):
        """Test StreamEvent with tool call info."""
        tc = ToolCallChunk(id="123", name="search", args='{"q": "test"}')
        event = StreamEvent(
            type=StreamEventType.TOOL_CALL_START,
            tool_call=tc,
            tool_name="search",
        )
        assert event.tool_call == tc
        assert event.tool_name == "search"


# =============================================================================
# StreamConfig Tests
# =============================================================================

class TestStreamConfig:
    """Tests for StreamConfig."""
    
    def test_stream_config_defaults(self):
        """Test default config values."""
        config = StreamConfig()
        assert config.emit_start_event is True
        assert config.emit_end_event is True
        assert config.include_tool_events is True
        assert config.include_accumulated is True
    
    def test_stream_config_minimal(self):
        """Test minimal config factory."""
        config = StreamConfig.minimal()
        assert config.emit_start_event is False
        assert config.emit_end_event is False
        assert config.include_tool_events is False
        assert config.include_accumulated is False
    
    def test_stream_config_full(self):
        """Test full config factory."""
        config = StreamConfig.full()
        assert config.emit_start_event is True
        assert config.emit_end_event is True
        assert config.include_tool_events is True
        assert config.include_accumulated is True


# =============================================================================
# ToolCallChunk Tests
# =============================================================================

class TestToolCallChunk:
    """Tests for ToolCallChunk."""
    
    def test_tool_call_chunk_basic(self):
        """Test basic ToolCallChunk."""
        tc = ToolCallChunk(id="abc", name="search", args='{"q": "test"}')
        assert tc.id == "abc"
        assert tc.name == "search"
        assert tc.args == '{"q": "test"}'
    
    def test_tool_call_chunk_is_complete(self):
        """Test is_complete method."""
        incomplete = ToolCallChunk(id="abc", name="search", args="")
        complete = ToolCallChunk(id="abc", name="search", args='{"q": "test"}')
        
        assert not incomplete.is_complete()
        assert complete.is_complete()
    
    def test_tool_call_chunk_accumulation(self):
        """Test accumulating args."""
        tc = ToolCallChunk(id="abc", name="search", args="")
        tc.args += '{"q":'
        tc.args += ' "test"}'
        
        assert tc.args == '{"q": "test"}'
        assert tc.is_complete()


# =============================================================================
# Callback Tests
# =============================================================================

class TestPrintStreamCallback:
    """Tests for PrintStreamCallback."""
    
    def test_print_callback_on_token(self, capsys):
        """Test token printing."""
        callback = PrintStreamCallback(end="", flush=True)
        callback.on_token("Hello")
        callback.on_token(" World")
        
        captured = capsys.readouterr()
        assert captured.out == "Hello World"
    
    def test_print_callback_with_prefix_suffix(self, capsys):
        """Test prefix and suffix."""
        callback = PrintStreamCallback(prefix=">> ", suffix=" <<\n", end="")
        callback.on_stream_start({})
        callback.on_token("test")
        callback.on_stream_end("")
        
        captured = capsys.readouterr()
        assert captured.out == ">> test <<\n"
    
    def test_print_callback_tool_call(self, capsys):
        """Test tool call notification."""
        callback = PrintStreamCallback(show_tool_calls=True)
        callback.on_tool_call("search", {"q": "test"})
        
        captured = capsys.readouterr()
        assert "[Calling search...]" in captured.out
    
    def test_print_callback_hide_tool_calls(self, capsys):
        """Test hiding tool call notifications."""
        callback = PrintStreamCallback(show_tool_calls=False)
        callback.on_tool_call("search", {"q": "test"})
        
        captured = capsys.readouterr()
        assert captured.out == ""


class TestCollectorStreamCallback:
    """Tests for CollectorStreamCallback."""
    
    def test_collector_basic(self):
        """Test basic collection."""
        collector = CollectorStreamCallback()
        collector.on_token("Hello")
        collector.on_token(" ")
        collector.on_token("World")
        
        assert collector.get_full_response() == "Hello World"
    
    def test_collector_tool_calls(self):
        """Test tool call collection."""
        collector = CollectorStreamCallback()
        collector.on_tool_call("search", {"q": "test"})
        collector.on_tool_call("fetch", {"url": "http://example.com"})
        
        tool_calls = collector.get_tool_calls()
        assert len(tool_calls) == 2
        assert tool_calls[0] == ("search", {"q": "test"})
    
    def test_collector_errors(self):
        """Test error collection."""
        collector = CollectorStreamCallback()
        collector.on_error(ValueError("test error"))
        
        assert collector.has_errors()
    
    def test_collector_clear(self):
        """Test clearing collector."""
        collector = CollectorStreamCallback()
        collector.on_token("test")
        collector.on_tool_call("search", {})
        collector.on_error(ValueError("err"))
        
        collector.clear()
        
        assert collector.get_full_response() == ""
        assert collector.get_tool_calls() == []
        assert not collector.has_errors()


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestChunkFromLangchain:
    """Tests for chunk_from_langchain function (uses native AIMessage)."""
    
    def test_string_content(self):
        """Test extracting string content."""
        from agenticflow.core.messages import AIMessage
        
        ai_chunk = AIMessage(content="Hello")
        chunk = chunk_from_langchain(ai_chunk, index=0)
        
        assert chunk.content == "Hello"
        assert chunk.index == 0
        assert chunk.raw == ai_chunk
    
    def test_empty_content(self):
        """Test chunk with empty content."""
        from agenticflow.core.messages import AIMessage
        
        ai_chunk = AIMessage(content="")
        chunk = chunk_from_langchain(ai_chunk, index=1)
        
        assert chunk.content == ""
        assert chunk.index == 1


class TestExtractToolCalls:
    """Tests for extract_tool_calls function."""
    
    def test_no_tool_calls(self):
        """Test chunk with no tool calls."""
        from agenticflow.core.messages import AIMessage
        
        msg = AIMessage(content="Hello")
        tool_calls = extract_tool_calls(msg)
        
        assert tool_calls == []
    
    def test_with_tool_calls(self):
        """Test extracting tool calls."""
        from agenticflow.core.messages import AIMessage
        
        msg = AIMessage(
            content="",
            tool_calls=[
                {"id": "123", "name": "search", "args": {"q": "test"}},
            ],
        )
        tool_calls = extract_tool_calls(msg)
        
        # May extract from both tool_call_chunks and tool_calls
        assert len(tool_calls) >= 1
        # At least one should have the correct data
        found = any(tc.name == "search" for tc in tool_calls)
        assert found, f"Expected 'search' tool call, got: {tool_calls}"


# =============================================================================
# Async Utility Tests
# =============================================================================

class TestCollectStream:
    """Tests for collect_stream function."""
    
    @pytest.mark.asyncio
    async def test_collect_stream(self):
        """Test collecting stream content."""
        async def mock_stream():
            yield StreamChunk(content="Hello")
            yield StreamChunk(content=" ")
            yield StreamChunk(content="World")
        
        content, tool_calls = await collect_stream(mock_stream())
        
        assert content == "Hello World"
        assert tool_calls == []


class TestPrintStream:
    """Tests for print_stream function."""
    
    @pytest.mark.asyncio
    async def test_print_stream(self, capsys):
        """Test printing and returning stream."""
        async def mock_stream():
            yield StreamChunk(content="Hello")
            yield StreamChunk(content=" ")
            yield StreamChunk(content="World")
        
        content = await print_stream(
            mock_stream(),
            prefix=">> ",
            suffix=" <<\n",
        )
        
        assert content == "Hello World"
        captured = capsys.readouterr()
        assert captured.out == ">> Hello World <<\n"


# =============================================================================
# StreamEventType Tests
# =============================================================================

class TestStreamEventType:
    """Tests for StreamEventType enum."""
    
    def test_all_event_types(self):
        """Test all event types exist."""
        assert StreamEventType.STREAM_START
        assert StreamEventType.TOKEN
        assert StreamEventType.TOOL_CALL_START
        assert StreamEventType.TOOL_CALL_ARGS
        assert StreamEventType.TOOL_CALL_END
        assert StreamEventType.TOOL_RESULT
        assert StreamEventType.STREAM_END
        assert StreamEventType.ERROR
    
    def test_event_type_values(self):
        """Test event type values are strings."""
        for event_type in StreamEventType:
            assert isinstance(event_type.value, str)


# =============================================================================
# Integration Tests
# =============================================================================

class TestStreamingIntegration:
    """Integration tests for streaming with Agent."""
    
    @pytest.mark.asyncio
    async def test_think_stream_integration(self):
        """Test think_stream with mock model."""
        from agenticflow import Agent
        from agenticflow.models.base import BaseChatModel
        from agenticflow.core.messages import AIMessage
        from dataclasses import dataclass, field
        from typing import Any, AsyncIterator
        
        @dataclass
        class MockModel(BaseChatModel):
            response: str = "Hi!"
            model: str = "mock"
            _tools: list = field(default_factory=list, repr=False)
            
            def _init_client(self) -> None:
                pass
            
            def invoke(self, messages: list[dict[str, Any]]) -> AIMessage:
                return AIMessage(content=self.response)
            
            async def ainvoke(self, messages: list[dict[str, Any]]) -> AIMessage:
                return AIMessage(content=self.response)
            
            def bind_tools(self, tools: list[Any], **kwargs) -> "MockModel":
                return self
            
            async def astream(
                self,
                messages: list[dict[str, Any]],
            ) -> AsyncIterator[AIMessage]:
                for char in self.response:
                    yield AIMessage(content=char)
        
        agent = Agent(name="TestAgent", model=MockModel(response="Hi!"))
        
        chunks = []
        async for chunk in agent.think_stream("Test prompt"):
            chunks.append(chunk.content)
        
        assert "".join(chunks) == "Hi!"
    
    @pytest.mark.asyncio
    async def test_stream_events_integration(self):
        """Test stream_events with mock model."""
        from agenticflow import Agent, StreamEventType
        from agenticflow.models.base import BaseChatModel
        from agenticflow.core.messages import AIMessage
        from dataclasses import dataclass, field
        from typing import Any, AsyncIterator
        
        @dataclass
        class MockModel(BaseChatModel):
            response: str = "Hi"
            model: str = "mock"
            _tools: list = field(default_factory=list, repr=False)
            
            def _init_client(self) -> None:
                pass
            
            def invoke(self, messages: list[dict[str, Any]]) -> AIMessage:
                return AIMessage(content=self.response)
            
            async def ainvoke(self, messages: list[dict[str, Any]]) -> AIMessage:
                return AIMessage(content=self.response)
            
            def bind_tools(self, tools: list[Any], **kwargs) -> "MockModel":
                return self
            
            async def astream(
                self,
                messages: list[dict[str, Any]],
            ) -> AsyncIterator[AIMessage]:
                for char in self.response:
                    yield AIMessage(content=char)
        
        agent = Agent(name="TestAgent", model=MockModel(response="Hi"))
        
        events = []
        async for event in agent.stream_events("Test"):
            events.append(event)
        
        # Should have: STREAM_START, TOKEN(s), STREAM_END
        event_types = [e.type for e in events]
        assert StreamEventType.STREAM_START in event_types
        assert StreamEventType.TOKEN in event_types
        assert StreamEventType.STREAM_END in event_types
        
        # First event should be start
        assert events[0].type == StreamEventType.STREAM_START
        
        # Last event should be end
        assert events[-1].type == StreamEventType.STREAM_END


class TestAgentStreamParameter:
    """Tests for stream parameter on Agent.chat() and Agent.think()."""
    
    @pytest.mark.asyncio
    async def test_chat_stream_parameter(self):
        """Test that chat(stream=True) returns an async iterator."""
        from agenticflow import Agent
        from agenticflow.models.base import BaseChatModel
        from agenticflow.core.messages import AIMessage
        from dataclasses import dataclass, field
        from typing import Any, AsyncIterator
        
        @dataclass
        class MockModel(BaseChatModel):
            response: str = "Hello from stream!"
            model: str = "mock"
            _tools: list = field(default_factory=list, repr=False)
            
            def _init_client(self) -> None:
                pass
            
            def invoke(self, messages: list[dict[str, Any]]) -> AIMessage:
                return AIMessage(content=self.response)
            
            async def ainvoke(self, messages: list[dict[str, Any]]) -> AIMessage:
                return AIMessage(content=self.response)
            
            def bind_tools(self, tools: list[Any], **kwargs) -> "MockModel":
                return self
            
            async def astream(
                self,
                messages: list[dict[str, Any]],
            ) -> AsyncIterator[AIMessage]:
                for char in self.response:
                    yield AIMessage(content=char)
        
        agent = Agent(name="TestAgent", model=MockModel())
        
        # Test stream=True returns async iterator
        result = agent.chat("Test", stream=True)
        assert hasattr(result, "__aiter__")
        
        chunks = []
        async for chunk in result:
            chunks.append(chunk.content)
        
        assert "".join(chunks) == "Hello from stream!"
    
    @pytest.mark.asyncio
    async def test_chat_no_stream_returns_coroutine(self):
        """Test that chat(stream=False) returns a coroutine."""
        from agenticflow import Agent
        from agenticflow.models.base import BaseChatModel
        from agenticflow.core.messages import AIMessage
        from dataclasses import dataclass, field
        from typing import Any
        import inspect
        
        @dataclass
        class MockModel(BaseChatModel):
            response: str = "Hello direct!"
            model: str = "mock"
            _tools: list = field(default_factory=list, repr=False)
            
            def _init_client(self) -> None:
                pass
            
            def invoke(self, messages: list[dict[str, Any]]) -> AIMessage:
                return AIMessage(content=self.response)
            
            async def ainvoke(self, messages: list[dict[str, Any]]) -> AIMessage:
                return AIMessage(content=self.response)
            
            def bind_tools(self, tools: list[Any], **kwargs) -> "MockModel":
                return self
        
        agent = Agent(name="TestAgent", model=MockModel())
        
        # Test stream=False returns coroutine
        result = agent.chat("Test", stream=False)
        assert inspect.iscoroutine(result)
        
        response = await result
        assert response == "Hello direct!"
    
    @pytest.mark.asyncio
    async def test_agent_default_stream_true(self):
        """Test Agent(stream=True) makes streaming the default."""
        from agenticflow import Agent
        from agenticflow.models.base import BaseChatModel
        from agenticflow.core.messages import AIMessage
        from dataclasses import dataclass, field
        from typing import Any, AsyncIterator
        
        @dataclass
        class MockModel(BaseChatModel):
            response: str = "Default stream!"
            model: str = "mock"
            _tools: list = field(default_factory=list, repr=False)
            
            def _init_client(self) -> None:
                pass
            
            def invoke(self, messages: list[dict[str, Any]]) -> AIMessage:
                return AIMessage(content=self.response)
            
            async def ainvoke(self, messages: list[dict[str, Any]]) -> AIMessage:
                return AIMessage(content=self.response)
            
            def bind_tools(self, tools: list[Any], **kwargs) -> "MockModel":
                return self
            
            async def astream(
                self,
                messages: list[dict[str, Any]],
            ) -> AsyncIterator[AIMessage]:
                for char in self.response:
                    yield AIMessage(content=char)
        
        # Create agent with stream=True as default
        agent = Agent(name="StreamAgent", model=MockModel(), stream=True)
        
        # chat() without stream param should use agent default (streaming)
        result = agent.chat("Test")
        assert hasattr(result, "__aiter__")
        
        chunks = []
        async for chunk in result:
            chunks.append(chunk.content)
        
        assert "".join(chunks) == "Default stream!"
    
    @pytest.mark.asyncio
    async def test_think_stream_parameter(self):
        """Test that think(stream=True) returns an async iterator."""
        from agenticflow import Agent
        from agenticflow.models.base import BaseChatModel
        from agenticflow.core.messages import AIMessage
        from dataclasses import dataclass, field
        from typing import Any, AsyncIterator
        
        @dataclass
        class MockModel(BaseChatModel):
            response: str = "Thinking..."
            model: str = "mock"
            _tools: list = field(default_factory=list, repr=False)
            
            def _init_client(self) -> None:
                pass
            
            def invoke(self, messages: list[dict[str, Any]]) -> AIMessage:
                return AIMessage(content=self.response)
            
            async def ainvoke(self, messages: list[dict[str, Any]]) -> AIMessage:
                return AIMessage(content=self.response)
            
            def bind_tools(self, tools: list[Any], **kwargs) -> "MockModel":
                return self
            
            async def astream(
                self,
                messages: list[dict[str, Any]],
            ) -> AsyncIterator[AIMessage]:
                for char in self.response:
                    yield AIMessage(content=char)
        
        agent = Agent(name="TestAgent", model=MockModel())
        
        # Test stream=True returns async iterator
        result = agent.think("Test", stream=True)
        assert hasattr(result, "__aiter__")
        
        chunks = []
        async for chunk in result:
            chunks.append(chunk.content)
        
        assert "".join(chunks) == "Thinking..."
    
    def test_stream_config_default(self):
        """Test that AgentConfig.stream defaults to False."""
        from agenticflow.agent.config import AgentConfig
        
        config = AgentConfig(name="Test")
        assert config.stream is False
    
    def test_stream_config_true(self):
        """Test that AgentConfig.stream can be set to True."""
        from agenticflow.agent.config import AgentConfig
        
        config = AgentConfig(name="Test", stream=True)
        assert config.stream is True