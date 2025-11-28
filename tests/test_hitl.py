"""
Tests for Human-in-the-Loop (HITL) functionality.
"""

import pytest
from datetime import datetime

from agenticflow.agent.hitl import (
    InterruptReason,
    DecisionType,
    PendingAction,
    HumanDecision,
    InterruptedState,
    InterruptedException,
    AbortedException,
    DecisionRequiredException,
    GuidanceResult,
    HumanResponse,
    should_interrupt,
)


# =============================================================================
# Test should_interrupt function
# =============================================================================

class TestShouldInterrupt:
    """Tests for should_interrupt function."""
    
    def test_should_interrupt_none_config(self):
        """None config should never interrupt."""
        assert should_interrupt("any_tool", {}, None) is False
    
    def test_should_interrupt_empty_config(self):
        """Empty config should never interrupt."""
        assert should_interrupt("any_tool", {}, {}) is False
    
    def test_should_interrupt_bool_true(self):
        """True rule should always interrupt."""
        rules = {"delete_file": True}
        assert should_interrupt("delete_file", {}, rules) is True
    
    def test_should_interrupt_bool_false(self):
        """False rule should never interrupt."""
        rules = {"read_file": False}
        assert should_interrupt("read_file", {}, rules) is False
    
    def test_should_interrupt_callable_true(self):
        """Callable returning True should interrupt."""
        rules = {"write_file": lambda name, args: True}
        assert should_interrupt("write_file", {}, rules) is True
    
    def test_should_interrupt_callable_false(self):
        """Callable returning False should not interrupt."""
        rules = {"write_file": lambda name, args: False}
        assert should_interrupt("write_file", {}, rules) is False
    
    def test_should_interrupt_callable_with_args(self):
        """Callable should receive tool name and args."""
        rules = {
            "write_file": lambda name, args: args.get("path", "").startswith("/important/")
        }
        assert should_interrupt("write_file", {"path": "/tmp/test"}, rules) is False
        assert should_interrupt("write_file", {"path": "/important/data.txt"}, rules) is True
    
    def test_should_interrupt_wildcard_bool(self):
        """Wildcard rule should apply to unlisted tools."""
        rules = {"*": True}
        assert should_interrupt("any_tool", {}, rules) is True
        assert should_interrupt("another_tool", {}, rules) is True
    
    def test_should_interrupt_wildcard_callable(self):
        """Wildcard callable should receive correct args."""
        rules = {"*": lambda name, args: name.startswith("dangerous_")}
        assert should_interrupt("dangerous_delete", {}, rules) is True
        assert should_interrupt("safe_read", {}, rules) is False
    
    def test_should_interrupt_specific_overrides_wildcard(self):
        """Specific rule should override wildcard."""
        rules = {
            "*": True,  # Default: require approval
            "read_file": False,  # Exception: auto-approve reads
        }
        assert should_interrupt("read_file", {}, rules) is False
        assert should_interrupt("delete_file", {}, rules) is True
    
    def test_should_interrupt_unknown_tool_no_wildcard(self):
        """Unknown tool without wildcard should not interrupt."""
        rules = {"delete_file": True}
        assert should_interrupt("unknown_tool", {}, rules) is False


# =============================================================================
# Test PendingAction
# =============================================================================

class TestPendingAction:
    """Tests for PendingAction dataclass."""
    
    def test_pending_action_creation(self):
        """Test basic PendingAction creation."""
        action = PendingAction(
            action_id="test-123",
            tool_name="delete_file",
            args={"path": "/test.txt"},
            agent_name="TestAgent",
        )
        
        assert action.action_id == "test-123"
        assert action.tool_name == "delete_file"
        assert action.args == {"path": "/test.txt"}
        assert action.agent_name == "TestAgent"
        assert action.reason == InterruptReason.TOOL_APPROVAL
    
    def test_pending_action_describe(self):
        """Test describe method."""
        action = PendingAction(
            action_id="test-123",
            tool_name="send_email",
            args={"to": "user@example.com", "subject": "Hello"},
            agent_name="TestAgent",
        )
        
        description = action.describe()
        assert "send_email" in description
        assert "to=" in description
        assert "user@example.com" in description
    
    def test_pending_action_serialization(self):
        """Test to_dict and from_dict."""
        original = PendingAction(
            action_id="test-123",
            tool_name="write_file",
            args={"path": "/test.txt", "content": "hello"},
            agent_name="TestAgent",
            reason=InterruptReason.TOOL_APPROVAL,
            context={"correlation_id": "corr-1"},
            metadata={"priority": "high"},
        )
        
        data = original.to_dict()
        restored = PendingAction.from_dict(data)
        
        assert restored.action_id == original.action_id
        assert restored.tool_name == original.tool_name
        assert restored.args == original.args
        assert restored.agent_name == original.agent_name
        assert restored.reason == original.reason
        assert restored.context == original.context
        assert restored.metadata == original.metadata


# =============================================================================
# Test HumanDecision
# =============================================================================

class TestHumanDecision:
    """Tests for HumanDecision dataclass."""
    
    def test_human_decision_approve(self):
        """Test approve factory method."""
        decision = HumanDecision.approve("action-123", feedback="Looks good!")
        
        assert decision.action_id == "action-123"
        assert decision.decision == DecisionType.APPROVE
        assert decision.feedback == "Looks good!"
        assert decision.modified_args is None
    
    def test_human_decision_reject(self):
        """Test reject factory method."""
        decision = HumanDecision.reject("action-123", feedback="Too dangerous")
        
        assert decision.decision == DecisionType.REJECT
        assert decision.feedback == "Too dangerous"
    
    def test_human_decision_edit(self):
        """Test edit factory method."""
        new_args = {"path": "/safe/location.txt"}
        decision = HumanDecision.edit("action-123", new_args, feedback="Changed path")
        
        assert decision.decision == DecisionType.EDIT
        assert decision.modified_args == new_args
        assert decision.feedback == "Changed path"
    
    def test_human_decision_skip(self):
        """Test skip factory method."""
        decision = HumanDecision.skip("action-123")
        
        assert decision.decision == DecisionType.SKIP
    
    def test_human_decision_abort(self):
        """Test abort factory method."""
        decision = HumanDecision.abort("action-123", feedback="Stop everything")
        
        assert decision.decision == DecisionType.ABORT
        assert decision.feedback == "Stop everything"
    
    def test_human_decision_guide(self):
        """Test guide factory method."""
        decision = HumanDecision.guide(
            "action-123",
            guidance="Archive the file first, then delete it",
            feedback="Be more careful"
        )
        
        assert decision.decision == DecisionType.GUIDE
        assert decision.guidance == "Archive the file first, then delete it"
        assert decision.feedback == "Be more careful"
    
    def test_human_decision_respond(self):
        """Test respond factory method."""
        decision = HumanDecision.respond(
            "action-123",
            response="Q4-Sales-Report",
            feedback="Use this naming convention"
        )
        
        assert decision.decision == DecisionType.RESPOND
        assert decision.response == "Q4-Sales-Report"
        assert decision.feedback == "Use this naming convention"
    
    def test_human_decision_respond_with_dict(self):
        """Test respond with complex response."""
        decision = HumanDecision.respond(
            "action-123",
            response={"name": "report", "format": "pdf", "include_charts": True}
        )
        
        assert decision.decision == DecisionType.RESPOND
        assert decision.response == {"name": "report", "format": "pdf", "include_charts": True}
    
    def test_human_decision_serialization(self):
        """Test to_dict and from_dict."""
        original = HumanDecision.edit(
            "action-123",
            modified_args={"path": "/new/path.txt"},
            feedback="Modified for safety",
        )
        
        data = original.to_dict()
        restored = HumanDecision.from_dict(data)
        
        assert restored.action_id == original.action_id
        assert restored.decision == original.decision
        assert restored.modified_args == original.modified_args
        assert restored.feedback == original.feedback


# =============================================================================
# Test InterruptedState
# =============================================================================

class TestInterruptedState:
    """Tests for InterruptedState dataclass."""
    
    def test_interrupted_state_creation(self):
        """Test basic InterruptedState creation."""
        pending = PendingAction(
            action_id="test-123",
            tool_name="delete_file",
            args={"path": "/test.txt"},
            agent_name="TestAgent",
        )
        
        state = InterruptedState(
            thread_id="thread-1",
            pending_actions=[pending],
            agent_state={"status": "waiting"},
        )
        
        assert state.thread_id == "thread-1"
        assert len(state.pending_actions) == 1
        assert state.is_interrupted is True
    
    def test_interrupted_state_not_interrupted(self):
        """Test is_interrupted when no pending actions."""
        state = InterruptedState(
            thread_id="thread-1",
            pending_actions=[],
        )
        
        assert state.is_interrupted is False
    
    def test_interrupted_state_serialization(self):
        """Test to_dict and from_dict."""
        pending = PendingAction(
            action_id="test-123",
            tool_name="delete_file",
            args={"path": "/test.txt"},
            agent_name="TestAgent",
        )
        
        original = InterruptedState(
            thread_id="thread-1",
            pending_actions=[pending],
            agent_state={"status": "waiting"},
            conversation_history=[{"role": "user", "content": "hello"}],
            interrupt_reason=InterruptReason.TOOL_APPROVAL,
            metadata={"flow": "test-flow"},
        )
        
        data = original.to_dict()
        restored = InterruptedState.from_dict(data)
        
        assert restored.thread_id == original.thread_id
        assert len(restored.pending_actions) == len(original.pending_actions)
        assert restored.pending_actions[0].action_id == pending.action_id
        assert restored.agent_state == original.agent_state
        assert restored.conversation_history == original.conversation_history
        assert restored.interrupt_reason == original.interrupt_reason
        assert restored.metadata == original.metadata


# =============================================================================
# Test GuidanceResult and HumanResponse
# =============================================================================

class TestGuidanceResult:
    """Tests for GuidanceResult dataclass."""
    
    def test_guidance_result_creation(self):
        """Test basic GuidanceResult creation."""
        pending = PendingAction(
            action_id="test-123",
            tool_name="delete_file",
            args={"path": "/test.txt"},
            agent_name="TestAgent",
        )
        
        result = GuidanceResult(
            action_id="test-123",
            guidance="Archive the file first before deleting",
            original_action=pending,
            feedback="Be careful with deletions",
            should_retry=True,
        )
        
        assert result.action_id == "test-123"
        assert result.guidance == "Archive the file first before deleting"
        assert result.original_action is pending
        assert result.should_retry is True
    
    def test_guidance_result_to_message(self):
        """Test to_message method."""
        pending = PendingAction(
            action_id="test-123",
            tool_name="delete_file",
            args={"path": "/test.txt"},
            agent_name="TestAgent",
        )
        
        result = GuidanceResult(
            action_id="test-123",
            guidance="Archive first, then delete",
            original_action=pending,
            feedback="Safety first",
        )
        
        message = result.to_message()
        assert "delete_file" in message
        assert "Archive first, then delete" in message
        assert "Safety first" in message
    
    def test_guidance_result_serialization(self):
        """Test to_dict method."""
        pending = PendingAction(
            action_id="test-123",
            tool_name="delete_file",
            args={"path": "/test.txt"},
            agent_name="TestAgent",
        )
        
        result = GuidanceResult(
            action_id="test-123",
            guidance="Archive first",
            original_action=pending,
        )
        
        data = result.to_dict()
        assert data["action_id"] == "test-123"
        assert data["guidance"] == "Archive first"
        assert data["original_action"]["tool_name"] == "delete_file"


class TestHumanResponse:
    """Tests for HumanResponse dataclass."""
    
    def test_human_response_string(self):
        """Test HumanResponse with string value."""
        pending = PendingAction(
            action_id="test-123",
            tool_name="ask_name",
            args={},
            agent_name="TestAgent",
        )
        
        response = HumanResponse(
            action_id="test-123",
            response="Alice",
            original_action=pending,
        )
        
        assert response.response == "Alice"
    
    def test_human_response_complex(self):
        """Test HumanResponse with complex value."""
        pending = PendingAction(
            action_id="test-123",
            tool_name="get_config",
            args={},
            agent_name="TestAgent",
        )
        
        response = HumanResponse(
            action_id="test-123",
            response={"name": "report", "format": "pdf"},
            original_action=pending,
        )
        
        assert response.response == {"name": "report", "format": "pdf"}
    
    def test_human_response_serialization(self):
        """Test to_dict method."""
        pending = PendingAction(
            action_id="test-123",
            tool_name="ask_name",
            args={},
            agent_name="TestAgent",
        )
        
        response = HumanResponse(
            action_id="test-123",
            response="Alice",
            original_action=pending,
            feedback="My name is Alice",
        )
        
        data = response.to_dict()
        assert data["response"] == "Alice"
        assert data["feedback"] == "My name is Alice"


# =============================================================================
# Test Exceptions
# =============================================================================

class TestExceptions:
    """Tests for HITL exceptions."""
    
    def test_interrupted_exception(self):
        """Test InterruptedException creation."""
        pending = PendingAction(
            action_id="test-123",
            tool_name="delete_file",
            args={"path": "/test.txt"},
            agent_name="TestAgent",
        )
        state = InterruptedState(
            thread_id="thread-1",
            pending_actions=[pending],
        )
        
        exc = InterruptedException(state, "Custom message")
        
        assert exc.state is state
        assert "Custom message" in str(exc)
    
    def test_decision_required_exception(self):
        """Test DecisionRequiredException creation."""
        pending = PendingAction(
            action_id="test-123",
            tool_name="delete_file",
            args={"path": "/test.txt"},
            agent_name="TestAgent",
        )
        
        exc = DecisionRequiredException(pending)
        
        assert exc.pending_action is pending
        assert "delete_file" in str(exc)
    
    def test_aborted_exception(self):
        """Test AbortedException creation."""
        decision = HumanDecision.abort("action-123", feedback="Emergency stop")
        
        exc = AbortedException(decision)
        
        assert exc.decision is decision
        assert "Emergency stop" in str(exc)


# =============================================================================
# Test Agent HITL Integration
# =============================================================================

class TestAgentHITL:
    """Tests for Agent HITL integration."""
    
    @pytest.fixture
    def simple_tool(self):
        """Create a simple tool for testing."""
        from agenticflow.tools.base import tool
        
        @tool
        def delete_file(path: str) -> str:
            """Delete a file."""
            return f"Deleted {path}"
        
        return delete_file
    
    @pytest.fixture
    def safe_tool(self):
        """Create a safe tool for testing."""
        from agenticflow.tools.base import tool
        
        @tool
        def read_file(path: str) -> str:
            """Read a file."""
            return f"Content of {path}"
        
        return read_file
    
    @pytest.mark.asyncio
    async def test_agent_interrupt_on_dangerous_tool(self, simple_tool):
        """Test that agent interrupts on configured tools."""
        from agenticflow import Agent
        
        agent = Agent(
            name="TestAgent",
            model=None,
            tools=[simple_tool],
            interrupt_on={"delete_file": True},
        )
        
        with pytest.raises(InterruptedException) as exc_info:
            await agent.act("delete_file", {"path": "/test.txt"})
        
        state = exc_info.value.state
        assert len(state.pending_actions) == 1
        assert state.pending_actions[0].tool_name == "delete_file"
    
    @pytest.mark.asyncio
    async def test_agent_no_interrupt_on_safe_tool(self, safe_tool):
        """Test that agent doesn't interrupt on auto-approved tools."""
        from agenticflow import Agent
        
        agent = Agent(
            name="TestAgent",
            model=None,
            tools=[safe_tool],
            interrupt_on={"read_file": False},
        )
        
        result = await agent.act("read_file", {"path": "/test.txt"})
        assert "Content of" in result
    
    @pytest.mark.asyncio
    async def test_agent_resume_approve(self, simple_tool):
        """Test resuming with approval."""
        from agenticflow import Agent
        
        agent = Agent(
            name="TestAgent",
            model=None,
            tools=[simple_tool],
            interrupt_on={"delete_file": True},
        )
        
        # Get interrupted
        with pytest.raises(InterruptedException) as exc_info:
            await agent.act("delete_file", {"path": "/test.txt"})
        
        pending = exc_info.value.state.pending_actions[0]
        
        # Resume with approval
        decision = HumanDecision.approve(pending.action_id)
        result = await agent.resume_action(decision)
        
        assert "Deleted" in result
        assert agent.is_interrupted is False
    
    @pytest.mark.asyncio
    async def test_agent_resume_reject(self, simple_tool):
        """Test resuming with rejection."""
        from agenticflow import Agent
        
        agent = Agent(
            name="TestAgent",
            model=None,
            tools=[simple_tool],
            interrupt_on={"delete_file": True},
        )
        
        # Get interrupted
        with pytest.raises(InterruptedException) as exc_info:
            await agent.act("delete_file", {"path": "/test.txt"})
        
        pending = exc_info.value.state.pending_actions[0]
        
        # Resume with rejection
        decision = HumanDecision.reject(pending.action_id)
        result = await agent.resume_action(decision)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_agent_resume_edit(self, simple_tool):
        """Test resuming with edit."""
        from agenticflow import Agent
        
        agent = Agent(
            name="TestAgent",
            model=None,
            tools=[simple_tool],
            interrupt_on={"delete_file": True},
        )
        
        # Get interrupted
        with pytest.raises(InterruptedException) as exc_info:
            await agent.act("delete_file", {"path": "/dangerous.txt"})
        
        pending = exc_info.value.state.pending_actions[0]
        
        # Resume with edited args
        new_args = {"path": "/safe/backup.txt"}
        decision = HumanDecision.edit(pending.action_id, new_args)
        result = await agent.resume_action(decision)
        
        assert "/safe/backup.txt" in result
    
    @pytest.mark.asyncio
    async def test_agent_resume_abort(self, simple_tool):
        """Test resuming with abort."""
        from agenticflow import Agent
        
        agent = Agent(
            name="TestAgent",
            model=None,
            tools=[simple_tool],
            interrupt_on={"delete_file": True},
        )
        
        # Get interrupted
        with pytest.raises(InterruptedException) as exc_info:
            await agent.act("delete_file", {"path": "/test.txt"})
        
        pending = exc_info.value.state.pending_actions[0]
        
        # Resume with abort
        decision = HumanDecision.abort(pending.action_id, feedback="Too risky")
        
        with pytest.raises(AbortedException) as abort_info:
            await agent.resume_action(decision)
        
        assert "Too risky" in abort_info.value.decision.feedback
    
    @pytest.mark.asyncio
    async def test_agent_pending_actions_property(self, simple_tool):
        """Test pending_actions property."""
        from agenticflow import Agent
        
        agent = Agent(
            name="TestAgent",
            model=None,
            tools=[simple_tool],
            interrupt_on={"delete_file": True},
        )
        
        assert agent.pending_actions == []
        assert agent.is_interrupted is False
        
        # Get interrupted
        with pytest.raises(InterruptedException):
            await agent.act("delete_file", {"path": "/test.txt"})
        
        assert len(agent.pending_actions) == 1
        assert agent.is_interrupted is True
    
    @pytest.mark.asyncio
    async def test_agent_clear_pending_actions(self, simple_tool):
        """Test clear_pending_actions method."""
        from agenticflow import Agent
        
        agent = Agent(
            name="TestAgent",
            model=None,
            tools=[simple_tool],
            interrupt_on={"delete_file": True},
        )
        
        # Get interrupted
        with pytest.raises(InterruptedException):
            await agent.act("delete_file", {"path": "/test.txt"})
        
        assert agent.is_interrupted is True
        
        agent.clear_pending_actions()
        
        assert agent.is_interrupted is False
        assert agent.pending_actions == []
    
    @pytest.mark.asyncio
    async def test_agent_resume_guide(self, simple_tool):
        """Test resuming with guidance."""
        from agenticflow import Agent
        
        agent = Agent(
            name="TestAgent",
            model=None,
            tools=[simple_tool],
            interrupt_on={"delete_file": True},
        )
        
        # Get interrupted
        with pytest.raises(InterruptedException) as exc_info:
            await agent.act("delete_file", {"path": "/test.txt"})
        
        pending = exc_info.value.state.pending_actions[0]
        
        # Resume with guidance
        decision = HumanDecision.guide(
            pending.action_id,
            guidance="Archive the file first, then delete it",
            feedback="Be more careful"
        )
        result = await agent.resume_action(decision)
        
        # Should return GuidanceResult
        assert isinstance(result, GuidanceResult)
        assert result.guidance == "Archive the file first, then delete it"
        assert result.original_action.tool_name == "delete_file"
        assert result.should_retry is True
        assert agent.is_interrupted is False
    
    @pytest.mark.asyncio
    async def test_agent_resume_respond(self, simple_tool):
        """Test resuming with direct response."""
        from agenticflow import Agent
        
        agent = Agent(
            name="TestAgent",
            model=None,
            tools=[simple_tool],
            interrupt_on={"delete_file": True},
        )
        
        # Get interrupted (simulating a "human input needed" scenario)
        with pytest.raises(InterruptedException) as exc_info:
            await agent.act("delete_file", {"path": "/test.txt"})
        
        pending = exc_info.value.state.pending_actions[0]
        
        # Resume with direct response
        decision = HumanDecision.respond(
            pending.action_id,
            response={"confirmed": True, "backup_path": "/backup/test.txt"}
        )
        result = await agent.resume_action(decision)
        
        # Should return HumanResponse
        assert isinstance(result, HumanResponse)
        assert result.response == {"confirmed": True, "backup_path": "/backup/test.txt"}
        assert result.original_action.tool_name == "delete_file"
        assert agent.is_interrupted is False
    
    @pytest.mark.asyncio
    async def test_guidance_result_has_correct_message(self, simple_tool):
        """Test that GuidanceResult.to_message() formats correctly."""
        from agenticflow import Agent
        
        agent = Agent(
            name="TestAgent",
            model=None,
            tools=[simple_tool],
            interrupt_on={"delete_file": True},
        )
        
        with pytest.raises(InterruptedException) as exc_info:
            await agent.act("delete_file", {"path": "/test.txt"})
        
        pending = exc_info.value.state.pending_actions[0]
        
        decision = HumanDecision.guide(
            pending.action_id,
            guidance="Check dependencies first",
            feedback="Safety matters"
        )
        result = await agent.resume_action(decision)
        
        message = result.to_message()
        assert "delete_file" in message
        assert "Check dependencies first" in message
        assert "Safety matters" in message


# =============================================================================
# Test AgentConfig with interrupt_on
# =============================================================================

class TestAgentConfigHITL:
    """Tests for AgentConfig interrupt_on field."""
    
    def test_config_with_interrupt_on(self):
        """Test creating config with interrupt_on."""
        from agenticflow import AgentConfig
        
        config = AgentConfig(
            name="TestAgent",
            interrupt_on={"delete_file": True, "read_file": False},
        )
        
        assert config.interrupt_on == {"delete_file": True, "read_file": False}
    
    def test_config_with_interrupt_on_builder(self):
        """Test with_interrupt_on builder method."""
        from agenticflow import AgentConfig
        
        config = AgentConfig(name="TestAgent").with_interrupt_on({
            "delete_file": True,
        })
        
        assert config.interrupt_on == {"delete_file": True}
    
    def test_config_interrupt_on_serialization(self):
        """Test that only bool rules are serialized."""
        from agenticflow import AgentConfig
        
        # Callable rules can't be serialized
        config = AgentConfig(
            name="TestAgent",
            interrupt_on={
                "delete_file": True,
                "read_file": False,
                "write_file": lambda n, a: True,  # Not serialized
            },
        )
        
        data = config.to_dict()
        
        # Only bool rules in serialized form
        assert data["interrupt_on"] == {"delete_file": True, "read_file": False}
    
    def test_config_builder_preserves_interrupt_on(self):
        """Test that builder methods preserve interrupt_on."""
        from agenticflow import AgentConfig
        
        config = AgentConfig(
            name="TestAgent",
            interrupt_on={"delete_file": True},
        )
        
        new_config = config.with_tools(["search"])
        
        assert new_config.interrupt_on == {"delete_file": True}
