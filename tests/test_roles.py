"""Tests for agent roles and behaviors."""

import pytest

from agenticflow.agent import (
    Agent,
    RoleBehavior,
    DelegationCommand,
    get_role_prompt,
    get_role_behavior,
    parse_delegation,
    has_final_answer,
    extract_final_answer,
    ROLE_PROMPTS,
    ROLE_BEHAVIORS,
)
from agenticflow.core.enums import AgentRole


class TestRolePrompts:
    """Test role-specific prompts."""
    
    def test_all_roles_have_prompts(self):
        """Every role should have a default prompt."""
        for role in AgentRole:
            prompt = get_role_prompt(role)
            assert prompt, f"Role {role} should have a prompt"
            assert len(prompt) > 50, f"Role {role} prompt too short"
    
    def test_supervisor_prompt_mentions_delegation(self):
        """Supervisor prompt should mention delegation."""
        prompt = get_role_prompt(AgentRole.SUPERVISOR)
        assert "delegate" in prompt.lower()
        assert "DELEGATE TO" in prompt or "delegate" in prompt.lower()
    
    def test_worker_prompt_mentions_tools(self):
        """Worker prompt should mention tools."""
        prompt = get_role_prompt(AgentRole.WORKER)
        assert "tool" in prompt.lower()
    
    def test_critic_prompt_mentions_review(self):
        """Critic prompt should mention review/feedback."""
        prompt = get_role_prompt(AgentRole.CRITIC)
        assert "review" in prompt.lower() or "feedback" in prompt.lower()


class TestRoleBehaviors:
    """Test role-specific behaviors."""
    
    def test_all_roles_have_behaviors(self):
        """Every role should have defined behaviors."""
        for role in AgentRole:
            behavior = get_role_behavior(role)
            assert isinstance(behavior, RoleBehavior)
    
    def test_supervisor_can_delegate(self):
        """Supervisors should be able to delegate."""
        behavior = get_role_behavior(AgentRole.SUPERVISOR)
        assert behavior.can_delegate is True
    
    def test_worker_cannot_delegate(self):
        """Workers should not be able to delegate."""
        behavior = get_role_behavior(AgentRole.WORKER)
        assert behavior.can_delegate is False
    
    def test_worker_can_use_tools(self):
        """Workers should be able to use tools."""
        behavior = get_role_behavior(AgentRole.WORKER)
        assert behavior.can_use_tools is True
    
    def test_supervisor_cannot_use_tools(self):
        """Supervisors delegate, they don't execute."""
        behavior = get_role_behavior(AgentRole.SUPERVISOR)
        assert behavior.can_use_tools is False
    
    def test_coordinator_can_route(self):
        """Coordinators should be able to route messages."""
        behavior = get_role_behavior(AgentRole.COORDINATOR)
        assert behavior.can_route_messages is True
    
    def test_coordinator_cannot_finish(self):
        """Coordinators route, they don't produce final answers."""
        behavior = get_role_behavior(AgentRole.COORDINATOR)
        assert behavior.can_finish is False


class TestDelegationParsing:
    """Test parsing of delegation commands from LLM output."""
    
    def test_parse_delegate_command(self):
        """Test parsing DELEGATE TO command."""
        text = "Let me analyze this. DELEGATE TO [researcher]: Find information about Python."
        cmd = parse_delegation(text)
        assert cmd is not None
        assert cmd.action == "delegate"
        assert cmd.target == "researcher"
        assert "Python" in cmd.task
    
    def test_parse_delegate_without_brackets(self):
        """Test parsing DELEGATE TO without brackets."""
        text = "DELEGATE TO analyst: Review the data"
        cmd = parse_delegation(text)
        assert cmd is not None
        assert cmd.action == "delegate"
        assert cmd.target == "analyst"
    
    def test_parse_final_answer(self):
        """Test parsing FINAL ANSWER command."""
        text = "Based on my analysis, FINAL ANSWER: The result is 42."
        cmd = parse_delegation(text)
        assert cmd is not None
        assert cmd.action == "final_answer"
        assert "42" in cmd.task
    
    def test_parse_route_command(self):
        """Test parsing ROUTE TO command."""
        text = "ROUTE TO [validator]: Please validate this output."
        cmd = parse_delegation(text)
        assert cmd is not None
        assert cmd.action == "route"
        assert cmd.target == "validator"
    
    def test_parse_broadcast_command(self):
        """Test parsing BROADCAST command."""
        text = "BROADCAST: All agents please check in."
        cmd = parse_delegation(text)
        assert cmd is not None
        assert cmd.action == "broadcast"
        assert "check in" in cmd.task
    
    def test_parse_no_command(self):
        """Test that regular text returns None."""
        text = "I'm just thinking about this problem."
        cmd = parse_delegation(text)
        assert cmd is None
    
    def test_has_final_answer_true(self):
        """Test detecting final answer."""
        text = "FINAL ANSWER: The result is complete."
        assert has_final_answer(text) is True
    
    def test_has_final_answer_false(self):
        """Test when no final answer present."""
        text = "I need to delegate this task."
        assert has_final_answer(text) is False
    
    def test_extract_final_answer(self):
        """Test extracting final answer content."""
        text = "After analysis, FINAL ANSWER: The project is approved."
        answer = extract_final_answer(text)
        assert answer is not None
        assert "approved" in answer


class TestAgentRoleIntegration:
    """Test Agent class role integration."""
    
    def test_agent_gets_role_prompt_by_default(self):
        """Agent should use role prompt if no instructions given."""
        # Note: We can't fully test without a model, but we can check the config
        from agenticflow.agent.config import AgentConfig
        
        config = AgentConfig(
            name="TestWorker",
            role=AgentRole.WORKER,
            model=None,  # No model for this test
        )
        # Worker role prompt should be set when Agent is created
        # (This is handled in Agent.__init__, not AgentConfig)
    
    def test_agent_role_behavior_property(self):
        """Agent should have role_behavior property."""
        from unittest.mock import MagicMock
        
        mock_model = MagicMock()
        agent = Agent(
            name="TestSupervisor",
            model=mock_model,
            role=AgentRole.SUPERVISOR,
        )
        
        assert agent.role == AgentRole.SUPERVISOR
        assert agent.can_delegate is True
        assert agent.can_finish is True
    
    def test_agent_factory_as_supervisor(self):
        """Test Agent.as_supervisor factory method."""
        from unittest.mock import MagicMock
        
        mock_model = MagicMock()
        supervisor = Agent.as_supervisor(
            name="Manager",
            model=mock_model,
            workers=["Alice", "Bob"],
        )
        
        assert supervisor.role == AgentRole.SUPERVISOR
        assert supervisor.can_delegate is True
        # Check workers are mentioned in prompt
        assert "Alice" in supervisor.instructions
        assert "Bob" in supervisor.instructions
    
    def test_agent_factory_as_worker(self):
        """Test Agent.as_worker factory method."""
        from unittest.mock import MagicMock
        
        mock_model = MagicMock()
        worker = Agent.as_worker(
            name="Analyst",
            model=mock_model,
            specialty="data analysis",
        )
        
        assert worker.role == AgentRole.WORKER
        assert worker.can_delegate is False
        assert "data analysis" in worker.instructions
    
    def test_agent_factory_as_critic(self):
        """Test Agent.as_critic factory method."""
        from unittest.mock import MagicMock
        
        mock_model = MagicMock()
        critic = Agent.as_critic(
            name="Reviewer",
            model=mock_model,
            criteria=["correctness", "clarity"],
        )
        
        assert critic.role == AgentRole.CRITIC
        assert "correctness" in critic.instructions
        assert "clarity" in critic.instructions
