"""Tests for agent roles and behaviors with clean 4-role system."""

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
    
    def test_worker_prompt_mentions_tools(self):
        """Worker prompt should mention tools."""
        prompt = get_role_prompt(AgentRole.WORKER)
        assert "tool" in prompt.lower()
    
    def test_reviewer_prompt_mentions_review(self):
        """Reviewer prompt should mention review/feedback."""
        prompt = get_role_prompt(AgentRole.REVIEWER)
        assert "review" in prompt.lower() or "quality" in prompt.lower()
    
    def test_autonomous_prompt_mentions_independence(self):
        """Autonomous prompt should mention independence."""
        prompt = get_role_prompt(AgentRole.AUTONOMOUS)
        assert "independent" in prompt.lower() or "autonomy" in prompt.lower() or "independently" in prompt.lower()


class TestRoleBehaviors:
    """Test role-specific behaviors with clean 4-role system."""
    
    def test_all_roles_have_behaviors(self):
        """Every role should have defined behaviors."""
        for role in AgentRole:
            behavior = get_role_behavior(role)
            assert isinstance(behavior, RoleBehavior)
    
    def test_supervisor_can_delegate(self):
        """Supervisors should be able to delegate."""
        behavior = get_role_behavior(AgentRole.SUPERVISOR)
        assert behavior.can_delegate is True
    
    def test_supervisor_can_finish(self):
        """Supervisors should be able to finish."""
        behavior = get_role_behavior(AgentRole.SUPERVISOR)
        assert behavior.can_finish is True
    
    def test_supervisor_cannot_use_tools(self):
        """Supervisors delegate, they don't execute."""
        behavior = get_role_behavior(AgentRole.SUPERVISOR)
        assert behavior.can_use_tools is False
    
    def test_worker_cannot_delegate(self):
        """Workers should not be able to delegate."""
        behavior = get_role_behavior(AgentRole.WORKER)
        assert behavior.can_delegate is False
    
    def test_worker_can_use_tools(self):
        """Workers should be able to use tools."""
        behavior = get_role_behavior(AgentRole.WORKER)
        assert behavior.can_use_tools is True
    
    def test_worker_cannot_finish(self):
        """Workers should not be able to finish."""
        behavior = get_role_behavior(AgentRole.WORKER)
        assert behavior.can_finish is False
    
    def test_autonomous_can_finish(self):
        """Autonomous agents can finish."""
        behavior = get_role_behavior(AgentRole.AUTONOMOUS)
        assert behavior.can_finish is True
    
    def test_autonomous_can_use_tools(self):
        """Autonomous agents can use tools."""
        behavior = get_role_behavior(AgentRole.AUTONOMOUS)
        assert behavior.can_use_tools is True
    
    def test_autonomous_cannot_delegate(self):
        """Autonomous agents don't delegate."""
        behavior = get_role_behavior(AgentRole.AUTONOMOUS)
        assert behavior.can_delegate is False
    
    def test_reviewer_can_finish(self):
        """Reviewers can finish (approve/reject)."""
        behavior = get_role_behavior(AgentRole.REVIEWER)
        assert behavior.can_finish is True
    
    def test_reviewer_cannot_delegate(self):
        """Reviewers don't delegate."""
        behavior = get_role_behavior(AgentRole.REVIEWER)
        assert behavior.can_delegate is False
    
    def test_reviewer_cannot_use_tools(self):
        """Reviewers review, they don't use tools."""
        behavior = get_role_behavior(AgentRole.REVIEWER)
        assert behavior.can_use_tools is False


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
    
    def test_agent_with_role_and_workers(self):
        """Test Agent with role=SUPERVISOR and workers parameter."""
        from unittest.mock import MagicMock
        
        mock_model = MagicMock()
        supervisor = Agent(
            name="Manager",
            model=mock_model,
            role=AgentRole.SUPERVISOR,
            workers=["Alice", "Bob"],
        )
        
        assert supervisor.role == AgentRole.SUPERVISOR
        assert supervisor.can_delegate is True
        assert "Alice" in supervisor.instructions
        assert "Bob" in supervisor.instructions
    
    def test_agent_with_role_and_specialty(self):
        """Test Agent with role=WORKER and specialty parameter."""
        from unittest.mock import MagicMock
        
        mock_model = MagicMock()
        worker = Agent(
            name="Analyst",
            model=mock_model,
            role=AgentRole.WORKER,
            specialty="data analysis",
        )
        
        assert worker.role == AgentRole.WORKER
        assert worker.can_delegate is False
        assert "data analysis" in worker.instructions
    
    def test_agent_with_role_and_criteria(self):
        """Test Agent with role=REVIEWER and criteria parameter."""
        from unittest.mock import MagicMock
        
        mock_model = MagicMock()
        reviewer = Agent(
            name="Reviewer",
            model=mock_model,
            role=AgentRole.REVIEWER,
            criteria=["correctness", "clarity"],
        )
        
        assert reviewer.role == AgentRole.REVIEWER
        assert "correctness" in reviewer.instructions
        assert "clarity" in reviewer.instructions
    
    def test_worker_capabilities(self):
        """Worker: can_tools=✅, can_finish=❌, can_delegate=❌."""
        from unittest.mock import MagicMock
        
        mock_model = MagicMock()
        worker = Agent(name="W", model=mock_model, role=AgentRole.WORKER)
        
        assert worker.can_use_tools is True
        assert worker.can_finish is False
        assert worker.can_delegate is False
    
    def test_supervisor_capabilities(self):
        """Supervisor: can_tools=❌, can_finish=✅, can_delegate=✅."""
        from unittest.mock import MagicMock
        
        mock_model = MagicMock()
        supervisor = Agent(name="S", model=mock_model, role=AgentRole.SUPERVISOR)
        
        assert supervisor.can_use_tools is False
        assert supervisor.can_finish is True
        assert supervisor.can_delegate is True
    
    def test_autonomous_capabilities(self):
        """Autonomous: can_tools=✅, can_finish=✅, can_delegate=❌."""
        from unittest.mock import MagicMock
        
        mock_model = MagicMock()
        auto = Agent(name="A", model=mock_model, role=AgentRole.AUTONOMOUS)
        
        assert auto.can_use_tools is True
        assert auto.can_finish is True
        assert auto.can_delegate is False
    
    def test_reviewer_capabilities(self):
        """Reviewer: can_tools=❌, can_finish=✅, can_delegate=❌."""
        from unittest.mock import MagicMock
        
        mock_model = MagicMock()
        reviewer = Agent(name="R", model=mock_model, role=AgentRole.REVIEWER)
        
        assert reviewer.can_use_tools is False
        assert reviewer.can_finish is True
        assert reviewer.can_delegate is False
