"""Comprehensive tests for the role system design and enforcement."""

import pytest

from agenticflow import Agent
from agenticflow.agent.roles import get_role_behavior, get_role_prompt
from agenticflow.core.enums import AgentRole
from agenticflow.models import ChatModel


class TestRoleCapabilities:
    """Test that role capabilities are correctly defined."""

    def test_worker_capabilities(self):
        """WORKER can use tools but cannot finish or delegate."""
        role = AgentRole.WORKER
        assert role.can_use_tools() is True
        assert role.can_finish() is False
        assert role.can_delegate() is False

    def test_supervisor_capabilities(self):
        """SUPERVISOR can finish and delegate but cannot use tools."""
        role = AgentRole.SUPERVISOR
        assert role.can_use_tools() is False
        assert role.can_finish() is True
        assert role.can_delegate() is True

    def test_autonomous_capabilities(self):
        """AUTONOMOUS can use tools and finish but cannot delegate."""
        role = AgentRole.AUTONOMOUS
        assert role.can_use_tools() is True
        assert role.can_finish() is True
        assert role.can_delegate() is False

    def test_reviewer_capabilities(self):
        """REVIEWER can finish but cannot use tools or delegate."""
        role = AgentRole.REVIEWER
        assert role.can_use_tools() is False
        assert role.can_finish() is True
        assert role.can_delegate() is False


class TestRoleBehaviors:
    """Test that RoleBehavior correctly reflects capabilities."""

    def test_worker_behavior(self):
        """WORKER behavior allows tools but not finish/delegate."""
        behavior = get_role_behavior(AgentRole.WORKER)
        assert behavior.can_use_tools is True
        assert behavior.can_finish is False
        assert behavior.can_delegate is False

    def test_supervisor_behavior(self):
        """SUPERVISOR behavior allows finish/delegate but not tools."""
        behavior = get_role_behavior(AgentRole.SUPERVISOR)
        assert behavior.can_use_tools is False
        assert behavior.can_finish is True
        assert behavior.can_delegate is True

    def test_autonomous_behavior(self):
        """AUTONOMOUS behavior allows tools and finish but not delegate."""
        behavior = get_role_behavior(AgentRole.AUTONOMOUS)
        assert behavior.can_use_tools is True
        assert behavior.can_finish is True
        assert behavior.can_delegate is False

    def test_reviewer_behavior(self):
        """REVIEWER behavior allows finish but not tools/delegate."""
        behavior = get_role_behavior(AgentRole.REVIEWER)
        assert behavior.can_use_tools is False
        assert behavior.can_finish is True
        assert behavior.can_delegate is False


class TestRolePrompts:
    """Test that role prompts are appropriate and complete."""

    def test_all_roles_have_prompts(self):
        """Every role has a default prompt."""
        for role in AgentRole:
            prompt = get_role_prompt(role)
            assert prompt
            assert len(prompt) > 50  # Non-trivial prompt

    def test_worker_prompt_mentions_tools(self):
        """WORKER prompt includes tool usage instructions."""
        prompt = get_role_prompt(AgentRole.WORKER, has_tools=True)
        assert "tool" in prompt.lower()
        assert "cannot finish" in prompt.lower() or "can not finish" in prompt.lower()

    def test_worker_prompt_without_tools(self):
        """WORKER prompt adapts when no tools available."""
        prompt = get_role_prompt(AgentRole.WORKER, has_tools=False)
        # Should not have tool-specific call format instructions
        assert "TOOL:" not in prompt or "tool_name" not in prompt

    def test_supervisor_prompt_mentions_delegation(self):
        """SUPERVISOR prompt includes delegation instructions."""
        prompt = get_role_prompt(AgentRole.SUPERVISOR)
        assert "delegate" in prompt.lower()
        assert "final answer" in prompt.lower()

    def test_supervisor_prompt_no_tools(self):
        """SUPERVISOR prompt explicitly says not to use tools."""
        prompt = get_role_prompt(AgentRole.SUPERVISOR)
        assert "do not use tools" in prompt.lower() or "not use tools" in prompt.lower()

    def test_autonomous_prompt_mentions_both(self):
        """AUTONOMOUS prompt mentions both tools and finishing."""
        prompt = get_role_prompt(AgentRole.AUTONOMOUS, has_tools=True)
        assert "tool" in prompt.lower()
        assert "final answer" in prompt.lower()

    def test_reviewer_prompt_mentions_approval(self):
        """REVIEWER prompt includes review/approval instructions."""
        prompt = get_role_prompt(AgentRole.REVIEWER)
        assert "review" in prompt.lower() or "approve" in prompt.lower()


class TestAgentFactoryMethods:
    """Test Agent factory methods create correct roles."""

    def test_worker_with_role_arg(self):
        """Agent with role=WORKER creates WORKER role."""
        model = ChatModel(model="gpt-4o-mini")
        agent = Agent(name="TestWorker", model=model, role=AgentRole.WORKER)
        
        assert agent.role == AgentRole.WORKER
        assert agent.can_use_tools is True
        assert agent.can_finish is False
        assert agent.can_delegate is False

    def test_supervisor_with_role_arg(self):
        """Agent with role=SUPERVISOR creates SUPERVISOR role."""
        model = ChatModel(model="gpt-4o-mini")
        agent = Agent(
            name="TestSupervisor",
            model=model,
            role=AgentRole.SUPERVISOR,
            workers=["worker1", "worker2"]
        )
        
        assert agent.role == AgentRole.SUPERVISOR
        assert agent.can_use_tools is False
        assert agent.can_finish is True
        assert agent.can_delegate is True

    def test_reviewer_with_role_arg(self):
        """Agent with role=REVIEWER creates REVIEWER role."""
        model = ChatModel(model="gpt-4o-mini")
        agent = Agent(name="TestReviewer", model=model, role=AgentRole.REVIEWER)
        
        assert agent.role == AgentRole.REVIEWER
        assert agent.can_use_tools is False
        assert agent.can_finish is True
        assert agent.can_delegate is False

    def test_default_role_is_worker(self):
        """Agent without explicit role defaults to WORKER."""
        model = ChatModel(model="gpt-4o-mini")
        agent = Agent(name="TestAgent", model=model)
        
        assert agent.role == AgentRole.WORKER
        assert agent.can_use_tools is True
        assert agent.can_finish is False
        assert agent.can_delegate is False


class TestRoleConsistency:
    """Test that role system is internally consistent."""

    def test_only_one_role_can_both_use_tools_and_delegate(self):
        """No role should be able to both use tools AND delegate."""
        for role in AgentRole:
            # This design constraint ensures clear separation of concerns
            if role.can_delegate():
                assert not role.can_use_tools(), \
                    f"{role.value} can both delegate AND use tools - breaks design!"

    def test_all_roles_except_worker_can_finish(self):
        """All roles except WORKER can finish."""
        for role in AgentRole:
            if role == AgentRole.WORKER:
                assert not role.can_finish()
            else:
                assert role.can_finish()

    def test_role_capabilities_are_distinct(self):
        """Each role has a unique combination of capabilities."""
        capabilities = {}
        for role in AgentRole:
            cap_tuple = (
                role.can_use_tools(),
                role.can_finish(),
                role.can_delegate()
            )
            assert cap_tuple not in capabilities.values(), \
                f"{role.value} has duplicate capabilities"
            capabilities[role] = cap_tuple


class TestRoleDesignPrinciples:
    """Test that the role system follows its design principles."""

    def test_separation_of_concerns(self):
        """Roles separate execution (tools) from coordination (delegation)."""
        # Workers execute, supervisors coordinate
        assert AgentRole.WORKER.can_use_tools()
        assert AgentRole.SUPERVISOR.can_delegate()
        assert not (AgentRole.WORKER.can_delegate() and AgentRole.SUPERVISOR.can_use_tools())

    def test_worker_requires_coordination(self):
        """WORKER cannot finish, forcing it to work within a team structure."""
        assert not AgentRole.WORKER.can_finish()
        assert not AgentRole.WORKER.can_delegate()
        # This ensures workers are part of coordinated workflows

    def test_autonomous_is_truly_independent(self):
        """AUTONOMOUS can work completely alone."""
        role = AgentRole.AUTONOMOUS
        # Can do work (tools) and finish without delegation
        assert role.can_use_tools()
        assert role.can_finish()
        # Perfect for single-agent scenarios

    def test_reviewer_focuses_on_judgment(self):
        """REVIEWER focuses purely on evaluation without tools."""
        role = AgentRole.REVIEWER
        assert role.can_finish()  # Can approve/reject
        assert not role.can_use_tools()  # No external actions
        assert not role.can_delegate()  # Direct judgment only


class TestRoleEnforcement:
    """Test that role capabilities are properly enforced at runtime."""

    def test_agent_inherits_role_capabilities(self):
        """Agent instances correctly inherit their role's capabilities."""
        model = ChatModel(model="gpt-4o-mini")
        
        for role in AgentRole:
            agent = Agent(name=f"Test{role.value}", model=model, role=role)
            
            # Agent capabilities should match role capabilities
            assert agent.can_use_tools == role.can_use_tools()
            assert agent.can_finish == role.can_finish()
            assert agent.can_delegate == role.can_delegate()

    def test_role_behavior_consistency(self):
        """RoleBehavior and AgentRole capabilities stay in sync."""
        for role in AgentRole:
            behavior = get_role_behavior(role)
            
            # Behavior should match role enum methods
            assert behavior.can_use_tools == role.can_use_tools()
            assert behavior.can_finish == role.can_finish()
            assert behavior.can_delegate == role.can_delegate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
