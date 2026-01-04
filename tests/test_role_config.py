"""Tests for RoleConfig classes."""

import pytest

from agenticflow import (
    Agent,
    AutonomousRole,
    CustomRole,
    ReviewerRole,
    SupervisorRole,
    WorkerRole,
)
from agenticflow.core.enums import AgentRole
from agenticflow.models.openai import OpenAIChat


class TestRoleConfigClasses:
    """Test RoleConfig dataclass objects."""

    def test_supervisor_role_creation(self):
        """Test SupervisorRole instantiation."""
        role = SupervisorRole(workers=["Alice", "Bob"])
        assert role.workers == ["Alice", "Bob"]
        assert role.get_role_type() == AgentRole.SUPERVISOR
        caps = role.get_capabilities()
        assert caps["can_finish"] is True
        assert caps["can_delegate"] is True
        assert caps["can_use_tools"] is False

    def test_supervisor_role_without_workers(self):
        """Test SupervisorRole can be created without workers."""
        role = SupervisorRole()
        assert role.workers is None
        assert role.get_role_type() == AgentRole.SUPERVISOR

    def test_supervisor_role_prompt_enhancement(self):
        """Test SupervisorRole enhances prompts with worker names."""
        role = SupervisorRole(workers=["Alice", "Bob", "Charlie"])
        base_prompt = "You are a supervisor."
        enhanced = role.enhance_prompt(base_prompt)
        assert "Alice" in enhanced
        assert "Bob" in enhanced
        assert "Charlie" in enhanced

    def test_worker_role_creation(self):
        """Test WorkerRole instantiation."""
        role = WorkerRole(specialty="data analysis")
        assert role.specialty == "data analysis"
        assert role.get_role_type() == AgentRole.WORKER
        caps = role.get_capabilities()
        assert caps["can_finish"] is False
        assert caps["can_delegate"] is False
        assert caps["can_use_tools"] is True

    def test_worker_role_without_specialty(self):
        """Test WorkerRole can be created without specialty."""
        role = WorkerRole()
        assert role.specialty is None
        assert role.get_role_type() == AgentRole.WORKER

    def test_worker_role_prompt_enhancement(self):
        """Test WorkerRole enhances prompts with specialty."""
        role = WorkerRole(specialty="financial analysis and forecasting")
        base_prompt = "You are a worker."
        enhanced = role.enhance_prompt(base_prompt)
        assert "financial analysis and forecasting" in enhanced

    def test_reviewer_role_creation(self):
        """Test ReviewerRole instantiation."""
        role = ReviewerRole(criteria=["accuracy", "clarity"])
        assert role.criteria == ["accuracy", "clarity"]
        assert role.get_role_type() == AgentRole.REVIEWER
        caps = role.get_capabilities()
        assert caps["can_finish"] is True
        assert caps["can_delegate"] is False
        assert caps["can_use_tools"] is False

    def test_reviewer_role_without_criteria(self):
        """Test ReviewerRole can be created without criteria."""
        role = ReviewerRole()
        assert role.criteria is None
        assert role.get_role_type() == AgentRole.REVIEWER

    def test_reviewer_role_prompt_enhancement(self):
        """Test ReviewerRole enhances prompts with criteria."""
        role = ReviewerRole(criteria=["accuracy", "clarity", "completeness"])
        base_prompt = "You are a reviewer."
        enhanced = role.enhance_prompt(base_prompt)
        assert "accuracy" in enhanced
        assert "clarity" in enhanced
        assert "completeness" in enhanced

    def test_autonomous_role_creation(self):
        """Test AutonomousRole instantiation."""
        role = AutonomousRole()
        assert role.get_role_type() == AgentRole.AUTONOMOUS
        caps = role.get_capabilities()
        assert caps["can_finish"] is True
        assert caps["can_delegate"] is False
        assert caps["can_use_tools"] is True

    def test_custom_role_with_defaults(self):
        """Test CustomRole with default values."""
        role = CustomRole()
        assert role.base_role == AgentRole.AUTONOMOUS
        assert role.can_finish is None
        assert role.can_delegate is None
        assert role.can_use_tools is None
        assert role.get_role_type() == AgentRole.AUTONOMOUS

    def test_custom_role_with_overrides(self):
        """Test CustomRole with capability overrides."""
        role = CustomRole(
            base_role=AgentRole.WORKER,
            can_finish=True,  # Override: workers normally can't finish
        )
        caps = role.get_capabilities()
        assert caps["can_finish"] is True  # Overridden
        assert caps["can_delegate"] is False  # From base
        assert caps["can_use_tools"] is True  # From base

    def test_custom_role_hybrid_capabilities(self):
        """Test CustomRole can create hybrid roles."""
        # Reviewer that can use tools
        role = CustomRole(base_role=AgentRole.REVIEWER, can_use_tools=True)
        caps = role.get_capabilities()
        assert caps["can_finish"] is True
        assert caps["can_delegate"] is False
        assert caps["can_use_tools"] is True  # Overridden


class TestAgentWithRoleConfig:
    """Test Agent instantiation with RoleConfig objects."""

    def test_agent_with_supervisor_role(self):
        """Test Agent accepts SupervisorRole."""
        model = OpenAIChat(model="gpt-4o-mini")
        role = SupervisorRole(workers=["Alice", "Bob"])
        agent = Agent(name="Manager", model=model, role=role)
        assert agent.role == AgentRole.SUPERVISOR
        assert agent.can_delegate is True
        assert agent.can_finish is True
        assert agent.can_use_tools is False

    def test_agent_with_worker_role(self):
        """Test Agent accepts WorkerRole."""
        model = OpenAIChat(model="gpt-4o-mini")
        role = WorkerRole(specialty="data analysis")
        agent = Agent(name="Analyst", model=model, role=role)
        assert agent.role == AgentRole.WORKER
        assert agent.can_delegate is False
        assert agent.can_finish is False
        assert agent.can_use_tools is True

    def test_agent_with_reviewer_role(self):
        """Test Agent accepts ReviewerRole."""
        model = OpenAIChat(model="gpt-4o-mini")
        role = ReviewerRole(criteria=["accuracy"])
        agent = Agent(name="QA", model=model, role=role)
        assert agent.role == AgentRole.REVIEWER
        assert agent.can_delegate is False
        assert agent.can_finish is True
        assert agent.can_use_tools is False

    def test_agent_with_autonomous_role(self):
        """Test Agent accepts AutonomousRole."""
        model = OpenAIChat(model="gpt-4o-mini")
        role = AutonomousRole()
        agent = Agent(name="Assistant", model=model, role=role)
        assert agent.role == AgentRole.AUTONOMOUS
        assert agent.can_delegate is False
        assert agent.can_finish is True
        assert agent.can_use_tools is True

    def test_agent_with_custom_role(self):
        """Test Agent accepts CustomRole."""
        model = OpenAIChat(model="gpt-4o-mini")
        role = CustomRole(base_role=AgentRole.WORKER, can_finish=True)
        agent = Agent(name="SpecialWorker", model=model, role=role)
        assert agent.role == AgentRole.WORKER
        assert agent.can_finish is True  # Overridden


class TestRoleConfigBackwardCompatibility:
    """Test backward compatibility with string/enum roles."""

    def test_string_role_still_works(self):
        """Test Agent still accepts string roles."""
        model = OpenAIChat(model="gpt-4o-mini")
        agent = Agent(name="Worker", model=model, role="worker")
        assert agent.role == AgentRole.WORKER

    def test_enum_role_still_works(self):
        """Test Agent still accepts AgentRole enum."""
        model = OpenAIChat(model="gpt-4o-mini")
        agent = Agent(name="Supervisor", model=model, role=AgentRole.SUPERVISOR)
        assert agent.role == AgentRole.SUPERVISOR

    def test_role_specific_args_still_work(self):
        """Test backward compat with role-specific arguments."""
        model = OpenAIChat(model="gpt-4o-mini")
        agent = Agent(
            name="Manager",
            model=model,
            role="supervisor",
            workers=["Alice", "Bob"],
        )
        assert agent.role == AgentRole.SUPERVISOR

    def test_capability_overrides_still_work(self):
        """Test backward compat with capability overrides."""
        model = OpenAIChat(model="gpt-4o-mini")
        agent = Agent(
            name="Worker",
            model=model,
            role="worker",
            can_finish=True,  # Override
        )
        assert agent.can_finish is True


class TestRoleConfigImmutability:
    """Test that RoleConfig objects are immutable."""

    def test_supervisor_role_is_frozen(self):
        """Test SupervisorRole is frozen (immutable)."""
        role = SupervisorRole(workers=["Alice"])
        with pytest.raises(AttributeError):
            role.workers = ["Bob"]  # type: ignore

    def test_worker_role_is_frozen(self):
        """Test WorkerRole is frozen (immutable)."""
        role = WorkerRole(specialty="analysis")
        with pytest.raises(AttributeError):
            role.specialty = "research"  # type: ignore

    def test_custom_role_is_frozen(self):
        """Test CustomRole is frozen (immutable)."""
        role = CustomRole(base_role=AgentRole.WORKER)
        with pytest.raises(AttributeError):
            role.base_role = AgentRole.SUPERVISOR  # type: ignore


class TestRoleConfigEquality:
    """Test RoleConfig equality and hashing."""

    def test_supervisor_role_equality(self):
        """Test SupervisorRole equality."""
        role1 = SupervisorRole(workers=["Alice", "Bob"])
        role2 = SupervisorRole(workers=["Alice", "Bob"])
        assert role1 == role2

    def test_supervisor_role_inequality(self):
        """Test SupervisorRole inequality."""
        role1 = SupervisorRole(workers=["Alice"])
        role2 = SupervisorRole(workers=["Bob"])
        assert role1 != role2

    def test_worker_role_equality(self):
        """Test WorkerRole equality."""
        role1 = WorkerRole(specialty="analysis")
        role2 = WorkerRole(specialty="analysis")
        assert role1 == role2

    def test_autonomous_role_equality(self):
        """Test AutonomousRole equality (no fields)."""
        role1 = AutonomousRole()
        role2 = AutonomousRole()
        assert role1 == role2

    def test_role_config_hashable(self):
        """Test RoleConfig objects can be used in sets (where applicable)."""
        # Only roles without mutable fields (lists) are hashable
        role1 = AutonomousRole()
        role2 = WorkerRole()  # Has specialty but it's optional and None
        role_set = {role1, role2}
        assert len(role_set) == 2
        assert role1 in role_set
        assert role2 in role_set
