"""
RoleConfig Objects - Modern Role API

Demonstrates the new RoleConfig object pattern for type-safe role definitions.

Key Features:
- Type-safe, immutable role configurations
- Built-in prompt enhancement
- Clear, explicit capability overrides
- Full IDE support with autocomplete

Usage:
    uv run python examples/basics/role_configs.py
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model

from agenticflow import (
    Agent,
    AutonomousRole,
    CustomRole,
    ReviewerRole,
    SupervisorRole,
    WorkerRole,
)
from agenticflow.core import AgentRole


async def demo_role_config_objects():
    """Demonstrate RoleConfig objects."""
    print("\n=== RoleConfig Objects ===\n")
    
    model = get_model()
    
    # SupervisorRole - coordinates workers
    print("1. SupervisorRole - Coordinates team members")
    supervisor_role = SupervisorRole(workers=["Alice", "Bob", "Charlie"])
    supervisor = Agent(
        name="Manager",
        model=model,
        role=supervisor_role,
        instructions="Coordinate the team to accomplish the task.",
    )
    print(f"   Role: {supervisor.role.value}")
    print(f"   Capabilities: finish={supervisor.can_finish}, "
          f"delegate={supervisor.can_delegate}, tools={supervisor.can_use_tools}")
    print(f"   Workers: {supervisor_role.workers}")
    
    # WorkerRole - executes tasks with specialty
    print("\n2. WorkerRole - Specialized task execution")
    worker_role = WorkerRole(specialty="data analysis and machine learning")
    worker = Agent(
        name="DataScientist",
        model=model,
        role=worker_role,
        instructions="Analyze data and build models.",
    )
    print(f"   Role: {worker.role.value}")
    print(f"   Capabilities: finish={worker.can_finish}, "
          f"delegate={worker.can_delegate}, tools={worker.can_use_tools}")
    print(f"   Specialty: {worker_role.specialty}")
    
    # ReviewerRole - evaluates with criteria
    print("\n3. ReviewerRole - Evaluation and approval")
    reviewer_role = ReviewerRole(
        criteria=["accuracy", "clarity", "completeness", "best practices"]
    )
    reviewer = Agent(
        name="QualityAssurance",
        model=model,
        role=reviewer_role,
        instructions="Review work against quality criteria.",
    )
    print(f"   Role: {reviewer.role.value}")
    print(f"   Capabilities: finish={reviewer.can_finish}, "
          f"delegate={reviewer.can_delegate}, tools={reviewer.can_use_tools}")
    print(f"   Criteria: {reviewer_role.criteria}")
    
    # AutonomousRole - independent operation
    print("\n4. AutonomousRole - Independent agent")
    autonomous_role = AutonomousRole()
    autonomous = Agent(
        name="Assistant",
        model=model,
        role=autonomous_role,
        instructions="Help users with their requests.",
    )
    print(f"   Role: {autonomous.role.value}")
    print(f"   Capabilities: finish={autonomous.can_finish}, "
          f"delegate={autonomous.can_delegate}, tools={autonomous.can_use_tools}")
    
    # CustomRole - hybrid capabilities
    print("\n5. CustomRole - Hybrid reviewer with tool access")
    custom_role = CustomRole(
        base_role=AgentRole.REVIEWER,
        can_use_tools=True,  # Override! Reviewers normally can't use tools
    )
    custom = Agent(
        name="TechnicalReviewer",
        model=model,
        role=custom_role,
        instructions="Review code and run automated checks.",
    )
    print(f"   Role: {custom.role.value}")
    print(f"   Capabilities: finish={custom.can_finish}, "
          f"delegate={custom.can_delegate}, tools={custom.can_use_tools}")


async def demo_prompt_enhancement():
    """Show how RoleConfig objects enhance prompts."""
    print("\n\n=== Prompt Enhancement ===\n")
    
    # SupervisorRole enhances with worker names
    supervisor_role = SupervisorRole(workers=["Researcher", "Writer", "Editor"])
    base_prompt = "You are a team supervisor."
    enhanced = supervisor_role.enhance_prompt(base_prompt)
    print("SupervisorRole prompt enhancement:")
    print(f"  Base: {base_prompt}")
    print(f"  Enhanced: {enhanced[:100]}...")
    
    # WorkerRole enhances with specialty
    worker_role = WorkerRole(specialty="financial modeling and forecasting")
    base_prompt = "You are a data analyst."
    enhanced = worker_role.enhance_prompt(base_prompt)
    print("\nWorkerRole prompt enhancement:")
    print(f"  Base: {base_prompt}")
    print(f"  Enhanced: {enhanced}")
    
    # ReviewerRole enhances with criteria
    reviewer_role = ReviewerRole(criteria=["accuracy", "completeness", "clarity"])
    base_prompt = "You review work."
    enhanced = reviewer_role.enhance_prompt(base_prompt)
    print("\nReviewerRole prompt enhancement:")
    print(f"  Base: {base_prompt}")
    print(f"  Enhanced: {enhanced[:100]}...")


async def demo_immutability():
    """Demonstrate that RoleConfig objects are immutable."""
    print("\n\n=== Immutability (Frozen Dataclasses) ===\n")
    
    role = SupervisorRole(workers=["Alice", "Bob"])
    print(f"Created: {role}")
    print(f"Workers: {role.workers}")
    
    try:
        role.workers = ["Charlie"]  # type: ignore
        print("❌ UNEXPECTED: Role was modified!")
    except AttributeError:
        print("✅ Role is immutable - modification prevented")
    
    print(f"Workers unchanged: {role.workers}")


async def demo_equality():
    """Show RoleConfig equality semantics."""
    print("\n\n=== Equality & Hashing ===\n")
    
    role1 = SupervisorRole(workers=["Alice", "Bob"])
    role2 = SupervisorRole(workers=["Alice", "Bob"])
    role3 = SupervisorRole(workers=["Charlie"])
    
    print(f"role1 == role2: {role1 == role2} (same workers)")
    print(f"role1 == role3: {role1 == role3} (different workers)")
    
    # Show different role types
    worker = WorkerRole(specialty="analysis")
    print(f"supervisor == worker: {role1 == worker} (different types)")


async def demo_backward_compatibility():
    """Show backward compatibility with string/enum roles."""
    print("\n\n=== Backward Compatibility ===\n")
    
    model = get_model()
    
    # Old API: String role
    old_style = Agent(
        name="Worker",
        model=model,
        role="worker",  # String still works
        specialty="data analysis",  # Parameters still work
    )
    print(f"String role: {old_style.role.value}")
    
    # Old API: Enum role
    old_enum = Agent(
        name="Supervisor",
        model=model,
        role=AgentRole.SUPERVISOR,  # Enum still works
        workers=["Alice", "Bob"],
    )
    print(f"Enum role: {old_enum.role.value}")
    
    # New API: RoleConfig object
    new_style = Agent(
        name="Worker",
        model=model,
        role=WorkerRole(specialty="data analysis"),  # RoleConfig object
    )
    print(f"RoleConfig object: {new_style.role.value}")
    
    print("\n✅ All three approaches work! RoleConfig is recommended for new code.")


async def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("  RoleConfig Objects - Modern Role API")
    print("=" * 70)
    
    await demo_role_config_objects()
    await demo_prompt_enhancement()
    await demo_immutability()
    await demo_equality()
    await demo_backward_compatibility()
    
    print("\n" + "=" * 70)
    print("✅ All demonstrations complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
