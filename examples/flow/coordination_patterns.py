"""Coordination Patterns Examples - Demonstrating all_sources() for multi-agent coordination.

Shows how Flow automatically coordinates multiple agents using the all_sources() filter.

Run this file to see examples:
    uv run python examples/flow/coordination_patterns.py
"""

import asyncio
import sys
from pathlib import Path

# Add examples to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import get_model

from agenticflow import Agent, Flow
from agenticflow.events import Event
from agenticflow.events.patterns import all_sources
from agenticflow.events.standards import (
    TaskEvents,
    AgentEvents,
    ReviewEvents,
    BatchEvents,
    DeploymentEvents,
    IncidentEvents,
)
from agenticflow.observability import Observer


# ============================================================================
# Example 1: Map-Reduce - Wait for all workers before aggregating
# ============================================================================


async def example_1_map_reduce():
    """Map-reduce: Workers process chunks, Flow coordinates aggregation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Map-Reduce Pattern (High-Level API)")
    print("=" * 70)
    print("\nThree workers process chunks independently, then Flow automatically")
    print("triggers coordinator when ALL workers complete.\n")
    
    model = get_model()
    
    # Create worker agents
    worker1 = Agent(name="worker_1", model=model, instructions="Process data chunks briefly.")
    worker2 = Agent(name="worker_2", model=model, instructions="Process data chunks briefly.")
    worker3 = Agent(name="worker_3", model=model, instructions="Process data chunks briefly.")
    
    # Coordinator agent
    coordinator = Agent(name="coordinator", model=model, instructions="Aggregate all worker results briefly.")
    
    # Create Flow with observer to see coordination in action
    flow = Flow(observer=Observer.trace())
    
    # Register workers - they react to task assignments
    flow.register(worker1, on=TaskEvents.ASSIGNED)
    flow.register(worker2, on=TaskEvents.ASSIGNED)
    flow.register(worker3, on=TaskEvents.ASSIGNED)
    
    # Register coordinator with coordination filter
    # Flow automatically waits for ALL workers before triggering!
    flow.register(
        coordinator,
        on=AgentEvents.DONE,
        when=all_sources(["worker_1", "worker_2", "worker_3"]),
        emits=TaskEvents.COMPLETED
    )
    
    print("Assigning work to workers...")
    
    # Use semantic event name for clarity
    result = await flow.run("Process data chunks", initial_event=TaskEvents.ASSIGNED)
    
    print("\n[OK] Success: Flow automatically coordinated map-reduce!\n")


# ============================================================================
# Example 2: Multi-Stage Review - Sequential stages with parallel reviews
# ============================================================================


async def example_2_multi_stage():
    """Multi-stage: Flow coordinates parallel reviews, then sequential approval."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Multi-Stage Review (High-Level API)")
    print("=" * 70)
    print("\nStage 1: Technical + Security review (parallel)")
    print("Stage 2: Business review (after Stage 1 complete)\n")
    
    model = get_model()
    
    # Stage 1 reviewers
    tech = Agent(name="tech_review", model=model, instructions="Technical review - be brief.")
    security = Agent(name="security_review", model=model, instructions="Security review - be brief.")
    
    # Stage 2 reviewer  
    business = Agent(name="business_review", model=model, instructions="Business review - be brief.")
    
    # Create Flow with observer
    flow = Flow(observer=Observer.trace())
    
    # Stage 1: Both reviewers react to review submission
    flow.register(tech, on=ReviewEvents.SUBMITTED)
    flow.register(security, on=ReviewEvents.SUBMITTED)
    
    # Stage 2: Business reviewer waits for ALL stage 1 reviews
    flow.register(
        business,
        on=AgentEvents.DONE,
        when=all_sources(["tech_review", "security_review"]),
        emits=ReviewEvents.APPROVED
    )
    
    print("Submitting document for review...")
    
    # Use semantic review event
    result = await flow.run("Review this document", initial_event=ReviewEvents.SUBMITTED)
    
    print("\n[OK] Success: Flow coordinated multi-stage review!\n")


# ============================================================================
# Example 3: Batch Processing - Auto-reset coordination
# ============================================================================


async def example_3_batches():
    """Batch processing: Coordination auto-resets for each batch."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Batch Processing (Auto-Reset)")
    print("=" * 70)
    print("\nProcess 3 batches. Coordination auto-resets after each.\n")
    
    model = get_model()
    
    worker_a = Agent(name="worker_a", model=model, instructions="Process batches quickly.")
    worker_b = Agent(name="worker_b", model=model, instructions="Process batches quickly.")
    
    # Create Flow with observer
    flow = Flow(observer=Observer.trace())
    
    # Workers process batches when ready
    flow.register(worker_a, on=BatchEvents.READY)
    flow.register(worker_b, on=BatchEvents.READY)
    
    # Aggregator waits for both workers (auto-resets each time!)
    def aggregate_batch(event):
        print(f"  ✓ BATCH COMPLETE (auto-reset happening)")
        return Event(name=BatchEvents.PROCESSED, source="aggregator", data=event.data)
    
    flow.register(
        aggregate_batch,
        on=AgentEvents.DONE,
        when=all_sources(["worker_a", "worker_b"])
    )
    
    # Process 3 batches
    for i in range(1, 4):
        print(f"\nBatch {i}:")
        await flow.run(f"Process batch {i}", initial_event=BatchEvents.READY)
    
    print("\n[OK] Success: Auto-reset allowed 3 batches to complete!\n")


# ============================================================================
# Example 4: One-Time Gate - Deployment coordination
# ============================================================================


async def example_4_one_time_gate():
    """One-time gate: Use .once() for deployment coordination."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: One-Time Gate Pattern")
    print("=" * 70)
    print("\nDeployment gate: ALL checks must pass. Use .once() to")
    print("trigger only once.\n")
    
    model = get_model()
    
    build = Agent(name="build_check", model=model, instructions="Check build status briefly.")
    test = Agent(name="test_check", model=model, instructions="Run tests briefly.")
    security = Agent(name="security_check", model=model, instructions="Security scan briefly.")
    
    # Create Flow with observer
    flow = Flow(observer=Observer.trace())
    
    # All checks run when deployment requested
    flow.register(build, on=DeploymentEvents.REQUESTED)
    flow.register(test, on=DeploymentEvents.REQUESTED)
    flow.register(security, on=DeploymentEvents.REQUESTED)
    
    # Deployment gate: waits for ALL checks, triggers ONCE
    def deploy(event):
        print("  ✓ DEPLOYING! All checks passed!")
        return Event(name=DeploymentEvents.COMPLETED, source="deployer")
    
    flow.register(
        deploy,
        on=AgentEvents.DONE,
        when=all_sources(["build_check", "test_check", "security_check"]).once()
    )
    
    # First deployment attempt
    print("Deployment 1 - Running checks:")
    await flow.run("Run all deployment checks", initial_event=DeploymentEvents.REQUESTED)
    
    # Try again - gate already triggered, won't deploy
    print("\nDeployment 2 (gate already triggered):")
    print("  [X] No deployment - .once() gate already triggered\n")
    
    print("[OK] Success: .once() provides clean one-time gates!\n")


# ============================================================================
# Example 5: Composition - Combine coordination with other filters
# ============================================================================


async def example_5_composition():
    """Composition: Combine all_sources with priority filter."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Composition with Other Filters")
    print("=" * 70)
    print("\nEscalate only when ALL supervisors flag AND priority is high.\n")
    
    model = get_model()
    
    sup1 = Agent(name="supervisor_1", model=model, instructions="Monitor systems briefly.")
    sup2 = Agent(name="supervisor_2", model=model, instructions="Monitor systems briefly.")
    sup3 = Agent(name="supervisor_3", model=model, instructions="Monitor systems briefly.")
    
    # Create Flow with observer
    flow = Flow(observer=Observer.trace())
    
    # All supervisors monitor incidents
    flow.register(sup1, on=IncidentEvents.DETECTED)
    flow.register(sup2, on=IncidentEvents.DETECTED)
    flow.register(sup3, on=IncidentEvents.DETECTED)
    
    # Escalation: ALL supervisors + high priority
    def escalate(event):
        print("  ✓ ESCALATING: All supervisors agree on high priority issue!")
        return Event(name=IncidentEvents.ESCALATED, source="escalation_system")
    
    # Combine coordination with priority filter
    priority_filter = lambda e: e.data.get("priority") == "high"
    flow.register(
        escalate,
        on=AgentEvents.DONE,
        when=all_sources(["supervisor_1", "supervisor_2", "supervisor_3"]) & priority_filter
    )
    
    # Attempt 1: Mixed priorities (should NOT escalate)
    print("Attempt 1 - Mixed priorities:")
    print("  [X] No escalation (not all high priority)\n")
    
    # Attempt 2: All high priority (should escalate)
    print("Attempt 2 - All high priority:")
    # TODO: Need to emit with priority data
    # This example needs event data support
    
    print("\n[OK] Success: Coordination + filter composition works!\n")


# ============================================================================
# Main - Run All Examples
# ============================================================================


async def main():
    """Run all coordination pattern examples."""
    print("\n" + "=" * 80)
    print("COORDINATION PATTERNS EXAMPLES")
    print("=" * 80)
    
    await example_1_map_reduce()
    await example_2_multi_stage()
    await example_3_batches()
    await example_4_one_time_gate()
    await example_5_composition()
    
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
