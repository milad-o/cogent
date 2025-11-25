#!/usr/bin/env python3
"""Example: Policy-based topologies.

Demonstrates how easy it is to define custom topologies using policies.
"""

from agenticflow.agents import Agent, AgentConfig
from agenticflow.core.enums import AgentRole
from agenticflow.topologies import (
    BaseTopology,
    TopologyConfig,
    TopologyPolicy,
    HandoffRule,
    HandoffCondition,
    AgentPolicy,
    AcceptancePolicy,
    # Convenience classes
    SupervisorTopology,
    PipelineTopology,
    MeshTopology,
    HierarchicalTopology,
)


def create_mock_agents():
    """Create mock agents for examples."""
    from unittest.mock import MagicMock
    
    agents = {}
    configs = [
        ("gateway", AgentRole.ORCHESTRATOR, ["routing", "auth"]),
        ("validator", AgentRole.VALIDATOR, ["schema_check", "sanitize"]),
        ("processor", AgentRole.WORKER, ["transform", "enrich"]),
        ("storage", AgentRole.WORKER, ["db_write", "cache"]),
        ("notifier", AgentRole.WORKER, ["email", "slack"]),
    ]
    
    for name, role, tools in configs:
        config = AgentConfig(name=name, role=role, tools=tools)
        agent = MagicMock(spec=Agent)
        agent.config = config
        agent.name = name
        agent.role = role
        agents[name] = agent
    
    return agents


def example_1_prebuilt_topologies():
    """Example 1: Using prebuilt topology convenience classes."""
    print("\n=== Example 1: Prebuilt Topologies ===\n")
    
    agents = create_mock_agents()
    agent_list = list(agents.values())
    
    # Supervisor topology
    supervisor = SupervisorTopology(
        config=TopologyConfig(name="data-team"),
        agents=agent_list,
        supervisor_name="gateway",
    )
    print("Supervisor Topology:")
    print(f"  Supervisor: {supervisor.supervisor_name}")
    print(f"  Workers: {supervisor.worker_names}")
    print(f"  Policy allows gateway->validator: {supervisor.policy.can_handoff('gateway', 'validator', {})}")
    
    # Pipeline topology
    pipeline = PipelineTopology(
        config=TopologyConfig(name="etl-pipeline"),
        agents=agent_list[:4],
        stages=["gateway", "validator", "processor", "storage"],
    )
    print("\nPipeline Topology:")
    print(f"  Stages: {pipeline.stages}")
    print(f"  Policy allows validator->processor: {pipeline.policy.can_handoff('validator', 'processor', {})}")
    print(f"  Policy allows processor->validator: {pipeline.policy.can_handoff('processor', 'validator', {})}")  # Should be False


def example_2_custom_policy():
    """Example 2: Creating a custom topology with explicit policy."""
    print("\n=== Example 2: Custom Policy ===\n")
    
    agents = create_mock_agents()
    agent_list = list(agents.values())
    
    # Create custom policy
    policy = TopologyPolicy(entry_point="gateway")
    
    # Gateway can send to validator
    policy.add_rule("gateway", "validator", label="validate")
    
    # Validator can send to processor (success) or back to gateway (failure)
    policy.add_rule("validator", "processor", 
                    condition=HandoffCondition.ON_SUCCESS, 
                    label="valid")
    policy.add_rule("validator", "gateway", 
                    condition=HandoffCondition.ON_FAILURE, 
                    label="retry")
    
    # Processor sends to storage
    policy.add_rule("processor", "storage", label="save")
    
    # Storage sends to notifier
    policy.add_rule("storage", "notifier", label="notify")
    
    # Set agent policies (who can finish the workflow)
    policy.add_agent_policy(AgentPolicy("gateway", can_finish=False))
    policy.add_agent_policy(AgentPolicy("validator", can_finish=False))
    policy.add_agent_policy(AgentPolicy("processor", can_finish=False))
    policy.add_agent_policy(AgentPolicy("storage", can_finish=False))
    policy.add_agent_policy(AgentPolicy("notifier", can_finish=True))
    
    # Create topology with policy
    topology = BaseTopology(
        config=TopologyConfig(name="data-pipeline"),
        agents=agent_list,
        policy=policy,
    )
    
    print("Custom Policy Topology:")
    print(f"  Entry point: {policy.entry_point}")
    print(f"  Rules:")
    for rule in policy.rules:
        cond = f" ({rule.condition.value})" if rule.condition != HandoffCondition.ALWAYS else ""
        print(f"    {rule.source} -> {rule.target}{cond}: {rule.label}")
    
    # Get edges for diagram
    edges = policy.get_edges_for_diagram(list(agents.keys()))
    print(f"\n  Edges for diagram: {edges}")


def example_3_acceptance_policies():
    """Example 3: Controlling who accepts tasks from whom."""
    print("\n=== Example 3: Acceptance Policies ===\n")
    
    agents = create_mock_agents()
    agent_list = list(agents.values())
    
    # Create policy where processor only accepts from validator
    policy = TopologyPolicy(entry_point="gateway")
    
    # Allow all connections
    policy.add_rule("*", "*")
    
    # But processor only accepts from validator
    policy.add_agent_policy(AgentPolicy(
        agent_name="processor",
        acceptance=AcceptancePolicy.ACCEPT_LISTED,
        accept_from=["validator"],
    ))
    
    print("Acceptance Policy Test:")
    print(f"  gateway -> processor allowed: {policy.can_handoff('gateway', 'processor', {})}")  # False
    print(f"  validator -> processor allowed: {policy.can_handoff('validator', 'processor', {})}")  # True


def example_4_factory_methods():
    """Example 4: Using policy factory methods."""
    print("\n=== Example 4: Factory Methods ===\n")
    
    # Supervisor policy
    sup_policy = TopologyPolicy.supervisor(
        supervisor="manager",
        workers=["dev1", "dev2", "dev3"],
    )
    print("Supervisor Policy:")
    print(f"  Entry: {sup_policy.entry_point}")
    print(f"  manager->dev1: {sup_policy.can_handoff('manager', 'dev1', {})}")
    print(f"  dev1->dev2: {sup_policy.can_handoff('dev1', 'dev2', {})}")  # False (workers don't talk to each other)
    
    # Pipeline policy
    pipe_policy = TopologyPolicy.pipeline(
        stages=["fetch", "parse", "store"],
        allow_skip=False,
        allow_repeat=True,  # Can retry
    )
    print("\nPipeline Policy (with retry):")
    print(f"  fetch->parse: {pipe_policy.can_handoff('fetch', 'parse', {})}")
    print(f"  parse->fetch: {pipe_policy.can_handoff('parse', 'fetch', {})}")  # True (retry allowed)
    
    # Hierarchical policy
    hier_policy = TopologyPolicy.hierarchical(
        levels=[
            ["ceo"],
            ["cto", "cfo"],
            ["dev1", "dev2", "accountant"],
        ]
    )
    print("\nHierarchical Policy:")
    print(f"  ceo->cto: {hier_policy.can_handoff('ceo', 'cto', {})}")
    print(f"  cto->dev1: {hier_policy.can_handoff('cto', 'dev1', {})}")
    print(f"  dev1->ceo: {hier_policy.can_handoff('dev1', 'ceo', {})}")  # False (skip levels)
    print(f"  dev1->cto: {hier_policy.can_handoff('dev1', 'cto', {})}")  # True (report to superior)


def example_5_mermaid_diagram():
    """Example 5: Generate Mermaid diagram from policy."""
    print("\n=== Example 5: Mermaid Diagram ===\n")
    
    agents = create_mock_agents()
    agent_list = list(agents.values())[:4]  # gateway, validator, processor, storage
    
    # Custom policy with labels
    policy = TopologyPolicy(entry_point="gateway")
    policy.add_rule("gateway", "validator", label="validate")
    policy.add_rule("validator", "processor", label="valid")
    policy.add_rule("validator", "gateway", label="invalid")
    policy.add_rule("processor", "storage", label="save")
    
    topology = BaseTopology(
        config=TopologyConfig(name="data-pipeline"),
        agents=agent_list,
        policy=policy,
    )
    
    # Generate diagram
    mermaid = topology.draw_mermaid(title="Data Pipeline", show_tools=True)
    print("Mermaid Diagram:")
    print(mermaid)


if __name__ == "__main__":
    example_1_prebuilt_topologies()
    example_2_custom_policy()
    example_3_acceptance_policies()
    example_4_factory_methods()
    example_5_mermaid_diagram()
