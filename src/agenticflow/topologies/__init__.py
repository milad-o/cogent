"""Multi-agent topology patterns.

This module provides extensible multi-agent coordination patterns
built on LangGraph's StateGraph and Command primitives.

Topologies are defined by policies that specify handoff rules:
- Who can send tasks to whom
- Who accepts tasks from whom
- Under what conditions handoffs occur

Example:
    >>> from agenticflow.topologies import BaseTopology, TopologyConfig, TopologyPolicy
    >>>
    >>> # Simple pipeline with policy
    >>> policy = TopologyPolicy.pipeline(["researcher", "writer", "reviewer"])
    >>> topology = BaseTopology(
    ...     config=TopologyConfig(name="content-pipeline"),
    ...     agents=agents,
    ...     policy=policy,
    ... )
    >>>
    >>> # Or use convenience classes
    >>> topology = PipelineTopology(
    ...     config=TopologyConfig(name="content-pipeline"),
    ...     agents=agents,
    ... )
"""

from agenticflow.topologies.base import (
    BaseTopology,
    TopologyConfig,
    TopologyState,
    HandoffStrategy,
    # Convenience classes
    SupervisorTopology,
    PipelineTopology,
    MeshTopology,
    HierarchicalTopology,
)
from agenticflow.topologies.policies import (
    TopologyPolicy,
    AgentPolicy,
    HandoffRule,
    HandoffCondition,
    AcceptancePolicy,
    ExecutionMode,
)
from agenticflow.topologies.custom import CustomTopology, CustomTopologyConfig, Edge
from agenticflow.topologies.factory import TopologyFactory, TopologyType
from agenticflow.topologies.builder import (
    # Enums
    TopologyPattern,
    DelegationStrategy,
    CompletionCondition,
    # Policies
    DelegationPolicy,
    EventHooks,
    # Main spec
    TopologySpec,
    # Factory functions (preferred API)
    supervisor_topology,
    coordinator_topology,
    pipeline_topology,
    mesh_topology,
    hierarchical_topology,
    # Validation
    validate_roles,
    TOPOLOGY_ROLE_REQUIREMENTS,
)

__all__ = [
    # Base
    "BaseTopology",
    "TopologyConfig",
    "TopologyState",
    "HandoffStrategy",
    # Policies
    "TopologyPolicy",
    "AgentPolicy",
    "HandoffRule",
    "HandoffCondition",
    "AcceptancePolicy",
    "ExecutionMode",
    # Convenience classes
    "SupervisorTopology",
    "MeshTopology",
    "PipelineTopology",
    "HierarchicalTopology",
    # Custom topology (explicit edges)
    "CustomTopology",
    "CustomTopologyConfig",
    "Edge",
    # Factory (old API)
    "TopologyFactory",
    "TopologyType",
    # New API - Enums
    "TopologyPattern",
    "DelegationStrategy",
    "CompletionCondition",
    # New API - Policies
    "DelegationPolicy",
    "EventHooks",
    # New API - Main spec
    "TopologySpec",
    # New API - Factory functions (preferred)
    "supervisor_topology",
    "coordinator_topology",
    "pipeline_topology",
    "mesh_topology",
    "hierarchical_topology",
    # New API - Validation
    "validate_roles",
    "TOPOLOGY_ROLE_REQUIREMENTS",
]
