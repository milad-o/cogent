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
)
from agenticflow.topologies.custom import CustomTopology, CustomTopologyConfig, Edge
from agenticflow.topologies.factory import TopologyFactory, TopologyType

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
    # Convenience classes
    "SupervisorTopology",
    "MeshTopology",
    "PipelineTopology",
    "HierarchicalTopology",
    # Custom topology (explicit edges)
    "CustomTopology",
    "CustomTopologyConfig",
    "Edge",
    # Factory
    "TopologyFactory",
    "TopologyType",
]
