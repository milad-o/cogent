"""Multi-agent topology patterns.

This module provides extensible multi-agent coordination patterns
built on LangGraph's StateGraph and Command primitives.
"""

from agenticflow.topologies.base import (
    BaseTopology,
    TopologyConfig,
    TopologyState,
)
from agenticflow.topologies.supervisor import SupervisorTopology
from agenticflow.topologies.mesh import MeshTopology
from agenticflow.topologies.pipeline import PipelineTopology
from agenticflow.topologies.hierarchical import HierarchicalTopology
from agenticflow.topologies.factory import TopologyFactory, TopologyType

__all__ = [
    # Base
    "BaseTopology",
    "TopologyConfig",
    "TopologyState",
    # Patterns
    "SupervisorTopology",
    "MeshTopology",
    "PipelineTopology",
    "HierarchicalTopology",
    # Factory
    "TopologyFactory",
    "TopologyType",
]
