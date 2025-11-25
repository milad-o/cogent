"""Topology factory for easy instantiation.

Create topologies by type with sensible defaults.
"""

from enum import Enum
from typing import Any, Sequence

from langgraph.checkpoint.memory import MemorySaver

from agenticflow.agents import Agent
from agenticflow.events import EventBus
from agenticflow.memory import MemoryManager
from agenticflow.topologies.base import BaseTopology, TopologyConfig
from agenticflow.topologies.supervisor import SupervisorTopology
from agenticflow.topologies.mesh import MeshTopology
from agenticflow.topologies.pipeline import PipelineTopology
from agenticflow.topologies.hierarchical import HierarchicalTopology


class TopologyType(Enum):
    """Available topology patterns."""

    SUPERVISOR = "supervisor"
    MESH = "mesh"
    PIPELINE = "pipeline"
    HIERARCHICAL = "hierarchical"


class TopologyFactory:
    """Factory for creating topology instances.

    Simplifies topology creation with sensible defaults
    and automatic checkpointer setup.

    Example:
        >>> agents = [agent1, agent2, agent3]
        >>> topology = TopologyFactory.create(
        ...     TopologyType.SUPERVISOR,
        ...     "my-team",
        ...     agents,
        ...     supervisor_name="agent1",
        ... )
        >>> result = await topology.run("Do something")
    """

    _registry: dict[TopologyType, type[BaseTopology]] = {
        TopologyType.SUPERVISOR: SupervisorTopology,
        TopologyType.MESH: MeshTopology,
        TopologyType.PIPELINE: PipelineTopology,
        TopologyType.HIERARCHICAL: HierarchicalTopology,
    }

    @classmethod
    def create(
        cls,
        topology_type: TopologyType,
        name: str,
        agents: Sequence[Agent],
        *,
        description: str = "",
        event_bus: EventBus | None = None,
        memory_manager: MemoryManager | None = None,
        enable_checkpointing: bool = True,
        **kwargs: Any,
    ) -> BaseTopology:
        """Create a topology of the specified type.

        Args:
            topology_type: Type of topology to create.
            name: Name for the topology.
            agents: List of agents to include.
            description: Optional description.
            event_bus: Optional event bus for events.
            memory_manager: Optional memory manager.
            enable_checkpointing: Whether to enable state checkpointing.
            **kwargs: Additional arguments for specific topology types.

        Returns:
            Configured topology instance.

        Raises:
            ValueError: If topology type is not registered.
        """
        if topology_type not in cls._registry:
            raise ValueError(f"Unknown topology type: {topology_type}")

        topology_class = cls._registry[topology_type]

        # Create config
        config = TopologyConfig(
            name=name,
            description=description,
            enable_checkpointing=enable_checkpointing,
            **{k: v for k, v in kwargs.items() if k in TopologyConfig.__dataclass_fields__},
        )

        # Setup checkpointer
        checkpointer = None
        if enable_checkpointing:
            checkpointer = MemorySaver()

        # Filter kwargs for topology-specific params
        topology_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in TopologyConfig.__dataclass_fields__
        }

        return topology_class(
            config=config,
            agents=agents,
            event_bus=event_bus,
            memory_manager=memory_manager,
            checkpointer=checkpointer,
            **topology_kwargs,
        )

    @classmethod
    def register(
        cls,
        topology_type: TopologyType,
        topology_class: type[BaseTopology],
    ) -> None:
        """Register a custom topology type.

        Args:
            topology_type: Type identifier for the topology.
            topology_class: Topology class to register.
        """
        cls._registry[topology_type] = topology_class

    @classmethod
    def available_types(cls) -> list[TopologyType]:
        """Get list of available topology types.

        Returns:
            List of registered topology types.
        """
        return list(cls._registry.keys())

    @classmethod
    def quick_supervisor(
        cls,
        name: str,
        supervisor: Agent,
        workers: Sequence[Agent],
        **kwargs: Any,
    ) -> SupervisorTopology:
        """Quick helper to create supervisor topology.

        Args:
            name: Topology name.
            supervisor: The supervisor agent.
            workers: Worker agents.
            **kwargs: Additional options.

        Returns:
            Configured supervisor topology.
        """
        return cls.create(
            TopologyType.SUPERVISOR,
            name,
            [supervisor, *workers],
            supervisor_name=supervisor.config.name,
            **kwargs,
        )  # type: ignore

    @classmethod
    def quick_pipeline(
        cls,
        name: str,
        stages: Sequence[Agent],
        **kwargs: Any,
    ) -> PipelineTopology:
        """Quick helper to create pipeline topology.

        Args:
            name: Topology name.
            stages: Agents in pipeline order.
            **kwargs: Additional options.

        Returns:
            Configured pipeline topology.
        """
        return cls.create(
            TopologyType.PIPELINE,
            name,
            stages,
            stages=[a.config.name for a in stages],
            **kwargs,
        )  # type: ignore

    @classmethod
    def quick_mesh(
        cls,
        name: str,
        agents: Sequence[Agent],
        **kwargs: Any,
    ) -> MeshTopology:
        """Quick helper to create mesh topology.

        Args:
            name: Topology name.
            agents: Peer agents.
            **kwargs: Additional options.

        Returns:
            Configured mesh topology.
        """
        return cls.create(
            TopologyType.MESH,
            name,
            agents,
            **kwargs,
        )  # type: ignore
