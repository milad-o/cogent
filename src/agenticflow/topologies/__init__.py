"""Topologies - Multi-agent coordination patterns.

This module provides simple, native coordination patterns for multi-agent workflows:

- **Supervisor**: One agent coordinates and delegates to workers
- **Pipeline**: Sequential processing A → B → C  
- **Mesh**: All agents collaborate in rounds until consensus
- **Hierarchical**: Tree structure with delegation levels

Quick Start:
    >>> from agenticflow import Agent, ChatModel
    >>> from agenticflow.topologies import Supervisor, Pipeline, Mesh, AgentConfig
    >>>
    >>> model = ChatModel(provider="openai", model="gpt-4o-mini")
    >>> researcher = Agent(name="researcher", model=model)
    >>> writer = Agent(name="writer", model=model)
    >>> editor = Agent(name="editor", model=model)
    >>>
    >>> # Supervisor pattern
    >>> topology = Supervisor(
    ...     coordinator=AgentConfig(agent=researcher, role="coordinator"),
    ...     workers=[
    ...         AgentConfig(agent=writer, role="content writer"),
    ...         AgentConfig(agent=editor, role="editor"),
    ...     ]
    ... )
    >>> result = await topology.run("Write a blog post about AI")
    >>>
    >>> # Pipeline pattern  
    >>> topology = Pipeline(stages=[
    ...     AgentConfig(agent=researcher, role="research"),
    ...     AgentConfig(agent=writer, role="draft"),
    ...     AgentConfig(agent=editor, role="polish"),
    ... ])
    >>> result = await topology.run("Create technical documentation")
    >>>
    >>> # Mesh collaboration
    >>> topology = Mesh(
    ...     agents=[
    ...         AgentConfig(agent=researcher, role="business analyst"),
    ...         AgentConfig(agent=writer, role="technical analyst"),
    ...     ],
    ...     max_rounds=2,
    ... )
    >>> result = await topology.run("Evaluate this product idea")

With TeamMemory:
    Share state between agents during execution:
    
    >>> from agenticflow.memory import TeamMemory
    >>> team_memory = TeamMemory(team_id="content-team")
    >>> result = await topology.run("Write article", team_memory=team_memory)
    >>> # Check agent statuses and shared results
    >>> statuses = await team_memory.get_agent_statuses()
    >>> results = await team_memory.get_agent_results()

Convenience Functions:
    For quick setup without AgentConfig boilerplate:
    
    >>> from agenticflow.topologies import supervisor, pipeline, mesh
    >>>
    >>> topology = supervisor(coordinator=researcher, workers=[writer, editor])
    >>> topology = pipeline(stages=[researcher, writer, editor])
    >>> topology = mesh(agents=[analyst1, analyst2], max_rounds=2)
"""

from .core import AgentConfig, BaseTopology, TopologyConfig, TopologyResult, TopologyType
from .patterns import Hierarchical, Mesh, Pipeline, Supervisor, mesh, pipeline, supervisor
from .context import (
    ContextStrategy,
    SlidingWindowStrategy,
    SummarizationStrategy,
    RetrievalStrategy,
    StructuredHandoffStrategy,
    StructuredHandoff,
    BlackboardStrategy,
    CompositeStrategy,
)

__all__ = [
    # Core classes
    "AgentConfig",
    "BaseTopology",
    "TopologyConfig",
    "TopologyResult",
    "TopologyType",
    # Pattern classes
    "Supervisor",
    "Pipeline",
    "Mesh",
    "Hierarchical",
    # Convenience functions
    "supervisor",
    "pipeline",
    "mesh",
    # Context strategies
    "ContextStrategy",
    "SlidingWindowStrategy",
    "SummarizationStrategy",
    "RetrievalStrategy",
    "StructuredHandoffStrategy",
    "StructuredHandoff",
    "BlackboardStrategy",
    "CompositeStrategy",
]

