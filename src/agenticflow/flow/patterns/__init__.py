"""Flow patterns - pre-configured orchestration topologies.

Patterns are helper functions that configure a Flow for common
multi-agent coordination scenarios.

Available patterns:
- **pipeline**: Sequential processing (A → B → C)
- **supervisor**: Coordinator delegates to workers
- **mesh**: Collaborative multi-agent discussion
- **fanout**: Parallel processing with aggregation
- **saga**: Multi-step workflows with compensation

Example:
    ```python
    from agenticflow.flow import pipeline, supervisor, mesh

    # Sequential pipeline
    flow = pipeline([researcher, writer, editor])

    # Coordinator with workers
    flow = supervisor(manager, [analyst, developer, tester])

    # Collaborative discussion
    flow = mesh([expert1, expert2, expert3], max_rounds=3)
    ```
"""

from agenticflow.flow.patterns.mesh import (
    brainstorm,
    collaborative,
    mesh,
)
from agenticflow.flow.patterns.pipeline import (
    chain,
    pipeline,
)
from agenticflow.flow.patterns.supervisor import (
    coordinator,
    supervisor,
)

__all__ = [
    # Pipeline
    "pipeline",
    "chain",
    # Supervisor
    "supervisor",
    "coordinator",
    # Mesh
    "mesh",
    "collaborative",
    "brainstorm",
]
