"""
Visualization Example
=====================

Demonstrates how to generate Mermaid diagrams for agents and topologies.
Shows agents with their tools and topology coordination patterns.

Run:
    uv run python examples/visualization_example.py
"""

from agenticflow import (
    Agent,
    AgentConfig,
    AgentRole,
    EventBus,
    TopologyFactory,
    TopologyType,
)


def main() -> None:
    """Demonstrate Mermaid diagram generation."""
    print("=" * 60)
    print("AgenticFlow Visualization Example")
    print("=" * 60)

    # Create event bus (required for agents)
    event_bus = EventBus()

    # Create agents with tools
    researcher = Agent(
        config=AgentConfig(
            name="Researcher",
            role=AgentRole.RESEARCHER,
            description="Researches topics and gathers information",
            model_name="gpt-4o",
            tools=["web_search", "document_reader", "summarize"],
        ),
        event_bus=event_bus,
    )

    writer = Agent(
        config=AgentConfig(
            name="Writer",
            role=AgentRole.WORKER,
            description="Writes content based on research",
            model_name="gpt-4o",
            tools=["text_generator", "grammar_checker"],
        ),
        event_bus=event_bus,
    )

    editor = Agent(
        config=AgentConfig(
            name="Editor",
            role=AgentRole.CRITIC,
            description="Reviews and improves content",
            model_name="gpt-4o",
            tools=["style_checker", "plagiarism_detector"],
        ),
        event_bus=event_bus,
    )

    supervisor = Agent(
        config=AgentConfig(
            name="Supervisor",
            role=AgentRole.ORCHESTRATOR,
            description="Coordinates the content team",
            model_name="gpt-4o",
            tools=["task_scheduler", "progress_tracker"],
        ),
        event_bus=event_bus,
    )

    # ============================================================
    # Example 1: Single Agent Diagram
    # ============================================================
    print("\n" + "=" * 60)
    print("1. Single Agent Diagram")
    print("=" * 60)

    print("\nResearcher Agent (with tools):")
    print("-" * 40)
    mermaid_code = researcher.draw_mermaid(title="Research Agent")
    print(mermaid_code)

    print("\n\nResearcher Agent (with config details):")
    print("-" * 40)
    mermaid_code = researcher.draw_mermaid(
        title="Research Agent Details",
        show_config=True,
        theme="forest",
    )
    print(mermaid_code)

    # ============================================================
    # Example 2: Supervisor Topology Diagram
    # ============================================================
    print("\n" + "=" * 60)
    print("2. Supervisor Topology Diagram")
    print("=" * 60)

    supervisor_topology = TopologyFactory.create(
        TopologyType.SUPERVISOR,
        "content-team",
        agents=[supervisor, researcher, writer, editor],
        supervisor_name="Supervisor",
    )

    print("\nSupervisor Topology:")
    print("-" * 40)
    mermaid_code = supervisor_topology.draw_mermaid(
        title="Content Production Team",
        show_tools=True,
    )
    print(mermaid_code)

    # ============================================================
    # Example 3: Pipeline Topology Diagram
    # ============================================================
    print("\n" + "=" * 60)
    print("3. Pipeline Topology Diagram")
    print("=" * 60)

    pipeline_topology = TopologyFactory.create(
        TopologyType.PIPELINE,
        "content-pipeline",
        agents=[researcher, writer, editor],
    )

    print("\nPipeline Topology:")
    print("-" * 40)
    mermaid_code = pipeline_topology.draw_mermaid(
        title="Content Production Pipeline",
        direction="LR",
        show_tools=True,
    )
    print(mermaid_code)

    # ============================================================
    # Example 4: Mesh Topology Diagram
    # ============================================================
    print("\n" + "=" * 60)
    print("4. Mesh Topology Diagram")
    print("=" * 60)

    mesh_topology = TopologyFactory.create(
        TopologyType.MESH,
        "collaboration-mesh",
        agents=[researcher, writer, editor],
    )

    print("\nMesh Topology:")
    print("-" * 40)
    mermaid_code = mesh_topology.draw_mermaid(
        title="Collaborative Mesh Network",
        show_tools=False,  # Cleaner view
    )
    print(mermaid_code)

    # ============================================================
    # Example 5: Different Themes
    # ============================================================
    print("\n" + "=" * 60)
    print("5. Different Themes")
    print("=" * 60)

    themes = ["default", "dark", "forest", "neutral"]
    for theme in themes:
        print(f"\nTheme: {theme}")
        print("-" * 40)
        mermaid_code = researcher.draw_mermaid(
            title=f"Agent with {theme.capitalize()} Theme",
            theme=theme,
            show_tools=True,
        )
        # Just show the frontmatter for brevity
        lines = mermaid_code.split("\n")
        frontmatter = "\n".join(lines[:10])
        print(frontmatter)
        print("...")

    # ============================================================
    # Example 6: Using Diagram Classes Directly
    # ============================================================
    print("\n" + "=" * 60)
    print("6. Using Diagram Classes Directly")
    print("=" * 60)

    from agenticflow.visualization import (
        AgentDiagram,
        MermaidConfig,
        MermaidDirection,
        MermaidTheme,
        TopologyDiagram,
    )

    config = MermaidConfig(
        title="Custom Configuration",
        theme=MermaidTheme.FOREST,
        direction=MermaidDirection.LEFT_RIGHT,
        node_spacing=80,
        rank_spacing=100,
        show_tools=True,
        show_roles=True,
    )

    diagram = AgentDiagram(writer, config=config)
    print("\nCustom Agent Diagram:")
    print("-" * 40)
    print(diagram.to_mermaid())

    print("\nSVG URL for rendering:")
    print("-" * 40)
    print(diagram.get_svg_url()[:100] + "...")

    # ============================================================
    # Example 7: HTML Output (for notebooks/web)
    # ============================================================
    print("\n" + "=" * 60)
    print("7. HTML Output (for notebooks/web)")
    print("=" * 60)

    html = diagram.to_html()
    print("\nHTML snippet (truncated):")
    print("-" * 40)
    print(html[:300])
    print("...")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
The visualization module provides:

1. Agent.draw_mermaid() - Generate Mermaid code for an agent
2. Agent.draw_mermaid_png() - Generate PNG image (requires httpx)
3. BaseTopology.draw_mermaid() - Generate Mermaid code for a topology
4. BaseTopology.draw_mermaid_png() - Generate PNG image

Features:
- YAML frontmatter configuration (modern approach)
- Multiple themes: default, dark, forest, neutral, base
- Flexible direction: TB, TD, BT, LR, RL
- Show/hide tools, roles, and config
- HTML output for web/notebook embedding
- URL generation for mermaid.ink rendering

Copy any Mermaid code to:
- https://mermaid.live (live editor)
- GitHub/GitLab markdown
- Notion, Obsidian, or other markdown tools
""")


if __name__ == "__main__":
    main()
