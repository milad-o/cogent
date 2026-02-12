#!/usr/bin/env python3
"""
Demo: Knowledge Graph Capability

Shows how to use the KnowledgeGraph capability to give an agent
persistent memory of entities and relationships. The agent can
drill down through the graph to find answers.

Key features demonstrated:
1. Load knowledge from a data file
2. Visualize with Mermaid (static SVG)
3. Visualize with PyVis (interactive HTML)
4. Agent uses KG tools to explore and find answers
5. Multi-hop reasoning through relationships
6. Save/load for persistence
7. Multiple storage backends (memory, sqlite, json)
8. Clean programmatic API for direct access
"""

import asyncio
import tempfile
from pathlib import Path


async def demo():
    from cogent import Agent
    from cogent.capabilities import KnowledgeGraph

    def response_text(value: object) -> str:
        text = getattr(value, "content", value)
        return str(text).strip()

    print("=" * 60)
    print("🧠 Knowledge Graph Capability Demo")
    print("=" * 60)

    # === Step 1: Agent generates KnowledgeGraph from file ===
    print("\n🤖 Step 1: Agent Generates KnowledgeGraph")
    print("-" * 40)

    kg = KnowledgeGraph()
    data_file = Path(__file__).parent.parent / "data" / "company_knowledge.txt"
    data_text = data_file.read_text()

    agent = Agent(
        name="CompanyExpert",
        model="gpt4",
        instructions="""You are a company knowledge expert. You have access to a knowledge graph
containing information about employees, teams, projects, and technologies.

Use the available tools to explore the knowledge graph and find answers.
When asked about relationships or connections, drill down through the graph
to trace the path and provide complete answers.""",
        capabilities=[kg],
    )

    print(f"Agent tools: {[t.name for t in agent.all_tools]}")

    ingest_prompt = f"""Load the following dataset into the knowledge graph.

Rules:
- Each line is either:
  - entity|name|type|{{\"attr\": \"value\"}}
  - rel|source|relation|target
- Ignore empty lines and lines starting with #.
- Use the tools only: `kg_remember` for entities, `kg_connect` for relationships.
- If attributes JSON is missing, pass an empty dict.
- After processing, reply with a short confirmation only.

Dataset:
{data_text}
"""

    response = await agent.run(ingest_prompt)
    print(f"💡 {response_text(response)}")

    print(f"✅ Loaded from {data_file.name}:")
    print(f"   Graph stats: {kg.stats()}")

    # Create output directory for all visualizations
    output_dir = Path(__file__).parent / "kg_outputs"
    output_dir.mkdir(exist_ok=True)

    # === Step 2: Visualize the inferred graph ===
    print("\n🗺️  Step 2: Knowledge Graph (Mermaid)")
    print("-" * 40)
    mermaid_code = kg.mermaid(direction="LR", group_by_type=True, max_entities=50)
    print(mermaid_code)

    svg_path = output_dir / "knowledge_graph.svg"
    print(f"\n🖼️  Saving SVG to: {output_dir.name}/knowledge_graph.svg")
    print("    (Requires Mermaid CLI: npm install -g @mermaid-js/mermaid-cli)")
    from cogent.graph.visualization import render_mermaid_to_image

    await render_mermaid_to_image(mermaid_code, str(svg_path), format="svg")

    # === Step 3: Interactive visualization (PyVis) ===
    print("\n🌐 Step 3: Interactive Visualization (PyVis)")
    print("-" * 40)
    html_path = kg.interactive(
        output_path=output_dir / "knowledge_graph.html",
        height="750px",
        width="100%",
        color_by_type=True,  # Color nodes by entity type
        show_type_in_label=True,  # Show type in labels
        relationship_color="#2B7CE9",
        max_entities=50,
    )
    print(f"✅ Interactive HTML saved to: {output_dir.name}/knowledge_graph.html")
    print(f"   - Open in browser: file://{html_path.absolute()}")

    # === Step 4: Agent drills down to find answers ===
    print("\n💬 Step 4: Agent Queries (Drill-down)")
    print("-" * 40)

    questions = [
        "Who is working on the ETL Pipeline project and what technologies does it use?",
        "Who does David Lee report to, and what team does that person lead?",
        "What Python experts do we have and what projects are they working on?",
    ]

    for q in questions:
        print(f"\n❓ {q}")
        response = await agent.run(q)
        print(f"💡 {response_text(response)}")

    # === Step 6: Agent updates knowledge ===
    print("\n➕ Step 6: Agent Adds Knowledge")
    print("-" * 40)

    response = await agent.run(
        "Remember that Frank Martinez is a new DevOps Engineer who joined Platform Team and is expert in Kubernetes",
    )
    print(f"Response: {response_text(response)}")

    # Verify Frank was added
    frank = kg.get_entity("Frank Martinez")
    if frank:
        print(f"\n✅ Frank added: {frank.type}, attrs={frank.attributes}")
        rels = kg.get_relationships("Frank Martinez", direction="outgoing")
        print(f"   Relationships: {[(r.relation, r.target_id) for r in rels]}")

    # === Step 6: Persistence Demo ===
    print("\n💾 Step 6: Persistence Demo")
    print("-" * 40)

    # Demo 1: Save memory graph to file
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "knowledge_backup.json"
        kg.save(save_path)
        print(f"✅ Saved to: {save_path}")
        print(f"   File size: {save_path.stat().st_size} bytes")

        # Create new graph and load
        from cogent.capabilities import KnowledgeGraph

        kg2 = KnowledgeGraph(backend="memory")
        kg2.load(save_path)
        print(f"✅ Loaded into new graph: {kg2.stats()}")

        # Demo 2: SQLite backend (persistent by default)
        db_path = Path(tmpdir) / "knowledge.db"
        with KnowledgeGraph(backend="sqlite", path=db_path) as kg_sqlite:
            kg_sqlite.graph.add_entity(
                "Test", "Demo", {"note": "SQLite persists automatically"}
            )
            print("\n✅ SQLite backend:")
            print(f"   Path: {db_path}")
            print(f"   Stats: {kg_sqlite.stats()}")

        # Reopen to verify persistence
        with KnowledgeGraph(backend="sqlite", path=db_path) as kg_sqlite2:
            test = kg_sqlite2.get_entity("Test")
            print(
                f"   Reloaded: {test.id if test else 'Not found'} - {test.attributes if test else {}}"
            )

        # Demo 3: JSON backend (auto-saves)
        json_path = Path(tmpdir) / "knowledge.json"
        kg_json = KnowledgeGraph(backend="json", path=json_path, auto_save=True)
        kg_json.graph.add_entity("Auto", "Demo", {"note": "Auto-saved on change"})
        print("\n✅ JSON backend (auto-save):")
        print(f"   Path: {json_path}")
        print(f"   File exists: {json_path.exists()}")


if __name__ == "__main__":
    asyncio.run(demo())
