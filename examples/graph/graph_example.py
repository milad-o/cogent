"""Realistic Graph Module Example - Tech Company Knowledge Graph

This example demonstrates building a knowledge graph for a technology company,
including employees, projects, technologies, and their relationships.

Features demonstrated:
- Creating entities and relationships
- Querying with filters
- Pattern matching
- Path finding
- Visualization (Mermaid, Graphviz, GraphML)
- Storage backends (Memory, File, SQL)
"""

import asyncio
from pathlib import Path

from cogent.graph import Graph, Entity, Relationship


async def build_company_graph() -> Graph:
    """Build a comprehensive company knowledge graph."""
    graph = Graph()  # Uses MemoryStorage and NetworkXEngine by default

    # --- Add People (Employees) ---
    print("Adding employees...")
    await graph.add_entity(
        "alice",
        "Person",
        name="Alice Chen",
        role="Engineering Manager",
        department="Engineering",
        years_experience=8,
    )
    await graph.add_entity(
        "bob",
        "Person",
        name="Bob Kumar",
        role="Senior Software Engineer",
        department="Engineering",
        years_experience=5,
    )
    await graph.add_entity(
        "carol",
        "Person",
        name="Carol Martinez",
        role="Data Scientist",
        department="Data Science",
        years_experience=6,
    )
    await graph.add_entity(
        "dave",
        "Person",
        name="Dave Johnson",
        role="Product Manager",
        department="Product",
        years_experience=7,
    )
    await graph.add_entity(
        "eve",
        "Person",
        name="Eve Williams",
        role="DevOps Engineer",
        department="Engineering",
        years_experience=4,
    )

    # --- Add Companies ---
    print("Adding companies...")
    await graph.add_entity(
        "techcorp",
        "Company",
        name="TechCorp Inc.",
        industry="Technology",
        size="500-1000",
        founded=2015,
    )
    await graph.add_entity(
        "cloudprovider",
        "Company",
        name="CloudProvider",
        industry="Cloud Services",
        size="10000+",
    )

    # --- Add Projects ---
    print("Adding projects...")
    await graph.add_entity(
        "platform",
        "Project",
        name="ML Platform",
        status="active",
        priority="high",
        budget=500000,
    )
    await graph.add_entity(
        "api",
        "Project",
        name="Customer API",
        status="active",
        priority="medium",
        budget=200000,
    )
    await graph.add_entity(
        "analytics",
        "Project",
        name="Analytics Dashboard",
        status="planning",
        priority="medium",
        budget=150000,
    )

    # --- Add Technologies ---
    print("Adding technologies...")
    await graph.add_entity("python", "Technology", name="Python", category="Language")
    await graph.add_entity(
        "postgresql", "Technology", name="PostgreSQL", category="Database"
    )
    await graph.add_entity("react", "Technology", name="React", category="Frontend")
    await graph.add_entity("docker", "Technology", name="Docker", category="DevOps")
    await graph.add_entity(
        "kubernetes", "Technology", name="Kubernetes", category="DevOps"
    )

    # --- Add Relationships ---
    print("Adding relationships...")

    # Employment relationships
    await graph.add_relationship("alice", "works_at", "techcorp", title="Eng Manager")
    await graph.add_relationship(
        "bob", "works_at", "techcorp", title="Senior Engineer"
    )
    await graph.add_relationship(
        "carol", "works_at", "techcorp", title="Data Scientist"
    )
    await graph.add_relationship("dave", "works_at", "techcorp", title="PM")
    await graph.add_relationship("eve", "works_at", "techcorp", title="DevOps Eng")

    # Management relationships
    await graph.add_relationship("alice", "manages", "bob")
    await graph.add_relationship("alice", "manages", "eve")
    await graph.add_relationship("dave", "manages", "platform")
    await graph.add_relationship("dave", "manages", "api")

    # Project assignments
    await graph.add_relationship("alice", "leads", "platform")
    await graph.add_relationship("bob", "works_on", "platform", role="Backend Lead")
    await graph.add_relationship("carol", "works_on", "platform", role="ML Engineer")
    await graph.add_relationship("eve", "works_on", "platform", role="Infrastructure")
    await graph.add_relationship("bob", "works_on", "api", role="Tech Lead")

    # Technology usage
    await graph.add_relationship("platform", "uses", "python")
    await graph.add_relationship("platform", "uses", "postgresql")
    await graph.add_relationship("platform", "uses", "kubernetes")
    await graph.add_relationship("api", "uses", "python")
    await graph.add_relationship("api", "uses", "postgresql")
    await graph.add_relationship("analytics", "uses", "react")
    await graph.add_relationship("analytics", "uses", "python")

    # Technology dependencies
    await graph.add_relationship("kubernetes", "depends_on", "docker")

    # Business relationships
    await graph.add_relationship("techcorp", "partners_with", "cloudprovider")
    await graph.add_relationship("platform", "deployed_on", "cloudprovider")

    print(f"\nGraph built successfully!")
    stats = await graph.stats()
    print(f"Statistics: {stats}")

    return graph


async def demonstrate_queries(graph: Graph):
    """Demonstrate various query capabilities."""
    print("\n" + "=" * 60)
    print("QUERY DEMONSTRATIONS")
    print("=" * 60)

    # --- 1. Simple entity queries ---
    print("\n1. Find all employees in Engineering department:")
    engineers = await graph.find_entities(
        entity_type="Person", attributes={"department": "Engineering"}
    )
    for person in engineers:
        print(f"   - {person.attributes.get('name')} ({person.attributes.get('role')})")

    # --- 2. Find high-priority projects ---
    print("\n2. Find high-priority projects:")
    high_priority = await graph.find_entities(
        entity_type="Project", attributes={"priority": "high"}
    )
    for project in high_priority:
        print(
            f"   - {project.attributes.get('name')} (Budget: ${project.attributes.get('budget'):,})"
        )

    # --- 3. Get relationships for a specific person ---
    print("\n3. Alice's relationships:")
    alice_rels = await graph.get_relationships(source_id="alice")
    for rel in alice_rels:
        target = await graph.get_entity(rel.target_id)
        print(f"   - {rel.relation} → {target.attributes.get('name', rel.target_id)}")

    # --- 4. Pattern matching - Find who works on what projects ---
    print("\n4. Pattern: People working on projects")
    pattern = {
        "source": {"type": "Person"},
        "relation": "works_on",
        "target": {"type": "Project"},
    }
    result = await graph.match(pattern)
    # Show first 5 relationships
    for rel in result.relationships[:5]:
        person = await graph.get_entity(rel.source_id)
        project = await graph.get_entity(rel.target_id)
        print(
            f"   - {person.attributes.get('name')} works on {project.attributes.get('name')} "
            f"as {rel.attributes.get('role', 'team member')}"
        )

    # --- 5. Multi-hop pattern - Technology stack for projects ---
    print("\n5. Pattern: Project → uses → Technology")
    tech_pattern = {
        "source": {"type": "Project"},
        "relation": "uses",
        "target": {"type": "Technology"},
    }
    tech_result = await graph.match(tech_pattern)

    # Group by project
    from collections import defaultdict

    projects_tech = defaultdict(list)
    for rel in tech_result.relationships:
        project = await graph.get_entity(rel.source_id)
        tech = await graph.get_entity(rel.target_id)
        project_name = project.attributes.get("name")
        tech_name = tech.attributes.get("name")
        projects_tech[project_name].append(tech_name)

    for project, technologies in projects_tech.items():
        print(f"   - {project}: {', '.join(technologies)}")

    # --- 6. Find paths between entities ---
    print("\n6. Path from Bob to CloudProvider:")
    path = await graph.find_path("bob", "cloudprovider")
    if path:
        print(f"   Found path with {len(path)} nodes:")
        for i, node_id in enumerate(path):
            node = await graph.get_entity(node_id)
            print(
                f"      {i+1}. {node.attributes.get('name', node_id)} ({node.entity_type})"
            )
    else:
        print("   No path found")

    # --- 7. Get all DevOps technologies ---
    print("\n7. DevOps technologies:")
    devops_tech = await graph.find_entities(
        entity_type="Technology", attributes={"category": "DevOps"}
    )
    for tech in devops_tech:
        print(f"   - {tech.attributes.get('name')}")

    # --- 8. Who does Alice manage? ---
    print("\n8. Alice's direct reports:")
    alice_manages = await graph.get_relationships(source_id="alice", relation="manages")
    for rel in alice_manages:
        if rel.target_id.startswith("person") or await graph.get_entity(
            rel.target_id
        ):
            target = await graph.get_entity(rel.target_id)
            if target and target.entity_type == "Person":
                print(
                    f"   - {target.attributes.get('name')} ({target.attributes.get('role')})"
                )


async def demonstrate_visualization(graph: Graph):
    """Demonstrate visualization capabilities."""
    print("\n" + "=" * 60)
    print("VISUALIZATION DEMONSTRATIONS")
    print("=" * 60)

    # Output directory next to the example file
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # --- 1. Mermaid diagram (left-to-right, grouped by type) ---
    print("\n1. Generating Mermaid diagram (grouped by type)...")
    mermaid_code = await graph.to_mermaid(
        direction="LR", group_by_type=True, scheme="default", title="TechCorp Knowledge Graph"
    )
    mermaid_file = output_dir / "company_graph.mmd"
    await graph.save_diagram(str(mermaid_file), format="mermaid", group_by_type=True, title="TechCorp Knowledge Graph")
    print(f"   Saved to: {mermaid_file}")
    print(f"   Lines: {len(mermaid_code.splitlines())}")

    # --- 2. Graphviz DOT format ---
    print("\n2. Generating Graphviz DOT diagram...")
    dot_file = output_dir / "company_graph.dot"
    await graph.save_diagram(str(dot_file), format="graphviz", title="TechCorp Org Chart")
    print(f"   Saved to: {dot_file}")

    # --- 3. GraphML for graph analysis tools ---
    print("\n3. Generating GraphML (for Gephi/yEd)...")
    graphml_file = output_dir / "company_graph.graphml"
    await graph.save_diagram(str(graphml_file), format="graphml", title="TechCorp Knowledge Graph")
    print(f"   Saved to: {graphml_file}")

    # --- 4. Show preview of Mermaid code ---
    print("\n4. Mermaid code preview (first 15 lines):")
    for line in mermaid_code.splitlines()[:15]:
        print(f"   {line}")
    print("   ...")


async def demonstrate_storage_backends(graph: Graph):
    """Demonstrate different storage backends."""
    print("\n" + "=" * 60)
    print("STORAGE BACKEND DEMONSTRATIONS")
    print("=" * 60)

    from cogent.graph import FileStorage, SQLStorage

    # Output directory next to the example file
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Clean up old files
    (output_dir / "company_graph.json").unlink(missing_ok=True)
    (output_dir / "company_graph.db").unlink(missing_ok=True)

    # --- 1. Save to file ---
    print("\n1. Saving graph to JSON file...")
    file_graph = Graph(storage=FileStorage(str(output_dir / "company_graph.json")))

    # Copy entities and relationships
    entities = await graph.get_all_entities()
    relationships = await graph.get_relationships()  # Get all relationships

    for entity in entities:
        await file_graph.add_entity(
            entity.id, entity.entity_type, **entity.attributes
        )
    for rel in relationships:
        await file_graph.add_relationship(
            rel.source_id, rel.relation, rel.target_id, **rel.attributes
        )

    print(f"   Saved {len(entities)} entities and {len(relationships)} relationships")
    file_stats = await file_graph.stats()
    print(f"   File storage stats: {file_stats}")

    # --- 2. SQL storage (SQLite with async driver) ---
    print("\n2. Saving graph to SQLite database...")
    sql_graph = Graph(storage=SQLStorage(f"sqlite+aiosqlite:///{output_dir / 'company_graph.db'}"))

    for entity in entities:
        await sql_graph.add_entity(entity.id, entity.entity_type, **entity.attributes)
    for rel in relationships:
        await sql_graph.add_relationship(
            rel.source_id, rel.relation, rel.target_id, **rel.attributes
        )

    print(f"   Saved to SQLite database")
    sql_stats = await sql_graph.stats()
    print(f"   SQL storage stats: {sql_stats}")

    # --- 3. Query from SQL storage ---
    print("\n3. Querying from SQLite...")
    sql_people = await sql_graph.find_entities(entity_type="Person")
    print(f"   Found {len(sql_people)} people in SQL storage")


async def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("COGENT GRAPH MODULE - REALISTIC EXAMPLE")
    print("TechCorp Inc. Knowledge Graph")
    print("=" * 60)

    # Create output directory next to this example file
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Build the graph
    graph = await build_company_graph()

    # Demonstrate queries
    await demonstrate_queries(graph)

    # Demonstrate visualization
    await demonstrate_visualization(graph)

    # Demonstrate storage backends
    await demonstrate_storage_backends(graph)

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE!")
    print("=" * 60)
    print("\nGenerated files in:", output_dir.absolute())
    print("  - company_graph.mmd (Mermaid)")
    print("  - company_graph.dot (Graphviz)")
    print("  - company_graph.graphml (GraphML)")
    print("  - company_graph.json (File storage)")
    print("  - company_graph.db (SQLite storage)")
    print("\nTry opening the .mmd file in a Mermaid viewer!")


if __name__ == "__main__":
    asyncio.run(main())
