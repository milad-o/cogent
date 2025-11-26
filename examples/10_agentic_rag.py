#!/usr/bin/env python3
"""
Example 10: Agentic RAG with Nature-themed Text
================================================

This example demonstrates an agentic RAG (Retrieval-Augmented Generation) system
that uses multiple specialized agents to process and answer questions about a
nature-themed document.

The system consists of:
- Retriever: Finds relevant chunks from the document
- Analyzer: Interprets and synthesizes information
- Reviewer: Validates answers for accuracy
- Supervisor: Coordinates the workflow

Text Source: "The Secret Garden" by Frances Hodgson Burnett (Public Domain)
Available at: https://www.gutenberg.org/cache/epub/113/pg113.txt
"""

import asyncio
from pathlib import Path

from agenticflow import Agent, AgentRole, BaseTopology
from agenticflow.topologies import SupervisorTopology, TopologyConfig


# Nature-themed sample text URL (public domain - The Secret Garden)
NATURE_TEXT_URL = "https://www.gutenberg.org/cache/epub/113/pg113.txt"
NATURE_TEXT_NAME = "the_secret_garden.txt"


async def download_sample_text(url: str, filename: str) -> Path:
    """Download a sample text for the example."""
    import httpx

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    text_path = data_dir / filename

    if text_path.exists():
        print(f"✓ Text already exists: {text_path}")
        return text_path

    print(f"⬇ Downloading nature text from {url}...")
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        text_path.write_bytes(response.content)

    print(f"✓ Downloaded: {text_path} ({text_path.stat().st_size / 1024:.1f} KB)")
    return text_path


def create_rag_agents() -> list[Agent]:
    """Create the RAG agent team with specialized roles."""

    # Retriever: Searches and retrieves relevant document chunks
    retriever = Agent(
        name="Retriever",
        role=AgentRole.WORKER,
        instructions="""You are a document retrieval specialist.
            Your job is to find and extract the most relevant passages from 
            the source document that relate to the user's question.
            
            Focus on:
            - Exact matches to key terms
            - Semantically related content
            - Context surrounding relevant passages
            
            Always cite chapter or section references when possible.""",
        tools=["text_search", "chunk_retriever", "semantic_search"],
    )

    # Analyzer: Interprets and synthesizes retrieved information
    analyzer = Agent(
        name="Analyzer",
        role=AgentRole.WORKER,
        instructions="""You are an information analyst specializing in nature topics.
            Your job is to synthesize retrieved passages into coherent, accurate answers.
            
            Guidelines:
            - Connect information from multiple sources
            - Identify patterns and themes
            - Provide context from the document
            - Distinguish between facts stated in the document vs. inferences""",
        tools=["text_summarizer", "theme_extractor"],
    )

    # Reviewer: Validates answers for accuracy and completeness
    reviewer = Agent(
        name="Reviewer",
        role=AgentRole.REVIEWER,
        instructions="""You are a quality assurance specialist for RAG systems.
            Your job is to verify that answers are:
            
            1. Accurate - Supported by the source document
            2. Complete - Address all aspects of the question
            3. Well-cited - Reference specific parts of the document
            4. Honest - Acknowledge when information is uncertain or missing
            
            Provide specific feedback for improvement if needed.""",
        tools=["fact_checker", "citation_validator"],
    )

    # Supervisor: Coordinates the RAG workflow
    supervisor = Agent(
        name="RAG_Coordinator",
        role=AgentRole.SUPERVISOR,
        instructions="""You coordinate a RAG (Retrieval-Augmented Generation) team.
            
            Workflow:
            1. Receive user question
            2. Direct Retriever to find relevant passages
            3. Pass passages to Analyzer for synthesis
            4. Send answer to Reviewer for validation
            5. Return final answer or iterate if needed
            
            Ensure answers are grounded in the source document.""",
    )

    return [supervisor, retriever, analyzer, reviewer]


def create_rag_topology(agents: list[Agent]) -> BaseTopology:
    """Create the RAG topology with supervisor pattern."""
    supervisor = agents[0]  # First agent is the supervisor

    # Build supervisor topology
    topology = SupervisorTopology(
        config=TopologyConfig(name="rag-system"),
        agents=agents,
        supervisor_name=supervisor.name,
    )

    return topology


def visualize_rag_system(topology: BaseTopology) -> str:
    """Generate Mermaid diagram for the RAG system."""
    from agenticflow.visualization import MermaidConfig, TopologyDiagram

    config = MermaidConfig(
        show_tools=True,
        show_roles=True,
        title="Agentic RAG System",
    )

    diagram = TopologyDiagram(topology, config=config)
    return diagram.to_mermaid()


async def simulate_rag_query(question: str) -> None:
    """Simulate a RAG query through the agent system."""
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")

    # In a real implementation, this would:
    # 1. Load and chunk the text
    # 2. Create embeddings for semantic search
    # 3. Execute the actual agent workflow

    # For this example, we show the workflow structure
    print("RAG Workflow Steps:")
    print("  1. 📥 Supervisor receives question")
    print("  2. 🔍 Retriever searches document for relevant passages")
    print("  3. 📊 Analyzer synthesizes information into answer")
    print("  4. ✅ Reviewer validates accuracy and citations")
    print("  5. 📤 Final answer returned to user")
    print()


async def main() -> None:
    """Run the Agentic RAG example."""
    print("=" * 60)
    print("Agentic RAG Example - The Secret Garden")
    print("=" * 60)

    # Download sample text
    text_path = None
    try:
        text_path = await download_sample_text(NATURE_TEXT_URL, NATURE_TEXT_NAME)

        # Show a snippet from the text
        if text_path and text_path.exists():
            content = text_path.read_text(encoding="utf-8", errors="ignore")
            # Find a nature-related excerpt
            lines = content.split("\n")
            nature_lines = []
            for i, line in enumerate(lines):
                if "garden" in line.lower() and len(line) > 50:
                    nature_lines = lines[i : i + 5]
                    break

            if nature_lines:
                print("\n📖 Sample from the text:")
                print("-" * 40)
                for line in nature_lines:
                    if line.strip():
                        print(f"  {line.strip()[:80]}...")
                print("-" * 40)

    except Exception as e:
        print(f"⚠ Could not download text: {e}")
        print("  Continuing with example structure...")

    # Create agents
    print("\n📦 Creating RAG Agents...")
    agents = create_rag_agents()
    for agent in agents:
        tools_str = ", ".join(agent.config.tools) if agent.config.tools else "none"
        print(f"  • {agent.name} ({agent.role.value}) - tools: {tools_str}")

    # Create topology
    print("\n🔗 Building RAG Topology...")
    topology = create_rag_topology(agents)
    print(f"  Topology name: {topology.config.name}")
    print(f"  Agent count: {len(topology.agents)}")

    # Visualize
    print("\n📊 Generating Mermaid Diagram...")
    mermaid = visualize_rag_system(topology)
    print(mermaid)

    # Save diagram
    diagram_path = Path(__file__).parent / "diagrams" / "7_agentic_rag.mmd"
    diagram_path.parent.mkdir(exist_ok=True)
    diagram_path.write_text(mermaid)
    print(f"\n✓ Diagram saved to: {diagram_path}")

    # Simulate queries
    sample_questions = [
        "What is the secret garden like when Mary first discovers it?",
        "How does the garden help the children heal?",
        "What role does nature play in the story's themes?",
    ]

    for question in sample_questions:
        await simulate_rag_query(question)

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)

    if text_path:
        print(f"\nText location: {text_path}")
    print("\nTo implement a full RAG system, you would need:")
    print("  • Text chunking (langchain_text_splitters)")
    print("  • Embeddings (langchain_openai.OpenAIEmbeddings)")
    print("  • Vector store (chromadb, faiss, InMemoryVectorStore)")
    print("  • LLM integration for agent execution")


if __name__ == "__main__":
    asyncio.run(main())
