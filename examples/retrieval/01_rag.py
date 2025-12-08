"""RAG (Retrieval-Augmented Generation) Example with Observability.

Demonstrates:
- VectorStore with observability events
- RAG capability with citation and bibliography configuration
- Observer for detailed logging instead of print statements
- Automatic bibliography generation
"""

import asyncio
import sys
from pathlib import Path

# Add examples dir to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_model, get_embeddings

from agenticflow import Agent
from agenticflow.capabilities import RAG
from agenticflow.capabilities.rag import RAGConfig
from agenticflow.observability import Observer
from agenticflow.retriever import DenseRetriever
from agenticflow.vectorstore import VectorStore

# Sample documents about a fictional company with rich metadata
DOCUMENTS = [
    {
        "content": "TechFlow Inc. was founded in 2018 by Sarah Chen and Marcus Williams. "
        "The company started as a small AI consultancy in San Francisco.",
        "metadata": {"source": "company_history.md", "page": 1, "author": "HR Dept", "date": "2024-01"},
    },
    {
        "content": "TechFlow's flagship product is FlowAI, an enterprise automation platform "
        "that uses machine learning to optimize business workflows. It was launched in 2020.",
        "metadata": {"source": "products.md", "page": 1, "author": "Product Team", "date": "2024-03"},
    },
    {
        "content": "As of 2024, TechFlow employs over 500 people across offices in San Francisco, "
        "New York, London, and Singapore. The company reported $150M in revenue last year.",
        "metadata": {"source": "company_overview.md", "page": 2, "author": "Executive Team", "date": "2024-06"},
    },
    {
        "content": "TechFlow's main competitors include AutomateNow, WorkflowPro, and AIStream. "
        "The company differentiates through its advanced NLP capabilities.",
        "metadata": {"source": "market_analysis.md", "page": 5, "author": "Strategy", "date": "2024-02"},
    },
    {
        "content": "The engineering team at TechFlow uses a microservices architecture built on "
        "Kubernetes. They practice continuous deployment with over 100 releases per week.",
        "metadata": {"source": "engineering.md", "page": 3, "author": "CTO Office", "date": "2024-04"},
    },
]


async def main() -> None:
    # Get model and embeddings from config
    model = get_model()
    embeddings = get_embeddings()

    # Initialize vectorstore - events will be emitted automatically
    store = VectorStore(embeddings=embeddings)

    # Add documents - vectorstore will emit VECTORSTORE_ADD event
    await store.add_texts(
        texts=[doc["content"] for doc in DOCUMENTS],
        metadatas=[doc["metadata"] for doc in DOCUMENTS],
    )

    # Configure RAG - simple! Most users only need top_k
    rag_config = RAGConfig(
        top_k=3,
        bibliography=True,  # Enable bibliography generation
        bibliography_fields=("author", "date"),  # Include in references
    )

    # Create retriever from vectorstore
    retriever = DenseRetriever(store)

    # Create RAG capability
    rag = RAG(retriever, config=rag_config)

    # Create agent with observability
    agent = Agent(
        name="Research Assistant",
        instructions=(
            "You are a research assistant that answers questions about TechFlow Inc. "
            "Use the RAG tool to find relevant information before answering. "
            "Always cite your sources using [1], [2] notation."
        ),
        model=model,
        capabilities=[rag],
    )

    # Attach detailed observer - this will show vectorstore events, tool calls, etc.
    observer = Observer.detailed()
    agent.add_observer(observer)

    # Test queries - observer will show all activity
    queries = [
        "Who founded TechFlow and when?",
        "What is FlowAI and when was it launched?",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("=" * 60)
        response = await agent.run(query)
        
        # Format response with bibliography
        formatted = rag.format_response_with_bibliography(str(response))
        print(f"\n{formatted}")


if __name__ == "__main__":
    asyncio.run(main())
