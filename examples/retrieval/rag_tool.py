"""
Example: RAG with Agent Using Retriever Tool

Demonstrates how to expose a retriever as an agent tool via `as_tool`, then
give it to an autonomous agent that can decide when and how to use it.

The agent receives questions and autonomously:
1. Decides whether to search the knowledge base
2. Formulates search queries
3. Uses retrieved context to answer questions

Usage:
    uv run python examples/retrieval/rag_tool.py
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_embeddings, get_model

from agenticflow.agent import Agent
from agenticflow.vectorstore import Document, VectorStore


DOCUMENTS = [
    Document(
        text="""Python 3.13 introduces free-threading (experimental no-GIL),
        improved error messages, and a new multiline REPL. Performance
        improvements average around 5%.""",
        metadata={"source": "python_3_13.md", "category": "release_notes"},
    ),
    Document(
        text="""API rate limiting policies: Free tier 100 rpm, 1k per day;
        Pro tier 1k rpm; Enterprise is custom. Exceeding limits returns 429
        with Retry-After header.""",
        metadata={"source": "api_limits.md", "category": "policies"},
    ),
    Document(
        text="""Machine learning best practices include clean data, baselines,
        cross-validation, and monitoring for drift in production. Always
        validate on held-out test sets.""",
        metadata={"source": "ml_best_practices.md", "category": "guides"},
    ),
    Document(
        text="""Database optimization tips: Use indexes for frequent queries,
        normalize schema to 3NF, denormalize for read-heavy workloads, and
        always use connection pooling.""",
        metadata={"source": "database_optimization.md", "category": "guides"},
    ),
    Document(
        text="""Security checklist: Enable 2FA, rotate keys quarterly, use
        HTTPS everywhere, validate all inputs, and implement rate limiting
        on all public endpoints.""",
        metadata={"source": "security_checklist.md", "category": "policies"},
    ),
]


async def demo_agent_with_rag() -> None:
    """Demonstrate an agent using a retriever tool for RAG."""
    print("=" * 70)
    print("RAG with Autonomous Agent")
    print("=" * 70)
    
    # Build vectorstore and create search tool
    print("\nSetting up knowledge base...")
    embeddings = get_embeddings()
    vectorstore = VectorStore(embeddings=embeddings)
    await vectorstore.add_documents(DOCUMENTS)
    
    # Create search tool using convenient as_retriever() API
    search_tool = vectorstore.as_retriever().as_tool(
        name="search_knowledge_base",
        description=(
            "Search the internal knowledge base for relevant information. "
            "Use this when you need factual information to answer questions. "
            "Returns relevant passages with scores and metadata."
        ),
        k_default=3,
        include_scores=True,
        include_metadata=True,
    )
    
    # Create agent with the search tool
    model = get_model()
    agent = Agent(
        name="KnowledgeAgent",
        model=model,
        tools=[search_tool],
        system_prompt="""You are a helpful assistant with access to a knowledge base.
        
When answering questions:
1. Use the search_knowledge_base tool to find relevant information
2. Cite sources from the metadata when providing answers
3. If information isn't in the knowledge base, say so clearly
4. Synthesize information from multiple sources when appropriate""",
    )
    
    print(f"Created agent with {len(agent.config.tools)} tool(s)")
    print(f"Knowledge base contains {vectorstore.count()} documents\n")
    
    # Test queries that should trigger tool usage
    queries = [
        "What is free-threading in Python 3.13?",
        "What are the API rate limits for the free tier?",
        "What are some machine learning best practices?",
        "Compare the security and database optimization recommendations.",
    ]
    
    for i, query in enumerate(queries, 1):
        print("\n" + "=" * 70)
        print(f"Query {i}: {query}")
        print("=" * 70)
        
        response = await agent.run(query)
        
        print(f"\nAnswer: {response}")


async def demo_direct_tool_usage() -> None:
    """Show direct tool usage without agent for comparison."""
    print("\n\n" + "=" * 70)
    print("Direct Tool Usage (No Agent)")
    print("=" * 70)
    
    # Build vectorstore
    embeddings = get_embeddings()
    vectorstore = VectorStore(embeddings=embeddings)
    await vectorstore.add_documents(DOCUMENTS)
    
    # Create tool
    search_tool = vectorstore.as_retriever().as_tool(
        name="search_docs",
        description="Search documentation",
        k_default=2,
        include_scores=True,
    )
    
    query = "What is free-threading?"
    print(f"\nQuery: {query}")
    
    # Call tool directly
    results = await search_tool.ainvoke({"query": query, "k": 2})
    
    print("\nResults:")
    for i, r in enumerate(results, 1):
        source = r.get("metadata", {}).get("source", "unknown")
        score = r.get("score", 0)
        print(f"\n{i}. [source={source}, score={score:.3f}]")
        print(f"   {r['text'][:100]}...")


def main() -> None:
    # Run both demos
    asyncio.run(demo_agent_with_rag())
    asyncio.run(demo_direct_tool_usage())


if __name__ == "__main__":
    main()
