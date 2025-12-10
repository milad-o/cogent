"""
Example: RAG with Retriever Tool API

Demonstrates how to expose a retriever as an agent tool via `as_tool`, then
use it to build a simple RAG flow.

Usage:
    uv run python examples/retrieval/32_rag_tool.py
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_embeddings, get_model

from agenticflow.retriever import DenseRetriever
from agenticflow.vectorstore import Document, VectorStore


DOCUMENTS = [
    Document(
        text="""Python 3.13 introduces free-threading (experimental no-GIL),
        improved error messages, and a new multiline REPL. Performance
        improvements average around 5%.""",
        metadata={"source": "python_3_13.md"},
    ),
    Document(
        text="""API rate limiting policies: Free tier 100 rpm, 1k per day;
        Pro tier 1k rpm; Enterprise is custom. Exceeding limits returns 429
        with Retry-After.""",
        metadata={"source": "api_limits.md"},
    ),
    Document(
        text="""Machine learning best practices include clean data, baselines,
        cross-validation, and monitoring for drift in production.""",
        metadata={"source": "ml_best_practices.md"},
    ),
]


async def run_rag_with_tool(query: str) -> None:
    # Build vectorstore
    embeddings = get_embeddings()
    vectorstore = VectorStore(embeddings=embeddings)
    await vectorstore.add_documents(DOCUMENTS)

    # Method 1: Traditional approach
    # retriever = DenseRetriever(vectorstore)
    # search_tool = retriever.as_tool(...)
    
    # Method 2: Convenient shortcut using as_retriever()
    search_tool = vectorstore.as_retriever().as_tool(
        name="search_docs",
        description="Search the internal knowledge base for relevant passages.",
        k_default=3,
        include_scores=True,
        include_metadata=True,
    )

    # Call the tool directly to fetch context
    results = await search_tool.ainvoke({"query": query, "k": 3})

    # Build context block
    context_lines = []
    for r in results:
        meta = r.get("metadata", {})
        source = meta.get("source", "unknown")
        score = r.get("score")
        prefix = f"[source={source} score={score:.3f}]" if score is not None else f"[source={source}]"
        context_lines.append(f"{prefix} {r['text']}")
    context_text = "\n".join(context_lines) if context_lines else "No results found."

    # Ask the model to answer using the retrieved context
    model = get_model()
    messages = [
        {
            "role": "system",
            "content": "Answer the question using only the provided context. Cite sources if available.",
        },
        {
            "role": "user",
            "content": f"Question: {query}\n\nContext:\n{context_text}\n\nAnswer concisely.",
        },
    ]
    response = await model.ainvoke(messages)

    print("\n=== RAG Result ===")
    print(response.content)
    print("\n=== Retrieved Context ===")
    for line in context_lines:
        print(f"- {line}")


def main() -> None:
    query = "What is free-threading in Python 3.13?"
    asyncio.run(run_rag_with_tool(query))


if __name__ == "__main__":
    main()
