"""
HyDE (Hypothetical Document Embeddings) Retriever

Generates hypothetical documents to bridge semantic gap between queries and documents.
Better for abstract/conceptual questions than direct embedding search.
"""

import asyncio

from cogent.documents import RecursiveCharacterSplitter
from cogent.models import OpenAIEmbedding, create_chat
from cogent.retriever import DenseRetriever, HyDERetriever
from cogent.vectorstore import Document, VectorStore

KNOWLEDGE_BASE = """
Exercise has profound effects on mental well-being. Physical activity
triggers endorphins that reduce stress and anxiety. Regular exercise
improves sleep quality, cognitive function, and self-esteem.

Quality sleep is fundamental to health. During sleep, the body repairs
tissues and consolidates memories. Poor sleep increases risk of obesity,
heart disease, and mental health disorders.

A balanced diet provides essential nutrients. Proteins build tissues,
complex carbs provide sustained energy, and healthy fats support brain
function and vitamin absorption.
"""


async def main() -> None:
    chat_model = create_chat("gpt4")
    embeddings = OpenAIEmbedding(model="text-embedding-3-small")

    # Setup vectorstore
    doc = Document(text=KNOWLEDGE_BASE, metadata={"source": "health.md"})
    splitter = RecursiveCharacterSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents([doc])

    store = VectorStore(embeddings=embeddings)
    await store.add_documents(chunks)

    # Create retrievers
    base_retriever = DenseRetriever(store)
    hyde_retriever = HyDERetriever(base_retriever, chat_model)

    # Compare regular vs HyDE retrieval
    query = "How can I feel happier?"
    print(f"\nQuery: {query!r}\n")

    print("Regular retrieval:")
    results = await base_retriever.retrieve(query, k=2, include_scores=True)
    for i, r in enumerate(results, 1):
        print(f"  [{i}] Score: {r.score:.3f} - {r.document.text[:60]}...")

    print("\nHyDE retrieval:")
    hypothetical = await hyde_retriever.generate_hypothetical(query)
    print(f"  Hypothetical: {hypothetical[:80]}...")

    results = await hyde_retriever.retrieve(query, k=2, include_scores=True)
    for i, r in enumerate(results, 1):
        print(f"  [{i}] Score: {r.score:.3f} - {r.document.text[:60]}...")

    # Multiple hypotheticals with fusion
    print("\n\nMultiple hypotheticals (n=3):")
    hyde_ensemble = HyDERetriever(base_retriever, chat_model, n_hypotheticals=3)
    results = await hyde_ensemble.retrieve(query, k=2, include_scores=True)
    for i, r in enumerate(results, 1):
        print(f"  [{i}] Score: {r.score:.3f} - {r.document.text[:60]}...")


if __name__ == "__main__":
    asyncio.run(main())
