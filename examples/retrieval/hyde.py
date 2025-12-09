"""
Example 34: HyDE (Hypothetical Document Embeddings) Retriever

HyDE improves retrieval for abstract/conceptual queries by first generating
a hypothetical document that would answer the query, then using that
document's embedding for similarity search.

┌─────────────────────────────────────────────────────────────────────────┐
│ How HyDE Works                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Query: "What are the benefits of exercise?"                            │
│                    │                                                    │
│                    ▼                                                    │
│  ┌─────────────────────────────────────┐                                │
│  │  LLM generates hypothetical doc:    │                                │
│  │  "Regular exercise provides many    │                                │
│  │   health benefits including..."     │                                │
│  └─────────────────────────────────────┘                                │
│                    │                                                    │
│                    ▼                                                    │
│  ┌─────────────────────────────────────┐                                │
│  │  Embed the hypothetical document    │                                │
│  └─────────────────────────────────────┘                                │
│                    │                                                    │
│                    ▼                                                    │
│  ┌─────────────────────────────────────┐                                │
│  │  Search vectorstore with embedding  │                                │
│  │  → Better matches than query alone  │                                │
│  └─────────────────────────────────────┘                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

Benefits of HyDE:
- Bridges semantic gap between short queries and long documents
- Works zero-shot (no training data needed)
- Especially good for abstract/conceptual questions

Usage:
    uv run python examples/34_hyde.py
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_embeddings, get_model

from agenticflow.retriever import DenseRetriever, HyDERetriever
from agenticflow.document import RecursiveCharacterSplitter
from agenticflow.vectorstore import VectorStore, Document


# Sample knowledge base about health and wellness
KNOWLEDGE_BASE = """
## Physical Activity and Health

Regular physical activity is essential for maintaining good health. 
The cardiovascular system benefits greatly from aerobic exercises like 
running, swimming, and cycling. These activities strengthen the heart 
muscle and improve blood circulation throughout the body.

Resistance training, such as weight lifting, helps build and maintain 
muscle mass. This is particularly important as we age, since muscle 
loss naturally occurs over time. Strong muscles support joint health 
and improve balance, reducing the risk of falls.

## Mental Health Benefits

Exercise has profound effects on mental well-being. Physical activity 
triggers the release of endorphins, often called "feel-good" hormones. 
These chemicals in the brain help reduce feelings of stress, anxiety, 
and depression.

Studies show that regular exercise improves sleep quality, enhances 
cognitive function, and boosts self-esteem. Even a 30-minute walk 
can significantly improve mood and mental clarity.

## Nutrition Fundamentals

A balanced diet provides the nutrients your body needs to function 
properly. Proteins are essential for building and repairing tissues. 
Good protein sources include lean meats, fish, eggs, and legumes.

Carbohydrates are the body's primary energy source. Complex carbs 
from whole grains, vegetables, and fruits provide sustained energy 
and important fiber for digestive health.

Healthy fats from sources like avocados, nuts, and olive oil support 
brain function and help absorb certain vitamins. Omega-3 fatty acids, 
found in fatty fish, are particularly beneficial for heart health.

## Sleep and Recovery

Quality sleep is fundamental to health. During sleep, the body repairs 
tissues, consolidates memories, and releases growth hormones. Adults 
typically need 7-9 hours of sleep per night for optimal functioning.

Poor sleep is linked to increased risk of obesity, heart disease, 
diabetes, and mental health disorders. Establishing a consistent 
sleep schedule and creating a restful environment can improve sleep quality.

## Stress Management

Chronic stress negatively impacts both physical and mental health. 
Prolonged stress can lead to high blood pressure, weakened immune 
function, and digestive problems.

Effective stress management techniques include meditation, deep 
breathing exercises, yoga, and spending time in nature. Social 
connections and hobbies also play important roles in stress reduction.
"""


async def main() -> None:
    model = get_model()
    embeddings = get_embeddings()

    # =========================================================================
    # Setup: Create vectorstore and retrievers
    # =========================================================================
    print("=" * 70)
    print("Setup: Creating vectorstore and retrievers")
    print("=" * 70)

    # Split the knowledge base into chunks
    doc = Document(text=KNOWLEDGE_BASE, metadata={"source": "health_guide.md"})
    splitter = RecursiveCharacterSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents([doc])
    print(f"Created {len(chunks)} chunks")

    # Create vectorstore and index
    store = VectorStore(embeddings=embeddings)
    await store.add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks")

    # Create retrievers
    base_retriever = DenseRetriever(store)
    hyde_retriever = HyDERetriever(base_retriever, model)
    
    print("\nRetrievers ready:")
    print(f"  - Base: {base_retriever}")
    print(f"  - HyDE: {hyde_retriever}")

    # =========================================================================
    # Comparison: Regular vs HyDE Retrieval
    # =========================================================================
    
    # Test queries - these are abstract/conceptual questions that benefit from HyDE
    test_queries = [
        "Why should I work out?",  # Abstract query
        "How can I feel happier?",  # Conceptual query  
        "What helps my brain work better?",  # Indirect query
    ]

    for query in test_queries:
        print("\n")
        print("═" * 70)
        print(f"  Query: {query!r}")
        print("═" * 70)

        # -------------------------------------------------------------------------
        # Regular retrieval (query embedding)
        # -------------------------------------------------------------------------
        print("\n┌─ Regular Retrieval (query → embedding → search)")
        results = await base_retriever.retrieve(query, k=2, include_scores=True)
        
        for i, r in enumerate(results, 1):
            print(f"│  [{i}] Score: {r.score:.3f}")
            text = r.document.text[:80].replace("\n", " ")
            print(f"│      {text}...")
        
        # -------------------------------------------------------------------------
        # HyDE retrieval (query → hypothetical doc → embedding → search)
        # -------------------------------------------------------------------------
        print("│")
        print("├─ HyDE Retrieval (query → hypothetical → embedding → search)")
        
        # Show the hypothetical document that gets generated
        hypothetical = await hyde_retriever.generate_hypothetical(query)
        print(f"│  Hypothetical doc: {hypothetical[:100].replace(chr(10), ' ')}...")
        print("│")
        
        results = await hyde_retriever.retrieve(query, k=2, include_scores=True)
        
        for i, r in enumerate(results, 1):
            print(f"│  [{i}] Score: {r.score:.3f}")
            text = r.document.text[:80].replace("\n", " ")
            print(f"│      {text}...")
        
        print("└" + "─" * 68)

    # =========================================================================
    # Advanced: Multiple Hypotheticals with Fusion
    # =========================================================================
    print("\n")
    print("═" * 70)
    print("  Advanced: Multiple Hypotheticals (n=3)")
    print("═" * 70)
    print("""
When generating multiple hypothetical documents, HyDE searches with each
and fuses the results using RRF (Reciprocal Rank Fusion) for better recall.
""")

    # Create HyDE with multiple hypotheticals
    hyde_ensemble = HyDERetriever(
        base_retriever,
        model,
        n_hypotheticals=3,  # Generate 3 different hypothetical docs
        include_original_query=True,  # Also search with original query
    )

    query = "What's the secret to living longer?"
    print(f"Query: {query!r}")
    print(f"Retriever: n_hypotheticals=3, include_original_query=True\n")

    results = await hyde_ensemble.retrieve(query, k=4, include_scores=True)
    
    print("Results (fused from 4 searches):")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] Score: {r.score:.3f}")
        text = r.document.text[:90].replace("\n", " ")
        print(f"      {text}...")

    # =========================================================================
    # Custom Prompt Template
    # =========================================================================
    print("\n")
    print("═" * 70)
    print("  Custom Prompt Template")
    print("═" * 70)
    
    # Domain-specific prompt for medical/health content
    medical_prompt = """You are a medical textbook. Write a detailed passage that would
appear in a chapter answering this question. Use technical medical terminology.

Question: {query}

Passage from medical textbook:"""

    hyde_medical = HyDERetriever(
        base_retriever,
        model,
        prompt_template=medical_prompt,
    )

    query = "What happens to your body when you don't sleep?"
    print(f"Query: {query!r}\n")
    
    # Show the hypothetical with custom prompt
    hypothetical = await hyde_medical.generate_hypothetical(query)
    print("Generated hypothetical (medical style):")
    print(f"  {hypothetical[:200].replace(chr(10), ' ')}...\n")
    
    results = await hyde_medical.retrieve(query, k=2, include_scores=True)
    print("Retrieved:")
    for i, r in enumerate(results, 1):
        text = r.document.text[:90].replace("\n", " ")
        print(f"  [{i}] {text}...")

    # =========================================================================
    # Using HyDE with RAG Capability
    # =========================================================================
    print("\n")
    print("═" * 70)
    print("  HyDE with Agent (RAG Capability)")
    print("═" * 70)
    
    from agenticflow import Agent
    from agenticflow.capabilities import RAG

    # Use HyDE as the retriever in RAG Capability
    rag = RAG(retriever=hyde_retriever)

    agent = Agent(
        name="HealthAdvisor",
        model=model,
        system_prompt="You are a helpful health advisor. Answer questions based on the provided context.",
        capabilities=[rag],
    )

    answer = await agent.run("I've been feeling really stressed lately. What can I do?")
    print(f"\nAgent response:\n{answer}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n")
    print("═" * 70)
    print("  SUMMARY")
    print("═" * 70)
    print("""
┌──────────────────────────────────────────────────────────────────────┐
│ HyDE (Hypothetical Document Embeddings)                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ WHEN TO USE:                                                         │
│   ✓ Abstract or conceptual queries                                   │
│   ✓ Queries phrased differently than document content                │
│   ✓ Zero-shot retrieval (no training data)                           │
│   ✓ When query-document semantic gap is large                        │
│                                                                      │
│ WHEN NOT TO USE:                                                     │
│   ✗ Simple keyword-matching queries                                  │
│   ✗ High-throughput scenarios (adds LLM latency)                     │
│   ✗ When documents already match query vocabulary                    │
│                                                                      │
│ OPTIONS:                                                             │
│   - n_hypotheticals: Generate multiple docs and fuse results         │
│   - include_original_query: Also search with original query          │
│   - prompt_template: Customize generation for your domain            │
│                                                                      │
│ USAGE:                                                               │
│   hyde = HyDERetriever(base_retriever, model)                        │
│   docs = await hyde.retrieve("abstract question", k=5)               │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
""")
    print("✓ Done")


if __name__ == "__main__":
    asyncio.run(main())
