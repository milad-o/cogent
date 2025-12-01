"""
Example: RAG with Per-File-Type Pipelines

Demonstrates how RAGAgent automatically uses optimal splitters for different
file types, treating markdown, code, JSON, and HTML differently.

Files in examples/data/:
- smarthome_docs.md    → MarkdownSplitter (preserves headers)
- smarthome_devices.py → CodeSplitter (preserves functions/classes)
- smarthome_config.json → RecursiveCharacterSplitter (default)
- smarthome_api.html   → HTMLSplitter (strips tags intelligently)

Configure your API keys in .env file (see config.py for details).
"""

import asyncio
from pathlib import Path

# Use the examples' config which auto-loads .env
from config import get_model, get_embeddings, settings

from agenticflow.prebuilt import RAGAgent, PipelineRegistry, DocumentPipeline


async def main():
    # Path to example data
    data_dir = Path(__file__).parent / "data"
    
    # Create model and embeddings from config
    model = get_model()
    embeddings = get_embeddings()
    
    print("=" * 60)
    print("RAG with Per-File-Type Pipelines")
    print(f"LLM: {settings.get_preferred_provider()} ({model})")
    print(f"Embeddings: {settings.get_preferred_embedding_provider()}")
    print("=" * 60)
    
    # ============================================================
    # Example 1: Default Pipelines (automatic per-type handling)
    # ============================================================
    print("\n📚 Example 1: Default Pipelines")
    print("-" * 40)
    
    rag = RAGAgent(
        model=model,
        embeddings=embeddings,
        verbose=True,  # Show progress
    )
    
    # Load all SmartHome files - each uses its optimal splitter
    await rag.load_documents([
        data_dir / "smarthome_docs.md",      # → MarkdownSplitter
        data_dir / "smarthome_devices.py",   # → CodeSplitter
        data_dir / "smarthome_config.json",  # → RecursiveCharacterSplitter
        data_dir / "smarthome_api.html",     # → HTMLSplitter
    ])
    
    print(f"\n✅ Loaded {rag.document_count} chunks from {len(rag.sources)} sources")
    
    # Ask questions that require different types of documents
    questions = [
        "What protocols does SmartHome support for device communication?",
        "How do I control light brightness in the code?",
        "What are the API rate limits?",
        "What rooms are configured in the system?",
    ]
    
    for q in questions:
        print(f"\n❓ {q}")
        answer = await rag.query(q)
        print(f"💬 {answer[:500]}...")
    
    # ============================================================
    # Example 2: Custom Pipelines (override defaults)
    # ============================================================
    print("\n" + "=" * 60)
    print("📚 Example 2: Custom Pipelines")
    print("-" * 40)
    
    from agenticflow.document import SemanticSplitter, RecursiveCharacterSplitter
    
    # Create custom pipelines with specific configurations
    custom_pipelines = PipelineRegistry(
        pipelines={
            # Use smaller chunks for markdown docs
            ".md": DocumentPipeline(
                splitter=RecursiveCharacterSplitter(chunk_size=500, chunk_overlap=100),
                metadata={"doc_type": "documentation"},
            ),
            # Add metadata to Python files
            ".py": DocumentPipeline(
                metadata={"doc_type": "source_code", "language": "python"},
            ),
        },
    )
    
    rag2 = RAGAgent(
        model=model,
        embeddings=embeddings,
        pipelines=custom_pipelines,
        top_k=3,  # Fewer results, more focused
    )
    
    # Load just markdown and Python
    await rag2.load_documents([
        data_dir / "smarthome_docs.md",
        data_dir / "smarthome_devices.py",
    ])
    
    print(f"\n✅ Loaded {rag2.document_count} chunks (smaller chunks = more)")
    
    # Ask a code-specific question
    answer = await rag2.query("Show me the ThermostatController class and explain its methods")
    print(f"\n❓ ThermostatController question")
    print(f"💬 {answer[:800]}...")
    
    # ============================================================
    # Example 3: RAG with Additional Tools
    # ============================================================
    print("\n" + "=" * 60)
    print("📚 Example 3: RAG + Custom Tools")
    print("-" * 40)
    
    from agenticflow import tool
    
    @tool
    def convert_temperature(celsius: float) -> str:
        """Convert Celsius to Fahrenheit.
        
        Args:
            celsius: Temperature in Celsius.
            
        Returns:
            Temperature in both units.
        """
        fahrenheit = (celsius * 9/5) + 32
        return f"{celsius}°C = {fahrenheit}°F"
    
    @tool  
    def calculate_energy_cost(kwh: float, rate: float = 0.15) -> str:
        """Calculate energy cost.
        
        Args:
            kwh: Energy consumption in kilowatt-hours.
            rate: Price per kWh (default: $0.15).
            
        Returns:
            Formatted cost string.
        """
        cost = kwh * rate
        return f"{kwh} kWh × ${rate}/kWh = ${cost:.2f}"
    
    rag3 = RAGAgent(
        model=model,
        embeddings=embeddings,
        tools=[convert_temperature, calculate_energy_cost],  # Extra tools!
    )
    
    await rag3.load_documents([data_dir / "smarthome_docs.md"])
    
    # This query might use both RAG search AND the calculator tool
    answer = await rag3.query(
        "What are the peak electricity rates in the SmartHome system? "
        "If I use 500 kWh during peak hours, what would it cost?"
    )
    print(f"\n❓ Energy cost question (uses RAG + tool)")
    print(f"💬 {answer}")
    
    # ============================================================
    # Example 4: RAG with Conversation Memory
    # ============================================================
    print("\n" + "=" * 60)
    print("📚 Example 4: RAG with Memory")
    print("-" * 40)
    
    rag4 = RAGAgent(
        model=model,
        embeddings=embeddings,
        memory=True,  # Enable conversation memory
    )
    
    await rag4.load_documents([data_dir / "smarthome_docs.md"])
    
    # Multi-turn conversation
    thread_id = "demo-session"
    
    r1 = await rag4.chat("What security features does SmartHome have?", thread_id=thread_id)
    print(f"\n❓ Q1: Security features?")
    print(f"💬 {r1[:400]}...")
    
    r2 = await rag4.chat("What about the encryption specifically?", thread_id=thread_id)
    print(f"\n❓ Q2: Encryption specifically? (follows up)")
    print(f"💬 {r2[:400]}...")
    
    r3 = await rag4.chat("And how often are keys rotated?", thread_id=thread_id)
    print(f"\n❓ Q3: Key rotation? (continues context)")
    print(f"💬 {r3[:400]}...")
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
