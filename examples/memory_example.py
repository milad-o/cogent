"""
Memory Example - demonstrates memory systems.

This example shows:
1. Checkpointers (short-term memory)
2. Stores (long-term memory)
3. Vector stores (semantic memory)
4. Memory with LangGraph integration

Usage:
    uv run python examples/memory_example.py
"""

import asyncio

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from agenticflow.memory import (
    # Checkpointers
    memory_checkpointer,
    create_checkpointer,
    # Stores
    memory_store,
    create_store,
    # Vector stores
    memory_vectorstore,
    create_vectorstore,
)


# =============================================================================
# Example 1: Short-term Memory (Checkpointers)
# =============================================================================

async def checkpointer_example():
    """Demonstrate checkpointers for short-term memory."""
    print("\n" + "=" * 60)
    print("Example 1: Short-term Memory (Checkpointers)")
    print("=" * 60)
    
    # Create an in-memory checkpointer
    checkpointer = memory_checkpointer()
    
    print(f"\n📝 Checkpointer type: {type(checkpointer).__name__}")
    
    # Checkpointers are used with LangGraph graphs
    # They save conversation state per thread_id
    print("""
   Checkpointers provide:
   - Per-thread conversation history
   - Automatic state persistence
   - Support for multiple backends (memory, sqlite, postgres)
   
   Usage with LangGraph:
   ```python
   graph = builder.compile(checkpointer=checkpointer)
   result = graph.invoke(
       input,
       {"configurable": {"thread_id": "user-123"}}
   )
   ```
   """)
    
    # Show available backends
    print("\n📋 Available checkpointer backends:")
    print("   - memory: In-memory (MemorySaver)")
    print("   - sqlite: SQLite database (SqliteSaver)")
    print("   - postgres: PostgreSQL database (PostgresSaver)")
    
    # Create with factory
    cp_memory = create_checkpointer("memory")
    print(f"\n✓ Created memory checkpointer: {type(cp_memory).__name__}")
    
    print("\n✅ Checkpointer example complete")


# =============================================================================
# Example 2: Long-term Memory (Stores)
# =============================================================================

async def store_example():
    """Demonstrate stores for long-term memory."""
    print("\n" + "=" * 60)
    print("Example 2: Long-term Memory (Stores)")
    print("=" * 60)
    
    # Create an in-memory store
    store = memory_store()
    
    print(f"\n📝 Store type: {type(store).__name__}")
    
    # Store user preferences (persists across threads)
    namespace = ("user", "alice", "preferences")
    
    store.put(namespace, "theme", {"value": "dark", "updated_at": "2024-01-01"})
    store.put(namespace, "language", {"value": "English"})
    store.put(namespace, "notifications", {"value": True, "email": True, "push": False})
    
    print(f"\n📦 Stored preferences for user 'alice':")
    
    # Retrieve all items in namespace
    items = store.search(namespace)
    for item in items:
        print(f"   {item.key}: {item.value}")
    
    # Get specific item
    theme = store.get(namespace, "theme")
    print(f"\n🔍 Retrieved theme: {theme.value if theme else 'not found'}")
    
    # Store memories for a different user
    bob_namespace = ("user", "bob", "preferences")
    store.put(bob_namespace, "theme", {"value": "light"})
    
    bob_items = store.search(bob_namespace)
    print(f"\n👤 Bob's preferences: {len(bob_items)} item(s)")
    
    # Delete an item
    store.delete(namespace, "notifications")
    remaining = store.search(namespace)
    print(f"\n🗑️ After deletion: {len(remaining)} items remaining")
    
    print("""
   Stores provide:
   - Namespaced key-value storage
   - Cross-thread memory (user memories, facts, etc.)
   - Suitable for user profiles, learned facts, etc.
   """)
    
    print("\n✅ Store example complete")


# =============================================================================
# Example 3: Semantic Store
# =============================================================================

async def semantic_store_example():
    """Demonstrate semantic search in stores."""
    print("\n" + "=" * 60)
    print("Example 3: Semantic Store")
    print("=" * 60)
    
    # Note: Semantic search requires embeddings model
    # For demo, we'll show the API without actual embeddings
    
    print("""
   Semantic stores add embedding-based search:
   
   ```python
   from langchain_openai import OpenAIEmbeddings
   
   embeddings = OpenAIEmbeddings()
   store = semantic_store(embeddings, dims=1536)
   
   # Store with automatic embedding
   store.put(namespace, key, {"text": "Content to embed"})
   
   # Search by similarity
   results = store.search(namespace, query="related query")
   ```
   
   Use cases:
   - Finding similar memories
   - Contextual retrieval
   - Semantic deduplication
   """)
    
    print("\n✅ Semantic store example complete")


# =============================================================================
# Example 4: Vector Stores
# =============================================================================

async def vectorstore_example():
    """Demonstrate vector stores for RAG."""
    print("\n" + "=" * 60)
    print("Example 4: Vector Stores")
    print("=" * 60)
    
    # Create sample documents
    docs = [
        Document(
            page_content="AgenticFlow is a multi-agent framework for Python.",
            metadata={"source": "docs", "section": "intro"}
        ),
        Document(
            page_content="Memory in AgenticFlow supports short-term and long-term storage.",
            metadata={"source": "docs", "section": "memory"}
        ),
        Document(
            page_content="Agents communicate through an event-driven architecture.",
            metadata={"source": "docs", "section": "events"}
        ),
        Document(
            page_content="Topologies define how agents coordinate: supervisor, mesh, pipeline.",
            metadata={"source": "docs", "section": "topologies"}
        ),
    ]
    
    print(f"\n📄 Sample documents: {len(docs)}")
    for i, doc in enumerate(docs, 1):
        print(f"   {i}. {doc.page_content[:50]}...")
    
    print("""
   Vector stores provide:
   - Document embedding and storage
   - Similarity search for RAG
   - Metadata filtering
   
   Available backends:
   - memory: In-memory (InMemoryVectorStore)
   - faiss: FAISS library
   - chroma: ChromaDB
   
   Example usage:
   ```python
   from langchain_openai import OpenAIEmbeddings
   
   embeddings = OpenAIEmbeddings()
   vectorstore = memory_vectorstore(embeddings)
   
   # Add documents
   vectorstore.add_documents(docs)
   
   # Similarity search
   results = vectorstore.similarity_search("multi-agent", k=2)
   ```
   """)
    
    print("\n📋 Vector store factory:")
    print("   create_vectorstore('memory')  # In-memory")
    print("   create_vectorstore('faiss')   # FAISS")
    print("   create_vectorstore('chroma')  # ChromaDB")
    
    print("\n✅ Vector store example complete")


# =============================================================================
# Example 5: Memory Integration Patterns
# =============================================================================

async def integration_patterns_example():
    """Demonstrate memory integration patterns."""
    print("\n" + "=" * 60)
    print("Example 5: Memory Integration Patterns")
    print("=" * 60)
    
    print("""
   📋 Memory Types Summary:
   
   1. CHECKPOINTER (Short-term Memory)
      - Per-conversation state
      - Automatic message history
      - Thread-scoped
      - Use: LangGraph compilation
      
   2. STORE (Long-term Memory)
      - Cross-conversation persistence
      - Namespaced key-value
      - User-scoped or global
      - Use: User preferences, learned facts
      
   3. VECTOR STORE (Semantic Memory)
      - Document embeddings
      - Similarity search
      - RAG retrieval
      - Use: Knowledge base, context injection

   📋 Integration Pattern:
   
   ```python
   from agenticflow.memory import (
       memory_checkpointer,
       memory_store,
   )
   from agenticflow.topologies import TopologyFactory, TopologyType
   
   # Create memory components
   checkpointer = memory_checkpointer()  # Thread-local state
   store = memory_store()                 # Cross-thread memory
   
   # Create topology with memory
   topology = TopologyFactory.create(
       TopologyType.SUPERVISOR,
       "my-team",
       agents=[...],
       checkpointer=checkpointer,
       store=store,
   )
   
   # Run - memory handled automatically
   result = await topology.run(
       "Process this request",
       config={"configurable": {"thread_id": "user-123"}}
   )
   ```
   """)
    
    print("\n✅ Integration patterns example complete")


# =============================================================================
# Example 6: Memory with Agents
# =============================================================================

async def memory_with_agents_example():
    """Demonstrate memory usage with agents."""
    print("\n" + "=" * 60)
    print("Example 6: Memory with Agents")
    print("=" * 60)
    
    from agenticflow import Agent, AgentConfig, EventBus
    
    event_bus = EventBus()
    checkpointer = memory_checkpointer()
    store = memory_store()
    
    # Store agent knowledge
    agent_namespace = ("agent", "researcher", "knowledge")
    store.put(agent_namespace, "specialty", {"value": "AI research"})
    store.put(agent_namespace, "style", {"value": "thorough and detailed"})
    
    # Agent can access its stored knowledge
    knowledge = store.search(agent_namespace)
    
    print(f"\n📦 Agent stored knowledge:")
    for item in knowledge:
        print(f"   {item.key}: {item.value['value']}")
    
    # Create agent (memory can be passed to agent for future use)
    config = AgentConfig(
        name="Researcher",
        description="An AI researcher agent with memory",
        metadata={
            "has_checkpointer": True,
            "has_store": True,
        },
    )
    
    agent = Agent(config=config, event_bus=event_bus)
    
    print(f"\n🤖 Agent: {agent.name}")
    print(f"   Memory metadata: {agent.config.metadata}")
    
    print("\n✅ Memory with agents example complete")


# =============================================================================
# Example 7: Conversation Memory Pattern
# =============================================================================

async def conversation_memory_example():
    """Demonstrate conversation memory pattern."""
    print("\n" + "=" * 60)
    print("Example 7: Conversation Memory Pattern")
    print("=" * 60)
    
    store = memory_store()
    
    # Simulate storing conversation context
    def save_conversation_summary(thread_id: str, summary: str):
        namespace = ("conversations", thread_id)
        store.put(namespace, "summary", {"text": summary})
    
    def get_conversation_summary(thread_id: str) -> str | None:
        namespace = ("conversations", thread_id)
        item = store.get(namespace, "summary")
        return item.value["text"] if item else None
    
    # Save summaries for different conversations
    save_conversation_summary("thread-1", "User asked about Python programming")
    save_conversation_summary("thread-2", "User needed help with data analysis")
    save_conversation_summary("thread-3", "User wanted to build a chatbot")
    
    print("\n💬 Stored conversation summaries:")
    for thread_id in ["thread-1", "thread-2", "thread-3"]:
        summary = get_conversation_summary(thread_id)
        print(f"   {thread_id}: {summary}")
    
    # List all conversations
    all_convos = []
    for i in range(1, 4):
        namespace = ("conversations", f"thread-{i}")
        items = store.search(namespace)
        if items:
            all_convos.append(f"thread-{i}")
    
    print(f"\n📋 Total conversations: {len(all_convos)}")
    
    print("\n✅ Conversation memory example complete")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all memory examples."""
    print("\n" + "🧠 " * 20)
    print("AgenticFlow Memory Examples")
    print("🧠 " * 20)
    
    await checkpointer_example()
    await store_example()
    await semantic_store_example()
    await vectorstore_example()
    await integration_patterns_example()
    await memory_with_agents_example()
    await conversation_memory_example()
    
    print("\n" + "=" * 60)
    print("All memory examples complete!")
    print("=" * 60)
    
    print("\n💡 Memory features:")
    print("   - Checkpointers: Thread-scoped conversation state")
    print("   - Stores: Cross-thread persistent memory")
    print("   - Vector stores: Semantic search and RAG")
    print("   - Factory functions for easy creation")
    print("   - Integration with LangGraph and agents")


if __name__ == "__main__":
    asyncio.run(main())
