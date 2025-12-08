"""
Example: Document Summarization with Agent Capability

Demonstrates using the Summarizer capability to enable agents to autonomously
summarize large documents using various strategies (map-reduce, refine, hierarchical).

Key features:
- Agent can decide when and how to summarize documents
- Multiple strategies for different use cases
- Works with text files, code, markdown, etc.

Usage:
    uv run python examples/35_summarizer.py

Requires:
    Configure API keys in .env (see config.py)
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from pathlib import Path

from config import get_model

from agenticflow import Agent
from agenticflow.capabilities import Summarizer, SummarizerConfig
from agenticflow.capabilities.filesystem import FileSystem


async def main() -> None:
    """Demonstrate the Summarizer capability."""
    
    model = get_model()
    data_dir = Path(__file__).parent / "data"
    
    print("=" * 70)
    print("Summarizer Capability Demo")
    print("=" * 70)
    
    # =========================================================================
    # Example 1: Direct tool usage (without agent)
    # =========================================================================
    print("\n📚 Example 1: Direct Summarizer Tool Usage\n")
    
    # Create summarizer capability with configuration
    summarizer = Summarizer(
        model=model,
        config=SummarizerConfig(
            chunk_size=4000,
            max_concurrent=3,
            default_strategy="map_reduce",
        ),
        allowed_paths=[str(data_dir)],
    )
    
    # Get available tools
    tools = summarizer.tools
    print(f"Available tools: {[t.name for t in tools]}")
    
    # Use the get_summary_strategies tool to see options
    strategies_tool = next(t for t in tools if t.name == "get_summary_strategies")
    strategies = strategies_tool.func()
    
    print("\nSummarization strategies:")
    for name, info in strategies["strategies"].items():
        print(f"  • {info['name']}: {info['description']}")
        print(f"    Best for: {', '.join(info['best_for'])}")
    
    # =========================================================================
    # Example 2: Summarize a text file
    # =========================================================================
    print("\n" + "=" * 70)
    print("📄 Example 2: Summarize a Text File")
    print("=" * 70)
    
    # Read the sample file
    sample_file = data_dir / "the_secret_garden.txt"
    if sample_file.exists():
        content = sample_file.read_text()
        print(f"\nFile: {sample_file.name}")
        print(f"Size: {len(content):,} characters")
        
        # Get the summarize_text tool
        summarize_text = next(t for t in tools if t.name == "summarize_text")
        
        # Summarize with map-reduce strategy
        print("\n🔄 Summarizing with map-reduce strategy...")
        result = await summarize_text.ainvoke({
            "text": content,
            "strategy": "map_reduce",
            "context": "classic literature excerpt",
        })
        
        print(f"\n📝 Summary ({result['summary_length']:,} chars, {result['reduction_ratio']}x reduction):")
        print("-" * 50)
        print(result["summary"])
        print("-" * 50)
        print(f"Chunks processed: {result['chunks_processed']}")
    else:
        print(f"⚠️  Sample file not found: {sample_file}")
    
    # =========================================================================
    # Example 3: Agent with Summarizer capability
    # =========================================================================
    print("\n" + "=" * 70)
    print("🤖 Example 3: Agent with Summarizer Capability")
    print("=" * 70)
    
    # Create an agent with summarizer and filesystem capabilities
    agent = Agent(
        name="DocumentAnalyzer",
        model=model,
        instructions="""You are a document analysis assistant. You can:
        1. Read files from the data directory
        2. Summarize documents using various strategies
        3. Answer questions about document content
        
        When asked to summarize, choose the appropriate strategy:
        - map_reduce: For large documents, fastest
        - refine: For documents needing coherent narrative
        - hierarchical: For very large documents (books)
        """,
        capabilities=[
            Summarizer(
                model=model,
                config=SummarizerConfig(chunk_size=4000),
                allowed_paths=[str(data_dir)],
            ),
            FileSystem(allowed_paths=[str(data_dir)]),
        ],
    )
    
    # Let the agent decide how to summarize
    print("\n🎯 Task: 'Summarize the company knowledge document briefly'")
    
    knowledge_file = data_dir / "company_knowledge.txt"
    if knowledge_file.exists():
        result = await agent.run(
            f"Please summarize the file at {knowledge_file}. "
            "Use the refine strategy for better coherence. "
            "Give me a brief executive summary."
        )
        
        print(f"\n🤖 Agent Response:")
        print("-" * 50)
        print(result.output)
    else:
        print(f"⚠️  File not found: {knowledge_file}")
    
    # =========================================================================
    # Example 4: Compare summarization strategies
    # =========================================================================
    print("\n" + "=" * 70)
    print("⚖️  Example 4: Compare Summarization Strategies")
    print("=" * 70)
    
    if sample_file.exists():
        content = sample_file.read_text()
        summarize_text = next(t for t in summarizer.tools if t.name == "summarize_text")
        
        strategies_to_compare = ["map_reduce", "refine"]
        
        for strategy in strategies_to_compare:
            print(f"\n📊 Strategy: {strategy}")
            result = await summarize_text.ainvoke({
                "text": content[:8000],  # Use first 8K chars for comparison
                "strategy": strategy,
                "context": "book excerpt",
            })
            print(f"   Chunks: {result['chunks_processed']}")
            print(f"   Reduction: {result['reduction_ratio']}x")
            print(f"   Summary length: {result['summary_length']} chars")
            # Show first 200 chars of summary
            preview = result["summary"][:200] + "..." if len(result["summary"]) > 200 else result["summary"]
            print(f"   Preview: {preview}")
    
    print("\n" + "=" * 70)
    print("✅ Summarizer capability demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
