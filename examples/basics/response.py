"""
Demo: Response Protocol - Structured Agent Results

The Response[T] protocol provides a structured way to work with agent results:
- Typed content with metadata
- Token usage tracking
- Tool call history
- Full conversation messages
- Error information

Every agent.run() and agent.think() returns a Response object.

Usage:
    uv run python examples/basics/response.py
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_model

from agenticflow import Agent


def search_web(query: str) -> str:
    """Search the web for information.
    
    Args:
        query: Search query
        
    Returns:
        Mock search results
    """
    return f"Found 3 results for '{query}': Python is a programming language..."


async def main():
    model = get_model()
    
    # Example 1: Basic Response
    print("\n=== Basic Response ===")
    
    agent = Agent(
        name="Assistant",
        model=model,
        instructions="Be helpful and concise.",
    )
    
    response = await agent.run("What is 2 + 2?")
    
    print(f"Content: {response.content}")
    print(f"Agent: {response.metadata.agent}")
    print(f"Model: {response.metadata.model}")
    print(f"Duration: {response.metadata.duration:.2f}s")
    
    if response.metadata.tokens:
        print(f"Tokens: {response.metadata.tokens.total_tokens}")
    
    # Example 2: With Tools  
    print("\n=== Response with Tools ===")
    
    agent_with_tools = Agent(
        name="Calculator",
        model=model,
        tools=[search_web],
    )
    
    response = await agent_with_tools.run("Use the search_web tool to find info about Python")
    
    print(f"Content: {response.content}")
    print(f"Tool calls: {len(response.tool_calls)}")
    
    for tc in response.tool_calls:
        print(f"  - {tc.tool_name} ({tc.duration:.3f}s)")
    
    # Example 3: Conversation Memory
    print("\n=== Multi-Turn Conversations ===")
    
    chatbot = Agent(name="Chatbot", model=model)
    
    r1 = await chatbot.run("My favorite color is blue", thread_id="conv-1")
    r2 = await chatbot.run("What's my favorite color?", thread_id="conv-1")
    
    print(f"Turn 1: {r1.content or '(received)'}")
    print(f"Turn 2: {r2.content or '(received)'}")
    
    # Example 4: Error Handling
    print("\n=== Error Handling ===")
    
    print(f"Success: {response.success}")
    if response.error:
        print(f"Error: {response.error.message}")
    
    # Example 5: Safe Content Access
    print("\n=== Safe Content Access ===")
    
    new_response = await agent.run("Say hello!")
    
    if new_response.success:
        print(f"Result: {new_response.content}")
    else:
        print(f"Error: {new_response.error.message if new_response.error else 'Unknown'}")



if __name__ == "__main__":
    asyncio.run(main())
