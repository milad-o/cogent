"""
Example: Using GitHub Models (Azure AI Foundry)

Demonstrates how to use GitHub Models for free-tier LLM inference
via Azure AI Foundry SDK.

Requirements:
    - GITHUB_TOKEN environment variable
    - uv add azure-ai-inference
"""

import asyncio
import os

from agenticflow.models.azure import AzureAIFoundryChat


async def main():
    """Run GitHub Models examples."""
    
    # Check for token
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("⚠️  Set GITHUB_TOKEN environment variable")
        print("   Get token: https://github.com/settings/tokens")
        return
    
    print("=" * 70)
    print("GitHub Models (Azure AI Foundry) Examples")
    print("=" * 70)
    
    # Example 1: Basic chat
    print("\n1. Basic Chat")
    print("-" * 70)
    
    llm = AzureAIFoundryChat.from_github(
        model="meta/Meta-Llama-3.1-8B-Instruct",
        token=token,
    )
    
    response = await llm.ainvoke([
        {"role": "user", "content": "What is the capital of France?"}
    ])
    
    print(f"Response: {response.content}")
    
    # Example 2: Multi-turn conversation
    print("\n2. Multi-turn Conversation")
    print("-" * 70)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What about Spain?"},
    ]
    
    response = await llm.ainvoke(messages)
    print(f"Response: {response.content}")
    
    # Example 3: Streaming
    print("\n3. Streaming Response")
    print("-" * 70)
    print("Response: ", end="", flush=True)
    
    async for chunk in llm.astream([
        {"role": "user", "content": "Give me 3 reasons to exercise daily."}
    ]):
        print(chunk.content, end="", flush=True)
    
    print("\n")
    
    # Example 4: Using create_chat factory
    print("\n4. Using Factory Function")
    print("-" * 70)
    
    from agenticflow.models import create_chat
    
    llm = create_chat(
        provider="github",
        model="meta/Meta-Llama-3.1-8B-Instruct",
        token=token,
    )
    
    response = await llm.ainvoke([
        {"role": "user", "content": "What is 2+2?"}
    ])
    
    print(f"Response: {response.content}")
    
    # Example 5: Different models
    print("\n5. Available Models")
    print("-" * 70)
    print("""
    Free GitHub Models:
    - meta/Meta-Llama-3.1-8B-Instruct
    - meta/Meta-Llama-3.1-70B-Instruct
    - meta/Meta-Llama-3.1-405B-Instruct
    - mistralai/Mistral-7B-Instruct-v0.2
    - mistralai/Mixtral-8x7B-Instruct-v0.1
    - microsoft/Phi-3-mini-4k-instruct
    - microsoft/Phi-3-medium-4k-instruct
    - cohere/command-r-plus
    
    See: https://github.com/marketplace/models
    """)
    
    print("\n" + "=" * 70)
    print("✓ All examples completed")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
