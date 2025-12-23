"""
Example: Flexible Model Input Types

Demonstrates how AgenticFlow models now accept various input formats:
- Simple strings
- List of dicts (standard format)
- List of message objects
- Mixed lists

This makes the API more user-friendly while maintaining backward compatibility.
"""

import asyncio
from agenticflow.core.messages import HumanMessage, SystemMessage, AIMessage
from agenticflow.models.mock import MockChatModel


async def main():
    """Demonstrate flexible model inputs."""
    
    print("=" * 70)
    print("Flexible Model Input Examples")
    print("=" * 70)
    
    # Create a mock model for demonstration
    model = MockChatModel(responses=[
        "Hello! How can I help you?",
        "Sure, I can help with that!",
        "Let me think about that...",
    ])
    
    # Example 1: Simple string input (most convenient)
    print("\n1. Simple String Input")
    print("-" * 70)
    response = await model.ainvoke("Hello, what can you do?")
    print(f"Input: \"Hello, what can you do?\"")
    print(f"Response: {response.content}")
    
    # Example 2: Traditional dict list (standard format)
    print("\n2. Traditional Dict List")
    print("-" * 70)
    response = await model.ainvoke([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Can you help me?"},
    ])
    print("Input: List of message dicts")
    print(f"Response: {response.content}")
    
    # Example 3: Message objects (type-safe)
    print("\n3. Message Objects")
    print("-" * 70)
    response = await model.ainvoke([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="I need help with Python."),
    ])
    print("Input: SystemMessage + HumanMessage objects")
    print(f"Response: {response.content}")
    
    # Example 4: Mixed list (flexible)
    print("\n4. Mixed List (Dicts + Objects)")
    print("-" * 70)
    response = await model.ainvoke([
        SystemMessage(content="You are a helpful assistant."),
        {"role": "user", "content": "What's the weather?"},
        AIMessage(content="Let me check that for you."),
        {"role": "user", "content": "Thanks!"},
    ])
    print("Input: Mixed SystemMessage, dicts, and AIMessage")
    print(f"Response: {response.content}")
    
    # Example 5: Sync invoke also works
    print("\n5. Synchronous Invoke")
    print("-" * 70)
    model.reset()  # Reset to start from first response
    response = model.invoke("Quick question!")
    print(f"Input: \"Quick question!\"")
    print(f"Response: {response.content}")
    
    # Example 6: All models support this
    print("\n6. Works with All Models")
    print("-" * 70)
    print("""
    All chat models now support flexible inputs:
    - OpenAIChat
    - AnthropicChat
    - GeminiChat
    - AzureOpenAIChat
    - AzureAIFoundryChat
    - CohereChat
    - CloudflareChat
    - GroqChat
    - OllamaChat
    - CustomChat
    - MockChatModel
    
    Example with any model:
        from agenticflow.models.openai import OpenAIChat
        
        llm = OpenAIChat()
        
        # All these work:
        response = await llm.ainvoke("What is 2+2?")
        response = await llm.ainvoke([{"role": "user", "content": "What is 2+2?"}])
        response = await llm.ainvoke([HumanMessage(content="What is 2+2?")])
    """)
    
    print("\n" + "=" * 70)
    print("Benefits:")
    print("=" * 70)
    print("""
    1. Convenience: Use simple strings for quick queries
    2. Type Safety: Use message objects for better IDE support
    3. Flexibility: Mix formats as needed
    4. Backward Compatible: All existing code continues to work
    5. Multimodal Support: Vision APIs work with complex content
    """)


if __name__ == "__main__":
    asyncio.run(main())
