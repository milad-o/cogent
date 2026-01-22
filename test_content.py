import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "examples"))
from config import get_model

from agenticflow import Agent

async def test():
    # Test the exact question from the example
    print("\n=== Testing: Name three closest stars to Earth ===")
    agent = Agent(name='Assistant', model=get_model(), verbose=True)
    r = await agent.run('Name three closest stars to Earth.')
    print(f'Content: [{r.content}]')
    print(f'Content length: {len(r.content) if r.content else 0}')
    print(f'Messages: {len(r.messages)}')
    
    # Try a different question
    print("\n=== Testing: What is Python? ===")
    agent2 = Agent(name='Assistant2', model=get_model(), verbose=True)
    r2 = await agent2.run('What is Python?')
    print(f'Content: [{r2.content}]')
    print(f'Content length: {len(r2.content) if r2.content else 0}')

asyncio.run(test())
