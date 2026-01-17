"""Quick test to see what traces Flow generates"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model

from agenticflow import Agent
from agenticflow.flow.reactive import ReactiveFlow
from agenticflow.observability import Observer


async def main():
    model = get_model()
    
    agent1 = Agent(name="Agent1", model=model, system_prompt="Say hello")
    agent2 = Agent(name="Agent2", model=model, system_prompt="Say goodbye")
    
    observer = Observer.trace()
    flow = ReactiveFlow(observer=observer)
    
    flow.register(agent1, on="start", emits="step1.done")
    flow.register(agent2, on="step1.done", emits="flow.done")
    
    await flow.run("test", initial_event="start")
    
    print("\nAll traces captured:")
    for observed in observer.events():
        print(f"  {observed.type.value}: {observed.data}")


if __name__ == "__main__":
    asyncio.run(main())
