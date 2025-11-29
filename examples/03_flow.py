"""
Demo: Flow Streaming

Watch flow execution updates in real-time using flow.stream().

Usage:
    uv run python examples/03_flow.py
"""

import asyncio
import os

from dotenv import load_dotenv

from agenticflow import Agent, Flow
from agenticflow.models.gemini import GeminiChat

load_dotenv()


async def main():
    model = GeminiChat(model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"))

    researcher = Agent(name="Researcher", model=model, instructions="Research the topic briefly.")
    writer = Agent(name="Writer", model=model, instructions="Write a short summary.")
    reviewer = Agent(name="Reviewer", model=model, instructions="Review and finalize briefly.")

    # verbose=True for basic progress, "verbose" for outputs, "debug" for everything
    flow = Flow(
        name="content",
        agents=[researcher, writer, reviewer],
        topology="pipeline",
        verbose="verbose",
    )

    print("Streaming execution updates:\n")
    async for event in flow.stream("Topic: AI in healthcare"):
        event_type = event.get("type", "unknown")
        if event_type == "status":
            print(f"  📌 Status: {event.get('data', '')}")
        elif event_type == "output":
            output = event.get('data', '')[:100]
            print(f"  📝 Output preview: {output}...")
        elif event_type == "complete":
            print("  ✅ Complete!")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
