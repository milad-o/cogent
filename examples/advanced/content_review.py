"""Content Review Pipeline - Agent as Tool with Memory + ACC + Cache.

Run: uv run python examples/advanced/content_review.py
"""

import asyncio
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from cogent import Agent, Observer, tool
from cogent.memory import Memory
from cogent.memory.acc import AgentCognitiveCompressor

load_dotenv()

ACCEPTANCE_THRESHOLD = 7
MAX_REVISIONS = 1


class ReviewDecision(BaseModel):
    """Editor's final decision."""

    status: Literal["approved", "needs_revision"]
    final_copy: str
    review_score: int = Field(ge=1, le=10)
    revision_count: int = 0


@tool
def get_product_info(product_name: str) -> str:
    """Get product specs."""
    return "SmartWatch: Heart rate, sleep tracking, 7-day battery, $299"


async def main():
    print("Content Review Pipeline\n" + "=" * 40)

    observer = Observer(level="trace")

    # ACC with model-based extraction (uses efficient model for semantic compression)
    # Options:
    #   extraction_mode="heuristic" - Fast, rule-based (default)
    #   extraction_mode="model" - LLM-based semantic extraction
    #   model="gpt-4o-mini" - Specify efficient model for extraction
    #   model=None - Uses agent's model when extraction_mode="model"
    acc = AgentCognitiveCompressor(
        max_constraints=5,
        max_entities=15,
        max_actions=10,
        max_context=10,
        extraction_mode="heuristic",  # Fast mode for comparison
    )
    memory = Memory(acc=acc)

    # Writer agent
    writer = Agent(
        name="CopyWriter",
        model="gpt-4o-mini",
        tools=[get_product_info],
        instructions="Write concise marketing copy. No superlatives, no salesy language.",
    )

    # Reviewer agent
    reviewer = Agent(
        name="ContentReviewer",
        model="gpt-4o-mini",
        instructions=f"Review copy. Score 1-10. APPROVED if >= {ACCEPTANCE_THRESHOLD}. Be concise.",
    )

    # Editor orchestrates
    editor = Agent(
        name="ContentEditor",
        model="gpt-4o-mini",
        tools=[
            writer.as_tool(description="Write marketing copy"),
            reviewer.as_tool(description="Review copy for compliance"),
        ],
        output=ReviewDecision,
        instructions=f"""Create copy, review it. If score < {ACCEPTANCE_THRESHOLD}, revise once. Return structured result.""",
        observer=observer,
        memory=memory,
        cache=True,
    )

    # Run first task
    result = await editor.run("Create a tweet for SmartWatch", thread_id="session-1")

    # Output
    print("\n" + "=" * 40)
    from cogent.agent.output import StructuredResult
    from cogent.core.response import Response

    if isinstance(result, Response) and isinstance(result.content, StructuredResult):
        if result.content.valid:
            d = result.content.data
            print(
                f"✅ {d.status.upper()} | Score: {d.review_score}/10 | Revisions: {d.revision_count}"
            )
            print(f"\n{d.final_copy}")
    else:
        print(result)

    # Stats after first run
    print(
        f"\n📊 Cache: {editor.cache.get_metrics()['cache_hit_rate']:.0%} hit rate"
        if editor.cache
        else ""
    )
    print(
        f"🧠 ACC: {len(acc.state.entities)} entities, {len(acc.state.actions)} actions"
    )

    # Second run - ACC should remember context from first run
    print("\n" + "=" * 40)
    print("Second run (ACC remembers context)...")
    result2 = await editor.run(
        "Now create a LinkedIn post for the same product", thread_id="session-1"
    )

    if isinstance(result2, Response) and isinstance(result2.content, StructuredResult):
        if result2.content.valid:
            d = result2.content.data
            print(f"✅ {d.status.upper()} | Score: {d.review_score}/10")
            print(f"\n{d.final_copy}")

    print(
        f"\n🧠 ACC after 2 runs: {len(acc.state.entities)} entities, {len(acc.state.actions)} actions"
    )


if __name__ == "__main__":
    asyncio.run(main())
