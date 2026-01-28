"""
Example: Agent as Tool - Content Review Pipeline

Demonstrates Cogent's approach: powerful single agents with tools,
using agent.as_tool() only when research shows it helps (verification workflows).

This example shows a realistic content review pipeline:
- Writer agent creates marketing copy
- Reviewer agent checks for issues (compliance, tone, clarity)
- Editor orchestrates with structured output and iterates until approved
- Full revision history tracked via Observer events

Run: uv run python examples/advanced/content_review.py
"""

import asyncio
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from cogent import Agent, Observer, tool

load_dotenv()


# === Configuration ===
ACCEPTANCE_THRESHOLD = 7  # Score >= 7 is approved
MAX_REVISIONS = 2  # Max revision cycles to prevent infinite loops


# === Structured Output Schema ===

class ReviewDecision(BaseModel):
    """Editor's decision after the review pipeline."""
    
    status: Literal["approved", "needs_revision"] = Field(
        description="Whether the content is approved or needs revision"
    )
    final_copy: str = Field(
        description="The final marketing copy"
    )
    review_score: int = Field(
        ge=1, le=10,
        description="Review score from 1-10"
    )
    revision_count: int = Field(
        default=0,
        description="Number of revision cycles completed"
    )
    issues: list[str] = Field(
        default_factory=list,
        description="List of issues found (empty if approved)"
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Suggestions for improvement (empty if approved)"
    )


# === Tools ===

@tool
def get_product_info(product_name: str) -> str:
    """Get product specifications and features."""
    products = {
        "cogent": {
            "tagline": "Production Agent Framework",
            "features": ["Tool-augmented agents", "Native model support", "Full observability"],
            "pricing": "Open source",
        },
        "smartwatch": {
            "tagline": "Your Health, Your Way",
            "features": ["Heart rate monitoring", "Sleep tracking", "7-day battery"],
            "pricing": "$299",
        },
    }
    info = products.get(product_name.lower(), {"error": "Product not found"})
    return str(info)


@tool  
def get_brand_guidelines() -> str:
    """Get brand voice and style guidelines."""
    return """
Brand Voice Guidelines:
- Tone: Professional but friendly, never salesy
- Avoid: Superlatives (best, greatest), pressure tactics, excessive emojis
- Include: Specific benefits, clear call-to-action
- Length: Keep copy under 100 words for social media
"""


async def main():
    """Content review pipeline using agents as tools."""
    print("\n" + "=" * 60)
    print("Content Review Pipeline - Agent as Tool")
    print("=" * 60)
    print(f"Acceptance threshold: {ACCEPTANCE_THRESHOLD}/10")
    print(f"Max revisions: {MAX_REVISIONS}")

    # Observer for full execution visibility
    # capture= automatically stores matching events for later retrieval
    observer = Observer(level="trace", capture=["tool.result"])

    # Writer Agent: Creates content using research tools
    writer = Agent(
        name="CopyWriter",
        model="gpt4",
        tools=[get_product_info, get_brand_guidelines],
        instructions="""You are a marketing copywriter.
        
When given a product, use get_product_info to learn about it,
then use get_brand_guidelines to match the brand voice.

Write engaging marketing copy that:
- Highlights key features and benefits
- Follows brand guidelines strictly (NO superlatives, NO pressure tactics)
- Is concise and professional
- Avoids excessive emojis and exclamation marks""",
    )

    # Reviewer Agent: Checks content for issues
    reviewer = Agent(
        name="ContentReviewer",
        model="gpt4",
        instructions=f"""You are a strict content compliance reviewer.

Review marketing copy for:
1. Brand guideline compliance (NO superlatives, NO salesy language)
2. Factual accuracy (matches product features)
3. Clarity and tone (professional, friendly)
4. Legal issues (no misleading claims)

Score guidelines:
- 9-10: Perfect, no issues
- 7-8: Minor issues, acceptable
- 5-6: Significant issues, needs revision
- 1-4: Major violations, must revise

Acceptance threshold is {ACCEPTANCE_THRESHOLD}/10.

Return a structured review:
- Issues: [list problems, or "None"]
- Score: 1-10
- Verdict: APPROVED (score >= {ACCEPTANCE_THRESHOLD}) or NEEDS_REVISION
- Suggestions: [specific changes needed if revision required]""",
    )

    # Editor Agent: Orchestrates with iteration
    editor = Agent(
        name="ContentEditor",
        model="gpt4",
        tools=[
            writer.as_tool(
                description="Write or revise marketing copy. Input: product name OR revision request with feedback"
            ),
            reviewer.as_tool(
                description="Review marketing copy for compliance. Input: the copy to review"
            ),
        ],
        output=ReviewDecision,
        instructions=f"""You are a content editor managing the review pipeline.

ACCEPTANCE THRESHOLD: {ACCEPTANCE_THRESHOLD}/10
MAX REVISIONS: {MAX_REVISIONS}

Process:
1. Use CopyWriter to create initial copy for the product
2. Use ContentReviewer to review the copy
3. If score >= {ACCEPTANCE_THRESHOLD}: APPROVED - return the copy
4. If score < {ACCEPTANCE_THRESHOLD} and revisions < {MAX_REVISIONS}:
   - Call CopyWriter again with the feedback: "Revise this copy: [COPY]. Issues: [ISSUES]. Suggestions: [SUGGESTIONS]"
   - Review the revised copy
   - Repeat until approved or max revisions reached
5. After {MAX_REVISIONS} revisions, return best version with status='needs_revision'

Track revision_count in your response.""",
        observer=observer,
    )

    # Run the pipeline
    print("\n📝 Task: Create Instagram post for SmartWatch\n")

    result = await editor.run(
        "Create an Instagram post for our SmartWatch product"
    )

    # Show revision history from captured events
    print("\n" + "=" * 60)
    print("REVISION HISTORY (from observer.history)")
    print("=" * 60)
    
    version = 0
    for event in observer.history("tool.result"):
        tool_name = event.data.get("tool_name", "")
        result = event.data.get("result", "")
        
        if tool_name == "CopyWriter":
            version += 1
            print(f"\n{'='*50}")
            print(f"VERSION {version} (from {tool_name})")
            print("="*50)
            print(f"\n📝 Copy:\n{result}")
        elif tool_name == "ContentReviewer":
            print(f"\n📋 Review:\n{result}")

    # Access structured output
    print("\n" + "=" * 60)
    print("FINAL DECISION")
    print("=" * 60)
    
    # result is Response[StructuredResult] when output= is configured
    # result.content is StructuredResult, .data is our Pydantic model
    from cogent.agent.output import StructuredResult
    from cogent.core.response import Response
    
    if isinstance(result, Response) and isinstance(result.content, StructuredResult):
        structured = result.content
        if structured.valid:
            decision: ReviewDecision = structured.data
            
            status_emoji = "✅" if decision.status == "approved" else "⚠️"
            print(f"\nStatus: {status_emoji} {decision.status.upper()}")
            print(f"Score: {decision.review_score}/10 (threshold: {ACCEPTANCE_THRESHOLD})")
            print(f"Revisions: {decision.revision_count}/{MAX_REVISIONS}")
            print(f"\nFinal Copy:\n{decision.final_copy}")
            
            if decision.issues:
                print(f"\nRemaining Issues ({len(decision.issues)}):")
                for issue in decision.issues:
                    print(f"  • {issue}")
        else:
            print(f"Structured output failed: {structured.error}")
    else:
        # Fallback: extract content from Response or use raw string
        content = result.content if isinstance(result, Response) else result
        print(f"\nRaw output:\n{content}")

    # Show observability stats
    print("\n" + "=" * 60)
    print("OBSERVABILITY STATS")
    print("=" * 60)
    print(f"\n{observer.summary()}")
    print(f"Captured events: {len(observer.captured)}")


if __name__ == "__main__":
    asyncio.run(main())
