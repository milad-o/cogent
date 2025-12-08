"""
Example 24: Structured Output

Demonstrates AgenticFlow's structured output capability for enforcing
response schemas on agent outputs.

Key features:
- Schema enforcement with Pydantic, dataclass, TypedDict, or JSON Schema
- Automatic validation with retry on errors
- Provider-native support where available (OpenAI, Anthropic)
- Clean API: just pass `output=YourSchema` to Agent

Run: uv run python examples/24_structured_output.py
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import Literal

from config import get_model

from agenticflow import Agent
from agenticflow.agent.output import ResponseSchema, OutputMethod


# =============================================================================
# Example 1: Simple Extraction with Pydantic
# =============================================================================

class ContactInfo(BaseModel):
    """Contact information extracted from text."""
    
    name: str = Field(description="Person's full name")
    email: str = Field(description="Email address")
    phone: str | None = Field(None, description="Phone number if available")
    company: str | None = Field(None, description="Company name if mentioned")


async def example_contact_extraction():
    """Extract structured contact info from text."""
    print("=" * 60)
    print("Example 1: Contact Extraction with Pydantic")
    print("=" * 60)
    
    agent = Agent(
        name="ContactExtractor",
        model=get_model(),
        output=ContactInfo,  # Enforce this schema
        instructions="You are an expert at extracting contact information from text.",
    )
    
    # Test texts
    texts = [
        "Hi, I'm John Doe from Acme Corp. You can reach me at john.doe@acme.com or call 555-123-4567.",
        "Contact Sarah at sarah@startup.io for more details.",
        "The manager, Bob Smith (bob@bigco.com), will be in touch.",
    ]
    
    for text in texts:
        print(f"\nInput: {text}")
        result = await agent.run(f"Extract contact info: {text}")
        
        if result.valid:
            contact = result.data
            print(f"  Name: {contact.name}")
            print(f"  Email: {contact.email}")
            print(f"  Phone: {contact.phone or 'N/A'}")
            print(f"  Company: {contact.company or 'N/A'}")
        else:
            print(f"  Error: {result.error}")


# =============================================================================
# Example 2: Data Classification with Enum
# =============================================================================

class SentimentAnalysis(BaseModel):
    """Sentiment analysis result."""
    
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    key_phrases: list[str] = Field(description="Key phrases that indicate sentiment")
    summary: str = Field(description="Brief summary of the sentiment")


async def example_sentiment_analysis():
    """Classify sentiment with structured output."""
    print("\n" + "=" * 60)
    print("Example 2: Sentiment Analysis with Enum")
    print("=" * 60)
    
    agent = Agent(
        name="SentimentAnalyzer",
        model=get_model(),
        output=SentimentAnalysis,
        instructions="You are an expert sentiment analyst. Analyze the sentiment of the given text.",
    )
    
    reviews = [
        "This product is absolutely amazing! Best purchase I've ever made.",
        "Terrible experience. The item broke after one day. Waste of money.",
        "It's okay, nothing special. Does what it says but nothing more.",
    ]
    
    for review in reviews:
        print(f"\nReview: {review[:50]}...")
        result = await agent.run(f"Analyze sentiment: {review}")
        
        if result.valid:
            analysis = result.data
            print(f"  Sentiment: {analysis.sentiment} ({analysis.confidence:.0%} confidence)")
            print(f"  Key phrases: {', '.join(analysis.key_phrases)}")
            print(f"  Summary: {analysis.summary}")


# =============================================================================
# Example 3: Using Dataclass
# =============================================================================

@dataclass
class MeetingAction:
    """Action item from a meeting."""
    
    task: str
    assignee: str
    priority: str  # high, medium, low
    due_date: str | None = None


async def example_meeting_actions():
    """Extract action items using dataclass schema."""
    print("\n" + "=" * 60)
    print("Example 3: Meeting Actions with Dataclass")
    print("=" * 60)
    
    agent = Agent(
        name="ActionExtractor",
        model=get_model(),
        output=MeetingAction,
        instructions="Extract the most important action item from meeting notes.",
    )
    
    meeting_notes = """
    Team sync call notes:
    - John needs to update the project timeline by Friday
    - Sarah will prepare the Q4 budget report (urgent!)
    - Bob to schedule client demo next week
    """
    
    print(f"Meeting notes: {meeting_notes[:80]}...")
    result = await agent.run(f"Extract key action: {meeting_notes}")
    
    if result.valid:
        action = result.data
        print(f"\n  Task: {action.task}")
        print(f"  Assignee: {action.assignee}")
        print(f"  Priority: {action.priority}")
        print(f"  Due: {action.due_date or 'Not specified'}")


# =============================================================================
# Example 4: Advanced Config with Retry
# =============================================================================

class ProductReview(BaseModel):
    """Validated product review."""
    
    rating: int = Field(ge=1, le=5, description="Rating from 1 to 5 stars")
    title: str = Field(max_length=100, description="Short review title")
    pros: list[str] = Field(description="List of positive points")
    cons: list[str] = Field(description="List of negative points")
    recommendation: bool = Field(description="Would recommend to others")


async def example_advanced_config():
    """Use ResponseSchema for fine-grained control."""
    print("\n" + "=" * 60)
    print("Example 4: Advanced Config with Validation Retry")
    print("=" * 60)
    
    # Full control over structured output behavior
    config = ResponseSchema(
        schema=ProductReview,
        method=OutputMethod.AUTO,  # Let AgenticFlow choose best method
        retry_on_error=True,       # Retry if validation fails
        max_retries=2,             # Up to 2 retries
        include_raw=True,          # Include raw response in result
    )
    
    agent = Agent(
        name="ReviewAnalyzer",
        model=get_model(),
        output=config,
        instructions="Parse product reviews into structured format. Be precise with ratings.",
    )
    
    review_text = """
    Amazing headphones! 10/10 would buy again.
    
    The Good:
    - Incredible sound quality
    - Super comfortable for long sessions
    - Battery lasts forever
    
    The Bad:
    - A bit pricey
    - Case could be better
    
    Overall: Definitely recommend these to anyone!
    """
    
    print(f"Review: {review_text[:60]}...")
    result = await agent.run(f"Parse this review: {review_text}")
    
    print(f"\n  Attempts: {result.attempts}")
    print(f"  Valid: {result.valid}")
    
    if result.valid:
        review = result.data
        print(f"  Rating: {'⭐' * review.rating}")
        print(f"  Title: {review.title}")
        print(f"  Pros: {', '.join(review.pros)}")
        print(f"  Cons: {', '.join(review.cons)}")
        print(f"  Recommends: {'Yes' if review.recommendation else 'No'}")
    
    if result.raw:
        print(f"\n  Raw output preview: {result.raw[:100]}...")


# =============================================================================
# Example 5: JSON Schema (for maximum flexibility)
# =============================================================================

async def example_json_schema():
    """Use raw JSON Schema for dynamic schemas."""
    print("\n" + "=" * 60)
    print("Example 5: Dynamic JSON Schema")
    print("=" * 60)
    
    # Define schema as JSON Schema dict (useful for dynamic schemas)
    event_schema = {
        "type": "object",
        "title": "CalendarEvent",
        "properties": {
            "title": {"type": "string", "description": "Event title"},
            "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
            "time": {"type": "string", "description": "Time in HH:MM format"},
            "duration_minutes": {"type": "integer", "description": "Duration in minutes"},
            "attendees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of attendee names",
            },
        },
        "required": ["title", "date", "time"],
    }
    
    agent = Agent(
        name="CalendarParser",
        model=get_model(),
        output=event_schema,
        instructions="Extract calendar event details from natural language.",
    )
    
    text = "Let's have a team lunch tomorrow at noon for about an hour. Invite Alice, Bob, and Carol."
    
    print(f"Input: {text}")
    result = await agent.run(f"Parse event: {text}")
    
    if result.valid:
        event = result.data
        print(f"\n  Event: {event}")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all examples."""
    print("\n🔧 AgenticFlow Structured Output Examples\n")
    
    await example_contact_extraction()
    await example_sentiment_analysis()
    await example_meeting_actions()
    await example_advanced_config()
    await example_json_schema()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
