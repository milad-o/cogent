"""
Example 24: Structured Output

Demonstrates Cogent's comprehensive structured output system for enforcing
response schemas on agent outputs.

Key features:
- Schema enforcement with Pydantic, dataclass, TypedDict, or JSON Schema
- Bare types: str, int, bool, float, Literal[...]
- Collections: list, list[T], set, set[T], tuple, tuple[T, ...]
- Union types: Union[A, B] for polymorphic responses
- Enum types: class MyEnum(str, Enum)
- None type: for confirmation responses
- dict type: for agent-decided dynamic structures
- Automatic validation with retry on errors
- Provider-native support where available (OpenAI, Anthropic)

Examples covered:
1. Bare Types - Direct values (str, int, Literal)
2. Contact Extraction - Pydantic models with validation
3. Sentiment Analysis - Literal enums
4. Meeting Actions - Dataclass schemas
5. Advanced Config - ResponseSchema with retry
6. JSON Schema - Raw dict schemas
7. Dynamic Structure - Agent-decided fields (dict)
8. Collections - list, set, tuple (bare and typed)
9. Union Types - Polymorphic responses
10. Enum Types - Type-safe choices
11. None Type - Confirmation responses
12. Complex Nested - Combining all features

Run: uv run python examples/basics/structured_output.py
"""

import asyncio
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from cogent import Agent

# =============================================================================
# Example 1: Bare Types - Simplest Possible
# =============================================================================

async def example_bare_types():
    """Bare types return values directly without wrapping."""
    print("=" * 60)
    print("Example 1: Bare Types (Direct Values)")
    print("=" * 60)
    
    # Bare Literal - constrained choices
    agent = Agent(
        name="Reviewer",
        model="gpt-4o-mini",
        output=Literal["PROCEED", "REVISE"],
        instructions="Review the implementation status and decide.",
    )
    result = await agent.run("Auth system 80% done, missing password reset")
    print(f"\n  Bare Literal: {result.content.data}")  # Direct value: "REVISE"
    
    # Bare str
    agent = Agent(
        name="Summarizer",
        model="gpt-4o-mini",
        output=str,
        instructions="Summarize in one word.",
    )
    result = await agent.run("Python is a programming language")
    print(f"  Bare str: {result.content.data}")  # Direct value: "Python"
    
    # Bare int
    agent = Agent(
        name="Counter",
        model="gpt-4o-mini",
        output=int,
        instructions="Count items.",
    )
    result = await agent.run("How many words: Hello world test")
    print(f"  Bare int: {result.content.data}")  # Direct value: 3
    
    # Single-field models (when you need named fields)
    class Priority(BaseModel):
        level: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    
    agent = Agent(name="Prioritizer", model="gpt-4o-mini", output=Priority)
    result = await agent.run("DB errors affecting 5% users")
    print(f"  Single-field model: {result.content.data.level}")  # Access via field name


# =============================================================================
# Example 2: Contact Extraction with Pydantic
# =============================================================================

class ContactInfo(BaseModel):
    """Contact information extracted from text."""

    name: str = Field(description="Person's full name")
    email: str = Field(description="Email address")
    phone: str | None = Field(None, description="Phone number if available")
    company: str | None = Field(None, description="Company name if mentioned")


async def example_contact_extraction():
    """Extract structured contact info from text."""
    print("\n" + "=" * 60)
    print("Example 2: Contact Extraction with Pydantic")
    print("=" * 60)

    agent = Agent(
        name="ContactExtractor",
        model="gpt4",
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
        structured = result.content

        if structured.valid:
            contact = structured.data
            print(f"  Name: {contact.name}")
            print(f"  Email: {contact.email}")
            print(f"  Phone: {contact.phone or 'N/A'}")
            print(f"  Company: {contact.company or 'N/A'}")
        else:
            print(f"  Error: {structured.error}")


# =============================================================================
# Example 3: Data Classification with Enum
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
    print("Example 3: Sentiment Analysis with Enum")
    print("=" * 60)

    agent = Agent(
        name="SentimentAnalyzer",
        model="gpt4",
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
        structured = result.content

        if structured.valid:
            analysis = structured.data
            print(f"  Sentiment: {analysis.sentiment} ({analysis.confidence:.0%} confidence)")
            print(f"  Key phrases: {', '.join(analysis.key_phrases)}")
            print(f"  Summary: {analysis.summary}")


# =============================================================================
# Example 4: Using Dataclass
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
    print("Example 4: Meeting Actions with Dataclass")
    print("=" * 60)

    agent = Agent(
        name="ActionExtractor",
        model="gpt4",
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
    structured = result.content

    if structured.valid:
        action = structured.data
        print(f"\n  Task: {action.task}")
        print(f"  Assignee: {action.assignee}")
        print(f"  Priority: {action.priority}")
        print(f"  Due: {action.due_date or 'Not specified'}")


# =============================================================================
# Example 5: Advanced Config with Retry
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
    print("Example 5: Advanced Config with Validation Retry")
    print("=" * 60)

    # Import only when needed for advanced use cases
    from cogent.agent.output import OutputMethod, ResponseSchema

    # Full control over structured output behavior
    config = ResponseSchema(
        schema=ProductReview,
        method=OutputMethod.AUTO,  # Let Cogent choose best method
        retry_on_error=True,       # Retry if validation fails
        max_retries=2,             # Up to 2 retries
        include_raw=True,          # Include raw response in result
    )

    agent = Agent(
        name="ReviewAnalyzer",
        model="gpt4",
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
    structured = result.content

    print(f"\n  Attempts: {structured.attempts}")
    print(f"  Valid: {structured.valid}")

    if structured.valid:
        review = structured.data
        print(f"  Rating: {'⭐' * review.rating}")
        print(f"  Title: {review.title}")
        print(f"  Pros: {', '.join(review.pros)}")
        print(f"  Cons: {', '.join(review.cons)}")
        print(f"  Recommends: {'Yes' if review.recommendation else 'No'}")

    if structured.raw:
        print(f"\n  Raw output preview: {structured.raw[:100]}...")


# =============================================================================
# Example 6: JSON Schema (for maximum flexibility)
# =============================================================================

async def example_json_schema():
    """Use raw JSON Schema for dynamic schemas."""
    print("\n" + "=" * 60)
    print("Example 6: Dynamic JSON Schema")
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
        model="gpt4",
        output=event_schema,
        instructions="Extract calendar event details from natural language.",
    )

    text = "Let's have a team lunch tomorrow at noon for about an hour. Invite Alice, Bob, and Carol."

    print(f"Input: {text}")
    result = await agent.run(f"Parse event: {text}")
    structured = result.content

    if structured.valid:
        event = structured.data
        print(f"\n  Event: {event}")


# =============================================================================
# Example 7: Dynamic Agent-Decided Structure
# =============================================================================

async def example_dynamic_structure():
    """Let the agent decide the output structure dynamically."""
    print("\n" + "=" * 60)
    print("Example 7: Agent-Decided Dynamic Structure")
    print("=" * 60)

    # Use dict for flexible JSON structure - agent decides fields
    agent = Agent(
        name="FlexibleAnalyzer",
        model="gpt-4o-mini",
        output=dict,  # Agent can return any dict structure
        instructions=(
            "Analyze the data and create a JSON response with whatever fields "
            "and structure you think best represents the information."
        ),
    )

    result = await agent.run("User feedback: Great UI, slow loading, missing dark mode, love the features")
    structured = result.content
    
    if structured.valid:
        analysis = structured.data
        print(f"\n  Agent-decided structure:")
        print(f"  Chose these fields: {list(analysis.keys())}")
        print(f"  Full response: {analysis}")

    # Compare: Predefined schema (from Example 3)
    print(f"\n  Contrast with Example 3 (predefined):")
    print(f"    Fixed fields: sentiment, confidence, key_phrases, summary")
    print(f"  vs Example 7 (dynamic):")
    print(f"    Agent chooses fields based on content")


# =============================================================================
# Example 8: Collections - Lists, Sets, Tuples
# =============================================================================

async def example_collections():
    """Collections - agent returns arrays with optional type constraints.
    
    Note: Bare collections (list, set, tuple) work best when wrapped in a model
    due to LLM API limitations with root-level arrays. Use bare collections for
    simple cases, or wrap in Pydantic models for reliability.
    """
    print("\n" + "=" * 60)
    print("Example 8: Collections (list, set, tuple)")
    print("=" * 60)
    
    # For bare collections, wrap in a model for reliability
    class Tags(BaseModel):
        """Tags extracted from text."""
        items: list[str]
    
    agent = Agent(
        name="TagExtractor",
        model="gpt-4o-mini",
        output=Tags,
        instructions="Extract relevant tags.",
    )
    result = await agent.run("Article about Python async programming with FastAPI")
    print(f"\n  list[str] in model: {result.content.data.items}")
    
    # set - unique items (wrapped)
    class UniqueCategories(BaseModel):
        """Unique categories extracted."""
        categories: set[str]
    
    agent = Agent(
        name="UniqueExtractor",
        model="gpt-4o-mini",
        output=UniqueCategories,
        instructions="Extract unique categories.",
    )
    result = await agent.run("Tags: ai, python, ai, automation, python, llm")
    print(f"\n  set[str] in model: {result.content.data.categories}")
    print(f"    Type: {type(result.content.data.categories)}")
    
    # tuple - fixed-length sequence (wrapped)
    class PlayerInfo(BaseModel):
        """Player information tuple."""
        data: tuple[str, int, float]
    
    agent = Agent(
        name="TripletExtractor",
        model="gpt-4o-mini",
        output=PlayerInfo,
        instructions="Extract player info as (name, age, score) tuple.",
    )
    result = await agent.run("Player: Sarah, 25 years old, score 95.5")
    print(f"\n  tuple[str, int, float] in model: {result.content.data.data}")
    print(f"    Type: {type(result.content.data.data)}")


# =============================================================================
# Example 9: Union Types - Polymorphic Responses
# =============================================================================

async def example_union_types():
    """Union types - agent chooses which schema based on content."""
    print("\n" + "=" * 60)
    print("Example 9: Union Types (Polymorphic)")
    print("=" * 60)
    
    from typing import Union
    
    class Success(BaseModel):
        """Successful operation result."""
        status: Literal["success"] = "success"
        result: str
        details: str | None = None
    
    class Failure(BaseModel):
        """Failed operation result."""
        status: Literal["failure"] = "failure"
        error: str
        code: int
    
    agent = Agent(
        name="OperationHandler",
        model="gpt-4o-mini",
        output=Union[Success, Failure],
        instructions="Analyze the operation and return appropriate status.",
    )
    
    # Should return Success
    result1 = await agent.run("Payment processed successfully, transaction ID: TXN-123")
    print(f"\n  Success case:")
    print(f"    Type: {type(result1.content.data).__name__}")
    print(f"    Data: {result1.content.data}")
    
    # Should return Failure
    result2 = await agent.run("Payment failed: insufficient funds, error code 402")
    print(f"\n  Failure case:")
    print(f"    Type: {type(result2.content.data).__name__}")
    print(f"    Data: {result2.content.data}")


# =============================================================================
# Example 10: Enum Types - Type-Safe Choices
# =============================================================================

async def example_enum_types():
    """Enum types - strongly typed choices with behavior."""
    print("\n" + "=" * 60)
    print("Example 10: Enum Types")
    print("=" * 60)
    
    from enum import Enum
    
    class Priority(str, Enum):
        """Task priority levels."""
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"
    
    agent = Agent(
        name="PriorityAssigner",
        model="gpt-4o-mini",
        output=Priority,
        instructions="Assign priority level based on task urgency.",
    )
    
    result = await agent.run("Production server down, users can't login")
    print(f"\n  Enum value: {result.content.data}")
    print(f"    Type: {type(result.content.data)}")
    print(f"    Is enum member: {isinstance(result.content.data, Priority)}")
    print(f"    Value: {result.content.data.value}")


# =============================================================================
# Example 11: None Type - Confirmation Responses
# =============================================================================

async def example_none_type():
    """None type - for actions that just need confirmation."""
    print("\n" + "=" * 60)
    print("Example 11: None Type (Confirmations)")
    print("=" * 60)
    
    agent = Agent(
        name="ActionExecutor",
        model="gpt-4o-mini",
        output=type(None),
        instructions="Acknowledge the action. Just confirm completion.",
    )
    
    result = await agent.run("Delete temporary files from cache")
    print(f"\n  Return value: {result.content.data}")
    print(f"    Type: {type(result.content.data)}")
    print(f"    Action confirmed: {result.content.data is None}")


# =============================================================================
# Example 12: Complex Nested Structures
# =============================================================================

async def example_nested_structures():
    """Complex nested types - combining all features."""
    print("\n" + "=" * 60)
    print("Example 12: Complex Nested Structures")
    print("=" * 60)
    
    from enum import Enum
    from typing import Union
    
    class TaskStatus(str, Enum):
        TODO = "todo"
        IN_PROGRESS = "in_progress"
        DONE = "done"
    
    @dataclass
    class Task:
        """A task in a project."""
        title: str
        status: TaskStatus
        tags: set[str]
        priority: int  # 1-5
    
    class ProjectAnalysis(BaseModel):
        """Analysis of a project's tasks."""
        total_tasks: int
        tasks: list[Task]
        completion_rate: float
        high_priority: list[str]  # Task titles
    
    agent = Agent(
        name="ProjectAnalyzer",
        model="gpt-4o-mini",
        output=ProjectAnalysis,
        instructions="Analyze the project and extract structured task data.",
    )
    
    result = await agent.run(
        "Project status: "
        "1. Implement auth (in progress, high priority, tags: backend, security) "
        "2. Design UI (todo, medium priority, tags: frontend, design) "
        "3. Write tests (done, low priority, tags: testing, backend)"
    )
    
    print(f"\n  Nested structure:")
    analysis = result.content.data
    print(f"    Total: {analysis.total_tasks} tasks")
    print(f"    Completion: {analysis.completion_rate:.0%}")
    print(f"    High priority: {analysis.high_priority}")
    for task in analysis.tasks:
        print(f"      - {task.title}: {task.status.value} (tags: {task.tags})")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all examples."""
    print("\n🔧 Cogent Structured Output Examples\n")

    await example_bare_types()
    await example_contact_extraction()
    await example_sentiment_analysis()
    await example_meeting_actions()
    await example_advanced_config()
    await example_json_schema()
    await example_dynamic_structure()
    await example_collections()
    await example_union_types()
    await example_enum_types()
    await example_none_type()
    await example_nested_structures()

    print("\n" + "=" * 60)
    print("✅ All 12 examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
