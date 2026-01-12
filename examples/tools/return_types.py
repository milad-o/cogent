"""
Tool Return Types Demo - Tool Chaining with Structured Outputs

Demonstrates how tool return types help the LLM understand structured outputs
and parse them to populate arguments for subsequent tool calls.

Usage:
    uv run python examples/tools/return_types.py

Key Pattern:
    Tool A returns structured data (dict/list) → LLM parses it → Tool B receives parsed args

This example shows:
1. get_user_profile → returns dict with user info
2. LLM extracts user's city from the dict
3. get_weather → called with extracted city
4. LLM extracts temperature from weather dict
5. recommend_activity → called with extracted temp and weather condition
"""

import asyncio
import json
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model, settings

from agenticflow import Agent, Flow, tool


# =============================================================================
# Tool Chain: User Profile → Weather → Activity Recommendation
# =============================================================================


@tool
def get_user_profile(user_id: str) -> dict[str, str | int]:
    """Get a user's profile information.

    Args:
        user_id: The user's unique identifier.

    Returns:
        A dictionary with name, city, age, and preferred_activities (comma-separated).
    """
    profiles = {
        "user_123": {
            "name": "Alice",
            "city": "Miami",
            "age": 28,
            "preferred_activities": "beach, swimming, outdoor dining",
        },
        "user_456": {
            "name": "Bob",
            "city": "Chicago",
            "age": 35,
            "preferred_activities": "museums, restaurants, theater",
        },
        "user_789": {
            "name": "Carol",
            "city": "Los Angeles",
            "age": 42,
            "preferred_activities": "hiking, yoga, coffee shops",
        },
    }
    return profiles.get(user_id, {"name": "Unknown", "city": "Unknown", "age": 0})


@tool
def get_weather(city: str) -> dict[str, int | str]:
    """Get current weather data for a city.

    Args:
        city: City name to query (e.g., "Miami", "Chicago").

    Returns:
        A dictionary with temp (int, Fahrenheit), humidity (int, %), 
        condition (str: sunny/cloudy/rainy), and wind_speed (int, mph).
    """
    weather_data = {
        "miami": {"temp": 85, "humidity": 78, "condition": "sunny", "wind_speed": 8},
        "chicago": {"temp": 42, "humidity": 55, "condition": "cloudy", "wind_speed": 15},
        "los angeles": {"temp": 72, "humidity": 35, "condition": "sunny", "wind_speed": 5},
        "new york": {"temp": 48, "humidity": 60, "condition": "rainy", "wind_speed": 12},
    }
    return weather_data.get(city.lower(), {"temp": 65, "humidity": 50, "condition": "partly cloudy", "wind_speed": 10})


@tool
def recommend_activity(
    temperature: int,
    condition: str,
    preferred_activities: str,
) -> dict[str, str | list[str]]:
    """Recommend activities based on weather and user preferences.

    Args:
        temperature: Current temperature in Fahrenheit.
        condition: Weather condition (sunny, cloudy, rainy).
        preferred_activities: User's preferred activities (comma-separated).

    Returns:
        A dictionary with recommendation (str), suitable_activities (list of str),
        and weather_note (str explaining the choice).
    """
    prefs = [p.strip() for p in preferred_activities.split(",")]
    
    # Determine suitable activities based on weather
    if condition == "rainy" or temperature < 45:
        # Indoor activities
        indoor = ["museums", "restaurants", "theater", "coffee shops", "yoga"]
        suitable = [p for p in prefs if p in indoor] or ["indoor activities"]
        weather_note = "Weather suggests indoor activities."
    elif temperature > 80 and condition == "sunny":
        # Hot weather activities
        hot_weather = ["beach", "swimming", "outdoor dining", "hiking"]
        suitable = [p for p in prefs if p in hot_weather] or ["water activities"]
        weather_note = "Great weather for outdoor/water activities!"
    else:
        # Mild weather - most activities work
        suitable = prefs[:3] if prefs else ["sightseeing"]
        weather_note = "Pleasant weather for most activities."
    
    return {
        "recommendation": f"Based on {temperature}°F and {condition} weather",
        "suitable_activities": suitable,
        "weather_note": weather_note,
    }


@tool
def create_itinerary(
    user_name: str,
    city: str,
    activities: list[str],
    weather_note: str,
) -> dict[str, str | list[str]]:
    """Create a personalized day itinerary.

    Args:
        user_name: Name of the user.
        city: City for the itinerary.
        activities: List of recommended activities.
        weather_note: Note about the weather conditions.

    Returns:
        A dictionary with title (str), morning/afternoon/evening (str each),
        and tips (list of str).
    """
    activity_list = activities if isinstance(activities, list) else [activities]
    
    return {
        "title": f"{user_name}'s Day in {city}",
        "morning": f"Start with {activity_list[0] if activity_list else 'breakfast'}",
        "afternoon": f"Enjoy {activity_list[1] if len(activity_list) > 1 else 'lunch and exploration'}",
        "evening": f"End with {activity_list[2] if len(activity_list) > 2 else 'dinner'}",
        "tips": [
            weather_note,
            f"Pack appropriately for {city}",
            "Stay hydrated!",
        ],
    }


# =============================================================================
# Second Chain: Order Processing
# =============================================================================


@tool
def lookup_product(product_id: str) -> dict[str, str | float | int]:
    """Look up product details by ID.

    Args:
        product_id: The product's SKU or ID.

    Returns:
        A dictionary with name (str), price (float), stock (int), and category (str).
    """
    products = {
        "SKU-001": {"name": "Wireless Headphones", "price": 79.99, "stock": 150, "category": "Electronics"},
        "SKU-002": {"name": "Running Shoes", "price": 129.99, "stock": 45, "category": "Footwear"},
        "SKU-003": {"name": "Coffee Maker", "price": 49.99, "stock": 0, "category": "Kitchen"},
    }
    return products.get(product_id, {"name": "Unknown", "price": 0.0, "stock": 0, "category": "Unknown"})


@tool
def calculate_order_total(
    price: float,
    quantity: int,
    apply_discount: bool = False,
) -> dict[str, float | str]:
    """Calculate the total for an order.

    Args:
        price: Unit price of the product.
        quantity: Number of units to order.
        apply_discount: Whether to apply a 10% discount.

    Returns:
        A dictionary with subtotal (float), discount (float), tax (float),
        total (float), and formatted_total (str).
    """
    subtotal = price * quantity
    discount = subtotal * 0.10 if apply_discount else 0.0
    taxable = subtotal - discount
    tax = taxable * 0.08  # 8% tax
    total = taxable + tax
    
    return {
        "subtotal": round(subtotal, 2),
        "discount": round(discount, 2),
        "tax": round(tax, 2),
        "total": round(total, 2),
        "formatted_total": f"${total:.2f}",
    }


@tool
def check_stock_and_reserve(
    product_name: str,
    stock: int,
    quantity: int,
) -> dict[str, bool | str | int]:
    """Check if stock is available and reserve it.

    Args:
        product_name: Name of the product.
        stock: Current stock level.
        quantity: Quantity to reserve.

    Returns:
        A dictionary with available (bool), reserved (int), 
        remaining_stock (int), and message (str).
    """
    if stock >= quantity:
        return {
            "available": True,
            "reserved": quantity,
            "remaining_stock": stock - quantity,
            "message": f"Reserved {quantity} x {product_name}",
        }
    else:
        return {
            "available": False,
            "reserved": 0,
            "remaining_stock": stock,
            "message": f"Insufficient stock for {product_name}. Only {stock} available.",
        }


# =============================================================================
# Main Demo
# =============================================================================


async def main() -> None:
    """Run the tool chaining demo."""

    print("=" * 70)
    print("Tool Chaining Demo: Structured Outputs → Parsed Arguments")
    print("=" * 70)
    print()

    # Show tool schemas
    tools = [
        get_user_profile, get_weather, recommend_activity, create_itinerary,
        lookup_product, calculate_order_total, check_stock_and_reserve,
    ]

    print("📋 Tool Return Types (LLM uses these to parse and chain):")
    print("-" * 70)
    for t in tools:
        print(f"  {t.name}: {t.return_info}")
    print()

    # Create agent
    model = get_model()

    assistant = Agent(
        name="Assistant",
        model=model,
        tools=tools,
        instructions="""You are a helpful assistant that chains tools together.
When a tool returns structured data (dict/list), extract the relevant fields
to use as arguments for the next tool call.

Chain pattern examples:
1. get_user_profile → extract city → get_weather → extract temp/condition → recommend_activity
2. lookup_product → extract price/stock → calculate_order_total and check_stock_and_reserve

Always show your reasoning about what data you're extracting from each tool's output.""",
    )

    flow = Flow(
        name="tool-chaining-demo",
        agents=[assistant],
        topology="pipeline",
        verbose=settings.verbose_level,
    )

    # Demo 1: User → Weather → Activity chain
    print("=" * 70)
    print("Demo 1: User Profile → Weather → Activity Recommendation → Itinerary")
    print("=" * 70)
    
    query1 = """
    Get user_123's profile, then check the weather in their city,
    and recommend activities based on the weather and their preferences.
    Finally, create a day itinerary for them.
    """
    
    print(f"Query: {query1.strip()}")
    print("-" * 70)
    
    result1 = await flow.run(query1)
    
    print()
    print("Result:")
    print(result1.output)
    print()

    # Demo 2: Product → Order Processing chain
    print("=" * 70)
    print("Demo 2: Product Lookup → Stock Check → Order Calculation")
    print("=" * 70)
    
    query2 = """
    Look up product SKU-002, check if we have enough stock for 3 units,
    and calculate the order total with a discount applied.
    """
    
    print(f"Query: {query2.strip()}")
    print("-" * 70)
    
    result2 = await flow.run(query2)
    
    print()
    print("Result:")
    print(result2.output)


if __name__ == "__main__":
    asyncio.run(main())
