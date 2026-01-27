"""Data Validator capability demonstration with LLM integration.

This example shows the DataValidator capability being used by an agent
to validate data during a real workflow.

Run with: uv run python examples/capabilities/data_validator_demo.py
"""

import asyncio

from cogent import Agent
from cogent.capabilities.data_validator import DataValidator


async def main():
    """Demonstrate DataValidator being used by an agent."""
    print("\n" + "=" * 60)
    print("DATA VALIDATOR CAPABILITY - AGENT DEMO")
    print("=" * 60)
    
    # Create agent with DataValidator capability
    agent = Agent(
        name="Data Quality Agent",
        model="gpt-4o-mini",
        capabilities=[DataValidator()],
        system_prompt="""You are a data quality specialist. 
        
Use the data validation tools to check data quality and provide feedback.
Always validate data before confirming it's acceptable.""",
    )
    
    print("\n" + "=" * 60)
    print("SCENARIO: User Registration Data Validation")
    print("=" * 60)
    
    user_data = {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "age": 28,
        "phone": "+1-555-123-4567",
        "website": "https://johndoe.com"
    }
    
    task = f"""Please validate this user registration data:

{user_data}

Check:
1. Schema: name (str), email (str), age (int), phone (str), website (str)
2. Email format
3. URL format  
4. Age range (18-120)
5. Data completeness

Provide a summary of validation results."""
    
    print(f"\nAgent task: {task}\n")
    print("-" * 60)
    
    result = await agent.run(task)
    
    print(f"\nAgent response:\n{result.content}")
    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
