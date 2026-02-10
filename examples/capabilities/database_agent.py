"""Database capability - agent integration demo.

Shows Database being used by an agent to query and analyze data.

Run with: uv run python examples/capabilities/database_agent_demo.py
"""

import asyncio
import tempfile
from pathlib import Path

from cogent import Agent
from cogent.capabilities.database import Database


async def setup_demo_database(db_path: str) -> None:
    """Create a demo database with sample data."""
    db = Database(connection_string=f"sqlite+aiosqlite:///{db_path}")

    try:
        # Create tables
        await db.execute("""
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                department TEXT,
                salary INTEGER,
                hire_date TEXT
            )
        """)

        await db.execute("""
            CREATE TABLE departments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                budget INTEGER
            )
        """)

        # Insert sample data
        employees = [
            ("Alice Johnson", "Engineering", 95000, "2020-01-15"),
            ("Bob Smith", "Engineering", 85000, "2021-03-20"),
            ("Carol Davis", "Sales", 75000, "2019-06-10"),
            ("David Wilson", "Sales", 80000, "2020-11-05"),
            ("Eve Martinez", "Marketing", 70000, "2021-08-12"),
            ("Frank Brown", "Engineering", 90000, "2018-09-25"),
        ]

        for name, dept, salary, hire_date in employees:
            await db.execute(
                "INSERT INTO employees (name, department, salary, hire_date) VALUES (:name, :dept, :salary, :hire_date)",
                params={
                    "name": name,
                    "dept": dept,
                    "salary": salary,
                    "hire_date": hire_date,
                },
            )

        departments = [
            ("Engineering", 500000),
            ("Sales", 300000),
            ("Marketing", 200000),
        ]

        for name, budget in departments:
            await db.execute(
                "INSERT INTO departments (name, budget) VALUES (:name, :budget)",
                params={"name": name, "budget": budget},
            )

        print("✓ Demo database created with sample data")

    finally:
        await db.close()


async def main():
    """Demonstrate Database being used by an agent."""
    print("\n" + "=" * 60)
    print("DATABASE CAPABILITY - AGENT DEMO")
    print("=" * 60)

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        # Setup demo data
        await setup_demo_database(db_path)

        agent = Agent(
            name="Data Analyst Agent",
            model="gpt-4o-mini",
            capabilities=[Database(connection_string=f"sqlite+aiosqlite:///{db_path}")],
            system_prompt="""You are a data analyst assistant.

Use SQL query tools to analyze the database and answer questions.
Always use parameterized queries for safety.""",
        )

        print("\n" + "=" * 60)
        print("SCENARIO: Employee Database Analysis")
        print("=" * 60)

        task = """Analyze the employees database and tell me:

1. How many employees are in each department?
2. What's the average salary by department?
3. Who are the top 3 highest-paid employees?
4. What's the total payroll across all departments?

The database has tables: employees (id, name, department, salary, hire_date) and departments (id, name, budget)."""

        print(f"\nAgent task: {task}\n")
        print("-" * 60)

        result = await agent.run(task)

        print(f"\nAgent response:\n{result.content}")
        print("\n" + "=" * 60)
        print("DEMO COMPLETE!")
        print("=" * 60)

    finally:
        # Cleanup
        Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
