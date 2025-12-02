"""
Example: Spreadsheet Capability

Agent analyzes sales data from CSV using spreadsheet tools.

Usage:
    uv run python examples/23_spreadsheet.py
"""

import asyncio
import tempfile
from pathlib import Path

from config import get_model, settings

from agenticflow import Agent, Flow
from agenticflow.capabilities import Spreadsheet


async def main() -> None:
    sales_data = [
        {"product": "Widget A", "region": "North", "sales": 1500, "quarter": "Q1"},
        {"product": "Widget B", "region": "South", "sales": 2300, "quarter": "Q1"},
        {"product": "Widget A", "region": "South", "sales": 1800, "quarter": "Q2"},
        {"product": "Widget B", "region": "North", "sales": 2100, "quarter": "Q2"},
        {"product": "Widget A", "region": "North", "sales": 1900, "quarter": "Q3"},
        {"product": "Widget B", "region": "South", "sales": 2500, "quarter": "Q3"},
        {"product": "Widget A", "region": "East", "sales": 1200, "quarter": "Q1"},
        {"product": "Widget B", "region": "West", "sales": 1700, "quarter": "Q2"},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        spreadsheet = Spreadsheet(allowed_paths=[tmpdir])

        # Write test data
        csv_path = Path(tmpdir) / "sales.csv"
        spreadsheet._write_csv(csv_path, sales_data)

        model = get_model()
        analyst = Agent(
            name="DataAnalyst",
            model=model,
            instructions="You analyze sales data. Use the spreadsheet tools to answer questions.",
            capabilities=[spreadsheet],
        )

        flow = Flow(
            name="analysis",
            agents=[analyst],
            verbose="debug",
        )

        result = await flow.run(
            f"Read the CSV file at {csv_path} and tell me the total sales by region."
        )
        print(f"\n{result.output}")


if __name__ == "__main__":
    asyncio.run(main())
