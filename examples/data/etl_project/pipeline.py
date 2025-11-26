"""
ETL Pipeline Runner.

Orchestrates the ETL process using components from etl.py.
"""

from etl import DataExtractor, DataTransformer, DataLoader


class ETLPipeline:
    """Orchestrates the full ETL process."""

    def __init__(self, source: str, destination: str):
        self.extractor = DataExtractor(source)
        self.transformer = DataTransformer(rules={"multiplier": 1.1})
        self.loader = DataLoader(destination)

    def run(self) -> dict:
        """Execute the full ETL pipeline."""
        # Extract
        self.extractor.connect()
        records = self.extractor.extract()
        print(f"Extracted {len(records)} records")

        # Transform
        transformed = self.transformer.transform(records)
        print(f"Transformed {len(transformed)} records")

        # Load
        count = self.loader.load(transformed)
        print(f"Loaded {count} records")

        self.extractor.close()
        return self.loader.get_stats()


def run_etl(source: str, destination: str) -> dict:
    """Convenience function to run ETL."""
    pipeline = ETLPipeline(source, destination)
    return pipeline.run()


if __name__ == "__main__":
    stats = run_etl("postgres://db/source", "s3://bucket/output")
    print(f"ETL Complete: {stats}")
