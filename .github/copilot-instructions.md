---
applyTo: **/*.py
---

# Project coding standards for Python files

## Python Version & Tooling
- We use Python 3.13, so ensure latest syntax and API are used and avoid deprecated features.
- We use `uv` for project management: `uv run`, `uv add`, `uv run uvicorn`, `uv run python`, `uv run pytest`, etc.
- Never suggest `pip`, `poetry`, or other package managers.

## Code Style & Quality
- Follow PEP 8 style guide for Python code.
- Use type hints for function parameters and return values.
- Maximum line length: 88 characters (Black formatter standard).
- Use meaningful variable and function names that describe their purpose.
- Prefer f-strings over `.format()` or `%` formatting.
- Use `pathlib.Path` instead of `os.path` for file operations.

## Documentation
- Comments should be toward a co-developer and self. Use clear and concise language.
- Add docstrings to all public functions, classes, and modules using Google or NumPy style.
- Keep comments up-to-date with code changes.
- Avoid obvious comments that repeat what the code does.

## Testing
- Test all new features and bug fixes with unit tests using pytest framework.
- If required, add new unit tests.
- Place tests in `tests/` directory mirroring the source structure.
- Use descriptive test function names: `test_<functionality>_<scenario>_<expected_outcome>`.
- Aim for high test coverage but focus on meaningful tests.
- Use fixtures for common test setup and teardown.
- Mock external dependencies to isolate unit tests.

## Error Handling
- Use specific exception types rather than generic `Exception`.
- Always include meaningful error messages.
- Use context managers (`with` statements) for resource management.
- Log errors appropriately before re-raising or handling.

## Imports
- Group imports in this order: standard library, third-party, local application.
- Use absolute imports over relative imports.
- Avoid wildcard imports (`from module import *`).
- Sort imports alphabetically within each group.

## Best Practices
- Follow SOLID principles and clean code practices.
- Keep functions small and focused on a single responsibility.
- Avoid deep nesting; extract complex logic into separate functions.
- Use list/dict comprehensions for simple transformations, but not at the cost of readability.
- Prefer explicit over implicit (Zen of Python).
- Use `Enum` for fixed sets of constants.
- Handle None values explicitly; use Optional[] type hints.

## Security
- Never hardcode credentials, API keys, or secrets in source code.
- Use environment variables or secure configuration management.
- Validate and sanitize all user inputs.
- Use parameterized queries for database operations to prevent SQL injection.

## Async Code
- Use `async`/`await` syntax consistently.
- Prefer `asyncio` standard library features.
- Don't mix blocking and async code without proper handling.

## Git & Version Control
- Write clear, concise commit messages.
- Keep commits atomic and focused on a single change.
- Reference issue numbers in commits when applicable.

## Frontend
- Use minimalistic React with Tailwind CSS for styling.
- Prioritize performance: minimize bundle size, lazy load components, optimize renders.
- All frontend code lives in the `frontend/` directory.
- Keep UI simple and functional; avoid over-engineering.

# Environment Variables and Secrets
- Use the `.env` in the project root to look for environment variables during development.
