"""
Structured Output - enforce response schemas on agent outputs.

AgenticFlow's approach to structured output:
- Agent-level configuration via `output` parameter
- Supports Pydantic, dataclass, TypedDict, and JSON Schema
- Automatic validation with configurable retry
- Provider-native support where available (OpenAI, Anthropic)
- Fallback to tool-based extraction for other providers

Usage:
    from pydantic import BaseModel, Field
    from agenticflow import Agent
    from agenticflow.models.openai import OpenAIChat

    class ContactInfo(BaseModel):
        '''Contact information extracted from text.'''
        name: str = Field(description="Person's full name")
        email: str = Field(description="Email address")
        phone: str | None = Field(None, description="Phone number if available")

    agent = Agent(
        name="Extractor",
        model=OpenAIChat(),
        output=ContactInfo,  # Enforce this schema
    )

    result = await agent.run("Extract: John Doe, john@acme.com, 555-1234")
    # result.structured -> ContactInfo(name="John Doe", email="john@acme.com", phone="555-1234")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, is_dataclass
from dataclasses import fields as dataclass_fields
from enum import Enum
from typing import (
    Any,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

# Type variable for schema
T = TypeVar("T")


class OutputMethod(str, Enum):
    """Method for enforcing structured output."""

    AUTO = "auto"  # Automatically choose best method
    NATIVE = "native"  # Use provider-native structured output (OpenAI json_schema, etc)
    TOOL = "tool"  # Use tool calling to extract structure
    PROMPT = "prompt"  # Use prompt engineering with JSON parsing


@dataclass
class ResponseSchema[T]:
    """Configuration for structured output.

    Controls how the agent enforces and validates response schemas.

    Args:
        schema: The schema to enforce (Pydantic model, dataclass, TypedDict, or JSON Schema dict)
        method: How to enforce the schema (auto, native, tool, or prompt)
        strict: Require exact schema match (no extra fields)
        retry_on_error: Retry with error feedback on validation failure
        max_retries: Maximum retry attempts for validation errors
        error_message: Custom error message template for retries
        include_raw: Include raw LLM response alongside parsed output

    Example:
        from pydantic import BaseModel
        from agenticflow.agent.output import ResponseSchema, OutputMethod

        class Review(BaseModel):
            rating: int
            summary: str

        # Auto-detect best method (recommended)
        config = ResponseSchema(Review)

        # Force tool-based extraction
        config = ResponseSchema(Review, method=OutputMethod.TOOL)

        # Custom error handling
        config = ResponseSchema(
            Review,
            retry_on_error=True,
            max_retries=3,
            error_message="Rating must be 1-5. Please fix: {error}",
        )
    """

    schema: type[T] | dict[str, Any]
    method: OutputMethod = OutputMethod.AUTO
    strict: bool = True
    retry_on_error: bool = True
    max_retries: int = 2
    error_message: str | None = None
    include_raw: bool = False

    # Internal cache
    _json_schema: dict[str, Any] | None = field(default=None, repr=False)
    _tool_definition: dict[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Cache JSON schema on creation."""
        self._json_schema = schema_to_json(self.schema)
        self._tool_definition = self._build_tool_definition()

    def _build_tool_definition(self) -> dict[str, Any]:
        """Build tool definition for tool-based extraction."""
        schema_name = get_schema_name(self.schema)
        description = get_schema_description(self.schema)

        return {
            "type": "function",
            "function": {
                "name": schema_name,
                "description": description or f"Extract {schema_name} from the input.",
                "parameters": self._json_schema,
            },
        }

    @property
    def json_schema(self) -> dict[str, Any]:
        """Get JSON Schema representation."""
        if self._json_schema is None:
            self._json_schema = schema_to_json(self.schema)
        return self._json_schema

    @property
    def tool_definition(self) -> dict[str, Any]:
        """Get tool definition for tool-based extraction."""
        if self._tool_definition is None:
            self._tool_definition = self._build_tool_definition()
        return self._tool_definition


@dataclass
class StructuredResult[T]:
    """Result from structured output extraction.

    Contains both the parsed structured data and metadata about the extraction.

    Attributes:
        data: The parsed and validated data (instance of the schema)
        raw: Raw LLM response (if include_raw=True)
        valid: Whether validation succeeded
        error: Validation error message if any
        attempts: Number of attempts before success
        method: The method used for extraction
    """

    data: T | None = None
    raw: str | None = None
    valid: bool = True
    error: str | None = None
    attempts: int = 1
    method: OutputMethod = OutputMethod.AUTO

    def __bool__(self) -> bool:
        """Returns True if extraction was successful."""
        return self.valid and self.data is not None

    def unwrap(self) -> T:
        """Get the data or raise an error if invalid.

        Returns:
            The parsed data.

        Raises:
            ValueError: If extraction failed.
        """
        if not self.valid or self.data is None:
            raise ValueError(f"Structured output extraction failed: {self.error}")
        return self.data


class OutputValidationError(Exception):
    """Raised when structured output validation fails."""

    def __init__(
        self,
        message: str,
        schema: type | dict,
        raw_output: str,
        validation_errors: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.schema = schema
        self.raw_output = raw_output
        self.validation_errors = validation_errors or []


# =============================================================================
# Schema Utilities
# =============================================================================

def get_schema_name(schema: type | dict[str, Any]) -> str:
    """Get the name of a schema.

    Args:
        schema: Pydantic model, dataclass, TypedDict, or JSON Schema dict.

    Returns:
        Schema name as string.
    """
    if isinstance(schema, dict):
        return schema.get("title", schema.get("name", "StructuredOutput"))
    return getattr(schema, "__name__", "StructuredOutput")


def get_schema_description(schema: type | dict[str, Any]) -> str:
    """Get the description of a schema.

    Args:
        schema: Pydantic model, dataclass, TypedDict, or JSON Schema dict.

    Returns:
        Schema description or empty string.
    """
    if isinstance(schema, dict):
        return schema.get("description", "")
    return getattr(schema, "__doc__", "") or ""


def is_pydantic_model(obj: Any) -> bool:
    """Check if object is a Pydantic model class."""
    try:
        from pydantic import BaseModel
        return isinstance(obj, type) and issubclass(obj, BaseModel)
    except ImportError:
        return False


def is_typed_dict(obj: Any) -> bool:
    """Check if object is a TypedDict class."""
    return (
        isinstance(obj, type)
        and hasattr(obj, "__annotations__")
        and hasattr(obj, "__total__")
    )


def schema_to_json(schema: type | dict[str, Any]) -> dict[str, Any]:
    """Convert a schema to JSON Schema format.

    Supports:
    - Pydantic models (uses model_json_schema)
    - Dataclasses (converts to JSON Schema)
    - TypedDict (converts to JSON Schema)
    - dict (assumes already JSON Schema)

    Args:
        schema: The schema to convert.

    Returns:
        JSON Schema dictionary.

    Raises:
        TypeError: If schema type is not supported.
    """
    # Already JSON Schema
    if isinstance(schema, dict):
        return schema

    # Pydantic model
    if is_pydantic_model(schema):
        return schema.model_json_schema()

    # Dataclass
    if is_dataclass(schema):
        return _dataclass_to_json_schema(schema)

    # TypedDict
    if is_typed_dict(schema):
        return _typed_dict_to_json_schema(schema)

    raise TypeError(
        f"Unsupported schema type: {type(schema).__name__}. "
        "Use Pydantic BaseModel, dataclass, TypedDict, or JSON Schema dict."
    )


def _dataclass_to_json_schema(cls: type) -> dict[str, Any]:
    """Convert a dataclass to JSON Schema."""
    properties = {}
    required = []

    hints = get_type_hints(cls)

    for f in dataclass_fields(cls):
        field_type = hints.get(f.name, Any)
        field_schema = _python_type_to_json_schema(field_type)

        # Add description from field metadata if available
        if f.metadata and "description" in f.metadata:
            field_schema["description"] = f.metadata["description"]

        properties[f.name] = field_schema

        # Check if required (no default and not Optional)
        if f.default is f.default_factory is type(None).__class__:
            if not _is_optional_type(field_type):
                required.append(f.name)

    return {
        "type": "object",
        "title": cls.__name__,
        "description": cls.__doc__ or "",
        "properties": properties,
        "required": required,
    }


def _typed_dict_to_json_schema(cls: type) -> dict[str, Any]:
    """Convert a TypedDict to JSON Schema."""
    properties = {}
    required = []

    hints = get_type_hints(cls)
    total = getattr(cls, "__total__", True)

    for name, field_type in hints.items():
        properties[name] = _python_type_to_json_schema(field_type)

        # All fields required if total=True
        if total and not _is_optional_type(field_type):
            required.append(name)

    return {
        "type": "object",
        "title": cls.__name__,
        "description": cls.__doc__ or "",
        "properties": properties,
        "required": required,
    }


def _python_type_to_json_schema(py_type: type) -> dict[str, Any]:
    """Convert a Python type annotation to JSON Schema."""
    # Handle None/NoneType
    if py_type is type(None):
        return {"type": "null"}

    # Basic types
    if py_type is str:
        return {"type": "string"}
    if py_type is int:
        return {"type": "integer"}
    if py_type is float:
        return {"type": "number"}
    if py_type is bool:
        return {"type": "boolean"}

    # Get origin for generic types
    origin = get_origin(py_type)
    args = get_args(py_type)

    # List/Sequence
    if origin in (list, tuple):
        item_schema = _python_type_to_json_schema(args[0]) if args else {}
        return {"type": "array", "items": item_schema}

    # Dict
    if origin is dict:
        value_schema = _python_type_to_json_schema(args[1]) if len(args) > 1 else {}
        return {"type": "object", "additionalProperties": value_schema}

    # Optional (Union with None)
    if origin is Union:
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            # Optional[X] = Union[X, None]
            return _python_type_to_json_schema(non_none_args[0])
        # Union of multiple types
        return {"anyOf": [_python_type_to_json_schema(a) for a in non_none_args]}

    # Literal
    try:
        from typing import Literal
        if origin is Literal:
            return {"enum": list(args)}
    except ImportError:
        pass

    # Enum
    if isinstance(py_type, type) and issubclass(py_type, Enum):
        return {"enum": [e.value for e in py_type]}

    # Nested Pydantic model
    if is_pydantic_model(py_type):
        return py_type.model_json_schema()

    # Nested dataclass
    if is_dataclass(py_type):
        return _dataclass_to_json_schema(py_type)

    # Fallback
    return {"type": "string"}


def _is_optional_type(py_type: type) -> bool:
    """Check if a type is Optional (Union with None)."""
    origin = get_origin(py_type)
    if origin is Union:
        return type(None) in get_args(py_type)
    return False


# =============================================================================
# Parsing and Validation
# =============================================================================

def parse_json_output(raw: str) -> dict[str, Any]:
    """Parse JSON from LLM output, handling common formats.

    Handles:
    - Plain JSON
    - JSON wrapped in markdown code blocks
    - JSON with leading/trailing text

    Args:
        raw: Raw LLM output string.

    Returns:
        Parsed JSON as dict.

    Raises:
        json.JSONDecodeError: If no valid JSON found.
    """
    # Try plain JSON first
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    import re

    # Match ```json ... ``` or ``` ... ```
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding JSON object in text
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try finding JSON array in text
    array_match = re.search(r"\[[\s\S]*\]", raw)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("No valid JSON found in output", raw, 0)


def validate_and_parse[T](
    raw: str | dict[str, Any],
    schema: type[T] | dict[str, Any],
) -> T:
    """Validate and parse output against a schema.

    Args:
        raw: Raw output (string or already parsed dict).
        schema: Schema to validate against.

    Returns:
        Validated and parsed instance.

    Raises:
        OutputValidationError: If validation fails.
    """
    # Parse if string
    if isinstance(raw, str):
        try:
            data = parse_json_output(raw)
        except json.JSONDecodeError as e:
            raise OutputValidationError(
                f"Failed to parse JSON: {e}",
                schema=schema,
                raw_output=raw if isinstance(raw, str) else str(raw),
            )
    else:
        data = raw

    # Validate based on schema type
    if isinstance(schema, dict):
        # JSON Schema - just return the data (no runtime validation)
        return data

    if is_pydantic_model(schema):
        try:
            return schema.model_validate(data)
        except Exception as e:
            raise OutputValidationError(
                f"Pydantic validation failed: {e}",
                schema=schema,
                raw_output=raw if isinstance(raw, str) else json.dumps(data),
                validation_errors=[str(e)],
            )

    if is_dataclass(schema):
        try:
            return schema(**data)
        except Exception as e:
            raise OutputValidationError(
                f"Dataclass validation failed: {e}",
                schema=schema,
                raw_output=raw if isinstance(raw, str) else json.dumps(data),
                validation_errors=[str(e)],
            )

    if is_typed_dict(schema):
        # TypedDict - return dict (no runtime validation)
        return data

    raise TypeError(f"Unsupported schema type: {type(schema)}")


# =============================================================================
# Prompt Engineering for Structured Output
# =============================================================================

STRUCTURED_OUTPUT_PROMPT = """
You must respond with valid JSON matching this schema:

```json
{schema}
```

Requirements:
- Output ONLY valid JSON, no other text
- Include all required fields
- Use the exact field names from the schema
- Match the specified types exactly

Respond with the JSON object now:
"""


def build_structured_prompt(
    task: str,
    config: ResponseSchema,
    error_context: str | None = None,
) -> str:
    """Build a prompt that requests structured output.

    Args:
        task: The original task.
        config: Output configuration.
        error_context: Previous error to include in retry prompt.

    Returns:
        Enhanced prompt requesting structured output.
    """
    schema_json = json.dumps(config.json_schema, indent=2)

    prompt_parts = [task]

    if error_context:
        error_msg = config.error_message or "Previous attempt failed: {error}"
        prompt_parts.append(f"\n\n⚠️ {error_msg.format(error=error_context)}")

    prompt_parts.append(STRUCTURED_OUTPUT_PROMPT.format(schema=schema_json))

    return "\n".join(prompt_parts)


# =============================================================================
# Provider Support Detection
# =============================================================================

def supports_native_structured_output(model: Any) -> bool:
    """Check if a model supports native structured output.

    Args:
        model: The chat model to check.

    Returns:
        True if provider supports native structured output.
    """
    model_name = type(model).__name__.lower()

    # OpenAI models support response_format with json_schema
    if "openai" in model_name:
        return True

    # Anthropic supports tool-use based structured output
    if "anthropic" in model_name:
        return True

    # Gemini supports response_schema
    if "gemini" in model_name:
        return True

    # Groq supports JSON mode
    return "groq" in model_name


def get_best_method(model: Any, config: ResponseSchema) -> OutputMethod:
    """Determine the best extraction method for a model.

    Args:
        model: The chat model.
        config: Output configuration.

    Returns:
        The best OutputMethod to use.
    """
    if config.method != OutputMethod.AUTO:
        return config.method

    # Check for native support
    if supports_native_structured_output(model):
        return OutputMethod.NATIVE

    # Check if model supports tool calling
    if hasattr(model, "bind_tools"):
        return OutputMethod.TOOL

    # Fallback to prompt engineering
    return OutputMethod.PROMPT


__all__ = [
    # Configuration
    "ResponseSchema",
    "OutputMethod",
    # Results
    "StructuredResult",
    # Errors
    "OutputValidationError",
    # Utilities
    "schema_to_json",
    "get_schema_name",
    "get_schema_description",
    "parse_json_output",
    "validate_and_parse",
    "build_structured_prompt",
    "supports_native_structured_output",
    "get_best_method",
    # Type checking
    "is_pydantic_model",
    "is_dataclass",
    "is_typed_dict",
]
