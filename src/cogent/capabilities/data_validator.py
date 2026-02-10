"""
Data Validation capability - comprehensive data quality validation.

Provides data validation with:
- Schema validation (Pydantic models, JSON Schema)
- Data quality checks (completeness, uniqueness, ranges)
- Format validation (email, URL, phone, dates)
- Type validation and coercion
- Custom validation rules

Example:
    ```python
    from cogent import Agent
    from cogent.capabilities import DataValidator

    agent = Agent(
        name="Data Quality Agent",
        model=model,
        capabilities=[DataValidator()],
    )

    # Agent can now validate data
    await agent.run("Validate this user data: {name: 'John', email: 'invalid'}")
    ```
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ValidationError

from cogent.capabilities.base import BaseCapability
from cogent.tools.base import BaseTool, tool

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validated_data: dict[str, Any] | None = None
    field_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "validated_data": self.validated_data,
            "field_count": self.field_count,
            "timestamp": self.timestamp.isoformat(),
        }


class DataValidator(BaseCapability):
    """
    Data Validation capability for comprehensive data quality checks.

    Provides validation for:
    - Schema validation (Pydantic models, type checking)
    - Data quality (completeness, uniqueness, ranges)
    - Format validation (email, URL, phone, dates)
    - Custom validation rules
    - Bulk data validation

    Args:
        strict_mode: Whether to fail on warnings (default: False)
        name: Capability name (default: "data_validator")

    Example:
        ```python
        # Schema validation
        validator = DataValidator()
        result = await validator.validate_schema(
            data={"name": "John", "age": 30},
            schema={"name": str, "age": int}
        )

        # Email validation
        result = await validator.validate_email("user@example.com")

        # Bulk validation
        results = await validator.validate_batch([
            {"name": "John", "age": 30},
            {"name": "Jane", "age": 25},
        ])
        ```
    """

    def __init__(
        self,
        strict_mode: bool = False,
        name: str = "data_validator",
    ):
        self._name = name
        self._strict_mode = strict_mode
        self._validation_history: list[ValidationResult] = []

        # Email regex pattern (RFC 5322 simplified)
        self._email_pattern = re.compile(
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        )

        # URL regex pattern
        self._url_pattern = re.compile(r"^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$")

        # Phone regex pattern (flexible)
        self._phone_pattern = re.compile(r"^[\d\s()+-]{10,}$")

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        mode = "strict" if self._strict_mode else "lenient"
        return f"Data validation and quality checks ({mode} mode)"

    @property
    def tools(self) -> list[BaseTool]:
        return [
            self._validate_schema_tool(),
            self._validate_format_tool(),
            self._validate_range_tool(),
            self._validate_completeness_tool(),
        ]

    async def validate_schema(
        self,
        data: dict[str, Any],
        schema: dict[str, type] | type[BaseModel] | None = None,
    ) -> ValidationResult:
        """
        Validate data against a schema.

        Args:
            data: Data to validate
            schema: Either dict of field:type or Pydantic model

        Returns:
            ValidationResult with errors and validated data
        """
        errors: list[str] = []
        warnings: list[str] = []
        validated_data = data.copy()

        # Pydantic model validation (preferred)
        if schema and isinstance(schema, type) and issubclass(schema, BaseModel):
            try:
                validated_model = schema.model_validate(data)
                validated_data = validated_model.model_dump()
                result = ValidationResult(
                    valid=True,
                    validated_data=validated_data,
                    field_count=len(validated_data),
                )
                self._validation_history.append(result)
                return result
            except ValidationError as e:
                for error in e.errors():
                    field = ".".join(str(x) for x in error["loc"])
                    errors.append(f"{field}: {error['msg']}")

                result = ValidationResult(
                    valid=False,
                    errors=errors,
                    field_count=len(data),
                )
                self._validation_history.append(result)
                return result

        # Simple type checking
        if schema and isinstance(schema, dict):
            for field_name, expected_type in schema.items():
                if field_name not in data:
                    errors.append(f"Missing required field: {field_name}")
                    continue

                value = data[field_name]
                if value is None:
                    warnings.append(f"Field '{field_name}' is None")
                    continue

                if not isinstance(value, expected_type):
                    actual_type = type(value).__name__
                    expected_name = expected_type.__name__
                    errors.append(
                        f"Field '{field_name}': expected {expected_name}, got {actual_type}"
                    )

        valid = len(errors) == 0 and (not self._strict_mode or len(warnings) == 0)

        result = ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            validated_data=validated_data if valid else None,
            field_count=len(data),
        )
        self._validation_history.append(result)
        return result

    async def validate_email(self, email: str) -> ValidationResult:
        """Validate email address format."""
        errors: list[str] = []

        if not email:
            errors.append("Email is empty")
        elif not self._email_pattern.match(email):
            errors.append(f"Invalid email format: {email}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            validated_data={"email": email} if len(errors) == 0 else None,
            field_count=1,
        )

    async def validate_url(self, url: str) -> ValidationResult:
        """Validate URL format."""
        errors: list[str] = []

        if not url:
            errors.append("URL is empty")
        elif not self._url_pattern.match(url):
            errors.append(f"Invalid URL format: {url}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            validated_data={"url": url} if len(errors) == 0 else None,
            field_count=1,
        )

    async def validate_phone(self, phone: str) -> ValidationResult:
        """Validate phone number format."""
        errors: list[str] = []

        if not phone:
            errors.append("Phone is empty")
        elif not self._phone_pattern.match(phone):
            errors.append(f"Invalid phone format: {phone}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            validated_data={"phone": phone} if len(errors) == 0 else None,
            field_count=1,
        )

    async def validate_range(
        self,
        value: int | float,
        min_value: int | float | None = None,
        max_value: int | float | None = None,
        field_name: str = "value",
    ) -> ValidationResult:
        """
        Validate numeric value is within range.

        Args:
            value: Value to validate
            min_value: Minimum allowed value (inclusive)
            max_value: Maximum allowed value (inclusive)
            field_name: Name of the field for error messages

        Returns:
            ValidationResult
        """
        errors: list[str] = []

        if min_value is not None and value < min_value:
            errors.append(f"{field_name} must be >= {min_value}, got {value}")

        if max_value is not None and value > max_value:
            errors.append(f"{field_name} must be <= {max_value}, got {value}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            validated_data={field_name: value} if len(errors) == 0 else None,
            field_count=1,
        )

    async def validate_completeness(
        self,
        data: dict[str, Any],
        required_fields: list[str],
    ) -> ValidationResult:
        """
        Validate data has all required fields and they are not empty.

        Args:
            data: Data to validate
            required_fields: List of required field names

        Returns:
            ValidationResult
        """
        errors: list[str] = []
        warnings: list[str] = []

        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
            elif data[field] is None:
                errors.append(f"Field '{field}' is None")
            elif isinstance(data[field], str) and not data[field].strip():
                errors.append(f"Field '{field}' is empty string")

        # Check for unexpected fields
        expected_set = set(required_fields)
        actual_set = set(data.keys())
        unexpected = actual_set - expected_set

        if unexpected:
            warnings.append(f"Unexpected fields: {', '.join(unexpected)}")

        valid = len(errors) == 0 and (not self._strict_mode or len(warnings) == 0)

        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            validated_data=data if valid else None,
            field_count=len(data),
        )

    async def validate_uniqueness(
        self,
        values: list[Any],
        field_name: str = "values",
    ) -> ValidationResult:
        """
        Validate list contains unique values.

        Args:
            values: List of values to check
            field_name: Name for error messages

        Returns:
            ValidationResult
        """
        errors: list[str] = []
        warnings: list[str] = []

        seen = set()
        duplicates = []

        for value in values:
            if value in seen:
                duplicates.append(str(value))
            else:
                seen.add(value)

        if duplicates:
            errors.append(
                f"{field_name} contains duplicates: {', '.join(duplicates[:5])}"
            )
            if len(duplicates) > 5:
                errors.append(f"... and {len(duplicates) - 5} more")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            validated_data={field_name: values} if len(errors) == 0 else None,
            field_count=len(values),
        )

    async def validate_batch(
        self,
        data_list: list[dict[str, Any]],
        schema: dict[str, type] | type[BaseModel] | None = None,
    ) -> list[ValidationResult]:
        """
        Validate multiple data items.

        Args:
            data_list: List of data items to validate
            schema: Schema for validation

        Returns:
            List of ValidationResults
        """
        results = []
        for data in data_list:
            result = await self.validate_schema(data, schema)
            results.append(result)
        return results

    @property
    def history(self) -> list[ValidationResult]:
        """Get validation history."""
        return self._validation_history.copy()

    def clear_history(self) -> int:
        """Clear validation history. Returns count cleared."""
        count = len(self._validation_history)
        self._validation_history.clear()
        return count

    # =========================================================================
    # Tool Generation
    # =========================================================================

    def _validate_schema_tool(self) -> BaseTool:
        validator = self

        @tool
        async def validate_data_schema(
            data: str,
            schema: str,
        ) -> str:
            """
            Validate data against a schema with type checking.

            Args:
                data: JSON string of data to validate
                schema: JSON string of schema (field:type pairs, e.g., '{"name": "str", "age": "int"}')

            Returns:
                Validation result with errors and warnings

            Example:
                validate_data_schema('{"name": "John", "age": 30}', '{"name": "str", "age": "int"}')
            """
            try:
                data_dict = json.loads(data)
            except json.JSONDecodeError:
                return f"Error: Invalid JSON in data: {data}"

            try:
                schema_dict = json.loads(schema)
            except json.JSONDecodeError:
                return f"Error: Invalid JSON in schema: {schema}"

            # Convert string type names to actual types
            type_mapping = {
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
            }

            typed_schema = {}
            for field_name, type_name in schema_dict.items():
                if type_name not in type_mapping:
                    return f"Error: Unknown type '{type_name}' for field '{field_name}'"
                typed_schema[field_name] = type_mapping[type_name]

            result = await validator.validate_schema(data_dict, typed_schema)

            lines = []
            if result.valid:
                lines.append("✓ Validation passed")
                lines.append(f"Fields validated: {result.field_count}")
            else:
                lines.append("✗ Validation failed")
                if result.errors:
                    lines.append(f"\nErrors ({len(result.errors)}):")
                    for error in result.errors:
                        lines.append(f"  - {error}")

            if result.warnings:
                lines.append(f"\nWarnings ({len(result.warnings)}):")
                for warning in result.warnings:
                    lines.append(f"  - {warning}")

            return "\n".join(lines)

        return validate_data_schema

    def _validate_format_tool(self) -> BaseTool:
        validator = self

        @tool
        async def validate_data_format(
            value: str,
            format_type: str,
        ) -> str:
            """
            Validate data format (email, URL, phone).

            Args:
                value: Value to validate
                format_type: Type of format ("email", "url", or "phone")

            Returns:
                Validation result

            Example:
                validate_data_format("user@example.com", "email")
                validate_data_format("https://example.com", "url")
                validate_data_format("+1-555-123-4567", "phone")
            """
            format_type = format_type.lower()

            if format_type == "email":
                result = await validator.validate_email(value)
            elif format_type == "url":
                result = await validator.validate_url(value)
            elif format_type == "phone":
                result = await validator.validate_phone(value)
            else:
                return f"Error: Unknown format type '{format_type}'. Use: email, url, phone"

            if result.valid:
                return f"✓ Valid {format_type}: {value}"
            else:
                return f"✗ Invalid {format_type}:\n  " + "\n  ".join(result.errors)

        return validate_data_format

    def _validate_range_tool(self) -> BaseTool:
        validator = self

        @tool
        async def validate_numeric_range(
            value: float,
            min_value: float | None = None,
            max_value: float | None = None,
            field_name: str = "value",
        ) -> str:
            """
            Validate numeric value is within range.

            Args:
                value: Numeric value to validate
                min_value: Minimum allowed value (inclusive, optional)
                max_value: Maximum allowed value (inclusive, optional)
                field_name: Name of field for error messages

            Returns:
                Validation result

            Example:
                validate_numeric_range(25, 0, 100, "age")
                validate_numeric_range(150.5, 0, None, "price")
            """
            result = await validator.validate_range(
                value=value,
                min_value=min_value,
                max_value=max_value,
                field_name=field_name,
            )

            if result.valid:
                range_str = ""
                if min_value is not None and max_value is not None:
                    range_str = f" (range: {min_value} to {max_value})"
                elif min_value is not None:
                    range_str = f" (min: {min_value})"
                elif max_value is not None:
                    range_str = f" (max: {max_value})"

                return f"✓ Valid {field_name}: {value}{range_str}"
            else:
                return f"✗ Invalid {field_name}:\n  " + "\n  ".join(result.errors)

        return validate_numeric_range

    def _validate_completeness_tool(self) -> BaseTool:
        validator = self

        @tool
        async def validate_data_completeness(
            data: str,
            required_fields: str,
        ) -> str:
            """
            Validate data has all required fields and they are not empty.

            Args:
                data: JSON string of data to validate
                required_fields: JSON array of required field names

            Returns:
                Validation result

            Example:
                validate_data_completeness('{"name": "John", "email": "john@example.com"}', '["name", "email"]')
            """
            try:
                data_dict = json.loads(data)
            except json.JSONDecodeError:
                return f"Error: Invalid JSON in data: {data}"

            try:
                fields_list = json.loads(required_fields)
            except json.JSONDecodeError:
                return f"Error: Invalid JSON in required_fields: {required_fields}"

            if not isinstance(fields_list, list):
                return "Error: required_fields must be a JSON array"

            result = await validator.validate_completeness(data_dict, fields_list)

            lines = []
            if result.valid:
                lines.append("✓ Completeness check passed")
                lines.append(f"All {len(fields_list)} required fields present")
            else:
                lines.append("✗ Completeness check failed")

            if result.errors:
                lines.append(f"\nErrors ({len(result.errors)}):")
                for error in result.errors:
                    lines.append(f"  - {error}")

            if result.warnings:
                lines.append(f"\nWarnings ({len(result.warnings)}):")
                for warning in result.warnings:
                    lines.append(f"  - {warning}")

            return "\n".join(lines)

        return validate_data_completeness
