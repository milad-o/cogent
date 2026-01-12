"""
Tests for tool return type extraction and schema generation.
"""

import pytest
from agenticflow.tools import tool
from agenticflow.tools.base import (
    BaseTool,
    _extract_return_info,
    _type_to_readable_string,
)


class TestReturnTypeExtraction:
    """Test return type info extraction from functions."""

    def test_simple_return_type(self) -> None:
        """Test extracting simple return types."""

        @tool
        def simple(x: int) -> str:
            """Do something."""
            return str(x)

        assert simple.return_info == "str"
        assert "Returns: str" in simple.description

    def test_complex_return_type(self) -> None:
        """Test extracting complex generic return types."""

        @tool
        def complex_return(x: int) -> dict[str, int]:
            """Get data."""
            return {"value": x}

        assert complex_return.return_info == "dict[str, int]"
        assert "Returns: dict[str, int]" in complex_return.description

    def test_list_return_type(self) -> None:
        """Test extracting list return types."""

        @tool
        def list_return(x: int) -> list[str]:
            """Get items."""
            return [str(x)]

        assert list_return.return_info == "list[str]"
        assert "Returns: list[str]" in list_return.description

    def test_optional_return_type(self) -> None:
        """Test extracting optional return types."""

        @tool
        def optional_return(x: int) -> str | None:
            """Maybe get value."""
            return str(x) if x > 0 else None

        assert "None" in optional_return.return_info
        assert "str" in optional_return.return_info

    def test_no_return_type(self) -> None:
        """Test function with no return type annotation."""

        @tool
        def no_return(x: int):
            """Do something."""
            return x

        assert no_return.return_info == ""
        assert "Returns:" not in no_return.description

    def test_none_return_type(self) -> None:
        """Test function returning None explicitly."""

        @tool
        def none_return(x: int) -> None:
            """Log something."""
            print(x)

        # None return type should not add "Returns:" since it's vacuous
        assert none_return.return_info == ""

    def test_docstring_returns_section(self) -> None:
        """Test extracting Returns section from docstring."""

        @tool
        def with_returns_doc(x: int) -> dict:
            """Process data.

            Args:
                x: Input value.

            Returns:
                A dictionary containing the result and metadata.
            """
            return {"result": x}

        assert "dict" in with_returns_doc.return_info
        assert "dictionary containing the result" in with_returns_doc.return_info

    def test_docstring_returns_combined_with_type(self) -> None:
        """Test that type and docstring Returns are combined."""

        @tool
        def combined(x: int) -> dict[str, int]:
            """Get weather data.

            Returns:
                Temperature and humidity values.
            """
            return {"temp": 75, "humidity": 45}

        assert "dict[str, int]" in combined.return_info
        assert "Temperature and humidity" in combined.return_info
        # Description should have both
        assert "dict[str, int] - Temperature and humidity" in combined.description

    def test_multiline_returns_section(self) -> None:
        """Test extracting multiline Returns section."""

        @tool
        def multiline_doc(x: int) -> str:
            """Process input.

            Returns:
                A formatted string containing
                the processed result.
            """
            return str(x)

        assert "formatted string" in multiline_doc.return_info
        assert "processed result" in multiline_doc.return_info


class TestTypeToReadableString:
    """Test the type to readable string conversion."""

    def test_basic_types(self) -> None:
        """Test basic Python types."""
        assert _type_to_readable_string(str) == "str"
        assert _type_to_readable_string(int) == "int"
        assert _type_to_readable_string(float) == "float"
        assert _type_to_readable_string(bool) == "bool"
        assert _type_to_readable_string(list) == "list"
        assert _type_to_readable_string(dict) == "dict"

    def test_generic_list(self) -> None:
        """Test generic list types."""
        assert _type_to_readable_string(list[str]) == "list[str]"
        assert _type_to_readable_string(list[int]) == "list[int]"

    def test_generic_dict(self) -> None:
        """Test generic dict types."""
        assert _type_to_readable_string(dict[str, int]) == "dict[str, int]"
        assert _type_to_readable_string(dict[str, list[int]]) == "dict[str, list[int]]"

    def test_union_types(self) -> None:
        """Test union types."""
        result = _type_to_readable_string(str | None)
        assert "str" in result
        assert "None" in result

    def test_none_type(self) -> None:
        """Test None type."""
        assert _type_to_readable_string(type(None)) == "None"


class TestToolDescriptionWithReturns:
    """Test that tool descriptions include return info."""

    def test_description_includes_returns(self) -> None:
        """Test that description is enhanced with return info."""

        @tool
        def search(query: str) -> str:
            """Search the web."""
            return f"Results for: {query}"

        assert search.description == "Search the web. Returns: str"

    def test_description_with_docstring_returns(self) -> None:
        """Test description with full docstring."""

        @tool
        def get_weather(city: str) -> dict[str, int]:
            """Get weather data for a city.

            Returns:
                A dictionary with temp, humidity, and wind_speed.
            """
            return {"temp": 75}

        expected = (
            "Get weather data for a city. "
            "Returns: dict[str, int] - A dictionary with temp, humidity, and wind_speed."
        )
        assert get_weather.description == expected

    def test_custom_description_still_gets_returns(self) -> None:
        """Test that custom description also gets return info appended."""

        @tool(description="Custom tool description")
        def custom(x: int) -> bool:
            """Original docstring."""
            return x > 0

        assert custom.description == "Custom tool description Returns: bool"

    def test_to_dict_includes_full_description(self) -> None:
        """Test that to_dict() includes the enhanced description."""

        @tool
        def my_tool(x: int) -> str:
            """Do something."""
            return str(x)

        schema = my_tool.to_dict()
        assert "Returns: str" in schema["function"]["description"]
