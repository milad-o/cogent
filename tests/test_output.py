"""Tests for structured output module."""

import json
import pytest
from dataclasses import dataclass, field
from typing import Literal

from agenticflow.agent.output import (
    ResponseSchema,
    OutputMethod,
    StructuredResult,
    OutputValidationError,
    schema_to_json,
    get_schema_name,
    get_schema_description,
    parse_json_output,
    validate_and_parse,
    build_structured_prompt,
    is_pydantic_model,
    is_typed_dict,
    supports_native_structured_output,
    get_best_method,
)


# =============================================================================
# Test Schemas
# =============================================================================

@dataclass
class ContactInfo:
    """Contact information for a person."""
    name: str
    email: str
    phone: str | None = None


@dataclass
class Review:
    """Product review with rating."""
    rating: int
    summary: str
    pros: list[str] = field(default_factory=list)


try:
    from pydantic import BaseModel, Field
    
    class PydanticContact(BaseModel):
        """Contact information using Pydantic."""
        name: str = Field(description="Person's full name")
        email: str = Field(description="Email address")
        phone: str | None = Field(None, description="Phone number")
    
    class PydanticReview(BaseModel):
        """Review with validation."""
        rating: int = Field(ge=1, le=5, description="Rating 1-5")
        sentiment: Literal["positive", "negative", "neutral"]
        summary: str = Field(description="Brief summary")
    
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False


# =============================================================================
# Schema Detection Tests
# =============================================================================

class TestSchemaDetection:
    """Test schema type detection functions."""
    
    def test_is_pydantic_model_with_dataclass(self):
        """Dataclass should not be detected as Pydantic."""
        assert is_pydantic_model(ContactInfo) is False
    
    @pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic not installed")
    def test_is_pydantic_model_with_pydantic(self):
        """Pydantic model should be detected."""
        assert is_pydantic_model(PydanticContact) is True
    
    def test_is_typed_dict(self):
        """TypedDict should be detected."""
        from typing import TypedDict
        
        class MyDict(TypedDict):
            name: str
            value: int
        
        assert is_typed_dict(MyDict) is True
        assert is_typed_dict(ContactInfo) is False


# =============================================================================
# Schema to JSON Tests
# =============================================================================

class TestSchemaToJson:
    """Test JSON schema generation."""
    
    def test_dataclass_to_json(self):
        """Dataclass should convert to JSON Schema."""
        schema = schema_to_json(ContactInfo)
        
        assert schema["type"] == "object"
        assert schema["title"] == "ContactInfo"
        assert "name" in schema["properties"]
        assert "email" in schema["properties"]
        assert "phone" in schema["properties"]
        assert schema["properties"]["name"]["type"] == "string"
    
    @pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic not installed")
    def test_pydantic_to_json(self):
        """Pydantic model should use model_json_schema."""
        schema = schema_to_json(PydanticContact)
        
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "description" in schema["properties"]["name"]
    
    def test_json_schema_passthrough(self):
        """JSON Schema dict should pass through unchanged."""
        input_schema = {
            "type": "object",
            "properties": {"foo": {"type": "string"}},
        }
        schema = schema_to_json(input_schema)
        assert schema == input_schema
    
    def test_typed_dict_to_json(self):
        """TypedDict should convert to JSON Schema."""
        from typing import TypedDict
        
        class Person(TypedDict):
            name: str
            age: int
        
        schema = schema_to_json(Person)
        
        assert schema["type"] == "object"
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["age"]["type"] == "integer"


# =============================================================================
# Schema Name/Description Tests
# =============================================================================

class TestSchemaMetadata:
    """Test schema metadata extraction."""
    
    def test_get_schema_name_dataclass(self):
        """Should get name from dataclass."""
        assert get_schema_name(ContactInfo) == "ContactInfo"
    
    def test_get_schema_name_dict(self):
        """Should get title from JSON Schema dict."""
        schema = {"title": "MySchema", "type": "object"}
        assert get_schema_name(schema) == "MySchema"
    
    def test_get_schema_description_dataclass(self):
        """Should get docstring from dataclass."""
        desc = get_schema_description(ContactInfo)
        assert "Contact information" in desc


# =============================================================================
# JSON Parsing Tests
# =============================================================================

class TestJsonParsing:
    """Test JSON extraction from LLM output."""
    
    def test_plain_json(self):
        """Parse plain JSON."""
        raw = '{"name": "John", "email": "john@example.com"}'
        result = parse_json_output(raw)
        assert result["name"] == "John"
    
    def test_json_with_markdown(self):
        """Parse JSON wrapped in markdown code block."""
        raw = """Here's the contact info:

```json
{"name": "Jane", "email": "jane@example.com"}
```

That's all!"""
        result = parse_json_output(raw)
        assert result["name"] == "Jane"
    
    def test_json_embedded_in_text(self):
        """Parse JSON embedded in text."""
        raw = """The extracted information is: {"name": "Bob", "email": "bob@test.com"} as requested."""
        result = parse_json_output(raw)
        assert result["name"] == "Bob"
    
    def test_invalid_json_raises(self):
        """Invalid JSON should raise error."""
        raw = "This is not JSON at all"
        with pytest.raises(json.JSONDecodeError):
            parse_json_output(raw)


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidation:
    """Test schema validation."""
    
    def test_validate_dataclass(self):
        """Validate and parse dataclass."""
        raw = '{"name": "Test", "email": "test@example.com"}'
        result = validate_and_parse(raw, ContactInfo)
        
        assert isinstance(result, ContactInfo)
        assert result.name == "Test"
        assert result.email == "test@example.com"
    
    def test_validate_dataclass_with_optional(self):
        """Optional fields should work."""
        raw = '{"name": "Test", "email": "test@example.com", "phone": "555-1234"}'
        result = validate_and_parse(raw, ContactInfo)
        
        assert result.phone == "555-1234"
    
    @pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic not installed")
    def test_validate_pydantic(self):
        """Validate Pydantic model."""
        raw = '{"name": "Test", "email": "test@example.com"}'
        result = validate_and_parse(raw, PydanticContact)
        
        assert isinstance(result, PydanticContact)
        assert result.name == "Test"
    
    @pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic not installed")
    def test_validate_pydantic_fails(self):
        """Invalid data should raise OutputValidationError."""
        raw = '{"rating": 10, "sentiment": "happy", "summary": "Great"}'
        
        with pytest.raises(OutputValidationError):
            validate_and_parse(raw, PydanticReview)
    
    def test_validate_json_schema(self):
        """JSON Schema validation returns dict."""
        schema = {"type": "object", "properties": {"foo": {"type": "string"}}}
        raw = '{"foo": "bar"}'
        result = validate_and_parse(raw, schema)
        
        assert result == {"foo": "bar"}


# =============================================================================
# ResponseSchema Tests
# =============================================================================

class TestResponseSchema:
    """Test ResponseSchema creation and properties."""
    
    def test_create_with_dataclass(self):
        """Create config from dataclass."""
        config = ResponseSchema(schema=ContactInfo)
        
        assert config.schema == ContactInfo
        assert config.method == OutputMethod.AUTO
        assert config.retry_on_error is True
    
    def test_create_with_method(self):
        """Create config with specific method."""
        config = ResponseSchema(schema=ContactInfo, method=OutputMethod.TOOL)
        
        assert config.method == OutputMethod.TOOL
    
    def test_json_schema_cached(self):
        """JSON schema should be cached."""
        config = ResponseSchema(schema=ContactInfo)
        
        schema1 = config.json_schema
        schema2 = config.json_schema
        
        assert schema1 is schema2
    
    def test_tool_definition(self):
        """Tool definition should be generated."""
        config = ResponseSchema(schema=ContactInfo)
        tool_def = config.tool_definition
        
        assert tool_def["type"] == "function"
        assert tool_def["function"]["name"] == "ContactInfo"


# =============================================================================
# StructuredResult Tests
# =============================================================================

class TestStructuredResult:
    """Test StructuredResult behavior."""
    
    def test_valid_result_bool(self):
        """Valid result should be truthy."""
        result = StructuredResult(data={"foo": "bar"}, valid=True)
        assert bool(result) is True
    
    def test_invalid_result_bool(self):
        """Invalid result should be falsy."""
        result = StructuredResult(data=None, valid=False, error="Failed")
        assert bool(result) is False
    
    def test_unwrap_valid(self):
        """Unwrap should return data for valid result."""
        data = {"foo": "bar"}
        result = StructuredResult(data=data, valid=True)
        assert result.unwrap() == data
    
    def test_unwrap_invalid_raises(self):
        """Unwrap should raise for invalid result."""
        result = StructuredResult(data=None, valid=False, error="Validation failed")
        
        with pytest.raises(ValueError, match="Validation failed"):
            result.unwrap()


# =============================================================================
# Prompt Building Tests
# =============================================================================

class TestPromptBuilding:
    """Test structured output prompt generation."""
    
    def test_build_basic_prompt(self):
        """Build prompt with schema."""
        config = ResponseSchema(schema=ContactInfo)
        prompt = build_structured_prompt("Extract contact info", config)
        
        assert "Extract contact info" in prompt
        assert "JSON" in prompt
        assert "name" in prompt
    
    def test_build_prompt_with_error(self):
        """Build retry prompt with error context."""
        config = ResponseSchema(schema=ContactInfo)
        prompt = build_structured_prompt(
            "Extract contact info",
            config,
            error_context="Missing required field: email",
        )
        
        assert "Missing required field: email" in prompt


# =============================================================================
# Provider Detection Tests
# =============================================================================

class TestProviderDetection:
    """Test provider capability detection."""
    
    def test_openai_supports_native(self):
        """OpenAI models should support native structured output."""
        class MockOpenAI:
            pass
        
        mock = MockOpenAI()
        mock.__class__.__name__ = "OpenAIChat"
        
        assert supports_native_structured_output(mock) is True
    
    def test_unknown_provider(self):
        """Unknown providers should not support native."""
        class MockUnknown:
            pass
        
        assert supports_native_structured_output(MockUnknown()) is False
    
    def test_get_best_method_auto(self):
        """Auto should detect best method."""
        class MockOpenAI:
            pass
        
        mock = MockOpenAI()
        mock.__class__.__name__ = "OpenAIChat"
        config = ResponseSchema(schema=ContactInfo)
        
        method = get_best_method(mock, config)
        assert method == OutputMethod.NATIVE
    
    def test_get_best_method_explicit(self):
        """Explicit method should be used."""
        class MockOpenAI:
            pass
        
        mock = MockOpenAI()
        mock.__class__.__name__ = "OpenAIChat"
        config = ResponseSchema(schema=ContactInfo, method=OutputMethod.PROMPT)
        
        method = get_best_method(mock, config)
        assert method == OutputMethod.PROMPT


# =============================================================================
# Integration Tests (without actual LLM)
# =============================================================================

class TestIntegration:
    """Integration tests for output processing."""
    
    def test_end_to_end_dataclass(self):
        """End-to-end test with dataclass."""
        config = ResponseSchema(schema=ContactInfo)
        raw_output = '{"name": "John Doe", "email": "john@example.com", "phone": "555-1234"}'
        
        data = validate_and_parse(raw_output, config.schema)
        
        assert data.name == "John Doe"
        assert data.email == "john@example.com"
        assert data.phone == "555-1234"
    
    @pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic not installed")
    def test_end_to_end_pydantic(self):
        """End-to-end test with Pydantic."""
        config = ResponseSchema(schema=PydanticReview)
        raw_output = '{"rating": 5, "sentiment": "positive", "summary": "Great product!"}'
        
        data = validate_and_parse(raw_output, config.schema)
        
        assert data.rating == 5
        assert data.sentiment == "positive"
