"""
Unit tests for MCP models and type safety.
"""

import pytest
from pydantic import ValidationError

from saber.models.mcp import (
    MCPToolCallRequest,
    MCPToolCallResponse,
    MCPToolListResponse,
    MCPToolSchema,
    MCPInputSchema,
    MCPPropertySchema,
    SessionContext,
    MCPConnectionError,
    MCPToolNotFoundError,
    MCPExecutionError,
    MCPTimeoutError
)


class TestMCPPropertySchema:
    """Test MCPPropertySchema model."""

    def test_property_schema_basic(self):
        """Test basic property schema creation."""
        prop = MCPPropertySchema(
            type="string",
            description="Test property"
        )

        assert prop.type == "string"
        assert prop.description == "Test property"
        assert prop.default is None
        assert prop.enum is None

    def test_property_schema_with_constraints(self):
        """Test property schema with constraints."""
        prop = MCPPropertySchema(
            type="integer",
            description="Age property",
            minimum=0,
            maximum=150,
            default=25
        )

        assert prop.minimum == 0
        assert prop.maximum == 150
        assert prop.default == 25

    def test_property_schema_enum(self):
        """Test property schema with enum values."""
        prop = MCPPropertySchema(
            type="string",
            description="Status property",
            enum=["active", "inactive", "pending"]
        )

        assert prop.enum == ["active", "inactive", "pending"]


class TestMCPInputSchema:
    """Test MCPInputSchema model."""

    def test_input_schema_creation(self):
        """Test input schema creation."""
        schema = MCPInputSchema(
            type="object",
            properties={
                "name": MCPPropertySchema(type="string", description="Name"),
                "age": MCPPropertySchema(type="integer", description="Age")
            },
            required=["name"]
        )

        assert schema.type == "object"
        assert len(schema.properties) == 2
        assert "name" in schema.properties
        assert "age" in schema.properties
        assert schema.required == ["name"]

    def test_input_schema_defaults(self):
        """Test input schema with defaults."""
        schema = MCPInputSchema(
            properties={}
        )

        assert schema.type == "object"
        assert schema.required == []


class TestMCPToolSchema:
    """Test MCPToolSchema model."""

    def test_tool_schema_creation(self):
        """Test tool schema creation."""
        input_schema = MCPInputSchema(
            properties={
                "command": MCPPropertySchema(type="string", description="Command to execute")
            },
            required=["command"]
        )

        tool = MCPToolSchema(
            name="bash",
            description="Execute CLI commands",
            inputSchema=input_schema
        )

        assert tool.name == "bash"
        assert tool.description == "Execute CLI commands"
        assert tool.inputSchema == input_schema


class TestMCPToolListResponse:
    """Test MCPToolListResponse model."""

    def test_tool_list_response_creation(self):
        """Test tool list response creation."""
        tool_schema = MCPToolSchema(
            name="test_tool",
            description="Test tool",
            inputSchema=MCPInputSchema(properties={})
        )

        response = MCPToolListResponse(
            tools=[tool_schema],
            session_id="test-session",
            episode_id="test-episode"
        )

        assert len(response.tools) == 1
        assert response.tools[0] == tool_schema
        assert response.session_id == "test-session"
        assert response.episode_id == "test-episode"

    def test_tool_list_response_minimal(self):
        """Test tool list response with minimal data."""
        response = MCPToolListResponse(tools=[])

        assert response.tools == []
        assert response.session_id is None
        assert response.episode_id is None


class TestMCPToolCallResponse:
    """Test MCPToolCallResponse model."""

    def test_tool_call_response_success(self):
        """Test successful tool call response."""
        response = MCPToolCallResponse(
            content=[{"type": "text", "text": "Success"}],
            isError=False
        )

        assert not response.isError
        assert len(response.content) == 1
        assert response.content[0]["text"] == "Success"

    def test_tool_call_response_error(self):
        """Test error tool call response."""
        response = MCPToolCallResponse(
            content=[{"type": "text", "text": "Error: Command failed"}],
            isError=True
        )

        assert response.isError
        assert "Error:" in response.content[0]["text"]


class TestMCPToolCallRequest:
    """Test MCPToolCallRequest model validation."""

    def test_tool_call_request_minimal(self):
        """Test minimal tool call request."""
        request = MCPToolCallRequest(tool_name="test", episode_id="test-episode")

        assert request.tool_name == "test"
        assert request.episode_id == "test-episode"
        assert request.arguments == {}
        assert request.timeout is None
        assert request.context is None

    def test_tool_call_request_full(self):
        """Test full tool call request."""
        request = MCPToolCallRequest(
            tool_name="bash",
            episode_id="test-episode",
            arguments={"command": "ls -la"},
            timeout=30.0,
            context={"episode_id": "test-episode"}
        )

        assert request.tool_name == "bash"
        assert request.episode_id == "test-episode"
        assert request.arguments == {"command": "ls -la"}
        assert request.timeout == 30.0
        assert request.context == {"episode_id": "test-episode"}

    def test_tool_call_request_validation_error(self):
        """Test tool call request validation error."""
        with pytest.raises(ValidationError):
            MCPToolCallRequest()  # Missing required tool_name


class TestSessionContext:
    """Test SessionContext model validation."""

    def test_session_context_minimal(self):
        """Test minimal session context."""
        context = SessionContext(
            session_id="test-session",
            episode_id="test-episode",
            client_id="test-client"
        )

        assert context.session_id == "test-session"
        assert context.episode_id == "test-episode"
        assert context.client_id == "test-client"
        assert context.task_id is None

    def test_session_context_full(self):
        """Test full session context."""
        context = SessionContext(
            session_id="test-session",
            episode_id="test-episode",
            task_id="test-task",
            client_id="test-client"
        )

        assert context.task_id == "test-task"

    def test_session_context_validation_error(self):
        """Test session context validation errors."""
        with pytest.raises(ValidationError):
            SessionContext()  # Missing required fields

        with pytest.raises(ValidationError):
            SessionContext(session_id="test")  # Missing episode_id and client_id


class TestMCPExceptions:
    """Test MCP exception types."""

    def test_mcp_connection_error(self):
        """Test MCPConnectionError."""
        error = MCPConnectionError("Connection failed")
        assert str(error) == "Connection failed"
        assert isinstance(error, Exception)

    def test_mcp_tool_not_found_error(self):
        """Test MCPToolNotFoundError."""
        error = MCPToolNotFoundError("Tool 'xyz' not found")
        assert "Tool 'xyz' not found" in str(error)

    def test_mcp_execution_error(self):
        """Test MCPExecutionError."""
        error = MCPExecutionError("Tool execution failed")
        assert "Tool execution failed" in str(error)

    def test_mcp_timeout_error(self):
        """Test MCPTimeoutError."""
        error = MCPTimeoutError("Request timed out")
        assert "Request timed out" in str(error)


class TestTypeCompatibility:
    """Test type compatibility and serialization."""

    def test_json_serialization(self):
        """Test JSON serialization of models."""
        request = MCPToolCallRequest(
            tool_name="test_tool",
            episode_id="test-episode",
            arguments={"param": "value"},
            timeout=30.0
        )

        # Test model_dump
        data = request.model_dump()
        assert data["tool_name"] == "test_tool"
        assert data["episode_id"] == "test-episode"
        assert data["arguments"] == {"param": "value"}
        assert data["timeout"] == 30.0

        # Test round-trip
        reconstructed = MCPToolCallRequest.model_validate(data)
        assert reconstructed == request

    def test_nested_model_serialization(self):
        """Test serialization of nested models."""
        tool_schema = MCPToolSchema(
            name="test_tool",
            description="Test tool",
            inputSchema=MCPInputSchema(
                properties={
                    "param": MCPPropertySchema(type="string", description="Parameter")
                },
                required=["param"]
            )
        )

        response = MCPToolListResponse(tools=[tool_schema])

        # Test serialization
        data = response.model_dump()
        assert len(data["tools"]) == 1
        assert data["tools"][0]["name"] == "test_tool"

        # Test round-trip
        reconstructed = MCPToolListResponse.model_validate(data)
        assert reconstructed == response


if __name__ == "__main__":
    pytest.main([__file__])
