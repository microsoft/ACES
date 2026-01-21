"""SABER Copilot Tool Bridge.

This module provides utilities to convert MCP tools to GitHub Copilot SDK
Tool format, enabling SABER sandbox tools to be used by the Copilot agent.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ...logging_config import LogCategory, get_saber_logger

logger = get_saber_logger(LogCategory.AGENT, __name__)

if TYPE_CHECKING:
    pass

# Type aliases for Copilot SDK types (avoid hard dependency at import time)
# The actual types are imported from copilot SDK when functions are called
ToolResult = dict[str, Any]
ToolInvocation = dict[str, Any]


@dataclass
class ToolCallRecord:
    """Record of a tool call for transcript tracking."""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    result: str
    is_error: bool = False


@dataclass
class ToolCallTracker:
    """Tracks tool calls during a Copilot session for transcript recording.

    This class accumulates tool calls made during agent execution so they
    can be added to the Inspect AI transcript for visibility.
    """

    calls: list[ToolCallRecord] = field(default_factory=list)

    def record_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
        is_error: bool = False,
        tool_call_id: str | None = None,
    ) -> ToolCallRecord:
        """Record a tool call.

        Args:
            tool_name: Name of the tool called
            arguments: Arguments passed to the tool
            result: Result text from the tool
            is_error: Whether the call resulted in an error
            tool_call_id: Optional ID (generated if not provided)

        Returns:
            The recorded ToolCallRecord
        """
        record = ToolCallRecord(
            tool_call_id=tool_call_id or f"call_{uuid.uuid4().hex[:8]}",
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            is_error=is_error,
        )
        self.calls.append(record)
        logger.debug(
            f"Recorded tool call: {tool_name}",
            extra={
                "tool_call_id": record.tool_call_id,
                "arguments_keys": list(arguments.keys()),
                "result_length": len(result),
                "is_error": is_error,
            },
        )
        return record

    def get_and_clear(self) -> list[ToolCallRecord]:
        """Get all recorded calls and clear the tracker.

        Returns:
            List of tool call records
        """
        calls = self.calls.copy()
        self.calls = []
        return calls

    def clear(self) -> None:
        """Clear all recorded calls."""
        self.calls = []


class Tool:
    """Copilot SDK Tool representation.

    This is a simplified representation compatible with the Copilot SDK Tool class.
    It allows us to define tools with name, description, parameters schema, and handler.
    """

    def __init__(
        self,
        name: str,
        description: str,
        handler: Callable[[ToolInvocation], Any],
        parameters: dict[str, Any] | None = None,
    ):
        self.name = name
        self.description = description
        self.handler = handler
        self.parameters = parameters


def _python_type_to_json_schema(type_name: str) -> str:
    """Convert Python type name to JSON Schema type.

    Args:
        type_name: Python type name (e.g., 'str', 'int', 'bool')

    Returns:
        JSON Schema type string
    """
    type_map = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
        "None": "null",
        "NoneType": "null",
    }
    return type_map.get(type_name, "string")


def _clean_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Clean a JSON schema by removing None values and empty fields.

    The Copilot SDK / OpenAI API doesn't accept None values in schemas.
    This function recursively removes all None values and empty containers.

    Args:
        schema: JSON schema dictionary (may have None values)

    Returns:
        Cleaned JSON schema with no None values
    """
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        # Skip None values entirely
        if value is None:
            continue

        # Recursively clean nested dicts
        if isinstance(value, dict):
            cleaned_value = _clean_json_schema(value)
            if cleaned_value:  # Only include non-empty dicts
                cleaned[key] = cleaned_value
        # Recursively clean items in lists
        elif isinstance(value, list):
            cleaned_list: list[Any] = [
                _clean_json_schema(item) if isinstance(item, dict) else item for item in value if item is not None
            ]
            if cleaned_list:  # Only include non-empty lists
                cleaned[key] = cleaned_list
        else:
            cleaned[key] = value

    return cleaned


def mcp_tool_to_copilot_tool(
    inspect_tool: Any,
    mcp_client: Any,
    tracker: ToolCallTracker | None = None,
) -> Tool:
    """Convert an Inspect AI tool or MCP tool to a Copilot SDK Tool.

    Creates a Copilot Tool that wraps the tool, allowing
    the Copilot agent to invoke SABER sandbox tools.

    Args:
        inspect_tool: Inspect AI tool function (decorated with @tool) or MCP tool object
        mcp_client: MCP client/server instance (used for context, may be needed for some tools)
        tracker: Optional ToolCallTracker to record calls for transcript visibility

    Returns:
        Copilot SDK Tool instance
    """
    # Get tool name - handle both inspect_ai tools (functions) and MCP tools (objects)
    tool_name = (
        getattr(inspect_tool, "name", None)  # MCP-style
        or getattr(inspect_tool, "__name__", None)  # Function-style
        or "unknown_tool"
    )

    # Get description - handle both styles
    tool_description = (
        getattr(inspect_tool, "description", None)  # MCP-style
        or getattr(inspect_tool, "__doc__", None)  # Function-style
        or f"Execute {tool_name}"
    )

    # Get input schema - try multiple approaches
    input_schema = None

    # 1. Try inspect_ai Tool's __TOOL_DESCRIPTION__ (set by ToolDef.as_tool())
    tool_desc = getattr(inspect_tool, "__TOOL_DESCRIPTION__", None)
    if tool_desc is not None and hasattr(tool_desc, "parameters") and tool_desc.parameters is not None:
        # ToolParams is a Pydantic model - convert to dict
        params = tool_desc.parameters
        if hasattr(params, "model_dump"):
            input_schema = params.model_dump()
        elif hasattr(params, "dict"):
            input_schema = params.dict()
        # Also get name and description from tool_desc if not already set
        if tool_desc.name:
            tool_name = tool_desc.name
        if tool_desc.description:
            tool_description = tool_desc.description

    # 2. Try MCP-style inputSchema
    if input_schema is None:
        input_schema = getattr(inspect_tool, "inputSchema", None) or getattr(inspect_tool, "input_schema", None)

    # 3. Try to build from function annotations as fallback
    if input_schema is None and hasattr(inspect_tool, "__annotations__"):
        props = {}
        required = []
        for param_name, param_type in inspect_tool.__annotations__.items():
            if param_name == "return":
                continue
            type_name = getattr(param_type, "__name__", str(param_type))
            props[param_name] = {"type": _python_type_to_json_schema(type_name)}
            required.append(param_name)
        if props:
            input_schema = {
                "type": "object",
                "properties": props,
                "required": required,
            }

    # Clean the schema to remove None values (Copilot SDK / OpenAI API rejects them)
    if input_schema is not None:
        input_schema = _clean_json_schema(input_schema)

    logger.debug(
        "Converting tool to Copilot tool",
        extra={
            "tool_name": tool_name,
            "has_schema": input_schema is not None,
            "schema_keys": list(input_schema.keys()) if input_schema else None,
            "schema_properties": list(input_schema.get("properties", {}).keys()) if input_schema else None,
        },
    )

    # Determine if this is a callable (inspect_ai tool) or an MCP tool object
    # MCP tool objects have a .name attribute explicitly set, functions have __name__
    # We check if the tool has .name as an instance attribute (not from __name__)
    is_mcp_style = (
        hasattr(inspect_tool, "name")
        and not callable(type(inspect_tool))
        or (hasattr(inspect_tool, "name") and inspect_tool.name != getattr(inspect_tool, "__name__", None))
    )
    is_callable_tool = not is_mcp_style and callable(inspect_tool)

    async def handler(invocation: ToolInvocation) -> ToolResult:
        """Handler that invokes the tool.

        Args:
            invocation: Copilot tool invocation with session_id, tool_call_id, etc.

        Returns:
            ToolResult with textResultForLlm and resultType
        """
        raw_arguments = invocation.get("arguments", {})

        # Filter arguments to only include parameters defined in the schema
        # The model sometimes adds extra params like "description" that the tool doesn't accept
        if input_schema and "properties" in input_schema:
            valid_params = set(input_schema["properties"].keys())
            arguments = {k: v for k, v in raw_arguments.items() if k in valid_params}
            filtered_params = set(raw_arguments.keys()) - valid_params
            if filtered_params:
                logger.debug(
                    "Filtered out unexpected tool arguments",
                    extra={
                        "tool_name": tool_name,
                        "filtered_params": list(filtered_params),
                        "valid_params": list(valid_params),
                    },
                )
        else:
            arguments = raw_arguments

        logger.info(
            "Executing tool via Copilot bridge",
            extra={
                "tool_name": tool_name,
                "tool_call_id": invocation.get("tool_call_id"),
                "session_id": invocation.get("session_id"),
                "is_callable": is_callable_tool,
                "argument_keys": list(arguments.keys()),
            },
        )

        try:
            if is_callable_tool:
                # Call the inspect_ai tool directly
                # The tool is a callable that may be sync or async
                import asyncio

                if asyncio.iscoroutinefunction(inspect_tool):
                    result = await inspect_tool(**arguments)
                else:
                    result = inspect_tool(**arguments)
                text_content = str(result) if result is not None else "Tool executed successfully"
            else:
                # MCP-style tool - call via mcp_client.call_tool()
                # This returns a ToolResult which is list[Content]
                import asyncio
                import traceback as tb

                try:
                    logger.debug(
                        "Calling MCP tool via call_tool",
                        extra={
                            "tool_name": tool_name,
                            "mcp_client_type": type(mcp_client).__name__,
                            "has_call_tool": hasattr(mcp_client, "call_tool"),
                        },
                    )
                    result = await mcp_client.call_tool(tool_name, arguments)
                    logger.debug(
                        "MCP tool call succeeded",
                        extra={"tool_name": tool_name, "result_type": type(result).__name__},
                    )
                except BaseException as async_err:
                    # Handle ExceptionGroup and other async errors
                    # Extract the actual error message from ExceptionGroup if present
                    error_msg = str(async_err)
                    full_traceback = tb.format_exc()
                    if hasattr(async_err, "exceptions"):
                        # ExceptionGroup - get the first sub-exception
                        sub_exceptions = list(async_err.exceptions)
                        if sub_exceptions:
                            error_msg = str(sub_exceptions[0])
                            # Get full traceback of sub-exception
                            full_traceback = "".join(
                                tb.format_exception(
                                    type(sub_exceptions[0]), sub_exceptions[0], sub_exceptions[0].__traceback__
                                )
                            )
                    logger.error(
                        "MCP tool call failed with error",
                        extra={
                            "tool_name": tool_name,
                            "error_type": type(async_err).__name__,
                            "error_msg": error_msg,
                            "full_traceback": full_traceback,
                        },
                    )
                    raise RuntimeError(f"MCP tool call failed: {error_msg}") from async_err

                # Extract text content from result
                # Result could be:
                # 1. A list of Content items (ToolResult from call_tool)
                # 2. An object with .content attribute (raw MCP result)
                # 3. A string
                text_content = ""
                content_items = None

                if isinstance(result, list):
                    # ToolResult is list[Content]
                    content_items = result
                elif hasattr(result, "content") and result.content:
                    # Raw MCP result with .content attribute
                    content_items = result.content
                elif isinstance(result, str):
                    text_content = result

                if content_items:
                    text_parts = []
                    for content_item in content_items:
                        if hasattr(content_item, "text"):
                            text_parts.append(content_item.text)
                        elif isinstance(content_item, str):
                            text_parts.append(content_item)
                        else:
                            # Try to convert to string
                            text_parts.append(str(content_item))
                    text_content = "\n".join(text_parts)

                # Check for error (only relevant for raw MCP results)
                is_error = getattr(result, "isError", False) or getattr(result, "is_error", False)
                if is_error:
                    logger.warning(
                        "MCP tool returned error",
                        extra={"tool_name": tool_name, "error": text_content[:200]},
                    )
                    # Record error in tracker for transcript
                    if tracker:
                        tracker.record_call(
                            tool_name=tool_name,
                            arguments=arguments,
                            result=text_content or "Tool execution failed",
                            is_error=True,
                            tool_call_id=invocation.get("tool_call_id"),
                        )
                    return {
                        "textResultForLlm": text_content or "Tool execution failed",
                        "resultType": "failure",
                        "error": text_content,
                    }

            logger.debug(
                "Tool completed",
                extra={"tool_name": tool_name, "result_length": len(text_content)},
            )

            # Record successful call in tracker for transcript
            if tracker:
                tracker.record_call(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=text_content,
                    is_error=False,
                    tool_call_id=invocation.get("tool_call_id"),
                )

            return {
                "textResultForLlm": text_content,
                "resultType": "success",
            }

        except Exception as e:
            logger.error(
                "Exception executing Inspect AI tool",
                extra={"tool_name": tool_name, "error": str(e)},
                exc_info=True,
            )
            error_text = f"Tool execution error: {type(e).__name__}: {str(e)}"

            # Record error in tracker for transcript
            if tracker:
                tracker.record_call(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=error_text,
                    is_error=True,
                    tool_call_id=invocation.get("tool_call_id"),
                )

            return {
                "textResultForLlm": error_text,
                "resultType": "failure",
                "error": str(e),
            }

    return Tool(
        name=tool_name,
        description=tool_description,
        handler=handler,
        parameters=input_schema,
    )


def convert_mcp_tools_to_copilot(
    mcp_tools: list[Any],
    mcp_client: Any,
    tracker: ToolCallTracker | None = None,
) -> list[Tool]:
    """Convert a list of MCP tools to Copilot SDK Tools.

    Args:
        mcp_tools: List of MCP tool objects
        mcp_client: MCP client instance for executing tool calls
        tracker: Optional ToolCallTracker to record calls for transcript visibility

    Returns:
        List of Copilot SDK Tool instances
    """
    if not mcp_tools:
        return []

    copilot_tools = []
    for mcp_tool in mcp_tools:
        copilot_tool = mcp_tool_to_copilot_tool(mcp_tool, mcp_client, tracker)
        copilot_tools.append(copilot_tool)

    logger.info(
        f"Converted {len(copilot_tools)} MCP tools to Copilot format",
        extra={"tool_names": [t.name for t in copilot_tools]},
    )

    return copilot_tools


def create_submit_tool(
    submission_handler: Callable[[str], Any],
    tracker: ToolCallTracker | None = None,
) -> Tool:
    """Create the submit_answer tool for SABER task completion.

    This tool allows the Copilot agent to submit its final answer,
    triggering SABER's evaluation and scoring pipeline.

    Args:
        submission_handler: Async function that handles answer submission.
                           Called with the answer string.
        tracker: Optional ToolCallTracker to record calls for transcript visibility

    Returns:
        Copilot SDK Tool for submitting answers
    """

    async def handler(invocation: ToolInvocation) -> ToolResult:
        """Handle answer submission.

        Args:
            invocation: Tool invocation with answer in arguments

        Returns:
            ToolResult indicating success/failure
        """
        arguments = invocation.get("arguments", {})
        answer = arguments.get("answer", "")

        logger.info(
            "Submitting answer via Copilot agent",
            extra={
                "tool_call_id": invocation.get("tool_call_id"),
                "answer_length": len(answer),
            },
        )

        try:
            await submission_handler(answer)

            result_text = "Answer submitted successfully. The task is now complete."

            # Record submission in tracker for transcript
            if tracker:
                tracker.record_call(
                    tool_name="submit_answer",
                    arguments={"answer": answer},
                    result=result_text,
                    is_error=False,
                    tool_call_id=invocation.get("tool_call_id"),
                )

            return {
                "textResultForLlm": result_text,
                "resultType": "success",
                "sessionLog": f"Submitted answer: {answer[:100]}{'...' if len(answer) > 100 else ''}",
            }

        except Exception as e:
            logger.error(
                "Error submitting answer",
                extra={"error": str(e)},
                exc_info=True,
            )
            error_text = "Failed to submit answer. Please try again."

            # Record error in tracker for transcript
            if tracker:
                tracker.record_call(
                    tool_name="submit_answer",
                    arguments={"answer": answer},
                    result=error_text,
                    is_error=True,
                    tool_call_id=invocation.get("tool_call_id"),
                )

            return {
                "textResultForLlm": error_text,
                "resultType": "failure",
                "error": str(e),
            }

    return Tool(
        name="submit_answer",
        description=(
            "Submit your final answer to complete the task. "
            "Use this tool when you have finished analyzing the problem and have a solution. "
            "Your answer should be comprehensive and address all aspects of the task."
        ),
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Your final answer or solution to the task",
                },
            },
            "required": ["answer"],
        },
    )


async def get_saber_mcp_tools(sandbox: Any) -> list[Any]:
    """Get MCP tools from SABER sandbox.

    Retrieves the list of available tools from the sandbox's MCP client.

    Args:
        sandbox: SABERSandboxEnvironment instance

    Returns:
        List of MCP tool objects

    Raises:
        RuntimeError: If MCP client is not available
    """
    # Get the actual sandbox if wrapped in proxy
    actual_sandbox = sandbox
    if hasattr(sandbox, "_sandbox"):
        actual_sandbox = sandbox._sandbox

    if actual_sandbox._mcp_client is None:
        raise RuntimeError("SABER MCP client not available. Ensure sample_init() has completed.")

    try:
        # Get tools from MCP server
        # The MCP client may be an MCPServerLocal which has a tools() method
        mcp_client = actual_sandbox._mcp_client

        # If it's an MCP server wrapper, call tools() to get the tool list
        if hasattr(mcp_client, "tools"):
            tools = await mcp_client.tools()
            logger.info(
                f"Retrieved {len(tools)} tools from MCP client",
                extra={"tool_count": len(tools)},
            )
            return list(tools)
        else:
            # Try list_tools for raw MCP client
            tools = await mcp_client.list_tools()
            return list(tools)

    except Exception as e:
        logger.error(f"Failed to get MCP tools: {e}", exc_info=True)
        raise


__all__ = [
    "Tool",
    "mcp_tool_to_copilot_tool",
    "convert_mcp_tools_to_copilot",
    "create_submit_tool",
    "get_saber_mcp_tools",
]
