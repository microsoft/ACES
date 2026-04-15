"""Copilot model provider for inspect_ai.

Pure conversion functions and CopilotModelAPI class for routing inference
through GitHub's Copilot infrastructure via the Python Copilot SDK.
SDK objects are accessed via ``getattr``/``hasattr`` to avoid hard imports.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass
from typing import ClassVar

from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ModelAPI,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.tool import ToolCall, ToolChoice, ToolInfo

from saber.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CopilotToolDef:
    """Lightweight tool definition for the Copilot SDK.

    Mirrors copilot.tools.Tool schema without requiring SDK import.
    Converted to SDK Tool at session creation time.
    """

    name: str
    description: str
    parameters: dict[str, object]


def _messages_to_prompt(messages: list[ChatMessage]) -> str:
    """Serialize inspect_ai ChatMessage list to a single prompt string.

    Format::

        [system]
        {content}

        [user]
        {content}

        [assistant]
        {content}

        [tool: {function_name}]
        {content}

    Args:
        messages: List of chat messages to serialize.

    Returns:
        Formatted prompt string. Empty string if messages is empty.
    """
    if not messages:
        return ""

    parts: list[str] = []
    for msg in messages:
        prefix = _role_prefix(msg)
        text = msg.text

        section = f"{prefix}\n{text}"

        # Append tool_call annotations for assistant messages
        if isinstance(msg, ChatMessageAssistant) and msg.tool_calls:
            call_lines: list[str] = []
            for tc in msg.tool_calls:
                args_json = json.dumps(tc.arguments, separators=(",", ":"))
                call_lines.append(f"[tool_call: {tc.function}({args_json})]")
            section = section + "\n" + "\n".join(call_lines)

        parts.append(section)

    return "\n\n".join(parts)


def _role_prefix(msg: ChatMessage) -> str:
    """Build the bracket-prefixed role tag for a message.

    Args:
        msg: A chat message.

    Returns:
        Role prefix string, e.g. ``[system]`` or ``[tool: func_name]``.
    """
    if isinstance(msg, ChatMessageTool) and msg.function:
        return f"[tool: {msg.function}]"
    return f"[{msg.role}]"


def _tool_info_to_sdk_tool(tool_info: ToolInfo) -> CopilotToolDef:
    """Convert an inspect_ai ToolInfo to a CopilotToolDef.

    Args:
        tool_info: The inspect_ai tool specification.

    Returns:
        A frozen ``CopilotToolDef`` with name, description, and JSON-Schema
        parameters.
    """
    return CopilotToolDef(
        name=tool_info.name,
        description=tool_info.description,
        parameters=tool_info.parameters.model_dump(exclude_none=True),
    )


def _sdk_response_to_model_output(
    response_data: object | None,
    model_name: str,
    usage: ModelUsage | None,
) -> ModelOutput:
    """Convert a Copilot SDK response to an inspect_ai ModelOutput.

    Accesses SDK response attributes via ``getattr``/``hasattr`` to avoid
    importing the Copilot SDK at module level.

    Args:
        response_data: An ``AssistantMessageData`` from the Copilot SDK,
            or ``None`` if no response was returned.
        model_name: Model identifier for the output.
        usage: Optional token usage information.

    Returns:
        A ``ModelOutput`` suitable for inspect_ai evaluation.
    """
    if response_data is None:
        return ModelOutput(
            model=model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content="", source="generate"),
                    stop_reason="unknown",
                ),
            ],
            usage=usage,
        )

    content: str = getattr(response_data, "content", "") or ""
    tool_requests: list[object] = getattr(response_data, "tool_requests", None) or []

    tool_calls: list[ToolCall] | None = None
    stop_reason: str = "stop"

    if tool_requests:
        stop_reason = "tool_calls"
        tool_calls = [_convert_tool_request(req) for req in tool_requests]

    return ModelOutput(
        model=model_name,
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content=content,
                    tool_calls=tool_calls,
                    source="generate",
                ),
                stop_reason=stop_reason,  # type: ignore[arg-type]
            ),
        ],
        usage=usage,
    )


def _convert_tool_request(req: object) -> ToolCall:
    """Convert a single SDK tool request to an inspect_ai ToolCall.

    Args:
        req: A tool request object with ``tool_call_id``, ``name``, and
            ``arguments`` attributes.

    Returns:
        An inspect_ai ``ToolCall``.
    """
    call_id: str = getattr(req, "tool_call_id", "") or ""
    function: str = getattr(req, "name", "") or ""
    raw_arguments = getattr(req, "arguments", {})

    arguments: dict[str, object]
    if isinstance(raw_arguments, str):
        arguments = json.loads(raw_arguments)
    elif isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        arguments = {}

    return ToolCall(
        id=call_id,
        function=function,
        arguments=arguments,
    )


def _extract_usage(events: list[object]) -> ModelUsage | None:
    """Extract token usage from Copilot SDK session events.

    Iterates over events looking for those whose ``.data`` has
    ``input_tokens`` / ``output_tokens`` attributes and sums them.

    Args:
        events: List of SDK ``SessionEvent`` objects.

    Returns:
        A ``ModelUsage`` with aggregated token counts, or ``None`` if no
        usage events were found.
    """
    input_tokens = 0
    output_tokens = 0
    found = False

    for event in events:
        data = getattr(event, "data", None)
        if data is not None and hasattr(data, "input_tokens"):
            found = True
            input_tokens += int(getattr(data, "input_tokens", 0) or 0)
            output_tokens += int(getattr(data, "output_tokens", 0) or 0)

    if not found:
        return None

    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


# ---------------------------------------------------------------------------
# CopilotModelAPI
# ---------------------------------------------------------------------------


class CopilotModelAPI(ModelAPI):
    """Copilot model API for GitHub Copilot-licensed inference.

    Uses the Python Copilot SDK (github-copilot-sdk) to route inference
    through GitHub's Copilot infrastructure. Creates a new session per
    ``generate()`` call; shares a single ``CopilotClient`` across all calls.
    """

    _client: ClassVar[object | None] = None
    _client_lock: ClassVar[asyncio.Lock | None] = None
    _client_refcount: ClassVar[int] = 0
    _github_token: ClassVar[str | None] = None

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig | None = None,
        **model_args: str,
    ) -> None:
        """Initialise the Copilot model API.

        Args:
            model_name: Model identifier, e.g. ``"gpt-4o"``.
            base_url: Unused — kept for ``ModelAPI`` compatibility.
            api_key: GitHub personal-access token.  Falls back to
                ``GITHUB_TOKEN`` env var, then ``gh`` CLI auth.
            config: Generation configuration.
            **model_args: Additional keyword arguments.  Recognised keys:
                ``timeout`` (seconds, default ``"120"``).

        Raises:
            RuntimeError: If no authentication source is available.
        """
        super().__init__(
            model_name, base_url, api_key, [], config or GenerateConfig()
        )
        self._timeout = int(model_args.get("timeout", "120"))

        token = api_key or os.environ.get("GITHUB_TOKEN")
        if not token:
            if not shutil.which("gh"):
                raise RuntimeError(
                    "Copilot model requires GITHUB_TOKEN env var, --api-key, "
                    "or 'gh' CLI. Install GitHub CLI (gh) or set GITHUB_TOKEN."
                )

        CopilotModelAPI._github_token = token
        CopilotModelAPI._client_refcount += 1

    @classmethod
    async def _get_or_create_client(cls) -> object:
        """Get or lazily create the shared ``CopilotClient`` singleton.

        Returns:
            The active ``CopilotClient`` instance.
        """
        if cls._client_lock is None:
            cls._client_lock = asyncio.Lock()

        async with cls._client_lock:
            if cls._client is not None:
                return cls._client

            from copilot import CopilotClient  # type: ignore[import-untyped]
            from copilot.client import SubprocessConfig  # type: ignore[import-untyped]

            subprocess_config = (
                SubprocessConfig(github_token=cls._github_token)
                if cls._github_token
                else SubprocessConfig()
            )
            client = CopilotClient(subprocess_config)
            cls._client = await client.__aenter__()
            return cls._client

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Generate a response using the Copilot SDK.

        Creates a new session per call, sends the prompt, extracts
        tool calls and usage, then cleans up the session.

        Args:
            input: Conversation history as chat messages.
            tools: Available tools for the model.
            tool_choice: Tool selection strategy.
            config: Generation configuration.

        Returns:
            A ``ModelOutput`` with the assistant response and usage data.
        """
        client = await self._get_or_create_client()

        from copilot.session import PermissionHandler  # type: ignore[import-untyped]

        sdk_tools = _convert_tools_for_sdk(tools) if tools else []

        session = await client.create_session(
            model=self.model_name,
            on_permission_request=PermissionHandler.approve_all,
            tools=sdk_tools,
        )

        try:
            prompt = _messages_to_prompt(input)

            try:
                response = await session.send_and_wait(
                    prompt, timeout=self._timeout
                )
            except TimeoutError:
                logger.warning(
                    "Copilot SDK send_and_wait timed out after %ds",
                    self._timeout,
                )
                response = None

            response_data = _extract_assistant_data(response)

            events = await session.get_messages()
            usage = _extract_usage(events)

            return _sdk_response_to_model_output(
                response_data, self.model_name, usage
            )
        finally:
            try:
                await client.delete_session(session.session_id)
            except Exception:
                logger.debug(
                    "Failed to delete Copilot session %s", session.session_id
                )

    async def aclose(self) -> None:
        """Shut down the shared client when the last instance closes."""
        CopilotModelAPI._client_refcount -= 1
        if (
            CopilotModelAPI._client_refcount <= 0
            and CopilotModelAPI._client is not None
        ):
            try:
                await CopilotModelAPI._client.__aexit__(None, None, None)
            except Exception:
                logger.debug("Error closing Copilot client", exc_info=True)
            finally:
                CopilotModelAPI._client = None
                CopilotModelAPI._client_refcount = 0


# ---------------------------------------------------------------------------
# SDK helper functions
# ---------------------------------------------------------------------------


def _convert_tools_for_sdk(tools: list[ToolInfo]) -> list[object]:
    """Convert inspect_ai ToolInfo list to Copilot SDK Tool objects.

    Requires the Copilot SDK to be installed.

    Args:
        tools: inspect_ai tool specifications.

    Returns:
        List of SDK ``Tool`` objects.
    """
    from copilot.tools import Tool as CopilotSdkTool  # type: ignore[import-untyped]

    async def _noop_handler(invocation: object) -> object:
        raise RuntimeError(
            "Tool handler should not be called — "
            "inspect_ai handles tool execution"
        )

    sdk_tools: list[object] = []
    for tool_info in tools:
        tool_def = _tool_info_to_sdk_tool(tool_info)
        sdk_tool = CopilotSdkTool(
            name=tool_def.name,
            description=tool_def.description,
            handler=_noop_handler,
            parameters=tool_def.parameters,
        )
        sdk_tools.append(sdk_tool)
    return sdk_tools


def _extract_assistant_data(response: object | None) -> object | None:
    """Extract ``AssistantMessageData`` from a ``SessionEvent``.

    Args:
        response: A ``SessionEvent`` or ``None``.

    Returns:
        The ``.data`` attribute of the event, or ``None``.
    """
    if response is None:
        return None
    return getattr(response, "data", response)
