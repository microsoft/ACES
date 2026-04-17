"""Copilot model provider for inspect_ai.

Pure conversion functions and CopilotModelAPI class for routing inference
through GitHub's Copilot infrastructure via the Python Copilot SDK.
SDK objects are accessed via ``getattr``/``hasattr`` to avoid hard imports.

IMPORTANT: The SDK is used as a PURE INFERENCE ENGINE. Built-in tools
(bash, file search, etc.) are disabled via ``available_tools=[]`` to
prevent the SDK CLI from executing tools on the host machine.  The
``send()`` + first-response pattern captures the model's first
``AssistantMessageData`` and returns immediately — before the SDK's
agent loop can execute any tools.
"""

from __future__ import annotations

import asyncio
import faulthandler
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import ClassVar

# Enable faulthandler for debugging — prints thread tracebacks on SIGUSR1
faulthandler.enable()
if hasattr(faulthandler, "register"):
    import signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)

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
from inspect_ai.model._registry import modelapi
from inspect_ai.tool import ToolCall, ToolChoice, ToolInfo

from saber.logging import get_logger

logger = get_logger(__name__)

_SESSION_DELETE_TIMEOUT = 10  # seconds to wait for session cleanup
_CLIENT_STOP_TIMEOUT = 15  # seconds to wait for SDK subprocess shutdown


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
                ``timeout`` (seconds, default ``"300"``).

        Raises:
            RuntimeError: If no authentication source is available.
        """
        super().__init__(
            model_name, base_url, api_key, [], config or GenerateConfig()
        )
        self._timeout = int(model_args.get("timeout", "300"))

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
            await client.start()
            cls._client = client
            return cls._client

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Generate a single-turn response using the Copilot SDK.

        Creates a new session per call with all built-in tools disabled
        (``available_tools=[]``).  Uses ``send()`` + event listener to
        capture the first ``AssistantMessageData`` and return immediately,
        preventing the SDK's agent loop from executing tools on the host.

        Args:
            input: Conversation history as chat messages.
            tools: Available tools for the model (passed as definitions
                so the model can propose tool calls, but never executed).
            tool_choice: Tool selection strategy.
            config: Generation configuration.

        Returns:
            A ``ModelOutput`` with the assistant response and usage data.
        """
        t_start = time.monotonic()
        return await self._generate_impl(input, tools, tool_choice, config, t_start)

    async def _generate_impl(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
        t_start: float,
    ) -> ModelOutput:
        """Internal generate implementation."""
        client = await self._get_or_create_client()
        t_client = time.monotonic()
        logger.info(
            "Copilot generate: client ready (%.1fs)",
            t_client - t_start,
        )

        from copilot import PermissionHandler  # type: ignore[import-untyped]
        from copilot.generated.session_events import (  # type: ignore[import-untyped]
            SessionEventType,
        )

        # Register tools so the model can see definitions and propose
        # tool_calls, but use _noop_tool_handler to prevent execution.
        sdk_tools = _convert_tools_for_sdk(tools) if tools else []

        try:
            session = await asyncio.wait_for(
                client.create_session(
                    model=self.model_name,
                    on_permission_request=PermissionHandler.approve_all,
                    tools=sdk_tools,
                    available_tools=[],  # Disable ALL built-in tools
                ),
                timeout=60,  # Session creation should be fast
            )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("Copilot create_session failed: %s", exc)
            return _empty_output(self.model_name)
        t_session = time.monotonic()
        logger.info(
            "Copilot generate: session created (%.1fs), model=%s, tools=%d",
            t_session - t_client,
            self.model_name,
            len(sdk_tools),
        )

        try:
            prompt = _messages_to_prompt(input)
            logger.info(
                "Copilot generate: prompt length=%d chars",
                len(prompt),
            )

            # Use send() + event listener to capture the FIRST assistant
            # message, then return immediately — before the SDK can execute
            # any tools.  This makes the SDK a pure inference engine.
            loop = asyncio.get_event_loop()
            first_response: asyncio.Future[object | None] = loop.create_future()

            def _on_event(event: object) -> None:
                if first_response.done():
                    return
                event_type = getattr(event, "type", None)
                data = getattr(event, "data", None)

                # Debug: log events with content
                if data is not None:
                    c = getattr(data, "content", None)
                    tr = getattr(data, "tool_requests", None)
                    if c or tr:
                        logger.debug(
                            "Copilot event: type=%s, content_len=%d, tool_requests=%d",
                            event_type,
                            len(c) if c else 0,
                            len(tr) if tr else 0,
                        )

                if event_type == SessionEventType.ASSISTANT_MESSAGE:
                    if not first_response.done():
                        first_response.set_result(data)
                elif event_type == SessionEventType.SESSION_ERROR:
                    msg = getattr(data, "message", None) or str(data)
                    if not first_response.done():
                        first_response.set_exception(RuntimeError(f"Copilot session error: {msg}"))
                elif event_type == SessionEventType.SESSION_IDLE:
                    if not first_response.done():
                        first_response.set_result(None)

            unsubscribe = session.on(_on_event)
            try:
                t_send = time.monotonic()
                await session.send(prompt)
                logger.info(
                    "Copilot generate: prompt sent (%.1fs), waiting for response...",
                    t_send - t_session,
                )

                try:
                    response_data = await asyncio.wait_for(
                        first_response, timeout=self._timeout
                    )
                except asyncio.CancelledError:
                    logger.warning(
                        "Copilot generate cancelled (likely time-limit exceeded)"
                    )
                    response_data = None
                    # Do NOT re-raise — return empty output so the eval can
                    # proceed to scoring instead of leaving the event loop dead.
                except asyncio.TimeoutError:
                    logger.warning(
                        "Copilot SDK response timed out after %ds",
                        self._timeout,
                    )
                    response_data = None
                except RuntimeError as exc:
                    # SESSION_ERROR events set_exception with RuntimeError.
                    logger.warning(
                        "Copilot session error: %s — returning empty response",
                        exc,
                    )
                    response_data = None
            finally:
                unsubscribe()

            t_response = time.monotonic()
            response_content = getattr(response_data, "content", None) if response_data else None
            response_trs = getattr(response_data, "tool_requests", None) if response_data else None
            logger.info(
                "Copilot generate: response received (%.1fs), "
                "content_len=%s, tool_requests=%s, data_type=%s",
                t_response - t_send,
                len(response_content) if response_content else 0,
                len(response_trs) if response_trs else 0,
                type(response_data).__name__ if response_data else "None",
            )

            usage: ModelUsage | None = None
            try:
                events = await session.get_messages()
                usage = _extract_usage(events)
                del events
            except (asyncio.CancelledError, Exception):
                logger.debug("Failed to retrieve usage from session", exc_info=True)

            result = _sdk_response_to_model_output(
                response_data, self.model_name, usage
            )
            t_done = time.monotonic()
            logger.info(
                "Copilot generate: total=%.1fs (client=%.1fs, session=%.1fs, "
                "send=%.1fs, response=%.1fs, finalize=%.1fs), "
                "usage=%s",
                t_done - t_start,
                t_client - t_start,
                t_session - t_client,
                t_send - t_session,
                t_response - t_send,
                t_done - t_response,
                usage,
            )
            return result
        finally:
            try:
                # Use shield to prevent nested cancellation from creating
                # orphaned timer handles in the event loop
                await asyncio.wait_for(
                    asyncio.shield(client.delete_session(session.session_id)),
                    timeout=_SESSION_DELETE_TIMEOUT,
                )
            except (asyncio.CancelledError, Exception):
                logger.debug(
                    "Failed to delete Copilot session %s",
                    session.session_id,
                    exc_info=True,
                )

    async def aclose(self) -> None:
        """Shut down the shared client when the last instance closes."""
        CopilotModelAPI._client_refcount -= 1
        if (
            CopilotModelAPI._client_refcount <= 0
            and CopilotModelAPI._client is not None
        ):
            try:
                await asyncio.wait_for(
                    CopilotModelAPI._client.stop(),
                    timeout=_CLIENT_STOP_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Copilot client stop timed out after %ds — force-killing",
                    _CLIENT_STOP_TIMEOUT,
                )
                try:
                    await CopilotModelAPI._client.force_stop()
                except Exception:
                    pass
            except Exception:
                logger.warning("Error closing Copilot client", exc_info=True)
            finally:
                CopilotModelAPI._client = None
                CopilotModelAPI._client_refcount = 0


# ---------------------------------------------------------------------------
# SDK helper functions
# ---------------------------------------------------------------------------


def _empty_output(model_name: str) -> ModelOutput:
    """Return a minimal empty ModelOutput for error paths."""
    return ModelOutput(
        model=model_name,
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(content="", source="generate"),
                stop_reason="unknown",
            ),
        ],
    )


async def _noop_tool_handler(invocation: object) -> object:
    """Return an error result — tool execution is forbidden in pure-inference mode.

    The SDK is used only for inference.  If the CLI somehow attempts to
    execute a tool, we return a descriptive error rather than raising,
    so the CLI can feed the error back to the model gracefully.
    """
    from copilot.tools import ToolResult  # type: ignore[import-untyped]

    return ToolResult(
        text_result_for_llm=(
            "Tool execution is disabled in pure-inference mode. "
            "Only the caller (e.g. Hyenas CLI) may execute tools."
        ),
        result_type="failure",
        error="tool execution disabled",
        tool_telemetry={},
    )


def _convert_tools_for_sdk(tools: list[ToolInfo]) -> list[object]:
    """Convert inspect_ai ToolInfo list to Copilot SDK Tool objects.

    Requires the Copilot SDK to be installed.

    Args:
        tools: inspect_ai tool specifications.

    Returns:
        List of SDK ``Tool`` objects.
    """
    from copilot.tools import Tool as CopilotSdkTool  # type: ignore[import-untyped]

    sdk_tools: list[object] = []
    for tool_info in tools:
        tool_def = _tool_info_to_sdk_tool(tool_info)
        sdk_tool = CopilotSdkTool(
            name=tool_def.name,
            description=tool_def.description,
            handler=_noop_tool_handler,
            parameters=tool_def.parameters,
            overrides_built_in_tool=True,
        )
        sdk_tools.append(sdk_tool)
    return sdk_tools


@modelapi(name="copilot")  # type: ignore[misc]
def copilot() -> type[ModelAPI]:
    """Register the Copilot model API provider.

    Requires the ``github-copilot-sdk`` package. Raises a descriptive
    error if the SDK is not installed.
    """
    try:
        import copilot as _copilot  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'copilot' model provider requires github-copilot-sdk. "
            "Install with: pip install github-copilot-sdk"
        ) from None
    return CopilotModelAPI
