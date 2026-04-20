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
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
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
_POOL_ACQUIRE_TIMEOUT = 120  # seconds to wait for a client from pool

# Sentinel used by _send_and_collect to signal that tool calls were captured
# from EXTERNAL_TOOL_REQUESTED events (not from ASSISTANT_MESSAGE).
_TOOL_CALLS_SENTINEL = object()


@dataclass(frozen=True)
class CopilotToolDef:
    """Lightweight tool definition for the Copilot SDK.

    Mirrors copilot.tools.Tool schema without requiring SDK import.
    Converted to SDK Tool at session creation time.
    """

    name: str
    description: str
    parameters: dict[str, object]


@dataclass
class _ConversationState:
    """Tracks an active multi-turn conversation with the Copilot SDK.

    Holds a session and its owning client so that subsequent ``generate()``
    calls for the same conversation can send only *delta* messages instead
    of re-serialising the entire history.
    """

    session: object  # CopilotSession
    client: object   # CopilotClient
    messages_sent: int = 0
    last_used: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------


def _conversation_key(messages: list[ChatMessage]) -> str:
    """Derive a stable key that is identical across all generate() calls
    for the same conversation.

    Searches for the first system and first user message by role (not by
    position) so the key stays constant as the message list grows with
    assistant/tool turns.

    Returns:
        16-char hex string (64 bits — collision-safe for ≤ thousands of
        concurrent conversations).
    """
    sys_text = ""
    user_text = ""
    for msg in messages:
        if msg.role == "system" and not sys_text:
            sys_text = msg.text[:500]
        elif msg.role == "user" and not user_text:
            user_text = msg.text[:500]
        if sys_text and user_text:
            break
    blob = f"system:{sys_text}|user:{user_text}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _format_tool_results(messages: list[ChatMessage]) -> str:
    """Format delta messages for a multi-turn continuation.

    Skips assistant messages (already in the SDK session history) and
    formats tool results for the model to consume as a user turn.

    Args:
        messages: The *delta* messages (``input[prev_count:]``).

    Returns:
        Formatted string to ``send()`` to the SDK session.
    """
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, ChatMessageAssistant):
            # The SDK already has this in its conversation history
            continue
        if isinstance(msg, ChatMessageTool):
            func = msg.function or "unknown"
            parts.append(f"[Tool Result: {func}]\n{msg.text}")
        else:
            # User or system message in the delta (rare but possible)
            parts.append(f"[{msg.role}]\n{msg.text}")
    return "\n\n".join(parts) if parts else ""


def _extract_system_content(messages: list[ChatMessage]) -> str | None:
    """Extract the system message content from the message list."""
    for msg in messages:
        if msg.role == "system":
            return msg.text
    return None


def _extract_first_user_content(messages: list[ChatMessage]) -> str:
    """Extract the first user message content from the message list."""
    for msg in messages:
        if msg.role == "user":
            return msg.text
    return ""


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
    through GitHub's Copilot infrastructure.  Maintains *persistent sessions*
    across ``generate()`` calls for the same conversation so the SDK server
    preserves multi-turn history.  A pool of ``CopilotClient`` instances
    (one subprocess each) provides parallel inference capacity.
    """

    _pool: ClassVar[asyncio.Queue[object] | None] = None
    _pool_lock: ClassVar[asyncio.Lock] = asyncio.Lock()  # Eager init — no race
    _pool_size: ClassVar[int] = 8
    _all_clients: ClassVar[list[object]] = []
    _instance_count: ClassVar[int] = 0
    _github_token: ClassVar[str | None] = None
    _shutting_down: ClassVar[bool] = False
    _conversations: ClassVar[dict[str, _ConversationState]] = {}

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
                ``timeout`` (seconds, default ``"300"``),
                ``pool_size`` (number of SDK subprocesses, default ``"4"``).

        Raises:
            RuntimeError: If no authentication source is available.
        """
        super().__init__(
            model_name, base_url, api_key, [], config or GenerateConfig()
        )
        self._timeout = int(model_args.get("timeout", "300"))
        self._pool_size_local = int(model_args.get("pool_size", "8"))
        CopilotModelAPI._pool_size = self._pool_size_local

        token = api_key or os.environ.get("GITHUB_TOKEN")
        if not token:
            if not shutil.which("gh"):
                raise RuntimeError(
                    "Copilot model requires GITHUB_TOKEN env var, --api-key, "
                    "or 'gh' CLI. Install GitHub CLI (gh) or set GITHUB_TOKEN."
                )

        CopilotModelAPI._github_token = token
        CopilotModelAPI._instance_count += 1

    @classmethod
    async def _create_single_client(cls) -> object:
        """Create and start a single CopilotClient instance."""
        from copilot import CopilotClient  # type: ignore[import-untyped]
        from copilot.client import SubprocessConfig  # type: ignore[import-untyped]

        subprocess_config = (
            SubprocessConfig(github_token=cls._github_token)
            if cls._github_token
            else SubprocessConfig()
        )
        client = CopilotClient(subprocess_config)
        await client.start()
        return client

    @classmethod
    async def _init_pool(cls) -> None:
        """Lazily create the client pool (idempotent under lock)."""
        if cls._pool is not None:
            return
        async with cls._pool_lock:
            if cls._pool is not None:
                return  # Another coroutine already initialized
            pool: asyncio.Queue[object] = asyncio.Queue(maxsize=cls._pool_size)
            clients: list[object] = []
            try:
                for i in range(cls._pool_size):
                    client = await cls._create_single_client()
                    clients.append(client)
                    pool.put_nowait(client)
                    logger.info("Copilot pool: client %d/%d started", i + 1, cls._pool_size)
            except Exception:
                # Stop any clients already started to avoid subprocess leaks
                for c in clients:
                    try:
                        await asyncio.wait_for(c.stop(), timeout=_CLIENT_STOP_TIMEOUT)  # type: ignore[union-attr]
                    except Exception:
                        pass
                raise
            cls._all_clients = clients
            cls._pool = pool
            cls._shutting_down = False

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Generate a response, reusing SDK sessions for multi-turn conversations.

        On the first call for a conversation, creates a new session with the
        system prompt set via ``system_message`` and sends the first user
        message.  On subsequent calls (same conversation key), sends only the
        *delta* messages (tool results) so the SDK server preserves full
        multi-turn history.

        Args:
            input: Conversation history as chat messages.
            tools: Available tools for the model (definitions only).
            tool_choice: Tool selection strategy.
            config: Generation configuration.

        Returns:
            A ``ModelOutput`` with the assistant response and usage data.
        """
        t_start = time.monotonic()
        await self._init_pool()

        conv_key = _conversation_key(input)
        cls = type(self)
        state = cls._conversations.get(conv_key)

        if state:
            # --- Continue existing conversation ---
            try:
                return await self._continue_conversation(
                    state, input, tools, config, t_start,
                )
            except (BrokenPipeError, ConnectionResetError, EOFError) as exc:
                logger.warning(
                    "Copilot session died during continuation (%s) — "
                    "evicting and starting fresh",
                    type(exc).__name__,
                )
                await self._evict_conversation(conv_key, replace_client=True)
                # Fall through to start a new conversation
            except Exception as exc:
                logger.warning(
                    "Copilot continuation failed (%s: %s) — "
                    "evicting and starting fresh",
                    type(exc).__name__,
                    exc,
                )
                await self._evict_conversation(conv_key, replace_client=False)
                # Fall through to start a new conversation

        # --- Start new conversation ---
        # If pool is empty, evict the oldest cached conversation to free a client
        if self._pool is not None and self._pool.empty() and cls._conversations:
            await self._evict_oldest_conversation()

        try:
            client = await asyncio.wait_for(
                self._pool.get(),  # type: ignore[union-attr]
                timeout=_POOL_ACQUIRE_TIMEOUT,
            )
        except TimeoutError:
            return _empty_output(self.model_name)

        try:
            return await self._start_conversation(
                conv_key, client, input, tools, config, t_start,
            )
        except (BrokenPipeError, ConnectionResetError, EOFError) as exc:
            logger.warning(
                "Copilot client subprocess died (%s) — discarding and replacing",
                type(exc).__name__,
            )
            await self._return_or_replace_client(client, failed=True)
            return _empty_output(self.model_name)
        except BaseException:
            # Any other error (including CancelledError): return client to pool
            await self._return_or_replace_client(client, failed=False)
            raise

    @classmethod
    async def _return_or_replace_client(cls, client: object, failed: bool) -> None:
        """Return a healthy client to pool, or replace a dead one."""
        if cls._shutting_down:
            # Pool is closing — stop this client instead of returning it
            try:
                await asyncio.wait_for(client.stop(), timeout=_CLIENT_STOP_TIMEOUT)  # type: ignore[union-attr]
            except Exception:
                logger.debug("Failed to stop client during shutdown", exc_info=True)
            return

        if not failed:
            await cls._pool.put(client)  # type: ignore[union-attr]
            return

        # Client died — remove from tracking, try to create replacement
        if client in cls._all_clients:
            cls._all_clients.remove(client)
        try:
            replacement = await cls._create_single_client()
            cls._all_clients.append(replacement)
            await cls._pool.put(replacement)  # type: ignore[union-attr]
            logger.info("Copilot pool: replaced dead client (pool size maintained)")
        except Exception:
            logger.warning(
                "Copilot pool: failed to create replacement client (pool size reduced to %d)",
                len(cls._all_clients),
            )

    async def _start_conversation(
        self,
        conv_key: str,
        client: object,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        config: GenerateConfig,
        t_start: float,
    ) -> ModelOutput:
        """Create a new SDK session and send the first prompt.

        The client is removed from the pool for the lifetime of the
        conversation and cached in ``_conversations``.
        """
        t_client = time.monotonic()
        logger.info(
            "Copilot generate [new]: client ready (%.1fs), conv=%s",
            t_client - t_start,
            conv_key[:8],
        )

        from copilot import PermissionHandler  # type: ignore[import-untyped]

        sdk_tools = _convert_tools_for_sdk(tools) if tools else []

        # Use system_message="replace" so the SDK's system prompt is exactly
        # the conversation's system content (no SDK guardrails mixed in).
        sys_content = _extract_system_content(input)
        sys_config: dict[str, object] | None = None
        if sys_content:
            sys_config = {"mode": "replace", "content": sys_content}

        # Whitelist ONLY the custom tools so the model receives their
        # definitions in the API request.  available_tools=[] would tell
        # the CLI to send *zero* tool definitions, forcing the model to
        # fake tool calls as garbled text.
        tool_names: list[str] = [t.name for t in sdk_tools]  # type: ignore[union-attr]

        try:
            session = await asyncio.wait_for(
                client.create_session(  # type: ignore[union-attr]
                    model=self.model_name,
                    on_permission_request=PermissionHandler.approve_all,
                    system_message=sys_config,
                    tools=sdk_tools,
                    available_tools=tool_names if tool_names else None,
                ),
                timeout=60,
            )
        except Exception as exc:
            logger.warning("Copilot create_session failed: %s", exc)
            # Return client to pool since we couldn't create a session
            await self._return_or_replace_client(client, failed=False)
            return _empty_output(self.model_name)

        t_session = time.monotonic()
        logger.info(
            "Copilot generate [new]: session created (%.1fs), model=%s, tools=%d",
            t_session - t_client,
            self.model_name,
            len(sdk_tools),
        )

        # Determine the prompt to send.  On a true first turn the input
        # only has system + user messages — send just the user content
        # (system is set via system_message config).  When starting a
        # fresh session for a continuation (e.g. after evicting a session
        # that had tool_calls), serialize the full history so the model
        # sees prior tool calls and results.
        has_tool_history = any(
            isinstance(m, (ChatMessageAssistant, ChatMessageTool))
            and not (isinstance(m, ChatMessageAssistant) and m == input[0])
            for m in input
        )
        if has_tool_history:
            # Full text serialization — skip system (already in system_message)
            non_system = [m for m in input if m.role != "system"]
            prompt = _messages_to_prompt(non_system)
            logger.info(
                "Copilot generate [new]: full-history prompt length=%d chars "
                "(continuation via fresh session)",
                len(prompt),
            )
        else:
            prompt = _extract_first_user_content(input)
            logger.info(
                "Copilot generate [new]: first-turn prompt length=%d chars",
                len(prompt),
            )

        result = await self._send_and_collect(
            session, client, prompt, t_start, t_session, conv_key,
        )

        # If the response contains tool_requests the SDK agent-loop is now
        # processing noop handler results and the session is in a transient
        # state.  Evict it so the next generate() for this conversation
        # starts a fresh session with the full text history.
        has_tool_calls = (
            result.choices
            and result.choices[0].message.tool_calls
        )
        cls = type(self)
        if has_tool_calls:
            logger.info(
                "Copilot generate [new]: response has tool_calls — "
                "evicting session to avoid stale agent-loop state, conv=%s",
                conv_key[:8],
            )
            # Return client to pool (session will be deleted)
            try:
                await asyncio.wait_for(
                    asyncio.shield(
                        client.delete_session(  # type: ignore[union-attr]
                            session.session_id  # type: ignore[union-attr]
                        )
                    ),
                    timeout=_SESSION_DELETE_TIMEOUT,
                )
            except (asyncio.CancelledError, Exception):
                logger.debug("Failed to delete session after tool_calls")
            await self._return_or_replace_client(client, failed=False)
        else:
            # Cache conversation state — client stays out of pool
            cls._conversations[conv_key] = _ConversationState(
                session=session,
                client=client,
                messages_sent=len(input),
                last_used=time.monotonic(),
            )
        return result

    async def _continue_conversation(
        self,
        state: _ConversationState,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        config: GenerateConfig,
        t_start: float,
    ) -> ModelOutput:
        """Send delta messages to an existing SDK session."""
        delta = input[state.messages_sent:]
        if not delta:
            logger.warning(
                "Copilot generate [continue]: no new messages (sent=%d, total=%d)",
                state.messages_sent,
                len(input),
            )
            return _empty_output(self.model_name)

        prompt = _format_tool_results(delta)
        if not prompt:
            logger.warning("Copilot generate [continue]: delta produced empty prompt")
            return _empty_output(self.model_name)

        logger.info(
            "Copilot generate [continue]: delta=%d msgs, prompt=%d chars, "
            "prev_sent=%d, total=%d",
            len(delta),
            len(prompt),
            state.messages_sent,
            len(input),
        )

        t_session = time.monotonic()
        result = await self._send_and_collect(
            state.session, state.client, prompt, t_start, t_session,
            _conversation_key(input),
        )

        # Update state
        state.messages_sent = len(input)
        state.last_used = time.monotonic()
        return result

    async def _send_and_collect(
        self,
        session: object,
        client: object,
        prompt: str,
        t_start: float,
        t_session: float,
        conv_key: str,
    ) -> ModelOutput:
        """Send a prompt and wait for the model response.

        Handles two response patterns:

        * **Text-only** (no tool calls): ``ASSISTANT_MESSAGE`` fires after the
          agent loop completes → captured directly.
        * **Tool calls**: ``EXTERNAL_TOOL_REQUESTED`` fires for each tool
          call *before* execution.  Since ``handler=None``, the CLI blocks
          waiting for a ``handle_pending_tool_call`` RPC that never comes.
          We collect the tool-call events and, after a short debounce,
          synthesise the ``ModelOutput`` ourselves.

        Shared by both ``_start_conversation`` and ``_continue_conversation``.
        """
        from copilot.generated.session_events import (  # type: ignore[import-untyped]
            SessionEventType,
        )

        loop = asyncio.get_running_loop()
        first_response: asyncio.Future[object | None] = loop.create_future()

        # Collect EXTERNAL_TOOL_REQUESTED events (tool calls before execution)
        pending_tool_calls: list[ToolCall] = []
        _debounce_handle: asyncio.TimerHandle | None = None

        def _resolve_with_tool_calls() -> None:
            """Resolve the future with collected tool calls."""
            if first_response.done():
                return
            # Synthesise a minimal response with tool_calls
            first_response.set_result(
                _TOOL_CALLS_SENTINEL  # Special marker — see below
            )

        def _on_event(event: object) -> None:
            nonlocal _debounce_handle
            if first_response.done():
                return
            event_type = getattr(event, "type", None)
            data = getattr(event, "data", None)

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

            if event_type == SessionEventType.EXTERNAL_TOOL_REQUESTED:
                # Capture tool call details from the event
                tool_name = getattr(data, "tool_name", None) or ""
                tool_call_id = getattr(data, "tool_call_id", None) or ""
                arguments = getattr(data, "arguments", None)
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (json.JSONDecodeError, TypeError):
                        arguments = {}
                elif not isinstance(arguments, dict):
                    arguments = arguments if arguments else {}
                pending_tool_calls.append(
                    ToolCall(
                        id=tool_call_id,
                        function=tool_name,
                        arguments=arguments,
                    )
                )
                logger.debug(
                    "Copilot EXTERNAL_TOOL_REQUESTED: %s (id=%s)",
                    tool_name,
                    tool_call_id,
                )
                # Debounce: resolve 500ms after the last tool-call event
                if _debounce_handle is not None:
                    _debounce_handle.cancel()
                _debounce_handle = loop.call_later(
                    0.5, _resolve_with_tool_calls
                )
            elif event_type == SessionEventType.ASSISTANT_MESSAGE:
                if not first_response.done():
                    first_response.set_result(data)
            elif event_type == SessionEventType.SESSION_ERROR:
                msg = getattr(data, "message", None) or str(data)
                if not first_response.done():
                    first_response.set_exception(
                        RuntimeError(f"Copilot session error: {msg}")
                    )
            elif event_type == SessionEventType.SESSION_IDLE:
                if not first_response.done():
                    first_response.set_result(None)

        unsubscribe = session.on(_on_event)  # type: ignore[union-attr]
        try:
            t_send = time.monotonic()
            await session.send(prompt)  # type: ignore[union-attr]
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
            except TimeoutError:
                logger.warning(
                    "Copilot SDK response timed out after %ds", self._timeout
                )
                response_data = None
            except RuntimeError as exc:
                logger.warning(
                    "Copilot session error: %s — returning empty response", exc
                )
                response_data = None
        finally:
            if _debounce_handle is not None:
                _debounce_handle.cancel()
            unsubscribe()

        t_response = time.monotonic()

        # Handle tool-call sentinel: build ModelOutput from collected events
        if response_data is _TOOL_CALLS_SENTINEL and pending_tool_calls:
            logger.info(
                "Copilot generate: captured %d tool_calls via "
                "EXTERNAL_TOOL_REQUESTED (%.1fs), conv=%s",
                len(pending_tool_calls),
                t_response - t_send,
                conv_key[:8],
            )
            return ModelOutput(
                model=self.model_name,
                choices=[
                    ChatCompletionChoice(
                        message=ChatMessageAssistant(
                            content="",
                            tool_calls=pending_tool_calls,
                            source="generate",
                        ),
                        stop_reason="tool_calls",
                    ),
                ],
                usage=None,
            )

        response_content = (
            getattr(response_data, "content", None) if response_data else None
        )
        response_trs = (
            getattr(response_data, "tool_requests", None) if response_data else None
        )
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
            events = await session.get_messages()  # type: ignore[union-attr]
            usage = _extract_usage(events)
            del events
        except (asyncio.CancelledError, Exception):
            logger.debug("Failed to retrieve usage from session", exc_info=True)

        result = _sdk_response_to_model_output(
            response_data, self.model_name, usage
        )
        t_done = time.monotonic()
        logger.info(
            "Copilot generate: total=%.1fs (session=%.1fs, send=%.1fs, "
            "response=%.1fs, finalize=%.1fs), conv=%s, usage=%s",
            t_done - t_start,
            t_session - t_start,
            t_send - t_session,
            t_response - t_send,
            t_done - t_response,
            conv_key[:8],
            usage,
        )
        return result

    @classmethod
    async def _evict_conversation(
        cls, conv_key: str, *, replace_client: bool
    ) -> None:
        """Remove a cached conversation, cleaning up session and client."""
        state = cls._conversations.pop(conv_key, None)
        if not state:
            return

        # Try to delete the session
        try:
            await asyncio.wait_for(
                asyncio.shield(
                    state.client.delete_session(  # type: ignore[union-attr]
                        state.session.session_id  # type: ignore[union-attr]
                    )
                ),
                timeout=_SESSION_DELETE_TIMEOUT,
            )
        except (asyncio.CancelledError, Exception):
            logger.debug("Failed to delete conversation session %s", conv_key[:8])

        # Return or replace the client
        await cls._return_or_replace_client(state.client, failed=replace_client)

    @classmethod
    async def _evict_oldest_conversation(cls) -> None:
        """Evict the least-recently-used conversation to free a client."""
        if not cls._conversations:
            return
        oldest_key = min(
            cls._conversations,
            key=lambda k: cls._conversations[k].last_used,
        )
        logger.info(
            "Copilot pool: evicting oldest conversation %s to free client",
            oldest_key[:8],
        )
        await cls._evict_conversation(oldest_key, replace_client=False)

    async def aclose(self) -> None:
        """Shut down the pool when the last instance closes."""
        CopilotModelAPI._instance_count -= 1
        if CopilotModelAPI._instance_count > 0:
            return

        CopilotModelAPI._shutting_down = True

        # Clean up all cached conversations (returns clients to pool)
        for conv_key in list(CopilotModelAPI._conversations):
            await self._evict_conversation(conv_key, replace_client=False)

        # Drain any clients in the queue
        if CopilotModelAPI._pool is not None:
            while not CopilotModelAPI._pool.empty():
                try:
                    CopilotModelAPI._pool.get_nowait()
                except asyncio.QueueEmpty:
                    break

        # Stop all tracked clients
        for client in CopilotModelAPI._all_clients:
            try:
                await asyncio.wait_for(
                    client.stop(), timeout=_CLIENT_STOP_TIMEOUT
                )
            except TimeoutError:
                logger.warning("Copilot pool: client stop timed out — force-killing")
                try:
                    await client.force_stop()
                except Exception:
                    pass
            except Exception:
                logger.warning("Error stopping Copilot client", exc_info=True)

        # Reset state
        CopilotModelAPI._pool = None
        CopilotModelAPI._all_clients = []
        CopilotModelAPI._conversations = {}
        CopilotModelAPI._instance_count = 0
        CopilotModelAPI._shutting_down = False


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
    """Return a benign success result — tool execution is handled externally.

    The SDK is used only for inference.  If the SDK somehow dispatches a
    tool execution event despite handler=None in the Tool definition, we
    return a success result so the SDK does NOT feed an error back to the
    model (which would cause retry loops and model confusion).
    """
    from copilot.tools import ToolResult  # type: ignore[import-untyped]

    return ToolResult(
        text_result_for_llm="OK",
        result_type="success",
        error=None,
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
        # handler=None prevents the SDK from executing tools internally.
        # Tool calls are captured via EXTERNAL_TOOL_REQUESTED events in
        # _send_and_collect() and returned as structured ToolCall objects
        # to the bridge/caller for external execution.
        sdk_tool = CopilotSdkTool(
            name=tool_def.name,
            description=tool_def.description,
            handler=None,  # type: ignore[arg-type]
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
