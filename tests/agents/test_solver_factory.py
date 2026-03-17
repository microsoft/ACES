"""Tests for create_saber_solver factory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.agents.models import AgentCapabilities
from saber.agents.solver_factory import (
    AGENT_CAPABILITIES,
    TOOL_CALL_LIMIT_MESSAGE,
    create_saber_solver,
)
from saber.tools.registry import ToolRegistry


class TestCreateSaberSolver:
    """create_saber_solver produces a callable solver."""

    def test_returns_callable(self) -> None:
        """create_saber_solver returns a callable (Solver)."""

        def mock_factory(**kwargs: object) -> object:
            """Two-level factory: factory() → create_with_prompts."""

            def create_with_prompts(**prompt_kwargs: object) -> object:
                return lambda state, generate: state

            return create_with_prompts

        result = create_saber_solver(
            agent_name="test_agent",
            agent_factory=mock_factory,
        )
        assert callable(result)

    def test_no_skills_dir_or_agent_persona_params(self) -> None:
        """create_saber_solver no longer accepts skills_dir / agent_persona as named params."""
        import inspect

        sig = inspect.signature(create_saber_solver)
        param_names = set(sig.parameters.keys())
        assert "skills_dir" not in param_names
        assert "agent_persona" not in param_names

    def test_kwargs_forwarded_to_factory(self) -> None:
        """create_saber_solver forwards **kwargs to agent_factory."""
        import inspect

        sig = inspect.signature(create_saber_solver)
        param_names = set(sig.parameters.keys())
        # kwargs should be accepted
        assert "kwargs" in param_names

    def test_accepts_tool_registry_param(self) -> None:
        """create_saber_solver accepts tool_registry instead of resolved_tools."""
        import inspect

        sig = inspect.signature(create_saber_solver)
        param_names = set(sig.parameters.keys())
        assert "tool_registry" in param_names
        assert "resolved_tools" not in param_names


class TestSaberSolverBehavior:
    """Behavioral tests for the solver returned by create_saber_solver."""

    @pytest.mark.asyncio
    async def test_solve_extracts_prompts_from_metadata(self) -> None:
        """Verify prompts are extracted from state.metadata and forwarded."""
        mock_inner_solver = AsyncMock(return_value=MagicMock())
        mock_create_with_prompts = MagicMock(return_value=mock_inner_solver)
        mock_factory = MagicMock(return_value=mock_create_with_prompts)

        solver = create_saber_solver(agent_name="test", agent_factory=mock_factory)

        state = MagicMock()
        state.metadata = {
            "instruction_prompt": "Do X",
            "assistant_prompt": "You are Y",
            "max_steps": 200,
        }
        generate = MagicMock()

        await solver(state, generate)

        # Factory was called (level 1)
        mock_factory.assert_called_once()
        # create_with_prompts was called (level 2) with correct kwargs
        mock_create_with_prompts.assert_called_once()
        call_kwargs = mock_create_with_prompts.call_args[1]
        assert call_kwargs["instruction_prompt"] == "Do X"
        assert call_kwargs["assistant_prompt"] == "You are Y"

    @pytest.mark.asyncio
    async def test_solve_calls_inner_solver(self) -> None:
        """The inner agent solver is invoked with state and generate."""
        sentinel_state = MagicMock(name="returned_state")
        mock_inner_solver = AsyncMock(return_value=sentinel_state)
        mock_create_with_prompts = MagicMock(return_value=mock_inner_solver)
        mock_factory = MagicMock(return_value=mock_create_with_prompts)

        solver = create_saber_solver(agent_name="test", agent_factory=mock_factory)

        state = MagicMock()
        state.metadata = {"max_steps": 200}
        generate = MagicMock()

        result = await solver(state, generate)

        mock_inner_solver.assert_awaited_once_with(state, generate)
        assert result is sentinel_state

    @pytest.mark.asyncio
    async def test_solve_handles_none_metadata_gracefully(self) -> None:
        """When metadata values are None, prompts default to empty strings."""
        mock_inner_solver = AsyncMock(return_value=MagicMock())
        mock_create_with_prompts = MagicMock(return_value=mock_inner_solver)
        mock_factory = MagicMock(return_value=mock_create_with_prompts)

        solver = create_saber_solver(agent_name="test", agent_factory=mock_factory)

        state = MagicMock()
        state.metadata = {
            "instruction_prompt": None,
            "assistant_prompt": None,
            "max_steps": 200,
        }
        generate = MagicMock()

        await solver(state, generate)

        call_kwargs = mock_create_with_prompts.call_args[1]
        # None values should become "" (not "None")
        assert call_kwargs["instruction_prompt"] == ""
        assert call_kwargs["assistant_prompt"] == ""

    @pytest.mark.asyncio
    async def test_solve_missing_metadata_defaults_empty(self) -> None:
        """When metadata has max_steps but no prompts, prompts default to empty strings."""
        mock_inner_solver = AsyncMock(return_value=MagicMock())
        mock_create_with_prompts = MagicMock(return_value=mock_inner_solver)
        mock_factory = MagicMock(return_value=mock_create_with_prompts)

        solver = create_saber_solver(agent_name="test", agent_factory=mock_factory)

        state = MagicMock()
        state.metadata = {"max_steps": 200}
        generate = MagicMock()

        await solver(state, generate)

        call_kwargs = mock_create_with_prompts.call_args[1]
        assert call_kwargs["instruction_prompt"] == ""
        assert call_kwargs["assistant_prompt"] == ""


class TestKwargsPassthrough:
    """Verify **kwargs passthrough from create_saber_solver to agent_factory."""

    @pytest.mark.asyncio
    async def test_kwargs_reach_agent_factory(self) -> None:
        """Verify persona_file and skills_dir kwargs are forwarded to agent factory."""
        captured_kwargs: dict[str, object] = {}

        def mock_factory(**kw: object) -> object:
            captured_kwargs.update(kw)

            def create_with_prompts(**prompt_kwargs: object) -> object:
                return AsyncMock(return_value=MagicMock())

            return create_with_prompts

        solver = create_saber_solver(
            agent_name="test",
            agent_factory=mock_factory,
            persona_file="/path/to/agent.md",
            skills_dir="/path/to/skills",
        )

        state = MagicMock()
        state.metadata = {"max_steps": 200}
        generate = MagicMock()

        await solver(state, generate)

        assert captured_kwargs.get("persona_file") == "/path/to/agent.md"
        assert captured_kwargs.get("skills_dir") == "/path/to/skills"

    @pytest.mark.asyncio
    async def test_no_kwargs_still_works(self) -> None:
        """When no extra kwargs are passed, agent_factory is called with no args."""
        captured_kwargs: dict[str, object] = {}

        def mock_factory(**kw: object) -> object:
            captured_kwargs.update(kw)

            def create_with_prompts(**prompt_kwargs: object) -> object:
                return AsyncMock(return_value=MagicMock())

            return create_with_prompts

        solver = create_saber_solver(
            agent_name="test",
            agent_factory=mock_factory,
        )

        state = MagicMock()
        state.metadata = {"max_steps": 200}
        generate = MagicMock()

        await solver(state, generate)

        assert captured_kwargs == {}


class TestNonCallableAgentRaises:
    """Non-callable return from agent factory must raise TypeError."""

    @pytest.mark.asyncio
    async def test_non_callable_agent_raises_type_error(self) -> None:
        """Non-callable return from agent factory should raise, not silently return."""

        def broken_factory() -> object:
            def create_with_prompts(**kwargs: object) -> str:
                return "not_a_solver"  # Returns string, not Solver

            return create_with_prompts

        solver = create_saber_solver("broken", broken_factory)

        state = MagicMock()
        state.metadata = {"max_steps": 200}
        generate = MagicMock()

        with pytest.raises(TypeError, match="non-callable"):
            await solver(state, generate)


class TestAgentCapabilitiesDict:
    """AGENT_CAPABILITIES module-level dict."""

    def test_has_react_entry(self) -> None:
        """react agent has capabilities entry."""
        assert "react" in AGENT_CAPABILITIES
        caps = AGENT_CAPABILITIES["react"]
        assert isinstance(caps, AgentCapabilities)
        assert caps.supports_tools is True

    def test_has_copilot_entry(self) -> None:
        """copilot agent has capabilities entry."""
        assert "copilot" in AGENT_CAPABILITIES
        caps = AGENT_CAPABILITIES["copilot"]
        assert isinstance(caps, AgentCapabilities)
        assert caps.supports_tools is True

    def test_has_claude_code_entry(self) -> None:
        """claude_code agent has capabilities entry."""
        assert "claude_code" in AGENT_CAPABILITIES
        caps = AGENT_CAPABILITIES["claude_code"]
        assert isinstance(caps, AgentCapabilities)
        assert caps.supports_tools is True

    def test_unknown_agent_not_in_dict(self) -> None:
        """Unknown agents don't have capabilities entries."""
        assert "nonexistent" not in AGENT_CAPABILITIES


class TestCapabilitiesGating:
    """Capabilities-gated kwarg forwarding."""

    @pytest.mark.asyncio
    async def test_tools_resolved_per_sample_from_metadata(self) -> None:
        """tools are resolved per-sample from state.metadata['tools']."""
        mock_inner_solver = AsyncMock(return_value=MagicMock())
        mock_create_with_prompts = MagicMock(return_value=mock_inner_solver)
        mock_factory = MagicMock(return_value=mock_create_with_prompts)

        registry = ToolRegistry()
        solver = create_saber_solver(
            agent_name="react",
            agent_factory=mock_factory,
            tool_registry=registry,
        )

        state = MagicMock()
        state.metadata = {
            "tools": {"bash": {"timeout": 120}},
            "max_steps": 200,
        }
        generate = MagicMock()

        await solver(state, generate)

        call_kwargs = mock_create_with_prompts.call_args[1]
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1

    @pytest.mark.asyncio
    async def test_different_samples_get_different_tools(self) -> None:
        """Two samples with different metadata['tools'] get different tool sets."""
        call_records: list[dict[str, object]] = []

        mock_inner_solver = AsyncMock(return_value=MagicMock())

        def capture_create(**kwargs: object) -> object:
            call_records.append(kwargs)
            return mock_inner_solver

        mock_factory = MagicMock(return_value=capture_create)

        registry = ToolRegistry()
        solver = create_saber_solver(
            agent_name="react",
            agent_factory=mock_factory,
            tool_registry=registry,
        )

        # Sample 1: only bash
        state1 = MagicMock()
        state1.metadata = {"tools": {"bash": {"timeout": 120}}, "max_steps": 200}
        generate = MagicMock()
        await solver(state1, generate)

        # Sample 2: bash + python
        mock_factory.reset_mock()
        mock_factory.return_value = capture_create
        state2 = MagicMock()
        state2.metadata = {"tools": {"bash": {"timeout": 120}, "python": {"timeout": 60}}, "max_steps": 200}
        await solver(state2, generate)

        assert len(call_records) == 2
        assert len(call_records[0]["tools"]) == 1  # type: ignore[arg-type]
        assert len(call_records[1]["tools"]) == 2  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_tool_resolution_caches_identical_configs(self) -> None:
        """Identical tool configs across samples reuse cached ResolvedTools."""
        mock_inner_solver = AsyncMock(return_value=MagicMock())
        mock_create_with_prompts = MagicMock(return_value=mock_inner_solver)
        mock_factory = MagicMock(return_value=mock_create_with_prompts)

        registry = ToolRegistry()
        # Spy on resolve to count calls
        original_resolve = registry.resolve
        resolve_calls: list[object] = []

        def tracking_resolve(tool_configs: object) -> object:
            resolve_calls.append(tool_configs)
            return original_resolve(tool_configs)  # type: ignore[arg-type]

        registry.resolve = tracking_resolve  # type: ignore[assignment]

        solver = create_saber_solver(
            agent_name="react",
            agent_factory=mock_factory,
            tool_registry=registry,
        )

        tools_meta = {"bash": {"timeout": 120}}
        generate = MagicMock()

        # Call twice with identical metadata
        for _ in range(2):
            state = MagicMock()
            state.metadata = {"tools": tools_meta, "max_steps": 200}
            await solver(state, generate)

        # resolve() should only be called once (second call uses cache)
        assert len(resolve_calls) == 1

    @pytest.mark.asyncio
    async def test_no_tools_in_metadata_no_tools_forwarded(self) -> None:
        """When metadata has no 'tools' key, no tools are forwarded."""
        mock_inner_solver = AsyncMock(return_value=MagicMock())
        mock_create_with_prompts = MagicMock(return_value=mock_inner_solver)
        mock_factory = MagicMock(return_value=mock_create_with_prompts)

        registry = ToolRegistry()
        solver = create_saber_solver(
            agent_name="react",
            agent_factory=mock_factory,
            tool_registry=registry,
        )

        state = MagicMock()
        state.metadata = {"max_steps": 200}  # no tools key
        generate = MagicMock()

        await solver(state, generate)

        call_kwargs = mock_create_with_prompts.call_args[1]
        assert "tools" not in call_kwargs

    @pytest.mark.asyncio
    async def test_tools_passed_for_copilot(self) -> None:
        """tools param IS forwarded for copilot (supports_tools=True)."""
        mock_inner_solver = AsyncMock(return_value=MagicMock())
        mock_create_with_prompts = MagicMock(return_value=mock_inner_solver)
        mock_factory = MagicMock(return_value=mock_create_with_prompts)

        registry = ToolRegistry()
        solver = create_saber_solver(
            agent_name="copilot",
            agent_factory=mock_factory,
            tool_registry=registry,
        )

        state = MagicMock()
        state.metadata = {"tools": {"bash": {"timeout": 120}}, "max_steps": 200}
        generate = MagicMock()

        await solver(state, generate)

        call_kwargs = mock_create_with_prompts.call_args[1]
        assert "tools" in call_kwargs

    @pytest.mark.asyncio
    async def test_tools_passed_for_claude_code(self) -> None:
        """tools param IS forwarded for claude_code."""
        mock_inner_solver = AsyncMock(return_value=MagicMock())
        mock_create_with_prompts = MagicMock(return_value=mock_inner_solver)
        mock_factory = MagicMock(return_value=mock_create_with_prompts)

        registry = ToolRegistry()
        solver = create_saber_solver(
            agent_name="claude_code",
            agent_factory=mock_factory,
            tool_registry=registry,
        )

        state = MagicMock()
        state.metadata = {"tools": {"bash": {"timeout": 120}}, "max_steps": 200}
        generate = MagicMock()

        await solver(state, generate)

        call_kwargs = mock_create_with_prompts.call_args[1]
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1

    @pytest.mark.asyncio
    async def test_unknown_agent_gets_tools_by_default(self) -> None:
        """Unknown agents default to supports_tools=True."""
        mock_inner_solver = AsyncMock(return_value=MagicMock())
        mock_create_with_prompts = MagicMock(return_value=mock_inner_solver)
        mock_factory = MagicMock(return_value=mock_create_with_prompts)

        registry = ToolRegistry()
        solver = create_saber_solver(
            agent_name="custom_unknown_agent",
            agent_factory=mock_factory,
            tool_registry=registry,
        )

        state = MagicMock()
        state.metadata = {"tools": {"bash": {"timeout": 180}}, "max_steps": 200}
        generate = MagicMock()

        await solver(state, generate)

        call_kwargs = mock_create_with_prompts.call_args[1]
        assert "tools" in call_kwargs


class TestPerSampleMaxSteps:
    """Per-sample max_steps propagation via state.tool_call_limit."""

    @pytest.mark.asyncio
    async def test_max_steps_from_metadata_sets_tool_call_limit(self) -> None:
        """max_steps in metadata should set state.tool_call_limit."""
        mock_inner_solver = AsyncMock(return_value=MagicMock())
        mock_create_with_prompts = MagicMock(return_value=mock_inner_solver)
        mock_factory = MagicMock(return_value=mock_create_with_prompts)

        solver = create_saber_solver(agent_name="react", agent_factory=mock_factory)

        state = MagicMock()
        state.metadata = {"max_steps": 100}
        # Track what tool_call_limit is set to
        captured_limit: list[int] = []
        original_setattr = type(state).__setattr__

        def tracking_setattr(self: object, name: str, value: object) -> None:
            if name == "tool_call_limit":
                captured_limit.append(value)  # type: ignore[arg-type]
            original_setattr(self, name, value)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(type(state), "__setattr__", tracking_setattr)
            generate = MagicMock()
            await solver(state, generate)

        assert 100 in captured_limit

    @pytest.mark.asyncio
    async def test_no_max_steps_raises_valueerror(self) -> None:
        """Without max_steps in metadata, ValueError is raised."""
        mock_inner_solver = AsyncMock(return_value=MagicMock())
        mock_create_with_prompts = MagicMock(return_value=mock_inner_solver)
        mock_factory = MagicMock(return_value=mock_create_with_prompts)

        solver = create_saber_solver(agent_name="react", agent_factory=mock_factory)

        state = MagicMock()
        state.metadata = {}
        generate = MagicMock()

        with pytest.raises(ValueError, match="must define a positive integer 'max_steps'"):
            await solver(state, generate)

    @pytest.mark.asyncio
    async def test_invalid_max_steps_raises_valueerror(self) -> None:
        """Non-int or non-positive max_steps should raise ValueError."""
        mock_inner_solver = AsyncMock(return_value=MagicMock())
        mock_create_with_prompts = MagicMock(return_value=mock_inner_solver)
        mock_factory = MagicMock(return_value=mock_create_with_prompts)

        solver = create_saber_solver(agent_name="react", agent_factory=mock_factory)

        for bad_value in [0, -5, "ten", None]:
            state = MagicMock()
            state.metadata = {"max_steps": bad_value}
            generate = MagicMock()

            with pytest.raises(ValueError, match="must define a positive integer 'max_steps'"):
                await solver(state, generate)

    @pytest.mark.asyncio
    async def test_none_metadata_raises_valueerror(self) -> None:
        """When metadata is None, ValueError is raised for missing max_steps."""
        mock_inner_solver = AsyncMock(return_value=MagicMock())
        mock_create_with_prompts = MagicMock(return_value=mock_inner_solver)
        mock_factory = MagicMock(return_value=mock_create_with_prompts)

        solver = create_saber_solver(agent_name="react", agent_factory=mock_factory)

        state = MagicMock()
        state.metadata = None
        generate = MagicMock()

        with pytest.raises(ValueError, match="must define a positive integer 'max_steps'"):
            await solver(state, generate)


class TestToolCallLimitRecovery:
    """When an agent raises LimitExceededError the solver injects a
    final-answer prompt and does one more tool-free generation."""

    @pytest.mark.asyncio
    async def test_limit_exceeded_injects_message_and_generates(self) -> None:
        """On LimitExceededError the solver injects the limit message,
        calls generate() with no tools, and returns the state."""
        from inspect_ai.model import ChatMessageUser
        from inspect_ai.model._model_output import ModelOutput
        from inspect_ai.util._limit import LimitExceededError

        # Inner solver that always raises LimitExceededError
        async def failing_solver(state: object, generate: object) -> object:
            raise LimitExceededError("tool_call", value=50, limit=50)

        mock_create_with_prompts = MagicMock(return_value=failing_solver)
        mock_factory = MagicMock(return_value=mock_create_with_prompts)

        solver = create_saber_solver(agent_name="react", agent_factory=mock_factory)

        # Build a state with a mutable messages list
        state = MagicMock()
        state.metadata = {"max_steps": 200}
        state.messages = []

        fake_output = ModelOutput.from_content(model="test", content="My final answer", stop_reason="stop")

        with patch("saber.agents.solver_factory.get_model") as mock_get_model:
            mock_model = AsyncMock()
            mock_model.generate = AsyncMock(return_value=fake_output)
            mock_get_model.return_value = mock_model

            result = await solver(state, MagicMock())

        # Verify the limit message was appended
        user_msgs = [m for m in state.messages if isinstance(m, ChatMessageUser)]
        assert len(user_msgs) == 1
        assert user_msgs[0].content == TOOL_CALL_LIMIT_MESSAGE

        # Verify generate was called with no tools
        mock_model.generate.assert_awaited_once()
        call_kwargs = mock_model.generate.call_args
        assert call_kwargs.kwargs.get("tools") == [] or call_kwargs[1].get("tools") == []

        # Verify the output was set and assistant message appended
        assert state.output is fake_output
        assert result is state


class TestToolCallLimitMessageReExport:
    """TOOL_CALL_LIMIT_MESSAGE is re-exported from solver_factory for backwards compat."""

    def test_importable_from_solver_factory(self) -> None:
        """TOOL_CALL_LIMIT_MESSAGE can be imported from solver_factory."""
        from saber.agents.solver_factory import TOOL_CALL_LIMIT_MESSAGE

        assert isinstance(TOOL_CALL_LIMIT_MESSAGE, str)
        assert "tool call limit" in TOOL_CALL_LIMIT_MESSAGE.lower()

    def test_matches_message_utils_constant(self) -> None:
        """solver_factory's TOOL_CALL_LIMIT_MESSAGE matches message_utils."""
        from saber.agents.message_utils import (
            TOOL_CALL_LIMIT_MESSAGE as ORIGINAL,
        )
        from saber.agents.solver_factory import (
            TOOL_CALL_LIMIT_MESSAGE as REEXPORT,
        )

        assert ORIGINAL == REEXPORT


class TestMaxStepsForwarding:
    """max_steps is forwarded in agent_kwargs when present in metadata."""

    @pytest.mark.asyncio
    async def test_max_steps_forwarded_in_agent_kwargs(self) -> None:
        """When metadata has max_steps, it is forwarded in agent_kwargs."""
        captured_prompt_kwargs: list[dict[str, object]] = []

        def mock_factory(**kw: object) -> object:
            def create_with_prompts(**prompt_kwargs: object) -> object:
                captured_prompt_kwargs.append(dict(prompt_kwargs))
                return AsyncMock(return_value=MagicMock())

            return create_with_prompts

        solver = create_saber_solver(agent_name="react", agent_factory=mock_factory)

        state = MagicMock()
        state.metadata = {"max_steps": 25}
        generate = MagicMock()

        await solver(state, generate)

        assert len(captured_prompt_kwargs) == 1
        assert captured_prompt_kwargs[0].get("max_steps") == 25

    @pytest.mark.asyncio
    async def test_max_steps_not_forwarded_when_absent(self) -> None:
        """When metadata has no max_steps, ValueError is raised."""
        def mock_factory(**kw: object) -> object:
            def create_with_prompts(**prompt_kwargs: object) -> object:
                return AsyncMock(return_value=MagicMock())

            return create_with_prompts

        solver = create_saber_solver(agent_name="react", agent_factory=mock_factory)

        state = MagicMock()
        state.metadata = {}
        generate = MagicMock()

        with pytest.raises(ValueError, match="must define a positive integer 'max_steps'"):
            await solver(state, generate)

    @pytest.mark.asyncio
    async def test_max_steps_not_forwarded_when_invalid(self) -> None:
        """When max_steps is invalid (0, negative, string), ValueError is raised."""
        for bad_value in [0, -5, "ten", None]:

            def mock_factory(**kw: object) -> object:
                def create_with_prompts(**prompt_kwargs: object) -> object:
                    return AsyncMock(return_value=MagicMock())

                return create_with_prompts

            solver = create_saber_solver(agent_name="react", agent_factory=mock_factory)

            state = MagicMock()
            state.metadata = {"max_steps": bad_value}
            generate = MagicMock()

            with pytest.raises(ValueError, match="must define a positive integer 'max_steps'"):
                await solver(state, generate)
    @pytest.mark.asyncio
    async def test_limit_exceeded_appends_assistant_message(self) -> None:
        """The model's response message is appended to state.messages."""
        from inspect_ai.model._model_output import ModelOutput
        from inspect_ai.util._limit import LimitExceededError

        async def failing_solver(state: object, generate: object) -> object:
            raise LimitExceededError("tool_call", value=10, limit=10)

        mock_create_with_prompts = MagicMock(return_value=failing_solver)
        mock_factory = MagicMock(return_value=mock_create_with_prompts)

        solver = create_saber_solver(agent_name="react", agent_factory=mock_factory)

        state = MagicMock()
        state.metadata = {"max_steps": 200}
        state.messages = []

        fake_output = ModelOutput.from_content(model="test", content="Here is my answer", stop_reason="stop")

        with patch("saber.agents.solver_factory.get_model") as mock_get_model:
            mock_model = AsyncMock()
            mock_model.generate = AsyncMock(return_value=fake_output)
            mock_get_model.return_value = mock_model

            await solver(state, MagicMock())

        # Last message should be the assistant's response
        assert len(state.messages) == 2  # user limit msg + assistant response
        assert state.messages[-1] is fake_output.message

    @pytest.mark.asyncio
    async def test_normal_flow_unaffected(self) -> None:
        """When no LimitExceededError is raised the solver works normally."""
        sentinel = MagicMock(name="returned_state")
        mock_inner_solver = AsyncMock(return_value=sentinel)
        mock_create_with_prompts = MagicMock(return_value=mock_inner_solver)
        mock_factory = MagicMock(return_value=mock_create_with_prompts)

        solver = create_saber_solver(agent_name="react", agent_factory=mock_factory)

        state = MagicMock()
        state.metadata = {"max_steps": 200}
        generate = MagicMock()

        result = await solver(state, generate)

        assert result is sentinel
        mock_inner_solver.assert_awaited_once_with(state, generate)

    @pytest.mark.asyncio
    async def test_limit_exceeded_patches_orphaned_tool_calls(self) -> None:
        """When state.messages contains assistant tool_calls without
        matching tool results, the solver injects dummy results before
        calling generate() to avoid OpenAI API 400 errors."""
        from inspect_ai.model import (
            ChatMessageAssistant,
            ChatMessageTool,
            ChatMessageUser,
        )
        from inspect_ai.model._model_output import ModelOutput
        from inspect_ai.tool import ToolCall
        from inspect_ai.util._limit import LimitExceededError

        async def failing_solver(state: object, generate: object) -> object:
            raise LimitExceededError("tool_call", value=50, limit=50)

        mock_create_with_prompts = MagicMock(return_value=failing_solver)
        mock_factory = MagicMock(return_value=mock_create_with_prompts)

        solver = create_saber_solver(agent_name="copilot", agent_factory=mock_factory)

        # State has assistant with 3 tool calls, only 1 has a result
        state = MagicMock()
        state.metadata = {"max_steps": 200}
        state.messages = [
            ChatMessageUser(content="Investigate the incident"),
            ChatMessageAssistant(
                content="I'll run some queries",
                tool_calls=[
                    ToolCall(id="call_1", function="bash", arguments={"cmd": "ls"}, type="function"),
                    ToolCall(id="call_2", function="report_intent", arguments={"intent": "test"}, type="function"),
                    ToolCall(id="call_3", function="bash", arguments={"cmd": "pwd"}, type="function"),
                ],
            ),
            ChatMessageTool(content="file1.txt", tool_call_id="call_1"),
            # call_2 and call_3 are "orphaned" — no tool result
        ]

        fake_output = ModelOutput.from_content(
            model="test", content="Final answer", stop_reason="stop"
        )

        with patch("saber.agents.solver_factory.get_model") as mock_get_model:
            mock_model = AsyncMock()
            mock_model.generate = AsyncMock(return_value=fake_output)
            mock_get_model.return_value = mock_model

            result = await solver(state, MagicMock())

        # Verify dummy tool results were injected for orphaned calls
        tool_msgs = [
            m for m in state.messages if isinstance(m, ChatMessageTool)
        ]
        tool_call_ids = {m.tool_call_id for m in tool_msgs}
        assert "call_1" in tool_call_ids  # original result preserved
        assert "call_2" in tool_call_ids  # dummy result injected
        assert "call_3" in tool_call_ids  # dummy result injected
        assert len(tool_msgs) == 3

        # Verify the generate was called (no 400 error)
        mock_model.generate.assert_awaited_once()

        # Verify the limit message comes AFTER the tool results
        user_limit_msgs = [
            m
            for m in state.messages
            if isinstance(m, ChatMessageUser) and "tool call limit" in str(m.content).lower()
        ]
        assert len(user_limit_msgs) == 1

        assert result is state
