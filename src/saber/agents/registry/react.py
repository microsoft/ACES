"""Default react agent implementation.

Uses inspect_ai's react() agent with SABER prompt injection.
Imports are deferred to avoid heavy inspect_ai loading at discovery time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inspect_ai.solver import Solver
    from inspect_ai.tool import Tool


def create_agent(**kwargs: object) -> Callable[..., Solver]:
    """Create a React agent factory with SABER integration.

    Returns a callable that accepts prompt kwargs and returns a Solver.
    """

    def create_with_prompts(
        instruction_prompt: str = "",
        assistant_prompt: str = "",
        submit_prompt: str = "",
        continue_prompt: str = "",
        submit: bool | None = None,
        tools: Sequence[Tool] | None = None,
        **extra_kwargs: object,
    ) -> Solver:
        from inspect_ai.agent import AgentPrompt, AgentSubmit, as_solver, react
        from inspect_ai.tool import bash, python

        submit_enabled = submit if submit is not None else True

        resolved_tools: list[Tool] = list(tools) if tools else [bash(), python()]

        agent = react(
            prompt=AgentPrompt(
                instructions=instruction_prompt or None,
                assistant_prompt=assistant_prompt or None,
                submit_prompt=submit_prompt if submit_enabled else None,
            ),
            tools=resolved_tools,
            on_continue=continue_prompt or None,
            submit=AgentSubmit(answer_only=True) if submit_enabled else False,
            **kwargs,
        )

        return as_solver(agent)

    return create_with_prompts
