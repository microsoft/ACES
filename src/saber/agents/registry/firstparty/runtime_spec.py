"""Runtime specification models for first-party agent runtimes."""

import os
import re
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, SkipValidation

if TYPE_CHECKING:
    from inspect_ai.util import SandboxEnvironment

    InvokeHook = Callable[
        ["SandboxEnvironment", dict[str, str]], Awaitable[dict[str, str]]
    ]

# At runtime Pydantic cannot resolve the SandboxEnvironment forward reference,
# so hook fields use SkipValidation with a generic callable annotation.
_HookField = Annotated[
    Callable[..., Awaitable[dict[str, str]]] | None,
    SkipValidation,
]


_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


class EnvSchema(BaseModel):
    """Describes the environment variables a runtime expects."""

    model_config = ConfigDict(frozen=True)

    bridge_injected: dict[str, str] = {}
    required: dict[str, str] = {}
    defaults: dict[str, str] = {}
    passthrough: list[str] = []


class AgentAlias(BaseModel):
    """Maps an agent name to its model alias and controlling env var."""

    model_config = ConfigDict(frozen=True)

    name: str
    model_alias: str
    env_var: str


class RuntimeSpec(BaseModel):
    """Declarative specification for a first-party agent runtime."""

    model_config = ConfigDict(frozen=True)

    name: str
    invoke_command: list[str]
    sandbox_compose: str | None = None
    permanent_compose: str | None = None
    env_schema: EnvSchema = EnvSchema()
    agents: list[AgentAlias] = []
    default_model_aliases: dict[str, str] = {}
    default_model: str | None = None
    pre_invoke_hook: _HookField = None
    post_invoke_hook: _HookField = None
    timeout: int = 1800
    port_base: int = 3000

    def build_model_aliases(self) -> dict[str, str] | None:
        """Return default_model_aliases if non-empty, else None."""
        return self.default_model_aliases if self.default_model_aliases else None

    def build_env(self, bridge_url: str, bridge_api_key: str) -> dict[str, str]:
        """Build the merged environment dict for runtime invocation.

        Merge order (later wins): defaults → required → passthrough → agent aliases → bridge_injected.

        Args:
            bridge_url: URL of the sandbox agent bridge.
            bridge_api_key: API key for bridge authentication.

        Returns:
            Merged environment dictionary.

        Raises:
            ValueError: If a required env var is not set in os.environ.
        """
        env: dict[str, str] = {}

        # 1. defaults
        env.update(self.env_schema.defaults)

        # 2. required (from os.environ, error if missing)
        for var_name, description in self.env_schema.required.items():
            value = os.environ.get(var_name)
            if value is None:
                msg = (
                    f"Required environment variable '{var_name}' is not set. "
                    f"Description: {description}"
                )
                raise ValueError(msg)
            env[var_name] = value

        # 3. passthrough (from os.environ, skip if missing)
        for var_name in self.env_schema.passthrough:
            value = os.environ.get(var_name)
            if value is not None:
                env[var_name] = value

        # 4. agent alias env vars
        for agent in self.agents:
            env[agent.env_var] = agent.model_alias

        # 5. bridge_injected (interpolate placeholders) — single-pass to prevent double-substitution
        bridge_values = {"bridge_url": bridge_url, "bridge_api_key": bridge_api_key}
        for key, template in self.env_schema.bridge_injected.items():

            def _replace_bridge(match: re.Match[str]) -> str:
                return bridge_values.get(match.group(1), match.group(0))

            env[key] = _PLACEHOLDER_RE.sub(_replace_bridge, template)

        return env


def interpolate_command(command: list[str], **values: str) -> list[str]:
    """Replace ``{key}`` tokens in command strings with provided values.

    Tokens whose keys are not in *values* are left as-is.

    Args:
        command: Command template tokens.
        **values: Substitution mapping.

    Returns:
        New list with placeholders resolved where possible.
    """

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return values.get(key, match.group(0))

    return [
        resolved
        for part in command
        if (resolved := _PLACEHOLDER_RE.sub(_replace, part))
    ]
