"""Tests for firstparty runtime_spec models."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from saber.agents.registry.firstparty.runtime_spec import (
    AgentAlias,
    EnvSchema,
    RuntimeSpec,
    interpolate_command,
)


# ── EnvSchema ──────────────────────────────────────────────────────────


class TestEnvSchema:
    def test_defaults(self) -> None:
        schema = EnvSchema()
        assert schema.bridge_injected == {}
        assert schema.required == {}
        assert schema.defaults == {}
        assert schema.passthrough == []

    def test_construction_with_values(self) -> None:
        schema = EnvSchema(
            bridge_injected={"URL": "{bridge_url}"},
            required={"API_KEY": "The API key"},
            defaults={"TIMEOUT": "30"},
            passthrough=["HOME"],
        )
        assert schema.bridge_injected == {"URL": "{bridge_url}"}
        assert schema.required == {"API_KEY": "The API key"}
        assert schema.defaults == {"TIMEOUT": "30"}
        assert schema.passthrough == ["HOME"]

    def test_frozen(self) -> None:
        schema = EnvSchema()
        with pytest.raises(ValidationError):
            schema.bridge_injected = {"a": "b"}  # type: ignore[misc]


# ── AgentAlias ─────────────────────────────────────────────────────────


class TestAgentAlias:
    def test_all_fields_required(self) -> None:
        alias = AgentAlias(
            name="Claude Code",
            model_alias="claude-sonnet-4-20250514",
            env_var="AGENT_MODEL",
        )
        assert alias.name == "Claude Code"
        assert alias.model_alias == "claude-sonnet-4-20250514"
        assert alias.env_var == "AGENT_MODEL"

    def test_missing_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            AgentAlias(name="x", model_alias="y")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        alias = AgentAlias(name="x", model_alias="y", env_var="Z")
        with pytest.raises(ValidationError):
            alias.name = "changed"  # type: ignore[misc]


# ── RuntimeSpec ────────────────────────────────────────────────────────


class TestRuntimeSpec:
    def test_required_fields(self) -> None:
        spec = RuntimeSpec(name="agent", invoke_command=["run", "--flag"])
        assert spec.name == "agent"
        assert spec.invoke_command == ["run", "--flag"]

    def test_optional_defaults(self) -> None:
        spec = RuntimeSpec(name="a", invoke_command=["cmd"])
        assert spec.sandbox_compose is None
        assert spec.permanent_compose is None
        assert spec.env_schema == EnvSchema()
        assert spec.agents == []
        assert spec.default_model_aliases == {}
        assert spec.pre_invoke_hook is None
        assert spec.post_invoke_hook is None
        assert spec.timeout == 1800
        assert spec.port_base == 3000

    def test_frozen(self) -> None:
        spec = RuntimeSpec(name="a", invoke_command=["cmd"])
        with pytest.raises(ValidationError):
            spec.name = "changed"  # type: ignore[misc]


# ── RuntimeSpec.build_model_aliases ────────────────────────────────────


class TestBuildModelAliases:
    def test_returns_none_when_empty(self) -> None:
        spec = RuntimeSpec(name="a", invoke_command=["cmd"])
        assert spec.build_model_aliases() is None

    def test_returns_dict_when_non_empty(self) -> None:
        aliases = {"gpt-4": "openai/gpt-4o"}
        spec = RuntimeSpec(
            name="a", invoke_command=["cmd"], default_model_aliases=aliases
        )
        assert spec.build_model_aliases() == aliases


# ── RuntimeSpec.build_env ──────────────────────────────────────────────


class TestBuildEnv:
    def test_merges_defaults(self) -> None:
        spec = RuntimeSpec(
            name="a",
            invoke_command=["cmd"],
            env_schema=EnvSchema(defaults={"K": "default_val"}),
        )
        env = spec.build_env(bridge_url="http://b", bridge_api_key="key123")
        assert env["K"] == "default_val"

    def test_required_from_os_environ(self) -> None:
        spec = RuntimeSpec(
            name="a",
            invoke_command=["cmd"],
            env_schema=EnvSchema(required={"MY_VAR": "needed"}),
        )
        with patch.dict(os.environ, {"MY_VAR": "from_env"}):
            env = spec.build_env(bridge_url="http://b", bridge_api_key="k")
        assert env["MY_VAR"] == "from_env"

    def test_required_missing_raises(self) -> None:
        spec = RuntimeSpec(
            name="a",
            invoke_command=["cmd"],
            env_schema=EnvSchema(required={"MISSING_VAR": "some description"}),
        )
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="MISSING_VAR"):
                spec.build_env(bridge_url="http://b", bridge_api_key="k")

    def test_passthrough_from_os_environ(self) -> None:
        spec = RuntimeSpec(
            name="a",
            invoke_command=["cmd"],
            env_schema=EnvSchema(passthrough=["PASS_VAR"]),
        )
        with patch.dict(os.environ, {"PASS_VAR": "passed"}):
            env = spec.build_env(bridge_url="http://b", bridge_api_key="k")
        assert env["PASS_VAR"] == "passed"

    def test_passthrough_missing_is_silently_skipped(self) -> None:
        spec = RuntimeSpec(
            name="a",
            invoke_command=["cmd"],
            env_schema=EnvSchema(passthrough=["NOT_SET"]),
        )
        with patch.dict(os.environ, {}, clear=True):
            env = spec.build_env(bridge_url="http://b", bridge_api_key="k")
        assert "NOT_SET" not in env

    def test_agent_alias_env_vars(self) -> None:
        spec = RuntimeSpec(
            name="a",
            invoke_command=["cmd"],
            agents=[
                AgentAlias(
                    name="Agent1",
                    model_alias="model-x",
                    env_var="AGENT_MODEL",
                ),
            ],
        )
        env = spec.build_env(bridge_url="http://b", bridge_api_key="k")
        assert env["AGENT_MODEL"] == "model-x"

    def test_bridge_injected_interpolation(self) -> None:
        spec = RuntimeSpec(
            name="a",
            invoke_command=["cmd"],
            env_schema=EnvSchema(
                bridge_injected={
                    "SERVER_URL": "{bridge_url}/api",
                    "AUTH": "{bridge_api_key}",
                }
            ),
        )
        env = spec.build_env(bridge_url="http://localhost:9000", bridge_api_key="sec")
        assert env["SERVER_URL"] == "http://localhost:9000/api"
        assert env["AUTH"] == "sec"

    def test_merge_order(self) -> None:
        """Later layers override earlier ones."""
        spec = RuntimeSpec(
            name="a",
            invoke_command=["cmd"],
            env_schema=EnvSchema(
                defaults={"K": "default"},
                required={"K": "overridden by required"},
                passthrough=["K"],
                bridge_injected={"K": "bridge_wins"},
            ),
            agents=[
                AgentAlias(name="agent", model_alias="model-x", env_var="K"),
            ],
        )
        with patch.dict(os.environ, {"K": "from_env"}):
            env = spec.build_env(bridge_url="http://b", bridge_api_key="k")
        # bridge_injected is applied last → wins
        assert env["K"] == "bridge_wins"


# ── RuntimeSpec hooks ──────────────────────────────────────────────────


class TestRuntimeSpecHooks:
    def test_hook_field_stores_callable(self) -> None:
        async def my_hook(sandbox, env):  # type: ignore[no-untyped-def]
            return env

        spec = RuntimeSpec(name="a", invoke_command=["cmd"], pre_invoke_hook=my_hook)
        assert spec.pre_invoke_hook is my_hook
        assert spec.post_invoke_hook is None


# ── interpolate_command ────────────────────────────────────────────────


class TestInterpolateCommand:
    def test_replaces_tokens(self) -> None:
        result = interpolate_command(
            ["run", "--url={url}", "--port={port}"], url="http://x", port="8080"
        )
        assert result == ["run", "--url=http://x", "--port=8080"]

    def test_leaves_non_tokens_alone(self) -> None:
        result = interpolate_command(["echo", "hello"], foo="bar")
        assert result == ["echo", "hello"]

    def test_unresolved_placeholders_left_as_is(self) -> None:
        result = interpolate_command(["--later={deferred}"], other="val")
        assert result == ["--later={deferred}"]

    def test_partial_replacement(self) -> None:
        result = interpolate_command(
            ["{a}-{b}"], a="hello"
        )
        assert result == ["hello-{b}"]

    def test_empty_command(self) -> None:
        assert interpolate_command([]) == []
