"""Hyenas config.yaml generator for SABER bridge mode."""

from __future__ import annotations

import yaml

from saber.agents.registry.firstparty.runtime_spec import AgentAlias
from saber.agents.registry.firstparty.runtimes.hyenas_models import (
    HyenasConfig,
    HyenasEndpoint,
    HyenasStageAgentConfig,
)

# Stage field names that HyenasConfig recognizes (kebab-case).
_KNOWN_STAGES = frozenset(
    {"prepare-stage", "scan-stage", "validate-stage", "prove-stage"}
)


def _collect_stage_configs(
    agents: list[AgentAlias],
) -> dict[str, dict[str, HyenasStageAgentConfig]]:
    """Parse agent env_vars and group into nested stage → agent → models dicts.

    Each ``AgentAlias.env_var`` has the form ``"stage-name.agent-name.models"``.
    Agents sharing the same stage+agent path have their model_alias values
    collected into one list, preserving insertion order.

    Args:
        agents: List of AgentAlias definitions.

    Returns:
        Mapping of stage name → agent name → HyenasStageAgentConfig.
    """
    # stage -> agent -> ordered list of model aliases
    grouped: dict[str, dict[str, list[str]]] = {}

    for agent in agents:
        parts = agent.env_var.split(".")
        if len(parts) != 3:  # noqa: PLR2004
            msg = (
                f"AgentAlias env_var must have exactly 3 dot-separated parts "
                f"(stage.agent.models), got: {agent.env_var!r}"
            )
            raise ValueError(msg)

        stage_name, agent_name, _field = parts

        if stage_name not in _KNOWN_STAGES:
            msg = (
                f"Unknown stage {stage_name!r} in env_var {agent.env_var!r}, "
                f"expected one of {sorted(_KNOWN_STAGES)}"
            )
            raise ValueError(msg)

        grouped.setdefault(stage_name, {}).setdefault(agent_name, []).append(
            agent.model_alias
        )

    return {
        stage: {
            agent_name: HyenasStageAgentConfig(models=models)
            for agent_name, models in agents_map.items()
        }
        for stage, agents_map in grouped.items()
    }


def generate_hyenas_config(
    agents: list[AgentAlias],
    bridge_port: int,
    default_model: str = "gpt-4o",
) -> str:
    """Generate a Hyenas config.yaml string for bridge mode.

    Args:
        agents: List of AgentAlias definitions from RuntimeSpec.
        bridge_port: Port number of the sandbox agent bridge.
        default_model: Model name for the root default.models entry.

    Returns:
        YAML string suitable for writing to config.yaml.
    """
    # Collect all unique model names (preserving first-seen order).
    seen: set[str] = set()
    unique_models: list[str] = []
    for model_name in [default_model, *(a.model_alias for a in agents)]:
        if model_name not in seen:
            seen.add(model_name)
            unique_models.append(model_name)

    endpoint = HyenasEndpoint(
        label="saber-bridge",
        type="openai",
        url=f"http://localhost:{bridge_port}/v1",
        models=unique_models,
    )

    stage_configs = _collect_stage_configs(agents)

    config = HyenasConfig(
        endpoints=[endpoint],
        default=HyenasStageAgentConfig(models=[default_model]),
        prepare_stage=stage_configs.get("prepare-stage"),
        scan_stage=stage_configs.get("scan-stage"),
        validate_stage=stage_configs.get("validate-stage"),
        prove_stage=stage_configs.get("prove-stage"),
    )

    data = config.model_dump(by_alias=True, exclude_none=True)
    return yaml.dump(data, default_flow_style=False, sort_keys=False)
