"""Hyenas config.yaml generator for SABER bridge mode."""

from __future__ import annotations

import yaml

from saber.agents.registry.firstparty.runtimes.hyenas_models import (
    HyenasConfig,
    HyenasEndpoint,
    HyenasStageAgentConfig,
)


def generate_hyenas_config(
    bridge_port: int,
    endpoint_models: list[str],
    default_model: str = "gpt-5.4",
    extra_config: dict[str, object] | None = None,
) -> str:
    """Generate a Hyenas config.yaml string for bridge mode.

    No stage-level overrides are emitted — Hyenas uses its own hardcoded
    model defaults per stage.

    Args:
        bridge_port: Port number of the sandbox agent bridge.
        endpoint_models: Model names to register at the bridge endpoint.
        default_model: Model name for the root default.models entry.
        extra_config: Optional dict to deep-merge into the generated config.

    Returns:
        YAML string suitable for writing to config.yaml.
    """
    # Collect all unique model names (preserving first-seen order).
    seen: set[str] = set()
    unique_models: list[str] = []
    for model_name in [default_model, *endpoint_models]:
        if model_name not in seen:
            seen.add(model_name)
            unique_models.append(model_name)

    endpoint = HyenasEndpoint(
        label="saber-bridge",
        type="openai",
        url=f"http://localhost:{bridge_port}/v1",
        models=unique_models,
    )

    config = HyenasConfig(
        endpoints=[endpoint],
        default=HyenasStageAgentConfig(models=[default_model]),
    )

    data = config.model_dump(by_alias=True, exclude_none=True)
    if extra_config:
        _deep_merge(data, extra_config)
    return yaml.dump(data, default_flow_style=False, sort_keys=False)


def _deep_merge(base: dict[str, object], overlay: dict[str, object]) -> None:
    """Recursively merge *overlay* into *base*, mutating *base* in place."""
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)  # type: ignore[arg-type]
        else:
            base[key] = value
