"""HYENAS_RUNTIME definition and hook functions."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from saber.agents.registry.firstparty.runtime_spec import (
    EnvSchema,
    RuntimeSpec,
)
from saber.agents.registry.firstparty.runtimes.hyenas_config import (
    generate_hyenas_config,
)
from saber.agents.registry.firstparty.runtimes.hyenas_hooks import (
    extract_hyenas_findings,
    inject_target_repo,
)
from saber.logging import get_logger

if TYPE_CHECKING:
    from inspect_ai.util import SandboxEnvironment

logger = get_logger(__name__)

_CONFIG_PATH = "/app/.hyenas-config/config.yaml"
_TOKEN_PATH = "/tmp/.copilot_token"
_DEFAULT_BRIDGE_PORT = 3000


async def hyenas_pre_invoke(
    sandbox: SandboxEnvironment,
    env: dict[str, str],
) -> dict[str, str]:
    """Generate Hyenas config.yaml and inject target repository.

    Args:
        sandbox: The sandbox environment.
        env: Environment dictionary (contains _BRIDGE_URL from bridge_injected).

    Returns:
        The (unmodified) env dict.
    """
    bridge_port = _parse_bridge_port(env.get("_BRIDGE_URL", ""))

    extra_config_json = env.get("_HYENAS_EXTRA_CONFIG")
    extra_config: dict[str, object] | None = None
    if extra_config_json:
        try:
            parsed = json.loads(extra_config_json)
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON in _HYENAS_EXTRA_CONFIG: {exc}"
            raise ValueError(msg) from exc
        if not isinstance(parsed, dict):
            msg = f"_HYENAS_EXTRA_CONFIG must be a JSON object, got {type(parsed).__name__}"
            raise TypeError(msg)
        extra_config = parsed

    config_yaml = generate_hyenas_config(
        bridge_port=bridge_port,
        endpoint_models=list(HYENAS_RUNTIME.default_model_aliases.keys()),
        extra_config=extra_config,
    )
    await sandbox.write_file(_CONFIG_PATH, config_yaml)

    # Inject target repo from metadata passed via env
    metadata: dict[str, str] = {}
    if "repo_tarball" in env:
        metadata["repo_tarball"] = env["repo_tarball"]
    elif "repo_path" in env:
        metadata["repo_path"] = env["repo_path"]

    if metadata:
        await inject_target_repo(sandbox, metadata)

    # Inject GITHUB_TOKEN securely via file (not via env to avoid ps aux leakage).
    # Pre-create with restricted perms to eliminate TOCTOU race (write_file uses
    # tee which inherits umask 022 → world-readable without this).
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        msg = "GITHUB_TOKEN must be set for Hyenas Copilot SDK authentication"
        raise ValueError(msg)
    result = await sandbox.exec(["install", "-m", "600", "/dev/null", _TOKEN_PATH])
    if not result.success:
        msg = f"Failed to create token file: {result.stderr}"
        raise RuntimeError(msg)
    await sandbox.write_file(_TOKEN_PATH, token)

    return env


async def hyenas_post_invoke(
    sandbox: SandboxEnvironment,
    env: dict[str, str],
) -> dict[str, str]:
    """Extract findings from Hyenas scan output.

    Args:
        sandbox: The sandbox environment.
        env: Environment dictionary.

    Returns:
        The env dict (findings logged, not returned via env).
    """
    findings = await extract_hyenas_findings(sandbox)
    if findings:
        logger.info("Extracted %d findings from Hyenas scan", len(findings))
    else:
        logger.info("No findings extracted from Hyenas scan")
    return env


HYENAS_RUNTIME = RuntimeSpec(
    name="hyenas",
    default_model="mockllm/model",
    sandbox_compose="runtimes/hyenas/compose/hyenas.sandbox.compose.yml",
    env_schema=EnvSchema(
        bridge_injected={
            "_BRIDGE_URL": "{bridge_url}",
        },
        passthrough=["_HYENAS_EXTRA_CONFIG"],
        defaults={
            "HYENAS_HEADLESS": "true",
            "LOG_LEVEL": "info",
            "prove_flag": "--no-prove",
        },
    ),
    agents=[],
    invoke_command=[
        "bash",
        "-c",
        "export GITHUB_TOKEN=$(cat /tmp/.copilot_token) && "
        "rm -f /tmp/.copilot_token && "
        "exec node dist/src/cli.js run "
        "--headless --no-copilot --no-check {prove_flag} "
        "--repo /workspace --hyenas /output/.hyenas "
        "--config /app/.hyenas-config/config.yaml",
    ],
    default_model_aliases={
        "gpt-5.4": "copilot/gpt-5.4",
        "claude-opus-4.6": "copilot/claude-opus-4.6",
        "claude-sonnet-4.6": "copilot/claude-sonnet-4.6",
        "gpt-4.1": "copilot/gpt-4.1",
    },
    pre_invoke_hook=hyenas_pre_invoke,
    post_invoke_hook=hyenas_post_invoke,
    timeout=3600,
    port_base=3000,
)


def _parse_bridge_port(bridge_url: str) -> int:
    """Extract port number from bridge URL.

    Args:
        bridge_url: URL like ``http://localhost:9100/v1``.

    Returns:
        Port number, or _DEFAULT_BRIDGE_PORT if URL has no explicit port.
    """
    parsed = urlparse(bridge_url)
    return parsed.port or _DEFAULT_BRIDGE_PORT
