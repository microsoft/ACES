"""HYENAS_RUNTIME definition and hook functions."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse

from saber.agents.registry.firstparty.runtime_spec import (
    AgentAlias,
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

    config_yaml = generate_hyenas_config(
        agents=HYENAS_RUNTIME.agents,
        bridge_port=bridge_port,
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
        defaults={
            "HYENAS_HEADLESS": "true",
            "LOG_LEVEL": "info",
        },
    ),
    agents=[
        # Stage-level defaults (required for --no-copilot validation).
        # Hyenas' verifyCopilotFreeRouting() checks stage.default.models
        # entries against configured endpoints; without these, hardcoded
        # defaults (gpt-5.4, claude-opus-4.6) would fail validation.
        AgentAlias(
            name="scan_default",
            model_alias="gpt-4o",
            env_var="scan-stage.default.models",
        ),
        AgentAlias(
            name="validate_default",
            model_alias="gpt-4o",
            env_var="validate-stage.default.models",
        ),
        AgentAlias(
            name="prove_default",
            model_alias="gpt-4o",
            env_var="prove-stage.default.models",
        ),
        # Scan stage — per-agent overrides
        AgentAlias(
            name="function_auditor",
            model_alias="gpt-4o",
            env_var="scan-stage.function-auditor.models",
        ),
        AgentAlias(
            name="variant_auditor",
            model_alias="claude-sonnet-4-20250514",
            env_var="scan-stage.variant-auditor.models",
        ),
        AgentAlias(
            name="file_enricher",
            model_alias="gpt-4o",
            env_var="scan-stage.file-enricher.models",
        ),
        # Validate stage — debater uses 3 models for consensus
        AgentAlias(
            name="debater_1",
            model_alias="claude-sonnet-4-20250514",
            env_var="validate-stage.debater.models",
        ),
        AgentAlias(
            name="debater_2",
            model_alias="gpt-4o",
            env_var="validate-stage.debater.models",
        ),
        AgentAlias(
            name="debater_3",
            model_alias="gpt-4.1",
            env_var="validate-stage.debater.models",
        ),
        # Prepare stage
        AgentAlias(
            name="scope_resolver",
            model_alias="claude-sonnet-4-20250514",
            env_var="prepare-stage.default.models",
        ),
    ],
    invoke_command=[
        "node",
        "dist/src/cli.js",
        "run",
        "--headless",
        "--no-copilot",
        "--no-prove",
        "--repo",
        "/workspace",
        "--hyenas",
        "/output/.hyenas",
        "--config",
        "/app/.hyenas-config/config.yaml",
    ],
    default_model_aliases={
        "gpt-4o": "copilot/gpt-4o",
        "claude-sonnet-4-20250514": "copilot/claude-sonnet-4-20250514",
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
