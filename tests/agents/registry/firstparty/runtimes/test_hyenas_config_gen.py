"""Tests for Hyenas config.yaml generator."""

from __future__ import annotations

import pytest
import yaml

from saber.agents.registry.firstparty.runtime_spec import AgentAlias
from saber.agents.registry.firstparty.runtimes.hyenas_config import (
    generate_hyenas_config,
)

SAMPLE_AGENTS = [
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
    AgentAlias(
        name="scope_resolver",
        model_alias="claude-sonnet-4-20250514",
        env_var="prepare-stage.default.models",
    ),
]


class TestGenerateHyenasConfigBasic:
    def test_single_agent_produces_valid_yaml(self) -> None:
        agents = [
            AgentAlias(
                name="auditor",
                model_alias="gpt-4o",
                env_var="scan-stage.function-auditor.models",
            ),
        ]
        result = generate_hyenas_config(agents, bridge_port=9100)
        parsed = yaml.safe_load(result)
        assert isinstance(parsed, dict)
        assert "endpoints" in parsed
        assert "default" in parsed

    def test_bridge_port_in_endpoint_url(self) -> None:
        agents = [
            AgentAlias(
                name="auditor",
                model_alias="gpt-4o",
                env_var="scan-stage.function-auditor.models",
            ),
        ]
        result = generate_hyenas_config(agents, bridge_port=8765)
        parsed = yaml.safe_load(result)
        assert parsed["endpoints"][0]["url"] == "http://localhost:8765/v1"

    def test_endpoint_label_and_type(self) -> None:
        agents = [
            AgentAlias(
                name="auditor",
                model_alias="gpt-4o",
                env_var="scan-stage.function-auditor.models",
            ),
        ]
        result = generate_hyenas_config(agents, bridge_port=9100)
        parsed = yaml.safe_load(result)
        ep = parsed["endpoints"][0]
        assert ep["label"] == "saber-bridge"
        assert ep["type"] == "openai"

    def test_key_absent_in_output(self) -> None:
        agents = [
            AgentAlias(
                name="auditor",
                model_alias="gpt-4o",
                env_var="scan-stage.function-auditor.models",
            ),
        ]
        result = generate_hyenas_config(agents, bridge_port=9100)
        parsed = yaml.safe_load(result)
        assert "key" not in parsed["endpoints"][0]


class TestGenerateHyenasConfigModels:
    def test_all_unique_models_in_endpoint(self) -> None:
        result = generate_hyenas_config(SAMPLE_AGENTS, bridge_port=9100)
        parsed = yaml.safe_load(result)
        endpoint_models = set(parsed["endpoints"][0]["models"])
        assert "gpt-4o" in endpoint_models
        assert "claude-sonnet-4-20250514" in endpoint_models
        assert "gpt-4.1" in endpoint_models

    def test_default_model_in_endpoint_models(self) -> None:
        result = generate_hyenas_config(
            SAMPLE_AGENTS, bridge_port=9100, default_model="special-model"
        )
        parsed = yaml.safe_load(result)
        assert "special-model" in parsed["endpoints"][0]["models"]

    def test_models_deduped(self) -> None:
        result = generate_hyenas_config(SAMPLE_AGENTS, bridge_port=9100)
        parsed = yaml.safe_load(result)
        models = parsed["endpoints"][0]["models"]
        assert len(models) == len(set(models))

    def test_default_models_entry(self) -> None:
        result = generate_hyenas_config(
            SAMPLE_AGENTS, bridge_port=9100, default_model="gpt-4o"
        )
        parsed = yaml.safe_load(result)
        assert parsed["default"]["models"] == ["gpt-4o"]

    def test_custom_default_model(self) -> None:
        result = generate_hyenas_config(
            SAMPLE_AGENTS, bridge_port=9100, default_model="my-custom-model"
        )
        parsed = yaml.safe_load(result)
        assert parsed["default"]["models"] == ["my-custom-model"]


class TestGenerateHyenasConfigStages:
    def test_stage_keys_are_kebab_case(self) -> None:
        result = generate_hyenas_config(SAMPLE_AGENTS, bridge_port=9100)
        parsed = yaml.safe_load(result)
        assert "scan-stage" in parsed
        assert "validate-stage" in parsed
        assert "prepare-stage" in parsed
        # Must NOT have snake_case keys
        assert "scan_stage" not in parsed
        assert "validate_stage" not in parsed
        assert "prepare_stage" not in parsed

    def test_debater_multi_model_grouped(self) -> None:
        """Three agents with same env_var path → single models list."""
        result = generate_hyenas_config(SAMPLE_AGENTS, bridge_port=9100)
        parsed = yaml.safe_load(result)
        debater_models = parsed["validate-stage"]["debater"]["models"]
        assert debater_models == ["claude-sonnet-4-20250514", "gpt-4o", "gpt-4.1"]

    def test_scan_stage_agents(self) -> None:
        result = generate_hyenas_config(SAMPLE_AGENTS, bridge_port=9100)
        parsed = yaml.safe_load(result)
        scan = parsed["scan-stage"]
        assert "function-auditor" in scan
        assert scan["function-auditor"]["models"] == ["gpt-4o"]
        assert "variant-auditor" in scan
        assert scan["variant-auditor"]["models"] == ["claude-sonnet-4-20250514"]
        assert "file-enricher" in scan
        assert scan["file-enricher"]["models"] == ["gpt-4o"]

    def test_prepare_stage(self) -> None:
        result = generate_hyenas_config(SAMPLE_AGENTS, bridge_port=9100)
        parsed = yaml.safe_load(result)
        assert parsed["prepare-stage"]["default"]["models"] == [
            "claude-sonnet-4-20250514"
        ]


class TestGenerateHyenasConfigErrors:
    def test_malformed_env_var_too_few_parts(self) -> None:
        agents = [
            AgentAlias(
                name="bad",
                model_alias="gpt-4o",
                env_var="scan-stage.models",
            ),
        ]
        with pytest.raises(ValueError, match="exactly 3 dot-separated parts"):
            generate_hyenas_config(agents, bridge_port=9100)

    def test_malformed_env_var_too_many_parts(self) -> None:
        agents = [
            AgentAlias(
                name="bad",
                model_alias="gpt-4o",
                env_var="scan-stage.agent.models.extra",
            ),
        ]
        with pytest.raises(ValueError, match="exactly 3 dot-separated parts"):
            generate_hyenas_config(agents, bridge_port=9100)

    def test_unknown_stage_name_raises(self) -> None:
        agents = [
            AgentAlias(
                name="bad",
                model_alias="gpt-4o",
                env_var="bogus-stage.agent.models",
            ),
        ]
        with pytest.raises(ValueError, match="Unknown stage"):
            generate_hyenas_config(agents, bridge_port=9100)


class TestGenerateHyenasConfigEdgeCases:
    def test_empty_agents_produces_valid_config(self) -> None:
        result = generate_hyenas_config([], bridge_port=9100)
        parsed = yaml.safe_load(result)
        assert "endpoints" in parsed
        assert "default" in parsed
        assert parsed["default"]["models"] == ["gpt-4o"]
        assert parsed["endpoints"][0]["models"] == ["gpt-4o"]

    def test_round_trip_yaml(self) -> None:
        result = generate_hyenas_config(SAMPLE_AGENTS, bridge_port=9100)
        parsed = yaml.safe_load(result)
        # Re-dump and re-parse
        redumped = yaml.dump(parsed, default_flow_style=False)
        reparsed = yaml.safe_load(redumped)
        assert parsed == reparsed

    def test_no_prove_stage_when_no_agents(self) -> None:
        result = generate_hyenas_config(SAMPLE_AGENTS, bridge_port=9100)
        parsed = yaml.safe_load(result)
        assert "prove-stage" not in parsed
