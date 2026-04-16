"""Tests for Hyenas config.yaml generator."""

from __future__ import annotations

import yaml

from saber.agents.registry.firstparty.runtimes.hyenas_config import (
    generate_hyenas_config,
)

SAMPLE_MODELS = ["gpt-5.4", "claude-opus-4.6", "claude-sonnet-4.6", "gpt-4.1"]


class TestGenerateHyenasConfigBasic:
    def test_produces_valid_yaml(self) -> None:
        result = generate_hyenas_config(
            bridge_port=9100, endpoint_models=SAMPLE_MODELS
        )
        parsed = yaml.safe_load(result)
        assert isinstance(parsed, dict)
        assert "endpoints" in parsed
        assert "default" in parsed

    def test_bridge_port_in_endpoint_url(self) -> None:
        result = generate_hyenas_config(
            bridge_port=8765, endpoint_models=SAMPLE_MODELS
        )
        parsed = yaml.safe_load(result)
        assert parsed["endpoints"][0]["url"] == "http://localhost:8765/v1"

    def test_endpoint_label_and_type(self) -> None:
        result = generate_hyenas_config(
            bridge_port=9100, endpoint_models=SAMPLE_MODELS
        )
        parsed = yaml.safe_load(result)
        ep = parsed["endpoints"][0]
        assert ep["label"] == "saber-bridge"
        assert ep["type"] == "openai"

    def test_key_absent_in_output(self) -> None:
        result = generate_hyenas_config(
            bridge_port=9100, endpoint_models=SAMPLE_MODELS
        )
        parsed = yaml.safe_load(result)
        assert "key" not in parsed["endpoints"][0]

    def test_no_stage_overrides(self) -> None:
        """Config must not contain any stage-level override keys."""
        result = generate_hyenas_config(
            bridge_port=9100, endpoint_models=SAMPLE_MODELS
        )
        parsed = yaml.safe_load(result)
        for key in ("prepare-stage", "scan-stage", "validate-stage", "prove-stage"):
            assert key not in parsed


class TestGenerateHyenasConfigModels:
    def test_all_endpoint_models_present(self) -> None:
        result = generate_hyenas_config(
            bridge_port=9100, endpoint_models=SAMPLE_MODELS
        )
        parsed = yaml.safe_load(result)
        ep_models = set(parsed["endpoints"][0]["models"])
        for m in SAMPLE_MODELS:
            assert m in ep_models

    def test_default_model_in_endpoint_models(self) -> None:
        result = generate_hyenas_config(
            bridge_port=9100,
            endpoint_models=SAMPLE_MODELS,
            default_model="special-model",
        )
        parsed = yaml.safe_load(result)
        assert "special-model" in parsed["endpoints"][0]["models"]

    def test_models_deduped(self) -> None:
        result = generate_hyenas_config(
            bridge_port=9100, endpoint_models=SAMPLE_MODELS
        )
        parsed = yaml.safe_load(result)
        models = parsed["endpoints"][0]["models"]
        assert len(models) == len(set(models))

    def test_default_models_entry(self) -> None:
        result = generate_hyenas_config(
            bridge_port=9100, endpoint_models=SAMPLE_MODELS
        )
        parsed = yaml.safe_load(result)
        assert parsed["default"]["models"] == ["gpt-5.4"]

    def test_custom_default_model(self) -> None:
        result = generate_hyenas_config(
            bridge_port=9100,
            endpoint_models=SAMPLE_MODELS,
            default_model="my-custom-model",
        )
        parsed = yaml.safe_load(result)
        assert parsed["default"]["models"] == ["my-custom-model"]


class TestGenerateHyenasConfigEdgeCases:
    def test_empty_models_produces_valid_config(self) -> None:
        result = generate_hyenas_config(bridge_port=9100, endpoint_models=[])
        parsed = yaml.safe_load(result)
        assert "endpoints" in parsed
        assert "default" in parsed
        assert parsed["default"]["models"] == ["gpt-5.4"]
        assert parsed["endpoints"][0]["models"] == ["gpt-5.4"]

    def test_round_trip_yaml(self) -> None:
        result = generate_hyenas_config(
            bridge_port=9100, endpoint_models=SAMPLE_MODELS
        )
        parsed = yaml.safe_load(result)
        # Re-dump and re-parse
        redumped = yaml.dump(parsed, default_flow_style=False)
        reparsed = yaml.safe_load(redumped)
        assert parsed == reparsed


class TestGenerateHyenasConfigExtraConfig:
    """Tests for the extra_config deep-merge support."""

    def test_extra_config_none_is_noop(self) -> None:
        baseline = generate_hyenas_config(
            bridge_port=9100, endpoint_models=SAMPLE_MODELS, extra_config=None
        )
        without = generate_hyenas_config(
            bridge_port=9100, endpoint_models=SAMPLE_MODELS
        )
        assert baseline == without

    def test_extra_config_adds_top_level_key(self) -> None:
        result = generate_hyenas_config(
            bridge_port=9100,
            endpoint_models=SAMPLE_MODELS,
            extra_config={"prove-stage": {"models": ["gpt-5.4"]}},
        )
        parsed = yaml.safe_load(result)
        assert parsed["prove-stage"] == {"models": ["gpt-5.4"]}

    def test_extra_config_deep_merges_nested(self) -> None:
        result = generate_hyenas_config(
            bridge_port=9100,
            endpoint_models=SAMPLE_MODELS,
            extra_config={"default": {"temperature": 0.5}},
        )
        parsed = yaml.safe_load(result)
        # Original key preserved
        assert parsed["default"]["models"] == ["gpt-5.4"]
        # New key merged in
        assert parsed["default"]["temperature"] == 0.5

    def test_extra_config_overwrites_scalar(self) -> None:
        result = generate_hyenas_config(
            bridge_port=9100,
            endpoint_models=SAMPLE_MODELS,
            extra_config={"default": {"models": ["override-model"]}},
        )
        parsed = yaml.safe_load(result)
        assert parsed["default"]["models"] == ["override-model"]
