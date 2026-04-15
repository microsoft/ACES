"""Tests for Hyenas Pydantic config models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from saber.agents.registry.firstparty.runtimes.hyenas_models import (
    HyenasConfig,
    HyenasEndpoint,
    HyenasFinding,
    HyenasStageAgentConfig,
)

# ── HyenasEndpoint ────────────────────────────────────────────────────


class TestHyenasEndpoint:
    def test_valid_construction(self) -> None:
        ep = HyenasEndpoint(
            label="saber-bridge",
            type="openai",
            url="http://localhost:9100/v1",
            models=["gpt-4o"],
        )
        assert ep.label == "saber-bridge"
        assert ep.type == "openai"
        assert ep.url == "http://localhost:9100/v1"
        assert ep.models == ["gpt-4o"]
        assert ep.key is None

    def test_with_key(self) -> None:
        ep = HyenasEndpoint(
            label="azure",
            type="azure",
            url="https://example.com/v1",
            key="sk-abc123",
            models=["gpt-4o"],
        )
        assert ep.key == "sk-abc123"

    def test_none_key_allowed(self) -> None:
        ep = HyenasEndpoint(
            label="bridge",
            type="openai",
            url="http://localhost:9100/v1",
            key=None,
            models=["gpt-4o"],
        )
        assert ep.key is None

    def test_empty_models_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HyenasEndpoint(
                label="bridge",
                type="openai",
                url="http://localhost:9100/v1",
                models=[],
            )

    def test_frozen(self) -> None:
        ep = HyenasEndpoint(
            label="bridge",
            type="openai",
            url="http://localhost:9100/v1",
            models=["gpt-4o"],
        )
        with pytest.raises(ValidationError):
            ep.label = "other"  # type: ignore[misc]

    def test_multiple_models(self) -> None:
        ep = HyenasEndpoint(
            label="bridge",
            type="openai",
            url="http://localhost:9100/v1",
            models=["gpt-4o", "claude-sonnet-4-20250514", "gpt-4.1"],
        )
        assert len(ep.models) == 3


# ── HyenasStageAgentConfig ────────────────────────────────────────────


class TestHyenasStageAgentConfig:
    def test_valid_construction(self) -> None:
        cfg = HyenasStageAgentConfig(models=["gpt-4o"])
        assert cfg.models == ["gpt-4o"]

    def test_multiple_models(self) -> None:
        cfg = HyenasStageAgentConfig(
            models=["claude-sonnet-4-20250514", "gpt-4o", "gpt-4.1"]
        )
        assert cfg.models == ["claude-sonnet-4-20250514", "gpt-4o", "gpt-4.1"]

    def test_frozen(self) -> None:
        cfg = HyenasStageAgentConfig(models=["gpt-4o"])
        with pytest.raises(ValidationError):
            cfg.models = ["other"]  # type: ignore[misc]


# ── HyenasConfig ──────────────────────────────────────────────────────


class TestHyenasConfig:
    @pytest.fixture()
    def minimal_config(self) -> HyenasConfig:
        return HyenasConfig(
            endpoints=[
                HyenasEndpoint(
                    label="bridge",
                    type="openai",
                    url="http://localhost:9100/v1",
                    models=["gpt-4o"],
                )
            ],
            default=HyenasStageAgentConfig(models=["gpt-4o"]),
        )

    def test_minimal_construction(self, minimal_config: HyenasConfig) -> None:
        assert len(minimal_config.endpoints) == 1
        assert minimal_config.default.models == ["gpt-4o"]
        assert minimal_config.scan_stage is None
        assert minimal_config.validate_stage is None
        assert minimal_config.prepare_stage is None
        assert minimal_config.prove_stage is None

    def test_kebab_case_aliases_in_dump(self) -> None:
        cfg = HyenasConfig(
            endpoints=[
                HyenasEndpoint(
                    label="bridge",
                    type="openai",
                    url="http://localhost:9100/v1",
                    models=["gpt-4o"],
                )
            ],
            default=HyenasStageAgentConfig(models=["gpt-4o"]),
            scan_stage={
                "function-auditor": HyenasStageAgentConfig(models=["gpt-4o"])
            },
            validate_stage={
                "debater": HyenasStageAgentConfig(
                    models=["claude-sonnet-4-20250514", "gpt-4o"]
                )
            },
        )
        dumped = cfg.model_dump(by_alias=True, exclude_none=True)
        assert "scan-stage" in dumped
        assert "scan_stage" not in dumped
        assert "validate-stage" in dumped
        assert "validate_stage" not in dumped

    def test_populate_by_name(self) -> None:
        """Construction with python field names works via populate_by_name."""
        cfg = HyenasConfig(
            endpoints=[
                HyenasEndpoint(
                    label="bridge",
                    type="openai",
                    url="http://localhost:9100/v1",
                    models=["gpt-4o"],
                )
            ],
            default=HyenasStageAgentConfig(models=["gpt-4o"]),
            prepare_stage={
                "default": HyenasStageAgentConfig(models=["gpt-4o"])
            },
        )
        assert cfg.prepare_stage is not None
        assert "default" in cfg.prepare_stage

    def test_construction_with_alias_keys(self) -> None:
        """Construction with kebab-case alias keys also works."""
        data = {
            "endpoints": [
                {
                    "label": "bridge",
                    "type": "openai",
                    "url": "http://localhost:9100/v1",
                    "models": ["gpt-4o"],
                }
            ],
            "default": {"models": ["gpt-4o"]},
            "scan-stage": {
                "function-auditor": {"models": ["gpt-4o"]}
            },
        }
        cfg = HyenasConfig(**data)
        assert cfg.scan_stage is not None

    def test_frozen(self, minimal_config: HyenasConfig) -> None:
        with pytest.raises(ValidationError):
            minimal_config.default = HyenasStageAgentConfig(models=["other"])  # type: ignore[misc]

    def test_exclude_none_omits_empty_stages(self) -> None:
        cfg = HyenasConfig(
            endpoints=[
                HyenasEndpoint(
                    label="bridge",
                    type="openai",
                    url="http://localhost:9100/v1",
                    models=["gpt-4o"],
                )
            ],
            default=HyenasStageAgentConfig(models=["gpt-4o"]),
        )
        dumped = cfg.model_dump(by_alias=True, exclude_none=True)
        assert "prepare-stage" not in dumped
        assert "scan-stage" not in dumped
        assert "validate-stage" not in dumped
        assert "prove-stage" not in dumped


# ── HyenasFinding ─────────────────────────────────────────────────────


class TestHyenasFinding:
    def test_minimal_construction(self) -> None:
        finding = HyenasFinding(file_path="/src/main.py")
        assert finding.file_path == "/src/main.py"
        assert finding.function_name is None
        assert finding.cwe_id is None
        assert finding.line_number is None
        assert finding.confidence is None
        assert finding.description is None

    def test_full_construction(self) -> None:
        finding = HyenasFinding(
            file_path="/src/main.py",
            function_name="process_input",
            cwe_id="CWE-79",
            line_number=42,
            confidence="CONFIRMED",
            description="XSS vulnerability in user input",
        )
        assert finding.function_name == "process_input"
        assert finding.cwe_id == "CWE-79"
        assert finding.line_number == 42
        assert finding.confidence == "CONFIRMED"
        assert finding.description == "XSS vulnerability in user input"

    def test_frozen(self) -> None:
        finding = HyenasFinding(file_path="/src/main.py")
        with pytest.raises(ValidationError):
            finding.file_path = "/other.py"  # type: ignore[misc]
