"""Pydantic models for the Hyenas config.yaml schema (BYOK endpoint config)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HyenasEndpoint(BaseModel):
    """A BYOK endpoint entry in Hyenas config.yaml.

    Attributes:
        label: Human-readable label for the endpoint.
        type: Provider type (e.g. "openai", "anthropic", "azure").
        url: Base URL of the endpoint.
        key: API key, or None for zero-auth (http://) endpoints.
        models: List of model names available at this endpoint.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    type: str
    url: str
    key: str | None = None
    models: list[str]

    @field_validator("models")
    @classmethod
    def models_must_not_be_empty(cls, v: list[str]) -> list[str]:
        """Validate that models list is non-empty."""
        if not v:
            msg = "models list must not be empty"
            raise ValueError(msg)
        return v


class HyenasStageAgentConfig(BaseModel):
    """Per-agent config within a Hyenas stage.

    Attributes:
        models: List of model names assigned to this agent.
    """

    model_config = ConfigDict(frozen=True)

    models: list[str]


class HyenasConfig(BaseModel):
    """Root Hyenas config.yaml structure.

    Uses kebab-case aliases for stage fields to match Hyenas YAML format.
    Construct with python field names (populate_by_name=True) and serialize
    with ``model_dump(by_alias=True)`` to produce kebab-case keys.

    Attributes:
        endpoints: List of BYOK endpoint definitions.
        default: Default agent config (root-level models).
        prepare_stage: Agents in the prepare stage.
        scan_stage: Agents in the scan stage.
        validate_stage: Agents in the validate stage.
        prove_stage: Agents in the prove stage.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    endpoints: list[HyenasEndpoint]
    default: HyenasStageAgentConfig
    prepare_stage: dict[str, HyenasStageAgentConfig] | None = Field(
        default=None, alias="prepare-stage"
    )
    scan_stage: dict[str, HyenasStageAgentConfig] | None = Field(
        default=None, alias="scan-stage"
    )
    validate_stage: dict[str, HyenasStageAgentConfig] | None = Field(
        default=None, alias="validate-stage"
    )
    prove_stage: dict[str, HyenasStageAgentConfig] | None = Field(
        default=None, alias="prove-stage"
    )


class HyenasFinding(BaseModel):
    """A single finding from Hyenas scan output.

    Attributes:
        file_path: Path to the file containing the finding.
        function_name: Name of the function where the finding was detected.
        cwe_id: CWE identifier (e.g. "CWE-79").
        line_number: Line number in the source file.
        confidence: Confidence level ("UNLIKELY", "SUSPECT", "LIKELY", "CONFIRMED").
        description: Human-readable description of the finding.
    """

    model_config = ConfigDict(frozen=True)

    file_path: str
    function_name: str | None = None
    cwe_id: str | None = None
    line_number: int | None = None
    confidence: str | None = None
    description: str | None = None
