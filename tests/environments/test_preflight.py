"""Tests for compose preflight validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from saber.environments.preflight import (
    ComposePreflightResult,
    PreflightFinding,
    validate_compose_file,
    validate_domain,
    validate_network_references,
)

# ── Model Tests ──────────────────────────────────────────────────────


class TestPreflightFinding:
    """Tests for PreflightFinding model."""

    def test_construction(self) -> None:
        finding = PreflightFinding(
            path="compose/sandbox.compose.yml",
            message="Missing pids limit",
            severity="warning",
        )
        assert finding.path == "compose/sandbox.compose.yml"
        assert finding.message == "Missing pids limit"
        assert finding.severity == "warning"

    def test_frozen(self) -> None:
        finding = PreflightFinding(path="f.yml", message="msg", severity="error")
        with pytest.raises(ValidationError):
            finding.path = "other.yml"  # type: ignore[misc]

    def test_error_severity(self) -> None:
        finding = PreflightFinding(path="f.yml", message="bad", severity="error")
        assert finding.severity == "error"


class TestComposePreflightResult:
    """Tests for ComposePreflightResult model."""

    def test_empty_is_passing(self) -> None:
        result = ComposePreflightResult()
        assert result.passed is True
        assert result.errors == ()
        assert result.warnings == ()

    def test_with_error_fails(self) -> None:
        error = PreflightFinding(path="f.yml", message="bad YAML", severity="error")
        result = ComposePreflightResult(findings=(error,))
        assert result.passed is False
        assert len(result.errors) == 1
        assert result.warnings == ()

    def test_with_only_warnings_passes(self) -> None:
        warn = PreflightFinding(path="f.yml", message="no pids", severity="warning")
        result = ComposePreflightResult(findings=(warn,))
        assert result.passed is True
        assert result.errors == ()
        assert len(result.warnings) == 1

    def test_mixed_findings(self) -> None:
        error = PreflightFinding(path="f.yml", message="bad", severity="error")
        warn = PreflightFinding(path="f.yml", message="no pids", severity="warning")
        result = ComposePreflightResult(findings=(error, warn))
        assert result.passed is False
        assert len(result.errors) == 1
        assert len(result.warnings) == 1

    def test_frozen(self) -> None:
        result = ComposePreflightResult()
        with pytest.raises(ValidationError):
            result.findings = ()  # type: ignore[misc]


# ── validate_compose_file Tests ──────────────────────────────────────


class TestValidateComposeFile:
    """Tests for validate_compose_file()."""

    def test_valid_file_no_findings(self, tmp_path: Path) -> None:
        """Valid compose with pids limits on all services → no findings."""
        compose = tmp_path / "compose" / "sandbox.compose.yml"
        compose.parent.mkdir(parents=True)
        compose.write_text(
            "services:\n  web:\n    image: nginx\n    deploy:\n      resources:\n        limits:\n          pids: 256\n"
        )
        findings = validate_compose_file(compose, tmp_path)
        assert findings == []

    def test_invalid_yaml_gives_error(self, tmp_path: Path) -> None:
        """Broken YAML → one error finding."""
        compose = tmp_path / "bad.yml"
        compose.write_text("services:\n  web:\n    - broken: [unterminated")
        findings = validate_compose_file(compose, tmp_path)
        assert len(findings) == 1
        assert findings[0].severity == "error"
        assert "YAML" in findings[0].message or "parse" in findings[0].message.lower()

    def test_missing_services_key_gives_error(self, tmp_path: Path) -> None:
        """Valid YAML but no 'services' key → error."""
        compose = tmp_path / "no_services.yml"
        compose.write_text("version: '3'\nnetworks:\n  default:\n")
        findings = validate_compose_file(compose, tmp_path)
        assert len(findings) == 1
        assert findings[0].severity == "error"
        assert "services" in findings[0].message.lower()

    def test_empty_file_gives_error(self, tmp_path: Path) -> None:
        """Empty YAML file → error."""
        compose = tmp_path / "empty.yml"
        compose.write_text("")
        findings = validate_compose_file(compose, tmp_path)
        assert len(findings) == 1
        assert findings[0].severity == "error"

    def test_missing_pids_gives_warning(self, tmp_path: Path) -> None:
        """Service without deploy.resources.limits.pids → warning."""
        compose = tmp_path / "no_pids.yml"
        compose.write_text("services:\n  web:\n    image: nginx\n")
        findings = validate_compose_file(compose, tmp_path)
        assert len(findings) == 1
        assert findings[0].severity == "warning"
        assert "pids" in findings[0].message.lower()

    def test_multiple_services_mixed_pids(self, tmp_path: Path) -> None:
        """One service with pids, one without → exactly one warning."""
        compose = tmp_path / "mixed.yml"
        compose.write_text(
            "services:\n"
            "  with_pids:\n"
            "    image: nginx\n"
            "    deploy:\n"
            "      resources:\n"
            "        limits:\n"
            "          pids: 256\n"
            "  without_pids:\n"
            "    image: redis\n"
        )
        findings = validate_compose_file(compose, tmp_path)
        assert len(findings) == 1
        assert findings[0].severity == "warning"
        assert "without_pids" in findings[0].message

    def test_permanent_skips_pids_check(self, tmp_path: Path) -> None:
        """is_permanent=True → no pids warning even when missing."""
        compose = tmp_path / "permanent.yml"
        compose.write_text("services:\n  db:\n    image: mysql\n")
        findings = validate_compose_file(compose, tmp_path, is_permanent=True)
        # Should have no findings — pids check is skipped for permanent services
        assert findings == []

    def test_relative_path_in_finding(self, tmp_path: Path) -> None:
        """Finding path should be relative to domain_root."""
        compose = tmp_path / "compose" / "sandbox.compose.yml"
        compose.parent.mkdir(parents=True)
        compose.write_text("services:\n  web:\n    image: nginx\n")
        findings = validate_compose_file(compose, tmp_path)
        assert findings[0].path == "compose/sandbox.compose.yml"

    def test_deploy_null_value_no_crash(self, tmp_path: Path) -> None:
        """deploy: with null value doesn't crash, still warns about pids."""
        compose = tmp_path / "null_deploy.yml"
        compose.write_text("services:\n  web:\n    image: nginx\n    deploy:\n")
        findings = validate_compose_file(compose, tmp_path)
        assert len(findings) == 1
        assert findings[0].severity == "warning"
        assert "pids" in findings[0].message.lower()

    def test_services_null_value_no_findings(self, tmp_path: Path) -> None:
        """services: with null value produces no findings (no services to check)."""
        compose = tmp_path / "null_services.yml"
        compose.write_text("services:\n")
        findings = validate_compose_file(compose, tmp_path)
        assert findings == []


# ── validate_network_references Tests ────────────────────────────────


class TestValidateNetworkReferences:
    """Tests for validate_network_references()."""

    def test_matching_networks_no_findings(self, tmp_path: Path) -> None:
        """Sandbox references ${SABER_PROJECT}_mynet, permanent defines mynet → pass."""
        sandbox = tmp_path / "sandbox.yml"
        sandbox.write_text(
            "services:\n"
            "  web:\n"
            "    image: nginx\n"
            "networks:\n"
            "  mynet:\n"
            "    external: true\n"
            "    name: ${SABER_PROJECT}_mynet\n"
        )
        permanent = tmp_path / "permanent.yml"
        permanent.write_text("services:\n  db:\n    image: mysql\nnetworks:\n  mynet:\n    driver: bridge\n")
        findings = validate_network_references(sandbox, permanent, tmp_path)
        assert findings == []

    def test_unmatched_network_gives_error(self, tmp_path: Path) -> None:
        """Sandbox references ${SABER_PROJECT}_missing but permanent doesn't define it → error."""
        sandbox = tmp_path / "sandbox.yml"
        sandbox.write_text(
            "services:\n"
            "  web:\n"
            "    image: nginx\n"
            "networks:\n"
            "  missing:\n"
            "    external: true\n"
            "    name: ${SABER_PROJECT}_missing\n"
        )
        permanent = tmp_path / "permanent.yml"
        permanent.write_text("services:\n  db:\n    image: mysql\nnetworks:\n  other_net:\n    driver: bridge\n")
        findings = validate_network_references(sandbox, permanent, tmp_path)
        assert len(findings) == 1
        assert findings[0].severity == "error"
        assert "missing" in findings[0].message

    def test_no_external_networks_no_findings(self, tmp_path: Path) -> None:
        """Sandbox with no networks section → no findings."""
        sandbox = tmp_path / "sandbox.yml"
        sandbox.write_text("services:\n  web:\n    image: nginx\n")
        permanent = tmp_path / "permanent.yml"
        permanent.write_text("services:\n  db:\n    image: mysql\n")
        findings = validate_network_references(sandbox, permanent, tmp_path)
        assert findings == []

    def test_non_saber_external_network_skipped(self, tmp_path: Path) -> None:
        """External network without ${SABER_PROJECT}_ prefix → skipped."""
        sandbox = tmp_path / "sandbox.yml"
        sandbox.write_text(
            "services:\n"
            "  web:\n"
            "    image: nginx\n"
            "networks:\n"
            "  external_net:\n"
            "    external: true\n"
            "    name: some_other_network\n"
        )
        permanent = tmp_path / "permanent.yml"
        permanent.write_text("services:\n  db:\n    image: mysql\n")
        findings = validate_network_references(sandbox, permanent, tmp_path)
        assert findings == []

    def test_missing_permanent_compose_gives_error(self, tmp_path: Path) -> None:
        """permanent_path doesn't exist → error finding."""
        sandbox = tmp_path / "sandbox.yml"
        sandbox.write_text(
            "services:\n"
            "  web:\n"
            "    image: nginx\n"
            "networks:\n"
            "  mynet:\n"
            "    external: true\n"
            "    name: ${SABER_PROJECT}_mynet\n"
        )
        nonexistent = tmp_path / "missing_permanent.yml"
        findings = validate_network_references(sandbox, nonexistent, tmp_path)
        assert len(findings) == 1
        assert findings[0].severity == "error"

    def test_permanent_compose_parse_failure(self, tmp_path: Path) -> None:
        """Permanent compose exists but is malformed YAML → error finding."""
        sandbox = tmp_path / "sandbox.yml"
        sandbox.write_text(
            "services:\n"
            "  web:\n"
            "    image: nginx\n"
            "networks:\n"
            "  mynet:\n"
            "    external: true\n"
            "    name: ${SABER_PROJECT}_mynet\n"
        )
        permanent = tmp_path / "permanent.yml"
        permanent.write_text("services:\n  db:\n    - broken: [unterminated")
        findings = validate_network_references(sandbox, permanent, tmp_path)
        assert len(findings) == 1
        assert findings[0].severity == "error"
        assert "parse" in findings[0].message.lower() or "Failed" in findings[0].message

    def test_networks_as_list_no_crash(self, tmp_path: Path) -> None:
        """networks: as a list instead of dict → no crash, no findings."""
        sandbox = tmp_path / "sandbox.yml"
        sandbox.write_text("services:\n  web:\n    image: nginx\nnetworks:\n  - default\n")
        permanent = tmp_path / "permanent.yml"
        permanent.write_text("services:\n  db:\n    image: mysql\n")
        findings = validate_network_references(sandbox, permanent, tmp_path)
        assert findings == []

    def test_multiple_networks_partial_match(self, tmp_path: Path) -> None:
        """Two sandbox networks: one matches permanent, one doesn't → one error."""
        sandbox = tmp_path / "sandbox.yml"
        sandbox.write_text(
            "services:\n"
            "  web:\n"
            "    image: nginx\n"
            "networks:\n"
            "  good:\n"
            "    external: true\n"
            "    name: ${SABER_PROJECT}_good\n"
            "  bad:\n"
            "    external: true\n"
            "    name: ${SABER_PROJECT}_bad\n"
        )
        permanent = tmp_path / "permanent.yml"
        permanent.write_text("services:\n  db:\n    image: mysql\nnetworks:\n  good:\n    driver: bridge\n")
        findings = validate_network_references(sandbox, permanent, tmp_path)
        assert len(findings) == 1
        assert findings[0].severity == "error"
        assert "bad" in findings[0].message


# ── validate_domain Tests ────────────────────────────────────────────


class TestValidateDomain:
    """Tests for validate_domain() orchestrator."""

    def test_valid_domain_passes(self, tmp_path: Path) -> None:
        """Domain with valid sandbox compose → passed."""
        compose_dir = tmp_path / "compose"
        compose_dir.mkdir()
        sandbox = compose_dir / "sandbox.compose.yml"
        sandbox.write_text(
            "services:\n  web:\n    image: nginx\n    deploy:\n      resources:\n        limits:\n          pids: 256\n"
        )
        result = validate_domain(tmp_path)
        assert result.passed is True
        assert result.findings == ()

    def test_missing_sandbox_gives_error(self, tmp_path: Path) -> None:
        """Missing sandbox compose file → error."""
        result = validate_domain(tmp_path)
        assert result.passed is False
        assert len(result.errors) == 1
        assert "not found" in result.errors[0].message.lower()

    def test_aggregates_sandbox_and_network_findings(self, tmp_path: Path) -> None:
        """Sandbox with missing pids + unmatched network → warning + error."""
        compose_dir = tmp_path / "compose"
        compose_dir.mkdir()
        sandbox = compose_dir / "sandbox.compose.yml"
        sandbox.write_text(
            "services:\n"
            "  web:\n"
            "    image: nginx\n"
            "networks:\n"
            "  mynet:\n"
            "    external: true\n"
            "    name: ${SABER_PROJECT}_mynet\n"
        )
        permanent = compose_dir / "permanent.compose.yml"
        permanent.write_text("services:\n  db:\n    image: mysql\n")
        result = validate_domain(
            tmp_path,
            permanent_compose="compose/permanent.compose.yml",
        )
        assert len(result.warnings) >= 1  # pids warning
        assert len(result.errors) >= 1  # network mismatch

    def test_no_permanent_skips_network_check(self, tmp_path: Path) -> None:
        """permanent_compose=None → only sandbox validated, no network check."""
        compose_dir = tmp_path / "compose"
        compose_dir.mkdir()
        sandbox = compose_dir / "sandbox.compose.yml"
        sandbox.write_text(
            "services:\n"
            "  web:\n"
            "    image: nginx\n"
            "    deploy:\n"
            "      resources:\n"
            "        limits:\n"
            "          pids: 256\n"
            "networks:\n"
            "  mynet:\n"
            "    external: true\n"
            "    name: ${SABER_PROJECT}_mynet\n"
        )
        result = validate_domain(tmp_path)
        assert result.passed is True  # No errors (network check skipped)
