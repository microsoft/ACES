"""Tests for saber.cli.discovery — domain discovery utilities."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from saber.cli.discovery import discover_domains, find_domains_root, resolve_domain


class TestFindDomainsRoot:
    """Tests for find_domains_root()."""

    def test_finds_parent_domains_dir(self, tmp_path: Path) -> None:
        """Finds domains/ when CWD is inside the workspace."""
        domains = tmp_path / "domains" / "test_domain"
        domains.mkdir(parents=True)
        (domains / "eval.yaml").write_text(yaml.dump({"slug": "test", "name": "Test", "description": "d"}))
        result = find_domains_root(start=tmp_path / "domains" / "test_domain")
        assert result == tmp_path / "domains"

    def test_finds_from_workspace_root(self, tmp_path: Path) -> None:
        """Finds domains/ when CWD is the workspace root."""
        domains = tmp_path / "domains" / "test_domain"
        domains.mkdir(parents=True)
        (domains / "eval.yaml").write_text(yaml.dump({"slug": "test", "name": "Test", "description": "d"}))
        result = find_domains_root(start=tmp_path)
        assert result == tmp_path / "domains"

    def test_raises_when_not_found(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when no domains/ exists."""
        with pytest.raises(FileNotFoundError, match="No domains/ directory"):
            find_domains_root(start=tmp_path)

    def test_ignores_domains_without_eval_yaml(self, tmp_path: Path) -> None:
        """domains/ with only empty dirs doesn't qualify."""
        (tmp_path / "domains" / "empty").mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            find_domains_root(start=tmp_path)


class TestDiscoverDomains:
    """Tests for discover_domains()."""

    @staticmethod
    def _write_domain(
        domains_root: Path,
        slug: str,
        name: str = "Test",
        permanent_env: str | None = None,
    ) -> Path:
        """Write a minimal domain with eval.yaml and optional global.yaml."""
        domain_dir = domains_root / slug
        domain_dir.mkdir(parents=True, exist_ok=True)
        (domain_dir / "eval.yaml").write_text(yaml.dump({"slug": slug, "name": name, "description": "d"}))
        if permanent_env is not None:
            tasks_dir = domain_dir / "tasks"
            tasks_dir.mkdir(exist_ok=True)
            compose_rel = f"compose/{permanent_env}.compose.yml"
            (tasks_dir / "global.yaml").write_text(
                yaml.dump(
                    {
                        "permanent_environment": {
                            "name": f"{slug}-databases",
                            "compose": compose_rel,
                        }
                    }
                )
            )
            compose_dir = domain_dir / "compose"
            compose_dir.mkdir(exist_ok=True)
            (compose_dir / f"{permanent_env}.compose.yml").write_text("services: {}")
        return domain_dir

    def test_returns_sorted_list(self, tmp_path: Path) -> None:
        """Returns domains sorted by slug."""
        self._write_domain(tmp_path, "zebra", "Zebra Domain")
        self._write_domain(tmp_path, "alpha", "Alpha Domain")
        result = discover_domains(tmp_path)
        assert len(result) == 2
        assert result[0].slug == "alpha"
        assert result[1].slug == "zebra"

    def test_skips_dirs_without_eval_yaml(self, tmp_path: Path) -> None:
        """Directories without eval.yaml are skipped."""
        self._write_domain(tmp_path, "valid")
        (tmp_path / "no_eval").mkdir()
        result = discover_domains(tmp_path)
        assert len(result) == 1
        assert result[0].slug == "valid"

    def test_resolves_permanent_compose(self, tmp_path: Path) -> None:
        """Resolves permanent compose from global.yaml."""
        self._write_domain(tmp_path, "mydom", permanent_env="databases")
        result = discover_domains(tmp_path)
        assert len(result) == 1
        assert result[0].permanent_compose is not None
        assert result[0].permanent_compose.name == "databases.compose.yml"
        assert result[0].permanent_project == "mydom-databases"

    def test_no_permanent_when_missing(self, tmp_path: Path) -> None:
        """No permanent compose when global.yaml lacks permanent_environment."""
        self._write_domain(tmp_path, "plain")
        result = discover_domains(tmp_path)
        assert len(result) == 1
        assert result[0].permanent_compose is None
        assert result[0].permanent_project is None

    def test_no_permanent_when_compose_file_missing(self, tmp_path: Path) -> None:
        """No permanent compose when the compose file doesn't exist."""
        domain_dir = tmp_path / "broken"
        domain_dir.mkdir()
        (domain_dir / "eval.yaml").write_text(yaml.dump({"slug": "broken", "name": "Broken", "description": "d"}))
        tasks_dir = domain_dir / "tasks"
        tasks_dir.mkdir()
        (tasks_dir / "global.yaml").write_text(
            yaml.dump(
                {
                    "permanent_environment": {
                        "name": "broken-permanent",
                        "compose": "compose/nonexistent.compose.yml",
                    }
                }
            )
        )
        result = discover_domains(tmp_path)
        assert result[0].permanent_compose is None

    def test_raises_on_invalid_eval_yaml(self, tmp_path: Path) -> None:
        """Domains with unparseable eval.yaml raise ValueError."""
        # Write a valid domain
        self._write_domain(tmp_path, "valid", "Valid Domain")
        # Write a domain with broken YAML that will cause load_domain_config to raise
        broken_dir = tmp_path / "broken"
        broken_dir.mkdir()
        (broken_dir / "eval.yaml").write_text(": invalid: yaml: [")

        with pytest.raises(ValueError, match="invalid eval.yaml"):
            discover_domains(tmp_path)


class TestResolveDomain:
    """Tests for resolve_domain()."""

    def test_finds_by_slug(self, tmp_path: Path) -> None:
        """Resolves domain by slug."""
        domain_dir = tmp_path / "mydom"
        domain_dir.mkdir()
        (domain_dir / "eval.yaml").write_text(yaml.dump({"slug": "mydom", "name": "My Dom", "description": "d"}))
        result = resolve_domain(tmp_path, "mydom")
        assert result.slug == "mydom"

    def test_finds_by_dir_name(self, tmp_path: Path) -> None:
        """Resolves domain by directory name when slug differs."""
        domain_dir = tmp_path / "dir_name"
        domain_dir.mkdir()
        (domain_dir / "eval.yaml").write_text(yaml.dump({"slug": "different_slug", "name": "X", "description": "d"}))
        result = resolve_domain(tmp_path, "dir_name")
        assert result.slug == "different_slug"

    def test_raises_bad_parameter_for_unknown(self, tmp_path: Path) -> None:
        """Raises typer.BadParameter for unknown domain."""
        import typer

        domain_dir = tmp_path / "existing"
        domain_dir.mkdir()
        (domain_dir / "eval.yaml").write_text(yaml.dump({"slug": "existing", "name": "X", "description": "d"}))
        with pytest.raises(typer.BadParameter, match="not found"):
            resolve_domain(tmp_path, "nonexistent")
