"""Tests for domain resource resolution utilities."""

from pathlib import Path

import pytest

from saber.domain.resources import resolve_redis_compose_path


class TestResolveRedisComposePath:
    """Tests for resolve_redis_compose_path()."""

    def test_resolve_redis_compose_path_returns_existing_path(self) -> None:
        """Resolved path should point to an existing file."""
        path = resolve_redis_compose_path()
        assert isinstance(path, Path)
        assert path.exists()
        assert path.name == "redis-compose.yml"

    def test_resolve_redis_compose_path_content_has_saber_redis_service(self) -> None:
        """The compose file must define the saber-redis service."""
        path = resolve_redis_compose_path()
        content = path.read_text(encoding="utf-8")
        assert "saber-redis:" in content

    def test_resolve_redis_compose_path_has_port_variable(self) -> None:
        """The compose file must reference SABER_REDIS_PORT for port mapping."""
        path = resolve_redis_compose_path()
        content = path.read_text(encoding="utf-8")
        assert "SABER_REDIS_PORT" in content

    def test_resolve_redis_compose_path_no_container_name(self) -> None:
        """The compose file must NOT include container_name (singleton risk)."""
        path = resolve_redis_compose_path()
        content = path.read_text(encoding="utf-8")
        assert "container_name" not in content

    def test_resolve_redis_compose_path_no_version_key(self) -> None:
        """The compose file must NOT include deprecated version key."""
        path = resolve_redis_compose_path()
        content = path.read_text(encoding="utf-8")
        # Ensure no top-level 'version:' key (deprecated in Docker Compose v2)
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("version:") or stripped.startswith("version :"):
                pytest.fail("Compose file should not contain deprecated 'version' key")

    def test_resolve_redis_compose_path_raises_when_resource_missing(self) -> None:
        """Should raise ResourceNotFoundError when package resource is missing."""
        from unittest.mock import patch

        from saber.domain.exceptions import ResourceNotFoundError

        with patch("saber.domain.resources.files", side_effect=ImportError("no module")):
            with pytest.raises(ResourceNotFoundError, match="redis-compose.yml"):
                resolve_redis_compose_path()
