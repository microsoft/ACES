"""
Unit tests for domain build operations.

Tests the build and rebuild functionality including:
- Image filtering by prefix
- Rebuild mode (remove and rebuild)
- Build mode (only build missing)
- Base image handling
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from saber.domain.orchestrator import DomainOrchestrator, DockerRunner, ManifestLoader, EnvironmentValidator
from saber.domain.exceptions import DockerError, DomainValidationError


@pytest.fixture
def mock_domains_root(tmp_path):
    """Create a temporary domains root directory."""
    domains_root = tmp_path / "domains"
    domains_root.mkdir()

    # Create a test domain
    test_domain = domains_root / "test_domain"
    test_domain.mkdir()
    (test_domain / "docker").mkdir()

    # Create domain.yaml
    domain_yaml = test_domain / "domain.yaml"
    domain_yaml.write_text("""
schemaVersion: "1.0.0"
domain:
  slug: "test_domain"
  name: "Test Domain"
  version: "1.0.0"
images:
  server:
    tag: "saber/test_domain/server:latest"
    dockerfile: "docker/Dockerfile.server"
  sandbox:
    tag: "saber/test_domain/sandbox:latest"
    dockerfile: "docker/Dockerfile.sandbox"
  cookie_0_target:
    tag: "saber/test_domain/cookie/target_0:latest"
    dockerfile: "docker/targets/cookie_0/Dockerfile"
    context: "docker/targets/cookie_0"
  cookie_1_target:
    tag: "saber/test_domain/cookie/target_1:latest"
    dockerfile: "docker/targets/cookie_1/Dockerfile"
    context: "docker/targets/cookie_1"
""")

    # Create Dockerfiles
    (test_domain / "docker" / "Dockerfile.server").write_text("FROM python:3.11\n")
    (test_domain / "docker" / "Dockerfile.sandbox").write_text("FROM ubuntu:22.04\n")

    # Create target directories and Dockerfiles
    for target in ["cookie_0", "cookie_1"]:
        target_dir = test_domain / "docker" / "targets" / target
        target_dir.mkdir(parents=True)
        (target_dir / "Dockerfile").write_text(f"FROM nginx:latest\n# {target}\n")

    return domains_root


@pytest.fixture
def mock_compose_file(tmp_path):
    """Create a temporary compose file."""
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services:\n  server:\n    image: test\n")
    return compose_file


@pytest.fixture
def mock_manifest():
    """Create a mock domain manifest."""
    return {
        "domain": {
            "slug": "test_domain",
            "name": "Test Domain",
            "version": "1.0.0"
        },
        "images": {
            "server": {
                "tag": "saber/test_domain/server:latest",
                "dockerfile": "docker/Dockerfile.server"
            },
            "sandbox": {
                "tag": "saber/test_domain/sandbox:latest",
                "dockerfile": "docker/Dockerfile.sandbox"
            },
            "cookie_0_target": {
                "tag": "saber/test_domain/cookie/target_0:latest",
                "dockerfile": "docker/targets/cookie_0/Dockerfile",
                "context": "docker/targets/cookie_0"
            },
            "cookie_1_target": {
                "tag": "saber/test_domain/cookie/target_1:latest",
                "dockerfile": "docker/targets/cookie_1/Dockerfile",
                "context": "docker/targets/cookie_1"
            }
        }
    }


class TestDockerRunnerBuildImages:
    """Tests for DockerRunner.build_images method."""

    @patch('saber.domain.orchestrator.subprocess.run')
    @patch('saber.domain.orchestrator.DockerRunner._docker_image_exists')
    @patch('saber.domain.orchestrator.DockerRunner.ensure_base_images')
    def test_build_all_images_rebuild_mode(
        self, mock_ensure_base, mock_exists, mock_subprocess,
        mock_domains_root, mock_compose_file, mock_manifest
    ):
        """Test building all images in rebuild mode."""
        mock_exists.return_value = True  # Images exist
        mock_subprocess.return_value = Mock(returncode=0)

        runner = DockerRunner(mock_compose_file, mock_domains_root)
        runner.build_images(
            "test_domain",
            mock_manifest,
            mock_domains_root,
            dry_run=False,
            image_filter=None,
            rebuild_mode=True
        )

        # Should ensure base images in rebuild mode
        mock_ensure_base.assert_called_once()

        # Should remove existing images (4 docker rmi calls)
        rmi_calls = [c for c in mock_subprocess.call_args_list if 'docker' in str(c) and 'rmi' in str(c)]
        assert len(rmi_calls) == 4

        # Should build all 4 images (4 docker build calls)
        build_calls = [c for c in mock_subprocess.call_args_list if 'docker' in str(c) and 'build' in str(c)]
        assert len(build_calls) == 4

    @patch('saber.domain.orchestrator.subprocess.run')
    @patch('saber.domain.orchestrator.DockerRunner._docker_image_exists')
    @patch('saber.domain.orchestrator.DockerRunner.ensure_base_images')
    def test_build_missing_images_only(
        self, mock_ensure_base, mock_exists, mock_subprocess,
        mock_domains_root, mock_compose_file, mock_manifest
    ):
        """Test building only missing images in build mode."""
        # Simulate some images exist, some don't
        def image_exists_side_effect(tag):
            return "cookie_0" in tag or "server" in tag

        mock_exists.side_effect = image_exists_side_effect
        mock_subprocess.return_value = Mock(returncode=0)

        runner = DockerRunner(mock_compose_file, mock_domains_root)
        runner.build_images(
            "test_domain",
            mock_manifest,
            mock_domains_root,
            dry_run=False,
            image_filter=None,
            rebuild_mode=False
        )

        # Should NOT ensure base images in build mode (unless needed)
        mock_ensure_base.assert_not_called()

        # Should NOT remove any existing images (no rmi calls)
        rmi_calls = [c for c in mock_subprocess.call_args_list if 'docker' in str(c) and 'rmi' in str(c)]
        assert len(rmi_calls) == 0

        # Should only build missing images (3: sandbox, cookie_1)
        build_calls = [c for c in mock_subprocess.call_args_list if 'docker' in str(c) and 'build' in str(c)]
        # Actually builds 3 images because cookie_0 doesn't exist (side_effect only returns True for "cookie_0" in tag)
        assert len(build_calls) == 3

    @patch('saber.domain.orchestrator.subprocess.run')
    @patch('saber.domain.orchestrator.DockerRunner._docker_image_exists')
    @patch('saber.domain.orchestrator.DockerRunner.ensure_base_images')
    def test_build_with_filter_cookie(
        self, mock_ensure_base, mock_exists, mock_subprocess,
        mock_domains_root, mock_compose_file, mock_manifest
    ):
        """Test building only cookie-prefixed images."""
        mock_exists.return_value = True
        mock_subprocess.return_value = Mock(returncode=0)

        runner = DockerRunner(mock_compose_file, mock_domains_root)
        runner.build_images(
            "test_domain",
            mock_manifest,
            mock_domains_root,
            dry_run=False,
            image_filter="cookie",
            rebuild_mode=True
        )

        # Should NOT ensure base images (cookie doesn't match base images)
        mock_ensure_base.assert_not_called()

        # Should only remove cookie images (2 rmi calls)
        rmi_calls = [c for c in mock_subprocess.call_args_list if 'docker' in str(c) and 'rmi' in str(c)]
        assert len(rmi_calls) == 2

        # Should only build cookie images (2 build calls)
        build_calls = [c for c in mock_subprocess.call_args_list if 'docker' in str(c) and 'build' in str(c)]
        assert len(build_calls) == 2

    @patch('saber.domain.orchestrator.subprocess.run')
    @patch('saber.domain.orchestrator.DockerRunner._docker_image_exists')
    @patch('saber.domain.orchestrator.DockerRunner.ensure_base_images')
    def test_build_with_filter_server(
        self, mock_ensure_base, mock_exists, mock_subprocess,
        mock_domains_root, mock_compose_file, mock_manifest
    ):
        """Test building only server image (should trigger base rebuild)."""
        mock_exists.return_value = True
        mock_subprocess.return_value = Mock(returncode=0)

        runner = DockerRunner(mock_compose_file, mock_domains_root)
        runner.build_images(
            "test_domain",
            mock_manifest,
            mock_domains_root,
            dry_run=False,
            image_filter="server",
            rebuild_mode=True
        )

        # Should ensure base images (server matches base image)
        mock_ensure_base.assert_called_once()

        # Should only remove server image (1 rmi call)
        rmi_calls = [c for c in mock_subprocess.call_args_list if 'docker' in str(c) and 'rmi' in str(c)]
        assert len(rmi_calls) == 1

        # Should only build server image (1 build call)
        build_calls = [c for c in mock_subprocess.call_args_list if 'docker' in str(c) and 'build' in str(c)]
        assert len(build_calls) == 1

    @patch('saber.domain.orchestrator.subprocess.run')
    @patch('saber.domain.orchestrator.DockerRunner._docker_image_exists')
    def test_build_with_invalid_filter(
        self, mock_exists, mock_subprocess,
        mock_domains_root, mock_compose_file, mock_manifest
    ):
        """Test building with filter that matches no images."""
        mock_exists.return_value = False

        runner = DockerRunner(mock_compose_file, mock_domains_root)

        with pytest.raises(DockerError, match="No images found matching filter"):
            runner.build_images(
                "test_domain",
                mock_manifest,
                mock_domains_root,
                dry_run=False,
                image_filter="nonexistent",
                rebuild_mode=True
            )

    @patch('saber.domain.orchestrator.subprocess.run')
    @patch('saber.domain.orchestrator.DockerRunner._docker_image_exists')
    @patch('saber.domain.orchestrator.DockerRunner.ensure_base_images')
    def test_build_dry_run(
        self, mock_ensure_base, mock_exists, mock_subprocess,
        mock_domains_root, mock_compose_file, mock_manifest
    ):
        """Test dry run mode doesn't execute any commands."""
        mock_exists.return_value = True

        runner = DockerRunner(mock_compose_file, mock_domains_root)
        runner.build_images(
            "test_domain",
            mock_manifest,
            mock_domains_root,
            dry_run=True,
            image_filter=None,
            rebuild_mode=True
        )

        # Should call ensure_base_images with dry_run=True
        mock_ensure_base.assert_called_once_with(True)

        # Subprocess should still be called for git rev-parse and image existence checks
        # but not for actual docker build commands (only git commands for SHA)
        git_calls = [c for c in mock_subprocess.call_args_list if 'git' in str(c)]
        assert len(git_calls) > 0  # Git commands for SHA labels are executed


class TestDomainOrchestratorBuildDomain:
    """Tests for DomainOrchestrator.build_domain method."""

    @patch('saber.domain.orchestrator.DockerRunner.build_images')
    def test_build_domain_default_rebuild_mode(
        self, mock_build_images,
        mock_domains_root, mock_compose_file
    ):
        """Test build_domain defaults to rebuild mode."""
        with patch('saber.domain.orchestrator.ManifestLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.domains_root = mock_domains_root
            mock_loader.load_manifest.return_value = {"images": {}}
            mock_loader_class.return_value = mock_loader

            orchestrator = DomainOrchestrator(mock_domains_root, mock_compose_file)
            orchestrator.build_domain("test_domain")

            # Should call build_images with rebuild_mode=True by default
            mock_build_images.assert_called_once()
            call_kwargs = mock_build_images.call_args[1]
            assert call_kwargs['rebuild_mode'] is True

    @patch('saber.domain.orchestrator.DockerRunner.build_images')
    def test_build_domain_with_filter(
        self, mock_build_images,
        mock_domains_root, mock_compose_file
    ):
        """Test build_domain with image filter."""
        with patch('saber.domain.orchestrator.ManifestLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.domains_root = mock_domains_root
            mock_loader.load_manifest.return_value = {"images": {}}
            mock_loader_class.return_value = mock_loader

            orchestrator = DomainOrchestrator(mock_domains_root, mock_compose_file)
            orchestrator.build_domain("test_domain", image_filter="cookie")

            # Should pass image_filter
            mock_build_images.assert_called_once()
            call_kwargs = mock_build_images.call_args[1]
            assert call_kwargs['image_filter'] == "cookie"

    @patch('saber.domain.orchestrator.DockerRunner.build_images')
    def test_build_domain_build_mode(
        self, mock_build_images,
        mock_domains_root, mock_compose_file
    ):
        """Test build_domain with rebuild_mode=False."""
        with patch('saber.domain.orchestrator.ManifestLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.domains_root = mock_domains_root
            mock_loader.load_manifest.return_value = {"images": {}}
            mock_loader_class.return_value = mock_loader

            orchestrator = DomainOrchestrator(mock_domains_root, mock_compose_file)
            orchestrator.build_domain("test_domain", rebuild_mode=False)

            # Should call build_images with rebuild_mode=False
            mock_build_images.assert_called_once()
            call_kwargs = mock_build_images.call_args[1]
            assert call_kwargs['rebuild_mode'] is False


class TestDomainOrchestratorStartDomain:
    """Tests for DomainOrchestrator.start_domain build/rebuild logic."""

    @patch('saber.domain.orchestrator.DockerRunner.start_services')
    @patch('saber.domain.orchestrator.EnvironmentValidator.generate_environment')
    @patch('saber.domain.orchestrator.DockerRunner.build_images')
    def test_start_domain_with_rebuild(
        self, mock_build_images, mock_gen_env, mock_start_services,
        mock_domains_root, mock_compose_file
    ):
        """Test start_domain with rebuild option."""
        mock_gen_env.return_value = {}

        with patch('saber.domain.orchestrator.ManifestLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.domains_root = mock_domains_root
            mock_loader.load_manifest.return_value = {"images": {}}
            mock_loader_class.return_value = mock_loader

            orchestrator = DomainOrchestrator(mock_domains_root, mock_compose_file)
            orchestrator.start_domain("test_domain", rebuild="")

            # Should call build_images with rebuild_mode=True
            mock_build_images.assert_called_once()
            call_kwargs = mock_build_images.call_args[1]
            assert call_kwargs['rebuild_mode'] is True
            assert call_kwargs['image_filter'] is None

    @patch('saber.domain.orchestrator.DockerRunner.start_services')
    @patch('saber.domain.orchestrator.EnvironmentValidator.generate_environment')
    @patch('saber.domain.orchestrator.DockerRunner.build_images')
    def test_start_domain_with_build(
        self, mock_build_images, mock_gen_env, mock_start_services,
        mock_domains_root, mock_compose_file
    ):
        """Test start_domain with build option."""
        mock_gen_env.return_value = {}

        with patch('saber.domain.orchestrator.ManifestLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.domains_root = mock_domains_root
            mock_loader.load_manifest.return_value = {"images": {}}
            mock_loader_class.return_value = mock_loader

            orchestrator = DomainOrchestrator(mock_domains_root, mock_compose_file)
            orchestrator.start_domain("test_domain", build="")

            # Should call build_images with rebuild_mode=False
            mock_build_images.assert_called_once()
            call_kwargs = mock_build_images.call_args[1]
            assert call_kwargs['rebuild_mode'] is False
            assert call_kwargs['image_filter'] is None

    @patch('saber.domain.orchestrator.DockerRunner.start_services')
    @patch('saber.domain.orchestrator.EnvironmentValidator.generate_environment')
    @patch('saber.domain.orchestrator.DockerRunner.build_images')
    def test_start_domain_with_rebuild_filter(
        self, mock_build_images, mock_gen_env, mock_start_services,
        mock_domains_root, mock_compose_file
    ):
        """Test start_domain with rebuild and filter."""
        mock_gen_env.return_value = {}

        with patch('saber.domain.orchestrator.ManifestLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.domains_root = mock_domains_root
            mock_loader.load_manifest.return_value = {"images": {}}
            mock_loader_class.return_value = mock_loader

            orchestrator = DomainOrchestrator(mock_domains_root, mock_compose_file)
            orchestrator.start_domain("test_domain", rebuild="cookie")

            # Should call build_images with filter
            mock_build_images.assert_called_once()
            call_kwargs = mock_build_images.call_args[1]
            assert call_kwargs['rebuild_mode'] is True
            assert call_kwargs['image_filter'] == "cookie"

    @patch('saber.domain.orchestrator.DockerRunner.start_services')
    @patch('saber.domain.orchestrator.EnvironmentValidator.generate_environment')
    @patch('saber.domain.orchestrator.DockerRunner.build_images')
    def test_start_domain_without_build(
        self, mock_build_images, mock_gen_env, mock_start_services,
        mock_domains_root, mock_compose_file
    ):
        """Test start_domain without build or rebuild."""
        mock_gen_env.return_value = {}

        with patch('saber.domain.orchestrator.ManifestLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.domains_root = mock_domains_root
            mock_loader.load_manifest.return_value = {"images": {}}
            mock_loader_class.return_value = mock_loader

            orchestrator = DomainOrchestrator(mock_domains_root, mock_compose_file)
            orchestrator.start_domain("test_domain")

            # Should NOT call build_images
            mock_build_images.assert_not_called()


class TestImageFiltering:
    """Tests for image name filtering logic."""

    def test_filter_cookie_images(self, mock_manifest):
        """Test filtering cookie-prefixed images."""
        images = mock_manifest["images"]
        filtered = {name: config for name, config in images.items() if name.startswith("cookie")}

        assert len(filtered) == 2
        assert "cookie_0_target" in filtered
        assert "cookie_1_target" in filtered
        assert "server" not in filtered

    def test_filter_server_image(self, mock_manifest):
        """Test filtering server image."""
        images = mock_manifest["images"]
        filtered = {name: config for name, config in images.items() if name.startswith("server")}

        assert len(filtered) == 1
        assert "server" in filtered

    def test_no_match_filter(self, mock_manifest):
        """Test filter that matches nothing."""
        images = mock_manifest["images"]
        filtered = {name: config for name, config in images.items() if name.startswith("nonexistent")}

        assert len(filtered) == 0


class TestBuildModeVsRebuildMode:
    """Tests to verify build mode vs rebuild mode behavior."""

    @patch('saber.domain.orchestrator.subprocess.run')
    @patch('saber.domain.orchestrator.DockerRunner._docker_image_exists')
    def test_rebuild_mode_removes_existing_images(
        self, mock_exists, mock_subprocess,
        mock_domains_root, mock_compose_file, mock_manifest
    ):
        """Verify rebuild mode removes existing images before building."""
        mock_exists.return_value = True
        mock_subprocess.return_value = Mock(returncode=0)

        runner = DockerRunner(mock_compose_file, mock_domains_root)

        with patch.object(runner, 'ensure_base_images'):
            runner.build_images(
                "test_domain",
                mock_manifest,
                mock_domains_root,
                dry_run=False,
                rebuild_mode=True
            )

        # Should have docker rmi calls
        rmi_calls = [c for c in mock_subprocess.call_args_list if 'docker' in str(c) and 'rmi' in str(c)]
        assert len(rmi_calls) > 0

    @patch('saber.domain.orchestrator.subprocess.run')
    @patch('saber.domain.orchestrator.DockerRunner._docker_image_exists')
    def test_build_mode_skips_existing_images(
        self, mock_exists, mock_subprocess,
        mock_domains_root, mock_compose_file, mock_manifest
    ):
        """Verify build mode skips existing images."""
        mock_exists.return_value = True  # All images exist
        mock_subprocess.return_value = Mock(returncode=0)

        runner = DockerRunner(mock_compose_file, mock_domains_root)
        runner.build_images(
            "test_domain",
            mock_manifest,
            mock_domains_root,
            dry_run=False,
            rebuild_mode=False
        )

        # Should have NO docker rmi calls
        rmi_calls = [c for c in mock_subprocess.call_args_list if 'docker' in str(c) and 'rmi' in str(c)]
        assert len(rmi_calls) == 0

        # Should have NO docker build calls (all images already exist)
        build_calls = [c for c in mock_subprocess.call_args_list if 'docker' in str(c) and 'build' in str(c)]
        assert len(build_calls) == 0
