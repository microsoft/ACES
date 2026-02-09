"""Core orchestration services for SABER domain management.

This module provides modular services for domain orchestration:
- ManifestLoader: Load and validate domain manifests
- EnvironmentValidator: Validate environment and generate variables
- DockerRunner: Execute Docker operations (for sandbox/permanent services)
- DomainOrchestrator: Coordinate the above services

Server deployment runs as a host subprocess (not in a container) to:
- Eliminate Docker-in-Docker issues
- Allow direct host path mounts for sandbox containers
- Simplify networking between server and managed containers
"""

import json
import os
import socket
import subprocess
import sys
import time
from importlib.resources import files
from pathlib import Path
from typing import IO, Any

import yaml

import saber.domain.package_resources

from .exceptions import DockerError, DomainNotFoundError, DomainValidationError
from .resources import resolve_schema_file


def detect_repo_structure(domains_root: Path) -> tuple[Path, Path]:
    """Detect repository structure and return appropriate paths.

    This utility walks up from domains_root to find the saber source code
    (src/saber/) and .env file, working with any repository structure.

    Args:
        domains_root: Path to the domains directory

    Returns:
        Tuple of (saber_src_path, env_file_path)

    Raises:
        DomainValidationError: If saber source cannot be found
    """
    current = domains_root
    saber_src = None
    env_file = None

    # Search up to 5 levels up
    for _ in range(5):
        current = current.parent

        # Look for saber source directory
        if saber_src is None:
            # Check if current directory has src/saber
            if (current / "src" / "saber").exists():
                saber_src = current
            # Check if there's an external/saber with src/saber
            elif (current / "external" / "saber" / "src" / "saber").exists():
                saber_src = current / "external" / "saber"

        # Look for .env file
        if env_file is None and (current / ".env").exists():
            env_file = current / ".env"

        # If we found both, we're done
        if saber_src and env_file:
            return saber_src, env_file

    # If we didn't find saber source, raise error
    if saber_src is None:
        raise DomainValidationError(
            domain="repo-structure",
            validation_errors=[
                f"Could not find saber source (src/saber/) walking up from: {domains_root}",
                "Searched up to 5 directory levels",
            ],
        )

    # If we found saber but no .env, raise a clear error
    if env_file is None:
        error_lines = [
            "❌ Required .env file not found!",
            "",
            f"Searched from {domains_root} up to {saber_src}",
            "",
            "To fix this issue:",
        ]

        # Check if .env.template exists
        if (saber_src / ".env.template").exists():
            error_lines.extend(
                [
                    f"  1. Copy the template: cp {saber_src}/.env.template {saber_src}/.env",
                    f"  2. Edit {saber_src}/.env with your API keys and configuration",
                ]
            )
        else:
            error_lines.extend(
                [
                    f"  1. Create {saber_src}/.env with your configuration",
                    "  2. Add required API keys (OpenAI, Anthropic, etc.)",
                ]
            )

        error_lines.extend(
            [
                "",
                "The .env file is required for:",
                "  - API keys (OpenAI, Anthropic, etc.)",
                "  - Azure credentials",
                "  - Other sensitive configuration",
            ]
        )

        raise DomainValidationError(
            domain="configuration",
            validation_errors=error_lines,
        )

    return saber_src, env_file


class ManifestLoader:
    """Service for loading and validating domain manifests."""

    def __init__(self, domains_root: Path):
        self.domains_root = domains_root.resolve()
        if not self.domains_root.exists():
            raise DomainNotFoundError("domains_root", f"Domains root directory does not exist: {self.domains_root}")

    def list_domains(self) -> list[str]:
        """List all available domains."""
        domains = []
        for path in self.domains_root.iterdir():
            if path.is_dir() and (path / "domain.yaml").exists():
                domains.append(path.name)
        return sorted(domains)

    def load_manifest(self, domain: str) -> dict[str, Any]:
        """Load and validate domain manifest.

        Args:
            domain: Domain name to load

        Returns:
            Dict containing validated manifest data

        Raises:
            DomainNotFoundError: If domain doesn't exist
            DomainValidationError: If manifest is invalid
        """
        domain_path = self.domains_root / domain
        manifest_path = domain_path / "domain.yaml"

        if not domain_path.exists():
            raise DomainNotFoundError(domain, str(self.domains_root))

        if not manifest_path.exists():
            raise DomainValidationError(domain, [f"Domain manifest not found: {manifest_path}"])

        # Load YAML
        try:
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)
                if not isinstance(manifest, dict):
                    raise DomainValidationError(domain, ["Manifest must be a YAML object/dictionary"])
        except yaml.YAMLError as e:
            raise DomainValidationError(domain, [f"Invalid YAML in manifest: {e}"]) from e

        # Validate against schema
        self._validate_schema(manifest, domain)

        # Validate domain structure
        self._validate_structure(domain, manifest)

        return manifest

    def _validate_schema(self, manifest: dict[str, Any], domain: str) -> None:
        """Validate manifest against JSON schema."""
        try:
            import jsonschema
        except ImportError:
            raise DomainValidationError(
                domain, ["jsonschema package required for validation. Install with: uv add jsonschema"]
            ) from None

        try:
            with resolve_schema_file() as schema_path:
                with open(schema_path) as f:
                    schema = json.load(f)
        except Exception as e:
            raise DomainValidationError(domain, [f"Failed to load validation schema: {e}"]) from e

        try:
            jsonschema.validate(manifest, schema)
        except jsonschema.ValidationError as e:
            error_path = " -> ".join(str(p) for p in e.absolute_path)
            raise DomainValidationError(domain, [f"Schema validation failed at {error_path}: {e.message}"]) from e

    def _validate_structure(self, domain: str, manifest: dict[str, Any]) -> None:
        """Validate required domain directory structure."""
        domain_path = self.domains_root / domain
        errors = []

        # Required directories (server-only architecture)
        required_dirs = ["server/config", "server/docker"]

        for dir_path in required_dirs:
            full_path = domain_path / dir_path
            if not full_path.exists():
                errors.append(f"Missing required directory: {dir_path}")

        # Required files based on manifest
        for service, image_config in manifest.get("images", {}).items():
            dockerfile = image_config.get("dockerfile")
            if dockerfile:
                dockerfile_path = domain_path / dockerfile
                if not dockerfile_path.exists():
                    errors.append(f"Missing Dockerfile for {service}: {dockerfile}")

        # Validate manifest domain slug matches directory name
        manifest_slug = manifest.get("domain", {}).get("slug")
        if manifest_slug != domain:
            errors.append(f"Domain slug mismatch: manifest declares '{manifest_slug}' but directory is '{domain}'")

        if errors:
            raise DomainValidationError(domain, errors)


class EnvironmentValidator:
    """Service for validating environment and generating variables."""

    def __init__(self, domains_root: Path):
        self.domains_root = domains_root.resolve()

    def generate_environment(
        self,
        domain: str,
        manifest: dict[str, Any],
        rest_port: int = 8000,
        mcp_port: int = 8001,
        log_level: str = "INFO",
        skip_image_check: bool = False,
        skip_port_check: bool = False,
    ) -> dict[str, str]:
        """Generate and validate environment variables for host subprocess deployment.

        Args:
            domain: Domain name
            manifest: Domain manifest
            rest_port: REST API port
            mcp_port: MCP protocol port
            log_level: Logging level
            skip_image_check: Skip Docker image existence check (for sandbox images)
            skip_port_check: Skip port availability check (for status checks)

        Returns:
            Dict of environment variables for server subprocess

        Raises:
            DomainValidationError: If validation fails
        """
        errors = []

        # Validate ports are available (unless skipping for status checks)
        if not skip_port_check:
            for port_name, port_num in [("REST", rest_port), ("MCP", mcp_port)]:
                if not self._is_port_available(port_num):
                    errors.append(f"Port {port_num} ({port_name}) is already in use")

        # Validate sandbox/permanent service images exist (unless building)
        # Note: server no longer runs in a container, so we skip server image validation
        if not skip_image_check:
            # First validate base sandbox image
            self._validate_base_sandbox_image(errors)
            # Then validate domain sandbox images (skip server image)
            self._validate_sandbox_images(manifest, errors)

        if errors:
            raise DomainValidationError(domain, errors)

        # Detect repository structure
        saber_src_path, env_file_path = self._detect_repo_structure()

        # Domain-specific paths
        domain_root = self.domains_root / domain
        config_dir = domain_root / "server" / "config"

        # Generate environment variables for host subprocess
        return {
            # Domain identification
            "SABER_DOMAIN": domain,
            # Host paths (not container paths)
            "SABER_CONFIG_DIR": str(config_dir),
            "SABER_DOMAINS_ROOT": str(self.domains_root),
            # Server configuration
            "SABER_HOST": "0.0.0.0",
            "SABER_PORT": str(rest_port),
            "SABER_MCP_PORT": str(mcp_port),
            "SABER_LOG_LEVEL": log_level,
            # Legacy env vars (for compatibility)
            "DOMAIN": domain,
            "DOMAINS_ROOT": str(self.domains_root),
            "REST_PORT": str(rest_port),
            "MCP_PORT": str(mcp_port),
            "LOG_LEVEL": log_level,
            "SABER_SRC": str(saber_src_path),
            "ENV_FILE": str(env_file_path),
        }

    def _detect_repo_structure(self) -> tuple[Path, Path]:
        """Detect repository structure and return appropriate paths.

        Returns:
            Tuple of (saber_src_path, env_file_path)

        Raises:
            DomainValidationError: If structure cannot be determined
        """
        return detect_repo_structure(self.domains_root)

    def _validate_sandbox_images(self, manifest: dict[str, Any], errors: list[str]) -> None:
        """Validate sandbox/permanent service images exist (not server image)."""
        images_config = manifest.get("images", {})
        if not images_config:
            # No images is fine - server runs on host, may not need sandboxes
            return

        # Validate all images EXCEPT server (server runs on host now)
        for image_name, image_config in images_config.items():
            if image_name == "server":
                # Skip server image validation - server runs as host subprocess
                continue

            if not isinstance(image_config, dict):
                errors.append(f"Image '{image_name}' configuration must be a dictionary")
                continue

            image_tag = image_config.get("tag")
            if not image_tag:
                errors.append(f"Image '{image_name}' missing required 'tag' field")
                continue

            if not self._docker_image_exists(image_tag):
                slug = manifest.get("domain", {}).get("slug", "<domain>")
                errors.append(
                    f"Docker image '{image_tag}' not found.\n"
                    f"  Build with inspect eval: uv run inspect eval domains/{slug} --model <model> -T build=true\n"
                    f"  Or use saber-domain CLI: uv run saber-domain build {slug} --build"
                )

    def _validate_base_sandbox_image(self, errors: list[str]) -> None:
        """Validate base sandbox image exists (server image no longer needed)."""
        try:
            # Load base images config from package resources
            base_images_file = files(saber.domain.package_resources) / "base-images.yaml"
            with base_images_file.open("r") as f:
                base_images_config = yaml.safe_load(f)

            # Only validate sandbox image, not server (server runs on host)
            sandbox_config = base_images_config["images"].get("sandbox")
            if sandbox_config:
                image_tag = sandbox_config["tag"]
                if not self._docker_image_exists(image_tag):
                    errors.append(
                        f"Base sandbox image '{image_tag}' not found.\n"
                        "  Build with inspect eval: uv run inspect eval domains/<domain> "
                        "--model <model> -T build=true\n"
                        "  Or use saber-domain CLI: uv run saber-domain build <domain> --build"
                    )
        except Exception as e:
            errors.append(f"Failed to validate base sandbox image: {e}")

    def _validate_all_images(self, manifest: dict[str, Any], errors: list[str]) -> None:
        """Validate all domain images exist."""
        images_config = manifest.get("images", {})
        # Note: images section is optional - domains may have no custom images if they
        # only use permanent containers with pre-built images

        # Validate all images defined in the manifest
        for image_name, image_config in images_config.items():
            if not isinstance(image_config, dict):
                errors.append(f"Image '{image_name}' configuration must be a dictionary")
                continue

            image_tag = image_config.get("tag")
            if not image_tag:
                errors.append(f"Image '{image_name}' missing required 'tag' field")
                continue

            if not self._docker_image_exists(image_tag):
                # Keep message lines under 120 chars for flake8
                slug = manifest.get("domain", {}).get("slug", "<domain>")
                errors.append(
                    f"Docker image '{image_tag}' not found.\n"
                    f"  Build with inspect eval: uv run inspect eval domains/{slug} --model <model> -T build=true\n"
                    f"  Or use saber-domain CLI: uv run saber-domain build {slug} --build"
                )

    def _validate_base_images(self, errors: list[str]) -> None:
        """Validate base images exist."""
        try:
            # Load base images config from package resources
            base_images_file = files(saber.domain.package_resources) / "base-images.yaml"
            with base_images_file.open("r") as f:
                base_images_config = yaml.safe_load(f)

            for _image_name, image_config in base_images_config["images"].items():
                image_tag = image_config["tag"]
                if not self._docker_image_exists(image_tag):
                    # Keep message lines under 120 chars for flake8
                    errors.append(
                        f"Base image '{image_tag}' not found.\n"
                        "  Build with inspect eval: uv run inspect eval domains/<domain> \
                                --model <model> -T build=true\n"
                        "  Or use saber-domain CLI: uv run saber-domain build <domain> --build"
                    )
        except Exception as e:
            errors.append(f"Failed to validate base images: {e}")

    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(("localhost", port))
                return result != 0
        except Exception:
            return False

    def _docker_image_exists(self, image_tag: str) -> bool:
        """Check if Docker image exists locally."""
        try:
            result = subprocess.run(["docker", "image", "inspect", image_tag], capture_output=True, check=False)
            return result.returncode == 0
        except Exception:
            return False


class DockerRunner:
    """Service for executing Docker operations (image building only).

    Note: Server deployment is now via subprocess, not docker-compose.
    This class only handles image building for sandboxes and permanent services.
    """

    def __init__(self, domains_root: Path):
        self.domains_root = domains_root

        # Validate Docker is available
        self._validate_docker()

    def _docker_image_exists(self, image_tag: str) -> bool:
        """Check if Docker image exists locally."""
        try:
            result = subprocess.run(["docker", "image", "inspect", image_tag], capture_output=True, check=False)
            return result.returncode == 0
        except Exception:
            return False

    def build_images(
        self,
        domain: str,
        manifest: dict[str, Any],
        domains_root: Path,
        dry_run: bool = False,
        image_filter: str | None = None,
        rebuild_mode: bool = True,
    ) -> None:
        """Build Docker images defined in domain manifest.

        Args:
            domain: Domain name
            manifest: Domain manifest
            domains_root: Path to domains directory
            dry_run: If True, show commands without executing
            image_filter: Optional substring to filter which images to build (e.g., 'server', 'cookie', 'sandbox')
            rebuild_mode: If True, remove and rebuild images. If False, only build missing images.

        Raises:
            DockerError: If build fails
        """
        domain_path = domains_root / domain
        images_config = manifest.get("images", {})

        if not images_config:
            raise DockerError("No images configuration found in manifest")

        # Filter images if requested
        if image_filter:
            images_to_build = {name: config for name, config in images_config.items() if image_filter in name}
            if not images_to_build:
                raise DockerError(
                    f"No images found matching filter '{image_filter}'. "
                    f"Available images: {', '.join(images_config.keys())}"
                )
            print(f"🔍 Filtered images matching '{image_filter}': {', '.join(images_to_build.keys())}")
        else:
            images_to_build = images_config

        # Ensure base images exist first
        # In rebuild mode: rebuild base images if no filter or filter matches base images
        # In build mode: ensure base images exist (build if missing) when needed
        # Note: Only sandbox base image is needed (server runs on host, not in container)
        needs_base_images = any("sandbox" in name for name in images_to_build.keys())

        if rebuild_mode and (image_filter is None or needs_base_images):
            # Rebuild mode: remove and rebuild base images
            print("🔍 Rebuilding base image dependencies...")
            self.ensure_base_images(dry_run)
        elif not rebuild_mode and needs_base_images:
            # Build mode: ensure base images exist (build if missing)
            print("🔍 Ensuring base images exist...")
            self._ensure_base_images_exist(dry_run)

        # Remove existing domain images for rebuild (only in rebuild mode)
        if rebuild_mode:
            print("🔄 Removing existing domain images for rebuild...")
            for _image_name, image_config in images_to_build.items():
                if isinstance(image_config, dict):
                    image_tag = image_config.get("tag")
                    if image_tag and self._docker_image_exists(image_tag):
                        if not dry_run:
                            try:
                                subprocess.run(["docker", "rmi", image_tag], check=False, capture_output=True)
                                print(f"  Removed {image_tag}")
                            except Exception:
                                pass
                        else:
                            print(f"  Would remove {image_tag}")

        # Build filtered images (in build mode, skip images that already exist)
        built_count = 0
        skipped_count = 0
        for image_name, image_config in images_to_build.items():
            if not isinstance(image_config, dict):
                raise DockerError(f"Image '{image_name}' configuration must be a dictionary")

            dockerfile = image_config.get("dockerfile")
            if not dockerfile:
                raise DockerError(f"Image '{image_name}' missing required 'dockerfile' field")

            image_tag = image_config.get("tag")
            if not image_tag:
                raise DockerError(f"Image '{image_name}' missing required 'tag' field")

            # In build mode (not rebuild), skip if image already exists
            if not rebuild_mode and self._docker_image_exists(image_tag):
                if not dry_run:
                    print(f"⏭️  Skipping {image_name} - image already exists: {image_tag}")
                else:
                    print(f"⏭️  Would skip {image_name} - image already exists: {image_tag}")
                skipped_count += 1
                continue

            dockerfile_path = domain_path / dockerfile
            if not dockerfile_path.exists():
                raise DockerError(f"Dockerfile not found: {dockerfile_path}")

            # Get build context (defaults to domain root)
            context = image_config.get("context", ".")
            context_path = domain_path / context
            if not context_path.exists():
                raise DockerError(f"Build context not found: {context_path}")

            build_args = image_config.get("buildArgs", {})
            labels = image_config.get("labels", {})

            # Prepare build command
            cmd = ["docker", "build", "-f", str(dockerfile_path), "-t", image_tag]

            # Add --no-cache when rebuilding to ensure fresh layers
            if rebuild_mode:
                cmd.append("--no-cache")

            cmd.append(str(context_path))

            # Add build args
            for key, value in build_args.items():
                cmd.extend(["--build-arg", f"{key}={value}"])

            # Add labels
            for key, value in labels.items():
                cmd.extend(["--label", f"{key}={value}"])

            # Add git metadata if available
            self._add_git_metadata(cmd, domain_path)

            if dry_run:
                print(f"Would build {image_name}: {' '.join(cmd)}")
                built_count += 1
                continue

            print(f"Building {image_name} image: {image_tag}")
            try:
                subprocess.run(cmd, check=True, cwd=domain_path)
                print(f"✓ Successfully built {image_tag}")
                built_count += 1
            except subprocess.CalledProcessError as e:
                raise DockerError(f"Failed to build {image_name} image {image_tag}: {e}") from e

        if not dry_run:
            if skipped_count > 0:
                print(
                    f"✓ Successfully built {built_count} images, "
                    f"skipped {skipped_count} existing images for domain '{domain}'"
                )
            else:
                print(f"✓ Successfully built {built_count} images for domain '{domain}'")

    def ensure_base_images(self, dry_run: bool = False) -> None:
        """Remove and rebuild all base images.

        Args:
            dry_run: If True, show commands without executing

        Raises:
            DockerError: If base image build fails
        """
        base_images_config = self._load_base_images_config()

        print("🔄 Removing existing base images for rebuild...")
        # Remove all existing base images
        for _image_name, image_config in base_images_config["images"].items():
            image_tag = image_config["tag"]
            if self._docker_image_exists(image_tag):
                if not dry_run:
                    try:
                        subprocess.run(["docker", "rmi", image_tag], check=False, capture_output=True)
                        print(f"  Removed {image_tag}")
                    except Exception:
                        pass  # Image might be in use, will fail later if needed
                else:
                    print(f"  Would remove {image_tag}")

        print(f"🔨 Building {len(base_images_config['images'])} base images...")

        for image_name, image_config in base_images_config["images"].items():
            self._build_or_pull_base_image(image_name, image_config, base_images_config, dry_run, no_cache=True)

    def _ensure_base_images_exist(self, dry_run: bool = False) -> None:
        """Ensure base images exist, building only if missing (incremental).

        Args:
            dry_run: If True, show commands without executing

        Raises:
            DockerError: If base image build fails
        """
        base_images_config = self._load_base_images_config()

        missing_images = []
        for image_name, image_config in base_images_config["images"].items():
            image_tag = image_config["tag"]
            if not self._docker_image_exists(image_tag):
                missing_images.append((image_name, image_config))

        if not missing_images:
            print("✓ All base images already exist")
            return

        print(f"🔨 Building/pulling {len(missing_images)} missing base image(s)...")

        for image_name, image_config in missing_images:
            self._build_or_pull_base_image(image_name, image_config, base_images_config, dry_run, no_cache=False)

    def _is_repo_checkout(self) -> bool:
        """Check if we're running from a saber repo checkout (vs installed package)."""
        # Check if docker/ directory exists relative to domains_root
        repo_root = self.domains_root.parent
        return (repo_root / "docker").exists()

    def _get_external_saber_path(self) -> Path | None:
        """Get path to external/saber if it exists (for oss_saber repo with gitsubmodule)."""
        repo_root = self.domains_root.parent
        external_saber = repo_root / "external" / "saber"
        if (external_saber / "docker").exists():
            return external_saber
        return None

    def _build_or_pull_base_image(
        self,
        image_name: str,
        image_config: dict[str, Any],
        base_images_config: dict[str, Any],
        dry_run: bool,
        no_cache: bool = False,
    ) -> None:
        """Build base image from repo or pull from registry.

        Priority order:
        1. If in saber repo checkout (docker/ exists): build from docker/
        2. Try to pull from ACR (fastest for most users)
        3. If ACR fails and external/saber exists: fallback to local build

        Args:
            no_cache: If True, pass --no-cache to docker build to ensure fresh layers
        """
        image_tag = image_config["tag"]
        labels = image_config.get("labels", {})
        dockerfile = image_config["dockerfile"]

        if self._is_repo_checkout():
            # Option 1: Build from docker/ directory (saber repo checkout)
            self._build_base_image_from_repo(image_name, image_tag, dockerfile, labels, dry_run, no_cache)
        else:
            # Option 2: Try ACR first, fallback to local build
            registry = base_images_config.get("registry", "")
            registry_image = image_config.get("registry_image", "")
            if registry and registry_image:
                # ACR configured - try to pull (will fallback to local build on failure)
                self._pull_base_image_from_registry(image_name, image_tag, registry, registry_image, dry_run)
            elif self._get_external_saber_path():
                # No ACR configured but have local source - build from it
                self._build_base_image_from_external_saber(image_name, image_tag, dockerfile, labels, dry_run, no_cache)
            else:
                raise DockerError(
                    f"Cannot build {image_name}: not in repo checkout and no registry configured. "
                    "Please run from a git checkout of the saber repository."
                )

    def _get_ado_token(self) -> str:
        """Get ADO token from environment or Azure CLI.

        Returns token from:
        1. ADO_TOKEN environment variable (for CI or manual override)
        2. Azure CLI `az account get-access-token` (for local dev with az login)
        3. Empty string if neither available
        """
        # Check environment first
        token = os.environ.get("ADO_TOKEN", "")
        if token:
            return token

        # Try Azure CLI
        try:
            result = subprocess.run(
                [
                    "az",
                    "account",
                    "get-access-token",
                    "--resource",
                    "499b84ac-1321-427f-aa17-267ca6975798",  # Azure DevOps resource ID
                    "--query",
                    "accessToken",
                    "-o",
                    "tsv",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # az CLI not available or timed out

        return ""

    def _build_base_image_from_repo(
        self,
        image_name: str,
        image_tag: str,
        dockerfile: str,
        labels: dict[str, str],
        dry_run: bool,
        no_cache: bool = False,
    ) -> None:
        """Build base image from docker/ directory in repo.

        For private ADO git dependencies, authentication is automatically handled via:
        1. ADO_TOKEN environment variable
        2. Azure CLI (az login) credentials

        Args:
            no_cache: If True, pass --no-cache to docker build to ensure fresh layers
        """
        try:
            repo_root = self.domains_root.parent
            dockerfile_path = repo_root / dockerfile

            if not dockerfile_path.exists():
                raise DockerError(f"Dockerfile not found: {dockerfile_path}")

            # Get ADO token for private git dependencies
            ado_token = self._get_ado_token()

            # Prepare build environment
            env = os.environ.copy()
            env["DOCKER_BUILDKIT"] = "1"
            if ado_token:
                env["ADO_TOKEN"] = ado_token

            cmd = ["docker", "build", "-f", str(dockerfile_path), "-t", image_tag]

            # Add --no-cache when rebuilding to ensure fresh layers
            if no_cache:
                cmd.append("--no-cache")

            # Pass ADO_TOKEN as secret for git authentication
            if ado_token:
                cmd.extend(["--secret", "id=ADO_TOKEN,env=ADO_TOKEN"])

            # Add labels
            for key, value in labels.items():
                cmd.extend(["--label", f"{key}={value}"])

            # Add build context (repo root for saber, or saber subdir for oss_saber)
            # The Dockerfile expects to COPY pyproject.toml, uv.lock, src/ from context
            if (repo_root / "src" / "saber").exists():
                # Standalone saber repo
                build_context = repo_root
            else:
                # oss_saber repo - use external/saber as context
                build_context = repo_root / "external" / "saber"

            cmd.append(str(build_context))

            if dry_run:
                print(f"Would build base image {image_name}: {' '.join(cmd)}")
                if ado_token:
                    print("ADO_TOKEN: [obtained from az CLI or environment]")
                else:
                    print("ADO_TOKEN: [not available - private deps may fail]")
                return

            print(f"Building base image: {image_tag}")
            if ado_token:
                print("🔑 Using ADO token for private git dependencies")
            else:
                print("⚠️  No ADO token available - private git dependencies may fail")
                print("   Run 'az login' or set ADO_TOKEN environment variable")

            subprocess.run(cmd, check=True, cwd=build_context, env=env)
            print(f"✓ Successfully built {image_tag}")

        except subprocess.CalledProcessError as e:
            raise DockerError(f"Failed to build base image {image_tag}: {e}") from e
        except Exception as e:
            raise DockerError(f"Failed to build base image: {e}") from e

    def _is_arm_host(self) -> bool:
        """Check if running on ARM architecture."""
        import platform

        machine = platform.machine().lower()
        return machine in ("arm64", "aarch64")

    def _pull_base_image_from_registry(
        self, image_name: str, image_tag: str, registry: str, registry_image: str, dry_run: bool
    ) -> None:
        """Pull base image from container registry and tag locally.

        Falls back to building from external/saber if available and pull fails,
        or if we're on ARM and ACR only has amd64 images.
        """
        # Check if we're on ARM - ACR images are currently amd64 only
        if self._is_arm_host():
            external_saber = self._get_external_saber_path()
            if external_saber:
                print("⚠️  ARM host detected - building locally instead of pulling amd64 image from ACR")
                base_images_config = self._load_base_images_config()
                image_config = base_images_config["images"].get(image_name.replace("saber/", "").split(":")[0], {})
                dockerfile = image_config.get("dockerfile", f"docker/Dockerfile.saber_{image_name}")
                labels = image_config.get("labels", {})
                self._build_base_image_from_external_saber(image_name, image_tag, dockerfile, labels, dry_run)
                return
            # No local source, try ACR anyway (will likely fail at runtime)
            print("⚠️  ARM host detected but no local source available - attempting ACR pull (may not work)")

        try:
            remote_image = f"{registry}/{registry_image}:latest"

            if dry_run:
                print(f"Would pull {remote_image} and tag as {image_tag}")
                return

            print(f"Pulling base image: {remote_image}")
            subprocess.run(["docker", "pull", remote_image], check=True)

            print(f"Tagging as {image_tag}")
            subprocess.run(["docker", "tag", remote_image, image_tag], check=True)

            print(f"✓ Successfully pulled and tagged {image_tag}")

        except subprocess.CalledProcessError as e:
            # Fallback: try to build from external/saber if available
            external_saber = self._get_external_saber_path()
            if external_saber:
                print(f"⚠️  Failed to pull from registry, falling back to local build from {external_saber}")
                # Load config to get dockerfile path
                base_images_config = self._load_base_images_config()
                image_config = base_images_config["images"].get(image_name.replace("saber/", "").split(":")[0], {})
                dockerfile = image_config.get("dockerfile", f"docker/Dockerfile.saber_{image_name}")
                labels = image_config.get("labels", {})
                self._build_base_image_from_external_saber(image_name, image_tag, dockerfile, labels, dry_run)
            else:
                raise DockerError(f"Failed to pull base image {image_tag}: {e}") from e

    def _build_base_image_from_external_saber(
        self,
        image_name: str,
        image_tag: str,
        dockerfile: str,
        labels: dict[str, str],
        dry_run: bool,
        no_cache: bool = False,
    ) -> None:
        """Build base image from external/saber/docker directory (oss_saber with gitsubmodule).

        Args:
            no_cache: If True, pass --no-cache to docker build to ensure fresh layers
        """
        try:
            external_saber = self._get_external_saber_path()
            if not external_saber:
                raise DockerError("external/saber not found")

            dockerfile_path = external_saber / dockerfile

            if not dockerfile_path.exists():
                raise DockerError(f"Dockerfile not found: {dockerfile_path}")

            # Get ADO token for private git dependencies
            ado_token = self._get_ado_token()

            # Prepare build environment
            env = os.environ.copy()
            env["DOCKER_BUILDKIT"] = "1"
            if ado_token:
                env["ADO_TOKEN"] = ado_token

            cmd = ["docker", "build", "-f", str(dockerfile_path), "-t", image_tag]

            # Add --no-cache when rebuilding to ensure fresh layers
            if no_cache:
                cmd.append("--no-cache")

            # Pass ADO_TOKEN as secret for git authentication
            if ado_token:
                cmd.extend(["--secret", "id=ADO_TOKEN,env=ADO_TOKEN"])

            # Add labels
            for key, value in labels.items():
                cmd.extend(["--label", f"{key}={value}"])

            # Build context is external/saber (contains pyproject.toml, uv.lock, src/)
            cmd.append(str(external_saber))

            if dry_run:
                print(f"Would build base image {image_name} from external/saber: {' '.join(cmd)}")
                if ado_token:
                    print("ADO_TOKEN: [obtained from az CLI or environment]")
                else:
                    print("ADO_TOKEN: [not available - private deps may fail]")
                return

            print(f"Building base image from external/saber: {image_tag}")
            if ado_token:
                print("🔑 Using ADO token for private git dependencies")
            else:
                print("⚠️  No ADO token available - private git dependencies may fail")
                print("   Run 'az login' or set ADO_TOKEN environment variable")

            subprocess.run(cmd, check=True, cwd=external_saber, env=env)
            print(f"✓ Successfully built {image_tag}")

        except subprocess.CalledProcessError as e:
            raise DockerError(f"Failed to build base image {image_tag}: {e}") from e
        except Exception as e:
            raise DockerError(f"Failed to build base image from external/saber: {e}") from e

    def _load_base_images_config(self) -> dict[str, Any]:
        """Load base images configuration."""
        try:
            base_images_file = files(saber.domain.package_resources) / "base-images.yaml"
            with base_images_file.open("r") as f:
                config: dict[str, Any] = yaml.safe_load(f)
                return config
        except Exception as e:
            raise DockerError(f"Failed to load base images configuration: {e}") from e

    def _validate_docker(self) -> None:
        """Validate Docker and Docker Compose are available."""
        for cmd in [["docker", "--version"], ["docker", "compose", "version"]]:
            try:
                subprocess.run(cmd, capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise DockerError(
                    f"Command '{' '.join(cmd)}' failed. Ensure Docker and Docker Compose are installed.",
                    command=" ".join(cmd),
                ) from None

    def _add_git_metadata(self, cmd: list[str], domain_path: Path) -> None:
        """Add git metadata labels to build command."""
        try:
            git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=domain_path, text=True).strip()
            cmd.extend(["--label", f"saber.git.sha={git_sha}"])
        except subprocess.CalledProcessError:
            pass


class DomainOrchestrator:
    """Main orchestrator coordinating all domain services.

    Server deployment runs as a host subprocess (not in a container) to:
    - Eliminate Docker-in-Docker issues
    - Allow direct host path mounts for sandbox containers
    - Simplify networking between server and managed containers
    """

    # Class-level storage for server processes (persists across instances)
    _running_servers: dict[str, subprocess.Popen] = {}

    def __init__(self, domains_root: Path):
        self.manifest_loader = ManifestLoader(domains_root)
        self.environment_validator = EnvironmentValidator(domains_root)
        self.docker_runner = DockerRunner(domains_root)
        self._domains_root = domains_root

    def list_domains(self) -> list[str]:
        """List all available domains."""
        return self.manifest_loader.list_domains()

    def get_running_domains(self) -> list[str]:
        """Get list of currently running domains by checking PID files.

        Returns:
            List of domain names that have running server processes
        """
        running = []

        # Scan all domains for PID files
        for domain in self.list_domains():
            domain_root = self._domains_root / domain
            pid_file = domain_root / "server" / "logs" / f"{domain}.pid"

            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text().strip())
                    # Check if process is actually running
                    os.kill(pid, 0)
                    running.append(domain)
                except (ValueError, OSError):
                    # Invalid PID or process not running - clean up stale file
                    try:
                        pid_file.unlink()
                    except OSError:
                        pass

        return running

    @staticmethod
    def get_all_running_saber_processes() -> list[tuple[int, str]]:
        """Find all running SABER server processes system-wide.

        Uses pgrep/ps to find any saber.server processes regardless of
        which domains root they were started from.

        Returns:
            List of (pid, cmdline) tuples for running SABER server processes
        """
        processes = []
        try:
            # Use pgrep to find saber.server processes
            result = subprocess.run(
                ["pgrep", "-f", "saber.server.*--start"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                for pid_str in result.stdout.strip().split("\n"):
                    if pid_str:
                        try:
                            pid = int(pid_str)
                            # Get the command line for this process
                            cmdline_path = Path(f"/proc/{pid}/cmdline")
                            if cmdline_path.exists():
                                cmdline = cmdline_path.read_text().replace("\x00", " ").strip()
                                processes.append((pid, cmdline))
                        except (ValueError, OSError):
                            pass
        except FileNotFoundError:
            # pgrep not available, try ps
            try:
                result = subprocess.run(
                    ["ps", "aux"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if "saber.server" in line and "--start" in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    pid = int(parts[1])
                                    processes.append((pid, line))
                                except ValueError:
                                    pass
            except FileNotFoundError:
                pass

        return processes

    def validate_domain(self, domain: str) -> dict[str, Any]:
        """Validate domain configuration and return manifest."""
        return self.manifest_loader.load_manifest(domain)

    def start_domain(
        self,
        domain: str,
        rest_port: int = 8000,
        mcp_port: int = 8001,
        log_level: str = "INFO",
        build: str | None = None,
        rebuild: str | None = None,
        dry_run: bool = False,
    ) -> None:
        """Start a domain server as a host subprocess.

        The server runs directly on the host machine, not in a container,
        which eliminates Docker-in-Docker issues when managing sandbox containers.
        """
        # Check if server is already running (before validation to avoid port conflict errors)
        status = self.get_domain_status(domain)
        if status.get("running", False) and status.get("health_status") == "healthy":
            print(f"Server for domain {domain} is already running (PID: {status.get('pid')})")
            return

        # Load and validate manifest
        manifest = self.manifest_loader.load_manifest(domain)

        # Build sandbox images if requested (server no longer needs image)
        if rebuild is not None:
            image_filter = rebuild if rebuild else None
            self.docker_runner.build_images(
                domain,
                manifest,
                self.manifest_loader.domains_root,
                dry_run,
                image_filter=image_filter,
                rebuild_mode=True,
            )
        elif build is not None:
            image_filter = build if build else None
            self.docker_runner.build_images(
                domain,
                manifest,
                self.manifest_loader.domains_root,
                dry_run,
                image_filter=image_filter,
                rebuild_mode=False,
            )

        # Generate and validate environment for host subprocess
        env_vars = self.environment_validator.generate_environment(
            domain,
            manifest,
            rest_port,
            mcp_port,
            log_level,
            skip_image_check=(rebuild is not None or build is not None),
        )

        # Start server as subprocess
        self._start_server_subprocess(domain, env_vars, dry_run)

    def _start_server_subprocess(self, domain: str, env_vars: dict[str, str], dry_run: bool = False) -> None:
        """Start SABER server as a host subprocess.

        Args:
            domain: Domain name
            env_vars: Environment variables for the server
            dry_run: If True, show commands without executing
        """
        if dry_run:
            print(f"Would start server subprocess for domain {domain} with environment:")
            for key, value in sorted(env_vars.items()):
                print(f"  {key}={value}")
            print(f"Would run: {sys.executable} -m saber.server")
            return

        # Check if already running (in-memory)
        if domain in DomainOrchestrator._running_servers:
            proc = DomainOrchestrator._running_servers[domain]
            if proc.poll() is None:
                print(f"Server for domain {domain} is already running (PID: {proc.pid})")
                return
            # Process has terminated, clean up
            del DomainOrchestrator._running_servers[domain]

        # Also check PID file for servers started by other processes
        domain_root = self._domains_root / domain
        pid_file = domain_root / "server" / "logs" / f"{domain}.pid"
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 0)  # Check if process exists
                print(f"Server for domain {domain} is already running (PID: {pid}, started externally)")
                return
            except (ValueError, OSError):
                # Invalid PID or process not running - clean up stale PID file
                try:
                    pid_file.unlink()
                except OSError:
                    pass

        # Prepare environment (inherit current env and add server-specific vars)
        server_env = os.environ.copy()
        server_env.update(env_vars)

        # Load .env file if it exists
        env_file = env_vars.get("ENV_FILE")
        if env_file and Path(env_file).exists():
            self._load_env_file(server_env, Path(env_file))

        # Set up log file for stdout/stderr
        domain_root = self._domains_root / domain
        log_dir = domain_root / "server" / "logs" / "server-logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file_path = log_dir / f"saber-server-{timestamp}.log"

        print(f"Starting server subprocess for domain {domain}...", flush=True)
        print(f"  Log file: {log_file_path}", flush=True)

        # Track if we started the server (for cleanup on interrupt)
        server_started = False

        # Set up signal handler for graceful cleanup on Ctrl+C
        # Only works in main thread (CLI usage); skip for thread pool (inspect_ai)
        import signal
        import threading

        in_main_thread = threading.current_thread() is threading.main_thread()
        original_sigint = None

        if in_main_thread:
            original_sigint = signal.getsignal(signal.SIGINT)

            def cleanup_on_interrupt(signum: int, frame: Any) -> None:
                print("\n\n⚠️  Interrupted! Cleaning up...", flush=True)
                if server_started:
                    self._stop_server_subprocess(domain)
                    self._cleanup_sandbox_containers(domain)
                # Restore original handler and re-raise
                signal.signal(signal.SIGINT, original_sigint)
                raise KeyboardInterrupt()

            signal.signal(signal.SIGINT, cleanup_on_interrupt)

        try:
            # Open log file for subprocess output
            log_file = open(log_file_path, "w")

            # Start server as subprocess
            # Note: --domain and --domains-root are passed via SABER_DOMAIN/SABER_DOMAINS_ROOT env vars
            proc = subprocess.Popen(
                [sys.executable, "-m", "saber.server", "--start"],
                env=server_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # Detach from terminal
            )

            server_started = True

            # Store process reference
            DomainOrchestrator._running_servers[domain] = proc

            # Write PID file for cross-process tracking
            pid_file = domain_root / "server" / "logs" / f"{domain}.pid"
            pid_file.write_text(str(proc.pid))

            # Store log file reference for cleanup
            self._store_log_file(domain, log_file)

            # Give server a moment to start or fail immediately
            time.sleep(2)

            # Check if process is still running
            if proc.poll() is not None:
                # Process terminated - read log file for error details
                log_file.close()
                with open(log_file_path) as f:
                    error_output = f.read()
                print(f"\n❌ Server process terminated immediately (exit code: {proc.returncode})", flush=True)
                print("Server output:", flush=True)
                print("=" * 80, flush=True)
                print(error_output or "(no output)", flush=True)
                print("=" * 80, flush=True)
                del DomainOrchestrator._running_servers[domain]
                raise DockerError(
                    f"Server subprocess failed to start (exit code: {proc.returncode})",
                    command=f"{sys.executable} -m saber.server",
                )

            print(f"✓ Server subprocess started (PID: {proc.pid})", flush=True)

            # Wait for health check
            self._wait_for_server_health(domain, env_vars)

            # Wait for permanent containers to be healthy
            self._wait_for_permanent_containers_health(domain)

        except KeyboardInterrupt:
            # Re-raise to let CLI handle it
            raise
        except Exception as e:
            if not isinstance(e, DockerError):
                raise DockerError(f"Failed to start server subprocess: {e}") from e
            raise
        finally:
            # Restore original signal handler (only if we set one up)
            if in_main_thread and original_sigint is not None:
                signal.signal(signal.SIGINT, original_sigint)

    def _load_env_file(self, env: dict[str, str], env_file: Path) -> None:
        """Load environment variables from .env file."""
        try:
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip()
                        # Remove surrounding quotes if present
                        if value and value[0] in ('"', "'") and value[-1] == value[0]:
                            value = value[1:-1]
                        env[key] = value
        except Exception:
            pass  # Best effort - .env file is optional

    # Log file storage for cleanup
    _log_files: dict[str, IO] = {}

    def _store_log_file(self, domain: str, log_file: IO) -> None:
        """Store log file reference for later cleanup."""
        DomainOrchestrator._log_files[domain] = log_file

    def _wait_for_server_health(self, domain: str, env_vars: dict[str, str], timeout: int = 60) -> None:
        """Wait for server to become healthy via HTTP health check.

        Args:
            domain: Domain name
            env_vars: Environment variables (for port info)
            timeout: Maximum time to wait in seconds

        Raises:
            DockerError: If server fails health check
        """
        rest_port = int(env_vars.get("REST_PORT", env_vars.get("SABER_PORT", "8000")))
        print(f"Waiting for server health check on port {rest_port}...", flush=True)

        start_time = time.time()
        attempt = 0

        while time.time() - start_time < timeout:
            attempt += 1

            # Check if process is still running
            if domain in DomainOrchestrator._running_servers:
                proc = DomainOrchestrator._running_servers[domain]
                if proc.poll() is not None:
                    elapsed = time.time() - start_time
                    exit_code = proc.returncode
                    print(
                        f"\n❌ Server crashed during health check after {elapsed:.0f}s (exit: {exit_code})",
                        flush=True,
                    )
                    self._show_subprocess_failure_diagnostics(domain)
                    raise DockerError(
                        f"Server process crashed during health check (exit code: {proc.returncode})",
                        command=f"{sys.executable} -m saber.server",
                    )

            # Try to connect to health endpoint
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(2)
                    if sock.connect_ex(("localhost", rest_port)) == 0:
                        elapsed = time.time() - start_time
                        print(f"✓ Server is responding (after {elapsed:.1f}s, {attempt} attempts)", flush=True)
                        return
            except Exception:
                pass

            # Show progress every 10 seconds
            if attempt % 5 == 0:
                elapsed = time.time() - start_time
                print(f"  Still waiting... ({elapsed:.0f}s elapsed, attempt {attempt})", flush=True)

            time.sleep(2)

        # Health check timeout
        elapsed = time.time() - start_time
        print(f"\n❌ Server health check TIMEOUT after {elapsed:.0f}s ({attempt} attempts)", flush=True)
        self._show_subprocess_failure_diagnostics(domain)
        self._stop_server_subprocess(domain)
        raise DockerError(
            f"Server failed to become healthy within {timeout}s.\n\n"
            "Common causes:\n"
            "  1. Server crash during startup (check logs above)\n"
            "  2. Missing dependencies or configuration\n"
            "  3. Port conflicts or network issues\n",
            command=f"{sys.executable} -m saber.server",
        )

    def _show_subprocess_failure_diagnostics(self, domain: str) -> None:
        """Show diagnostic information when server subprocess fails."""
        print("\nRetrieving diagnostic information...\n", flush=True)

        # Try to find and display the most recent server log file
        domain_root = self._domains_root / domain
        server_logs_dir = domain_root / "server" / "logs" / "server-logs"

        if server_logs_dir.exists():
            log_files = sorted(
                server_logs_dir.glob("saber-server-*.log*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            if log_files:
                latest_log = log_files[0]
                print(f"📄 Server log file: {latest_log}", flush=True)
                print("=" * 80, flush=True)

                try:
                    with open(latest_log) as f:
                        # Read last 100 lines
                        lines = f.readlines()
                        tail_lines = lines[-100:] if len(lines) > 100 else lines
                        print("".join(tail_lines), flush=True)
                except Exception as e:
                    print(f"Error reading log file: {e}", flush=True)

                print("=" * 80, flush=True)

    def _wait_for_permanent_containers_health(self, domain: str, timeout: int = 120) -> None:
        """Wait for permanent environment containers to become healthy.

        Args:
            domain: Domain name
            timeout: Maximum time to wait in seconds

        Raises:
            DockerError: If containers fail to become healthy
        """
        permanent_project_name = f"{domain}_permanent_environment"

        # Find containers belonging to permanent environment
        try:
            find_result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "-a",
                    "--filter",
                    f"label=com.docker.compose.project={permanent_project_name}",
                    "--format",
                    "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            container_names = [n.strip() for n in find_result.stdout.strip().split("\n") if n.strip()]
        except Exception:
            container_names = []

        if not container_names:
            # No permanent containers to wait for
            return

        print(f"Waiting for {len(container_names)} permanent container(s) to be healthy...", flush=True)

        start_time = time.time()
        while time.time() - start_time < timeout:
            all_healthy = True
            status_summary = []

            for container_name in container_names:
                try:
                    # Get container health status
                    result = subprocess.run(
                        [
                            "docker",
                            "inspect",
                            "--format",
                            "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
                            container_name,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    health_status = result.stdout.strip()

                    if health_status == "healthy":
                        status_summary.append(f"  ✓ {container_name}: healthy")
                    elif health_status == "none":
                        # Container has no healthcheck - consider it ready if running
                        run_result = subprocess.run(
                            ["docker", "inspect", "--format", "{{.State.Running}}", container_name],
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )
                        if run_result.stdout.strip() == "true":
                            status_summary.append(f"  ✓ {container_name}: running (no healthcheck)")
                        else:
                            status_summary.append(f"  ⏳ {container_name}: not running")
                            all_healthy = False
                    else:
                        status_summary.append(f"  ⏳ {container_name}: {health_status}")
                        all_healthy = False
                except Exception as e:
                    status_summary.append(f"  ❌ {container_name}: error ({e})")
                    all_healthy = False

            if all_healthy:
                for line in status_summary:
                    print(line, flush=True)
                print("✓ All permanent containers are healthy", flush=True)
                return

            # Show progress every 10 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                print(f"  Still waiting for containers... ({elapsed:.0f}s elapsed)", flush=True)

            time.sleep(2)

        # Timeout - show final status
        elapsed = time.time() - start_time
        print(f"\n⚠️  Permanent container health check timed out after {elapsed:.0f}s", flush=True)
        for line in status_summary:
            print(line, flush=True)
        # Don't fail - just warn, as sometimes healthchecks take longer

    def _stop_server_subprocess(self, domain: str, timeout: int = 10) -> bool:
        """Stop server subprocess gracefully.

        Args:
            domain: Domain name
            timeout: Seconds to wait for graceful shutdown

        Returns:
            True if stopped successfully, False otherwise
        """
        domain_root = self._domains_root / domain
        pid_file = domain_root / "server" / "logs" / f"{domain}.pid"

        # Try to get process from in-memory tracking first
        proc = DomainOrchestrator._running_servers.get(domain)
        pid = None

        if proc is None:
            # Not in memory - try to read PID file
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text().strip())
                except (ValueError, OSError):
                    pass

        if proc is None and pid is None:
            # No process tracked
            if pid_file.exists():
                pid_file.unlink()
            return True

        if proc is not None:
            # Check if already terminated
            if proc.poll() is not None:
                del DomainOrchestrator._running_servers[domain]
                self._close_log_file(domain)
                if pid_file.exists():
                    pid_file.unlink()
                return True
            pid = proc.pid

        print(f"Stopping server subprocess (PID: {pid})...", flush=True)

        try:
            import signal

            if pid is None:
                print("No PID found - server may have already stopped", flush=True)
                return True

            # Send SIGTERM for graceful shutdown
            os.kill(pid, signal.SIGTERM)

            # Wait for process to terminate
            for _ in range(timeout * 10):
                try:
                    os.kill(pid, 0)  # Check if process exists
                    time.sleep(0.1)
                except OSError:
                    # Process terminated
                    break
            else:
                # Still running after timeout - force kill
                print("Server did not stop gracefully, sending SIGKILL...", flush=True)
                os.kill(pid, signal.SIGKILL)
                time.sleep(0.5)

            print("✓ Server stopped", flush=True)

            # Cleanup
            if domain in DomainOrchestrator._running_servers:
                del DomainOrchestrator._running_servers[domain]
            self._close_log_file(domain)
            if pid_file.exists():
                pid_file.unlink()
            return True

        except ProcessLookupError:
            # Process already gone
            print("✓ Server already stopped", flush=True)
            if pid_file.exists():
                pid_file.unlink()
            return True
        except Exception as e:
            print(f"Warning: Failed to stop server subprocess: {e}", flush=True)
            return False

    def _close_log_file(self, domain: str) -> None:
        """Close log file if open."""
        if domain in DomainOrchestrator._log_files:
            try:
                DomainOrchestrator._log_files[domain].close()
            except Exception:
                pass
            del DomainOrchestrator._log_files[domain]

    def stop_domain(self, domain: str, dry_run: bool = False) -> None:
        """Stop a domain server subprocess."""
        # Validate domain exists
        self.manifest_loader.load_manifest(domain)

        if dry_run:
            print(f"Would stop server subprocess for domain {domain}")
            return

        # Stop server subprocess
        self._stop_server_subprocess(domain)

        # Also clean up any sandbox/permanent environment containers
        self._cleanup_sandbox_containers(domain)

    def _cleanup_sandbox_containers(self, domain: str) -> None:
        """Clean up sandbox and permanent environment containers for a domain."""
        # Clean up permanent environment containers if any
        permanent_project_name = f"{domain}_permanent_environment"

        try:
            # Find all containers belonging to permanent environment project
            find_result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "-aq",
                    "--filter",
                    f"label=com.docker.compose.project={permanent_project_name}",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            container_ids = find_result.stdout.strip().split()
            if container_ids and container_ids[0]:
                print(f"Stopping {len(container_ids)} permanent environment container(s)...", flush=True)
                for container_id in container_ids:
                    try:
                        subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, timeout=30)
                    except Exception:
                        pass
                print("✓ Permanent environment containers stopped", flush=True)

        except Exception as e:
            print(f"Warning: Failed to clean up permanent environment containers: {e}", flush=True)

    def build_domain(
        self, domain: str, image_filter: str | None = None, dry_run: bool = False, rebuild_mode: bool = True
    ) -> None:
        """Build domain images.

        Args:
            domain: Domain name
            image_filter: Optional substring filter for image names (e.g., 'server', 'cookie', 'sandbox')
            dry_run: Show commands without executing
            rebuild_mode: If True, remove and rebuild. If False, only build missing images.
        """
        # Load and validate manifest
        manifest = self.manifest_loader.load_manifest(domain)

        # Build images
        self.docker_runner.build_images(
            domain,
            manifest,
            self.manifest_loader.domains_root,
            dry_run,
            image_filter=image_filter,
            rebuild_mode=rebuild_mode,
        )

    def get_domain_status(self, domain: str) -> dict[str, Any]:
        """Get domain status by checking server subprocess and health endpoint."""
        # Load manifest
        try:
            manifest = self.manifest_loader.load_manifest(domain)
        except Exception as e:
            return {"running": False, "services": [], "error": f"Domain manifest unavailable: {e}"}

        # Check if subprocess is running (in-memory or via PID file)
        proc = DomainOrchestrator._running_servers.get(domain)
        pid = None

        if proc is None:
            # Not in memory - try to read PID file (started by another process or CLI)
            domain_root = self._domains_root / domain
            pid_file = domain_root / "server" / "logs" / f"{domain}.pid"
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text().strip())
                    # Check if process is still running
                    os.kill(pid, 0)
                except (ValueError, OSError):
                    # Invalid PID or process not running
                    pid = None

        if proc is None and pid is None:
            return {
                "running": False,
                "services": [],
                "health_status": "stopped",
                "domain": domain,
                "manifest": manifest,
            }

        # Check if in-memory process is still alive
        if proc is not None:
            if proc.poll() is not None:
                # Process has terminated
                del DomainOrchestrator._running_servers[domain]
                return {
                    "running": False,
                    "services": [],
                    "health_status": "stopped",
                    "domain": domain,
                    "manifest": manifest,
                    "exit_code": proc.returncode,
                }
            pid = proc.pid

        # Process is running, check health endpoint
        try:
            env_vars = self.environment_validator.generate_environment(
                domain, manifest, 8000, 8001, "INFO", skip_image_check=True, skip_port_check=True
            )
            rest_port = int(env_vars.get("REST_PORT", env_vars.get("SABER_PORT", "8000")))
        except Exception:
            rest_port = 8000

        health_status = "unknown"
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(2)
                if sock.connect_ex(("localhost", rest_port)) == 0:
                    health_status = "healthy"
                else:
                    health_status = "starting"
        except Exception:
            health_status = "starting"

        return {
            "running": True,
            "services": [
                {
                    "Name": f"{domain}-saber-server",
                    "State": "running",
                    "Status": f"Up (PID: {pid})",
                }
            ],
            "health_status": health_status,
            "domain": domain,
            "manifest": manifest,
            "pid": pid,
        }
