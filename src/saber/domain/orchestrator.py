"""Core orchestration services for SABER domain management.

This module provides modular services for domain orchestration:
- ManifestLoader: Load and validate domain manifests
- EnvironmentValidator: Validate environment and generate variables
- DockerRunner: Execute Docker operations
- DomainOrchestrator: Coordinate the above services
"""

import json
import os
import socket
import subprocess
import tempfile
import time
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, List

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

    def list_domains(self) -> List[str]:
        """List all available domains."""
        domains = []
        for path in self.domains_root.iterdir():
            if path.is_dir() and (path / "domain.yaml").exists():
                domains.append(path.name)
        return sorted(domains)

    def load_manifest(self, domain: str) -> Dict[str, Any]:
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
            with open(manifest_path, "r") as f:
                manifest = yaml.safe_load(f)
                if not isinstance(manifest, dict):
                    raise DomainValidationError(domain, ["Manifest must be a YAML object/dictionary"])
        except yaml.YAMLError as e:
            raise DomainValidationError(domain, [f"Invalid YAML in manifest: {e}"])

        # Validate against schema
        self._validate_schema(manifest, domain)

        # Validate domain structure
        self._validate_structure(domain, manifest)

        return manifest

    def _validate_schema(self, manifest: Dict[str, Any], domain: str) -> None:
        """Validate manifest against JSON schema."""
        try:
            import jsonschema
        except ImportError:
            raise DomainValidationError(
                domain, ["jsonschema package required for validation. Install with: uv add jsonschema"]
            )

        try:
            with resolve_schema_file() as schema_path:
                with open(schema_path, "r") as f:
                    schema = json.load(f)
        except Exception as e:
            raise DomainValidationError(domain, [f"Failed to load validation schema: {e}"])

        try:
            jsonschema.validate(manifest, schema)
        except jsonschema.ValidationError as e:
            error_path = " -> ".join(str(p) for p in e.absolute_path)
            raise DomainValidationError(domain, [f"Schema validation failed at {error_path}: {e.message}"])

    def _validate_structure(self, domain: str, manifest: Dict[str, Any]) -> None:
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
            errors.append(f"Domain slug mismatch: manifest declares '{manifest_slug}' " f"but directory is '{domain}'")

        if errors:
            raise DomainValidationError(domain, errors)


class EnvironmentValidator:
    """Service for validating environment and generating variables."""

    def __init__(self, domains_root: Path):
        self.domains_root = domains_root.resolve()

    def generate_environment(
        self,
        domain: str,
        manifest: Dict[str, Any],
        rest_port: int = 8000,
        mcp_port: int = 8001,
        log_level: str = "INFO",
        skip_image_check: bool = False,
        skip_port_check: bool = False,
    ) -> Dict[str, str]:
        """Generate and validate environment variables for server-only architecture.

        Args:
            domain: Domain name
            manifest: Domain manifest
            rest_port: REST API port
            mcp_port: MCP protocol port
            log_level: Logging level
            skip_image_check: Skip Docker image existence check
            skip_port_check: Skip port availability check (for status checks)

        Returns:
            Dict of environment variables for docker-compose

        Raises:
            DomainValidationError: If validation fails
        """
        errors = []

        # Validate ports are available (unless skipping for status checks)
        if not skip_port_check:
            for port_name, port_num in [("REST", rest_port), ("MCP", mcp_port)]:
                if not self._is_port_available(port_num):
                    errors.append(f"Port {port_num} ({port_name}) is already in use")

        # Validate all domain images exist (unless building)
        if not skip_image_check:
            # First validate base images
            self._validate_base_images(errors)
            # Then validate domain images
            self._validate_all_images(manifest, errors)

        if errors:
            raise DomainValidationError(domain, errors)

        # Detect repository structure for SABER source code mounting
        saber_src_path, env_file_path = self._detect_repo_structure()

        # Generate environment variables (server-only)
        return {
            "DOMAIN": domain,
            "DOMAINS_ROOT": str(self.domains_root),
            "SERVER_IMAGE": manifest["images"]["server"]["tag"],
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

    def _validate_all_images(self, manifest: Dict[str, Any], errors: List[str]) -> None:
        """Validate all domain images exist."""
        images_config = manifest.get("images", {})
        if not images_config:
            errors.append("No images configuration found in manifest")
            return

        # Validate server image (required)
        server_config = images_config.get("server")
        if not server_config:
            errors.append("No server image configuration found in manifest")
            return

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
                    (
                        "Docker image '%s' not found.\n"
                        "  Build with inspect eval: uv run inspect eval domains/%s --model <model> -T build=true\n"
                        "  Or use saber-domain CLI: uv run saber-domain build %s --build"
                    )
                    % (image_tag, slug, slug)
                )

    def _validate_base_images(self, errors: List[str]) -> None:
        """Validate base images exist."""
        try:
            # Load base images config from package resources
            base_images_file = files(saber.domain.package_resources) / "base-images.yaml"
            with base_images_file.open("r") as f:
                base_images_config = yaml.safe_load(f)

            for image_name, image_config in base_images_config["images"].items():
                image_tag = image_config["tag"]
                if not self._docker_image_exists(image_tag):
                    # Keep message lines under 120 chars for flake8
                    errors.append(
                        (
                            "Base image '%s' not found.\n"
                            "  Build with inspect eval: uv run inspect eval domains/<domain> \
                                --model <model> -T build=true\n"
                            "  Or use saber-domain CLI: uv run saber-domain build <domain> --build"
                        )
                        % (image_tag,)
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
    """Service for executing Docker operations."""

    def __init__(self, compose_file: Path, domains_root: Path):
        self.compose_file = compose_file.resolve()
        self.domains_root = domains_root
        if not self.compose_file.exists():
            raise DockerError(f"Compose file not found: {self.compose_file}", command=None)

        # Validate Docker is available
        self._validate_docker()

    def _docker_image_exists(self, image_tag: str) -> bool:
        """Check if Docker image exists locally."""
        try:
            result = subprocess.run(["docker", "image", "inspect", image_tag], capture_output=True, check=False)
            return result.returncode == 0
        except Exception:
            return False

    def _detect_repo_structure_for_stop(self, domains_root: Path) -> tuple[Path, Path]:
        """Detect repository structure for stop operations.

        Args:
            domains_root: Path to domains directory

        Returns:
            Tuple of (saber_src_path, env_file_path)
        """
        return detect_repo_structure(domains_root)

    def build_images(
        self,
        domain: str,
        manifest: Dict[str, Any],
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
            image_filter: Optional prefix to filter which images to build (e.g., 'server', 'cookie', 'sandbox')
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
            images_to_build = {name: config for name, config in images_config.items() if name.startswith(image_filter)}
            if not images_to_build:
                raise DockerError(
                    f"No images found matching filter '{image_filter}'. "
                    f"Available images: {', '.join(images_config.keys())}"
                )
            print(f"🔍 Filtered images by prefix '{image_filter}': {', '.join(images_to_build.keys())}")
        else:
            images_to_build = images_config

        # Ensure base images exist first
        # In rebuild mode: rebuild base images if no filter or filter matches base images
        # In build mode: ensure base images exist (build if missing) when needed
        needs_base_images = any(name in ["server", "sandbox"] for name in images_to_build.keys())

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
            for image_name, image_config in images_to_build.items():
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
            cmd = ["docker", "build", "-f", str(dockerfile_path), "-t", image_tag, str(context_path)]

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
                raise DockerError(f"Failed to build {image_name} image {image_tag}: {e}")

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
        for image_name, image_config in base_images_config["images"].items():
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
            dockerfile = image_config["dockerfile"]
            image_tag = image_config["tag"]
            labels = image_config.get("labels", {})

            # Handle package:// scheme for packaged Dockerfiles
            if dockerfile.startswith("package://"):
                package_path = dockerfile[len("package://") :]
                self._build_base_image_from_package(image_name, image_tag, package_path, labels, dry_run)
            else:
                raise DockerError(f"Unsupported dockerfile path format: {dockerfile}")

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

        print(f"🔨 Building {len(missing_images)} missing base image(s)...")

        for image_name, image_config in missing_images:
            dockerfile = image_config["dockerfile"]
            image_tag = image_config["tag"]
            labels = image_config.get("labels", {})

            # Handle package:// scheme for packaged Dockerfiles
            if dockerfile.startswith("package://"):
                package_path = dockerfile[len("package://") :]
                self._build_base_image_from_package(image_name, image_tag, package_path, labels, dry_run)
            else:
                raise DockerError(f"Unsupported dockerfile path format: {dockerfile}")

    def _build_base_image_from_package(
        self, image_name: str, image_tag: str, package_path: str, labels: Dict[str, str], dry_run: bool
    ) -> None:
        """Build base image from packaged Dockerfile using stdin (no temporary files).

        This approach pipes the Dockerfile content directly to docker build via stdin,
        avoiding the need to create temporary files in the repo root.
        """
        try:
            # Use repo root as build context to access external/saber
            repo_root = self.domains_root.parent

            # Detect if we're in a standalone saber repo or oss_saber repo
            # In standalone saber repo: copy the whole repo (.)
            # In oss_saber repo: copy external/saber subdirectory
            saber_src_path = "." if (repo_root / "src" / "saber").exists() else "external/saber"

            # Read Dockerfile content from package resources
            dockerfile_resource = files(saber.domain.package_resources) / package_path
            dockerfile_content = dockerfile_resource.read_text()

            # Prepare build command with Dockerfile from stdin (-f -)
            cmd = ["docker", "build", "-f", "-", "-t", image_tag]

            # Pass SABER_SRC_PATH as build arg
            cmd.extend(["--build-arg", f"SABER_SRC_PATH={saber_src_path}"])

            # Add labels
            for key, value in labels.items():
                cmd.extend(["--label", f"{key}={value}"])

            # Add build context (repo root)
            cmd.append(str(repo_root))

            if dry_run:
                print(f"Would build base image {image_name}: {' '.join(cmd)}")
                print(f"Dockerfile content from: {package_path}")
                print(f"Using SABER source path: {saber_src_path}")
                return

            print(f"Building base image: {image_tag} (SABER_SRC_PATH={saber_src_path})")
            subprocess.run(cmd, input=dockerfile_content, text=True, check=True, cwd=repo_root)
            print(f"✓ Successfully built {image_tag}")

        except subprocess.CalledProcessError as e:
            raise DockerError(f"Failed to build base image {image_tag}: {e}")
        except Exception as e:
            raise DockerError(f"Failed to prepare base image build: {e}")

    def _load_base_images_config(self) -> Dict[str, Any]:
        """Load base images configuration."""
        try:
            base_images_file = files(saber.domain.package_resources) / "base-images.yaml"
            with base_images_file.open("r") as f:
                config: Dict[str, Any] = yaml.safe_load(f)
                return config
        except Exception as e:
            raise DockerError(f"Failed to load base images configuration: {e}")

    def start_services(self, domain: str, env_vars: Dict[str, str], dry_run: bool = False) -> None:
        """Start domain services using docker-compose.

        Args:
            domain: Domain name
            env_vars: Environment variables
            dry_run: If True, show commands without executing

        Raises:
            DockerError: If start fails
        """
        if dry_run:
            print(f"Would start domain {domain} with environment:")
            for key, value in sorted(env_vars.items()):
                print(f"  {key}={value}")
            print(f"Would run: docker compose -f {self.compose_file} up -d")
            return

        # Create temporary env file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
            env_file = f.name

        try:
            cmd = [
                "docker",
                "compose",
                "-f",
                str(self.compose_file),
                "--env-file",
                env_file,
                "--project-name",
                domain,
                "up",
                "-d",
            ]

            print(f"Starting domain {domain}...", flush=True)
            subprocess.run(cmd, check=True)

            # Give container a moment to start or fail immediately
            time.sleep(2)

            # Check if container is still running (detect immediate crashes)
            container_name = f"{domain}-saber-server"
            check_result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Status}}", container_name],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if check_result.returncode == 0:
                container_status = check_result.stdout.strip()
                if container_status not in ["running", "starting"]:
                    # Container crashed immediately - show logs before failing
                    print(
                        f"\n❌ Container {container_name} crashed immediately (status: {container_status})", flush=True
                    )
                    self._show_failure_diagnostics(domain, env_vars, container_name, "Container crashed during startup")
                    raise DockerError(
                        f"Container {container_name} failed to start (status: {container_status})",
                        command=" ".join(cmd),
                    )

            print(f"✓ Domain {domain} containers started, checking health...", flush=True)

            # Wait for health checks
            self._wait_for_services(domain, env_vars)

        except subprocess.CalledProcessError as e:
            raise DockerError(f"Failed to start domain {domain}", command=" ".join(cmd), exit_code=e.returncode)
        finally:
            # Clean up temp env file
            try:
                os.unlink(env_file)
            except OSError:
                pass

    def stop_services(self, domain: str, domains_root: Path, dry_run: bool = False) -> None:
        """Stop domain services.

        Args:
            domain: Domain name
            domains_root: Path to domains directory
            dry_run: If True, show commands without executing

        Raises:
            DockerError: If stop fails
        """
        if dry_run:
            print(f"Would stop domain {domain}")
            return

        # Detect repository structure for required paths
        saber_src_path, env_file_path = self._detect_repo_structure_for_stop(domains_root)

        # Create minimal environment file for compose down
        minimal_env = {
            "DOMAIN": domain,
            "DOMAINS_ROOT": str(domains_root),
            "SERVER_IMAGE": "dummy",  # Not needed for 'down' but required by compose file
            "REST_PORT": "8000",  # Not needed for 'down' but required by compose file
            "MCP_PORT": "8001",  # Not needed for 'down' but required by compose file
            "LOG_LEVEL": "INFO",  # Not needed for 'down' but required by compose file
            "SABER_SRC": str(saber_src_path),  # Required by compose file volume mounts
            "ENV_FILE": str(env_file_path),  # Required by compose file volume mounts
        }

        # Create temporary env file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            for key, value in minimal_env.items():
                f.write(f"{key}={value}\n")
            env_file = f.name

        try:
            cmd = [
                "docker",
                "compose",
                "-f",
                str(self.compose_file),
                "--env-file",
                env_file,
                "--project-name",
                domain,
                "down",
            ]

            print(f"Stopping domain {domain}...")
            subprocess.run(cmd, check=True)
            print(f"✓ Domain {domain} stopped successfully")
        except subprocess.CalledProcessError as e:
            raise DockerError(f"Failed to stop domain {domain}", command=" ".join(cmd), exit_code=e.returncode)
        finally:
            # Clean up temporary env file
            try:
                os.unlink(env_file)
            except Exception:
                pass

    def _validate_docker(self) -> None:
        """Validate Docker and Docker Compose are available."""
        for cmd in [["docker", "--version"], ["docker", "compose", "version"]]:
            try:
                subprocess.run(cmd, capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise DockerError(
                    f"Command '{' '.join(cmd)}' failed. Ensure Docker and Docker Compose are installed.",
                    command=" ".join(cmd),
                )

    def _add_git_metadata(self, cmd: List[str], domain_path: Path) -> None:
        """Add git metadata labels to build command."""
        try:
            git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=domain_path, text=True).strip()
            cmd.extend(["--label", f"saber.git.sha={git_sha}"])
        except subprocess.CalledProcessError:
            pass

    def _show_failure_diagnostics(
        self, domain: str, env_vars: Dict[str, str], container_name: str, reason: str
    ) -> None:
        """Show diagnostic information when server fails to start.

        Args:
            domain: Domain name
            env_vars: Environment variables
            container_name: Name of the container
            reason: Brief reason for the failure
        """
        print("\nRetrieving diagnostic information...\n", flush=True)

        # Try to get container logs for debugging
        try:
            logs_result = subprocess.run(
                ["docker", "logs", "--tail", "100", container_name], capture_output=True, text=True, timeout=5
            )
            if logs_result.stdout or logs_result.stderr:
                print(f"Container logs from {container_name}:", flush=True)
                print("=" * 80, flush=True)
                if logs_result.stdout:
                    print(logs_result.stdout, flush=True)
                if logs_result.stderr:
                    print(logs_result.stderr, flush=True)
                print("=" * 80, flush=True)
        except Exception as e:
            print(f"Could not retrieve container logs: {e}", flush=True)

        # Try to find and display the most recent server log file with errors
        print("\nSearching for server log file errors...\n", flush=True)
        try:
            # Path to server logs directory (from domains_root)
            domains_root = Path(env_vars.get("DOMAINS_ROOT", "."))
            server_logs_dir = domains_root / domain / "server" / "logs" / "server-logs"

            if server_logs_dir.exists():
                # Find the most recent log file
                log_files = sorted(
                    server_logs_dir.glob("saber-server-*.log*"), key=lambda p: p.stat().st_mtime, reverse=True
                )

                if log_files:
                    latest_log = log_files[0]
                    print(f"📄 Most recent server log file: {latest_log}", flush=True)
                    print("=" * 80, flush=True)

                    # Read the log file and extract error lines
                    error_lines = []
                    warning_lines = []
                    try:
                        with open(latest_log, "r") as f:
                            for line in f:
                                line_lower = line.lower()
                                if (
                                    "error" in line_lower
                                    or "exception" in line_lower
                                    or "traceback" in line_lower
                                    or "failed" in line_lower
                                ):
                                    error_lines.append(line.rstrip())
                                elif "warning" in line_lower or "warn" in line_lower:
                                    warning_lines.append(line.rstrip())

                        if error_lines:
                            print(f"\n🔴 Found {len(error_lines)} error/exception lines:", flush=True)
                            print("-" * 80, flush=True)
                            # Show all errors (they're usually not that many)
                            for line in error_lines:
                                print(line, flush=True)
                            print("-" * 80, flush=True)
                        else:
                            print("No obvious errors found in log file.", flush=True)

                        if warning_lines and len(warning_lines) <= 20:
                            print(f"\n⚠️  Found {len(warning_lines)} warnings:", flush=True)
                            print("-" * 80, flush=True)
                            for line in warning_lines:
                                print(line, flush=True)
                            print("-" * 80, flush=True)
                        elif warning_lines:
                            print(f"\n⚠️  Found {len(warning_lines)} warnings (showing last 20):", flush=True)
                            print("-" * 80, flush=True)
                            for line in warning_lines[-20:]:
                                print(line, flush=True)
                            print("-" * 80, flush=True)

                    except Exception as e:
                        print(f"Error reading log file: {e}", flush=True)

                    print("=" * 80, flush=True)

                    # Print clear message about where to find the full log
                    print("\n💡 For complete details, view the full log file:", flush=True)
                    print(f"   {latest_log}", flush=True)
                    print(f"   (Use: cat {latest_log} | less)", flush=True)
                else:
                    print(f"No log files found in {server_logs_dir}", flush=True)
            else:
                print(f"Server logs directory not found: {server_logs_dir}", flush=True)

        except Exception as e:
            print(f"Could not access server log files: {e}", flush=True)

        # Stop the containers since they're unhealthy
        print("\nStopping unhealthy containers...", flush=True)
        try:
            subprocess.run(
                ["docker", "compose", "-f", str(self.compose_file), "-p", domain, "down"],
                capture_output=True,
                timeout=30,
            )
            print("✓ Containers stopped", flush=True)
        except Exception as e:
            print(f"Warning: Failed to stop containers: {e}", flush=True)

    def _wait_for_services(self, domain: str, env_vars: Dict[str, str], timeout: int = 60) -> None:
        """Wait for services to be healthy.

        Raises:
            DockerError: If services fail to become healthy within timeout
        """
        # Always check health for server in new architecture (no profiles)
        rest_port = int(env_vars.get("REST_PORT", "8000"))
        container_name = f"{domain}-saber-server"
        print(f"Waiting for server health check on port {rest_port}...", flush=True)

        start_time = time.time()
        attempt = 0
        while time.time() - start_time < timeout:
            attempt += 1

            # Check if container is still running before checking port
            check_result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Status}}", container_name],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if check_result.returncode == 0:
                container_status = check_result.stdout.strip()
                if container_status not in ["running", "starting"]:
                    # Container crashed during health check
                    elapsed = time.time() - start_time
                    print(
                        f"\n❌ Container crashed during health check after {elapsed:.0f}s (status: {container_status})",
                        flush=True,
                    )
                    self._show_failure_diagnostics(domain, env_vars, container_name, "Container crashed")
                    raise DockerError(
                        f"Container {container_name} crashed during health check (status: {container_status})",
                        command="docker compose up -d (health check)",
                    )

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

        # Health check timeout - show diagnostics
        elapsed = time.time() - start_time
        print(f"\n❌ Server health check TIMEOUT after {elapsed:.0f}s ({attempt} attempts)", flush=True)
        self._show_failure_diagnostics(domain, env_vars, container_name, "Health check timeout")

        # Stop the containers since they're unhealthy
        print("\nStopping unhealthy containers...", flush=True)
        try:
            subprocess.run(
                ["docker", "compose", "-f", str(self.compose_file), "-p", domain, "down"],
                capture_output=True,
                timeout=30,
            )
            print("✓ Containers stopped", flush=True)
        except Exception as e:
            print(f"Warning: Failed to stop containers: {e}", flush=True)

        # Raise error to fail the startup
        raise DockerError(
            f"Server failed to become healthy within {timeout}s.\n"
            f"The server containers have been stopped.\n\n"
            "Common causes:\n"
            "  1. Server crash during startup (check logs above)\n"
            "  2. Missing dependencies or configuration\n"
            f"  3. Port conflicts or network issues\n\n"
            f"To debug:\n"
            f"  1. Check full logs: docker logs {container_name}\n"
            f"  2. Try manual start: docker compose -p {domain} up\n"
            f"  3. Check domain configuration in domains/{domain}/",
            command="docker compose up -d (health check)",
        )


class DomainOrchestrator:
    """Main orchestrator coordinating all domain services."""

    def __init__(self, domains_root: Path, compose_file: Path):
        self.manifest_loader = ManifestLoader(domains_root)
        self.environment_validator = EnvironmentValidator(domains_root)
        self.docker_runner = DockerRunner(compose_file, domains_root)

    def list_domains(self) -> List[str]:
        """List all available domains."""
        return self.manifest_loader.list_domains()

    def validate_domain(self, domain: str) -> Dict[str, Any]:
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
        """Start a domain server with full validation and setup."""
        # Load and validate manifest
        manifest = self.manifest_loader.load_manifest(domain)

        # Rebuild server image if requested
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
        # Build missing images if requested
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

        # Generate and validate environment (server-only)
        env_vars = self.environment_validator.generate_environment(
            domain,
            manifest,
            rest_port,
            mcp_port,
            log_level,
            skip_image_check=(rebuild is not None or build is not None),
        )

        # Start server
        self.docker_runner.start_services(domain, env_vars, dry_run)

    def stop_domain(self, domain: str, dry_run: bool = False) -> None:
        """Stop a domain."""
        # Validate domain exists
        self.manifest_loader.load_manifest(domain)

        # Stop services
        self.docker_runner.stop_services(domain, self.manifest_loader.domains_root, dry_run)

    def build_domain(
        self, domain: str, image_filter: str | None = None, dry_run: bool = False, rebuild_mode: bool = True
    ) -> None:
        """Build domain images.

        Args:
            domain: Domain name
            image_filter: Optional prefix filter for image names (e.g., 'server', 'cookie', 'sandbox')
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

    def get_domain_status(self, domain: str) -> Dict[str, Any]:
        """Get domain status with full health information."""
        # Load manifest to get environment variables for proper compose context
        try:
            manifest = self.manifest_loader.load_manifest(domain)
        except Exception as e:
            return {"running": False, "services": [], "error": f"Domain manifest unavailable: {e}"}

        # Generate environment variables (reuse the same logic as start_domain)
        try:
            env_vars = self.environment_validator.generate_environment(
                domain, manifest, 8000, 8001, "INFO", skip_image_check=True, skip_port_check=True
            )
        except Exception as e:
            return {"running": False, "services": [], "error": f"Environment generation failed: {e}"}

        # Use docker compose ps with proper environment context for authoritative status
        cmd = [
            "docker",
            "compose",
            "-f",
            str(self.docker_runner.compose_file),
            "--project-name",
            domain,
            "ps",
            "--format",
            "json",
        ]

        try:
            # Create environment with necessary variables
            compose_env = os.environ.copy()
            compose_env.update(env_vars)

            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=compose_env)
            if result.stdout.strip():
                # Parse JSON output - docker compose ps --format json returns one JSON object per line
                services = []
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        service_info = json.loads(line)
                        services.append(service_info)

                # Check if any services are running (State == "running")
                running = any(service.get("State") == "running" for service in services)

                # Extract health information from the first service
                health_status = "unknown"
                if services:
                    # Check health from Status field (e.g., "Up 8 minutes (unhealthy)")
                    status_text = services[0].get("Status", "")
                    if "unhealthy" in status_text.lower():
                        health_status = "unhealthy"
                    elif "healthy" in status_text.lower():
                        health_status = "healthy"
                    elif "up" in status_text.lower() and running:
                        health_status = "healthy"  # Default to healthy if running and no explicit health info

                return {
                    "running": running,
                    "services": services,
                    "health_status": health_status,
                    "domain": domain,
                    "manifest": manifest,
                }
            else:
                return {"running": False, "services": [], "health_status": "stopped", "domain": domain}

        except subprocess.CalledProcessError as e:
            return {
                "running": False,
                "services": [],
                "error": f"Docker compose ps failed: {e}",
                "health_status": "error",
                "domain": domain,
            }
        except json.JSONDecodeError as e:
            return {
                "running": False,
                "services": [],
                "error": f"Failed to parse status: {e}",
                "health_status": "error",
                "domain": domain,
            }
