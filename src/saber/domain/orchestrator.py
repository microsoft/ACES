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
        required_dirs = ["server/config", "docker"]

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
            self._validate_base_images(errors, self.domains_root.parent / "external" / "saber")
            # Then validate domain images
            self._validate_all_images(manifest, errors)

        if errors:
            raise DomainValidationError(domain, errors)

        # Generate environment variables (server-only)
        return {
            "DOMAIN": domain,
            "DOMAINS_ROOT": str(self.domains_root),
            "SERVER_IMAGE": manifest["images"]["server"]["tag"],
            "REST_PORT": str(rest_port),
            "MCP_PORT": str(mcp_port),
            "LOG_LEVEL": log_level,
        }

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
                errors.append(f"Docker image '{image_tag}' not found. Use --build to create it.")

    def _validate_base_images(self, errors: List[str], saber_root: Path) -> None:
        """Validate base images exist."""
        try:
            # Load base images config directly
            base_images_file = saber_root / "src" / "saber" / "domain" / "package_resources" / "base-images.yaml"
            with open(base_images_file, "r") as f:
                base_images_config = yaml.safe_load(f)

            for image_name, image_config in base_images_config["images"].items():
                image_tag = image_config["tag"]
                if not self._docker_image_exists(image_tag):
                    errors.append(f"Base image '{image_tag}' not found. Use --build to create it.")
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

    def build_images(self, domain: str, manifest: Dict[str, Any], domains_root: Path, dry_run: bool = False) -> None:
        """Build all Docker images defined in domain manifest.

        Args:
            domain: Domain name
            manifest: Domain manifest
            domains_root: Path to domains directory
            dry_run: If True, show commands without executing

        Raises:
            DockerError: If build fails
        """
        domain_path = domains_root / domain
        images_config = manifest.get("images", {})

        if not images_config:
            raise DockerError("No images configuration found in manifest")

        # Ensure base images exist first
        print("🔍 Checking base image dependencies...")
        self.ensure_base_images(dry_run)

        # Build all images defined in manifest
        built_count = 0
        for image_name, image_config in images_config.items():
            if not isinstance(image_config, dict):
                raise DockerError(f"Image '{image_name}' configuration must be a dictionary")

            dockerfile = image_config.get("dockerfile")
            if not dockerfile:
                raise DockerError(f"Image '{image_name}' missing required 'dockerfile' field")

            image_tag = image_config.get("tag")
            if not image_tag:
                raise DockerError(f"Image '{image_name}' missing required 'tag' field")

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
                continue

            print(f"Building {image_name} image: {image_tag}")
            try:
                subprocess.run(cmd, check=True, cwd=domain_path)
                print(f"✓ Successfully built {image_tag}")
                built_count += 1
            except subprocess.CalledProcessError as e:
                raise DockerError(f"Failed to build {image_name} image {image_tag}: {e}")

        if not dry_run:
            print(f"✓ Successfully built {built_count} images for domain '{domain}'")

    def ensure_base_images(self, dry_run: bool = False) -> None:
        """Ensure base images exist, build if missing.

        Args:
            dry_run: If True, show commands without executing

        Raises:
            DockerError: If base image build fails
        """
        base_images_config = self._load_base_images_config()
        missing_images = []

        # Check which base images are missing
        for image_name, image_config in base_images_config["images"].items():
            image_tag = image_config["tag"]
            if not self._docker_image_exists(image_tag):
                missing_images.append((image_name, image_config))

        if not missing_images:
            print("✓ All base images are available")
            return

        print(f"🔨 Building {len(missing_images)} missing base images...")

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
        """Build base image from packaged Dockerfile with repo root context."""
        import tempfile

        try:
            # Use repo root as build context to access external/saber
            repo_root = self.domains_root.parent

            # Create temporary Dockerfile in repo root
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".dockerfile", dir=repo_root, delete=False
            ) as temp_dockerfile:
                # Copy Dockerfile content from package resources
                dockerfile_resource = files(saber.domain.package_resources) / package_path

                with dockerfile_resource.open("r") as src:
                    temp_dockerfile.write(src.read())

                temp_dockerfile_path = Path(temp_dockerfile.name)

            try:
                # Prepare build command with repo root as context
                cmd = ["docker", "build", "-f", str(temp_dockerfile_path), "-t", image_tag, str(repo_root)]

                # Add labels
                for key, value in labels.items():
                    cmd.extend(["--label", f"{key}={value}"])

                if dry_run:
                    print(f"Would build base image {image_name}: {' '.join(cmd)}")
                    return

                print(f"Building base image: {image_tag}")
                subprocess.run(cmd, check=True, cwd=repo_root)
                print(f"✓ Successfully built {image_tag}")

            finally:
                # Clean up temporary Dockerfile
                if temp_dockerfile_path.exists():
                    temp_dockerfile_path.unlink()

        except subprocess.CalledProcessError as e:
            raise DockerError(f"Failed to build base image {image_tag}: {e}")
        except Exception as e:
            raise DockerError(f"Failed to prepare base image build context: {e}")

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

            print(f"Starting domain {domain}...")
            subprocess.run(cmd, check=True)
            print(f"✓ Domain {domain} started successfully")

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

        # Create minimal environment file for compose down
        minimal_env = {
            "DOMAIN": domain,
            "DOMAINS_ROOT": str(domains_root),
            "SERVER_IMAGE": "dummy",  # Not needed for 'down' but required by compose file
            "REST_PORT": "8000",  # Not needed for 'down' but required by compose file
            "MCP_PORT": "8001",  # Not needed for 'down' but required by compose file
            "LOG_LEVEL": "INFO",  # Not needed for 'down' but required by compose file
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

    def _wait_for_services(self, domain: str, env_vars: Dict[str, str], timeout: int = 60) -> None:
        """Wait for services to be healthy."""
        profiles = set(env_vars.get("COMPOSE_PROFILES", "").split(","))
        if not profiles or "server" not in profiles:
            return

        rest_port = int(env_vars.get("REST_PORT", "8000"))
        print(f"Waiting for server health check on port {rest_port}...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(2)
                    if sock.connect_ex(("localhost", rest_port)) == 0:
                        print("✓ Server is responding")
                        return
            except Exception:
                pass

            time.sleep(2)

        print(f"⚠ Server health check timeout after {timeout}s")


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
        build: bool = False,
        dry_run: bool = False,
    ) -> None:
        """Start a domain server with full validation and setup."""
        # Load and validate manifest
        manifest = self.manifest_loader.load_manifest(domain)

        # Build server image if requested
        if build:
            self.docker_runner.build_images(domain, manifest, self.manifest_loader.domains_root, dry_run)

        # Generate and validate environment (server-only)
        env_vars = self.environment_validator.generate_environment(
            domain, manifest, rest_port, mcp_port, log_level, skip_image_check=build
        )

        # Start server
        self.docker_runner.start_services(domain, env_vars, dry_run)

    def stop_domain(self, domain: str, dry_run: bool = False) -> None:
        """Stop a domain."""
        # Validate domain exists
        self.manifest_loader.load_manifest(domain)

        # Stop services
        self.docker_runner.stop_services(domain, self.manifest_loader.domains_root, dry_run)

    def build_domain(self, domain: str, dry_run: bool = False) -> None:
        """Build domain images."""
        # Load and validate manifest
        manifest = self.manifest_loader.load_manifest(domain)

        # Build images
        self.docker_runner.build_images(domain, manifest, self.manifest_loader.domains_root, dry_run)

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
