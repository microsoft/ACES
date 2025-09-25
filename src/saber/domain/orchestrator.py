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
from pathlib import Path
from typing import Any, Dict, List

import yaml

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
    ) -> Dict[str, str]:
        """Generate and validate environment variables for server-only architecture.

        Args:
            domain: Domain name
            manifest: Domain manifest
            rest_port: REST API port
            mcp_port: MCP protocol port
            log_level: Logging level
            skip_image_check: Skip Docker image existence check

        Returns:
            Dict of environment variables for docker-compose

        Raises:
            DomainValidationError: If validation fails
        """
        errors = []

        # Validate ports are available
        for port_name, port_num in [("REST", rest_port), ("MCP", mcp_port)]:
            if not self._is_port_available(port_num):
                errors.append(f"Port {port_num} ({port_name}) is already in use")

        # Validate server image exists (unless building)
        if not skip_image_check:
            self._validate_server_image(manifest, errors)

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

    def _validate_server_image(self, manifest: Dict[str, Any], errors: List[str]) -> None:
        """Validate server Docker image exists."""
        server_config = manifest.get("images", {}).get("server")
        if not server_config:
            errors.append("No server image configuration found in manifest")
            return

        image_tag = server_config["tag"]
        if not self._docker_image_exists(image_tag):
            errors.append(f"Server Docker image '{image_tag}' not found. " f"Use --build to create it.")

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

    def __init__(self, compose_file: Path):
        self.compose_file = compose_file.resolve()
        if not self.compose_file.exists():
            raise DockerError(f"Compose file not found: {self.compose_file}", command=None)

        # Validate Docker is available
        self._validate_docker()

    def build_images(self, domain: str, manifest: Dict[str, Any], domains_root: Path, dry_run: bool = False) -> None:
        """Build server Docker image for domain.

        Args:
            domain: Domain name
            manifest: Domain manifest
            domains_root: Path to domains directory
            dry_run: If True, show commands without executing

        Raises:
            DockerError: If build fails
        """
        domain_path = domains_root / domain

        # Only build server image
        server_config = manifest.get("images", {}).get("server")
        if not server_config:
            raise DockerError("No server image configuration found in manifest")

        dockerfile_path = domain_path / server_config["dockerfile"]
        image_tag = server_config["tag"]
        build_args = server_config.get("buildArgs", {})
        labels = server_config.get("labels", {})

        # Prepare build command
        cmd = ["docker", "build", "-f", str(dockerfile_path), "-t", image_tag, str(domain_path)]

        # Add build args
        for key, value in build_args.items():
            cmd.extend(["--build-arg", f"{key}={value}"])

        # Add labels
        for key, value in labels.items():
            cmd.extend(["--label", f"{key}={value}"])

        # Add git metadata if available
        self._add_git_metadata(cmd, domain_path)

        if dry_run:
            print(f"Would build server: {' '.join(cmd)}")
            return

        print(f"Building server image: {image_tag}")
        try:
            subprocess.run(cmd, check=True, cwd=domain_path)
            print(f"✓ Successfully built {image_tag}")
        except subprocess.CalledProcessError as e:
            raise DockerError(f"Failed to build server image {image_tag}: {e}")

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

    def get_status(self, domain: str) -> Dict[str, Any]:
        """Get status of domain services.

        Args:
            domain: Domain name

        Returns:
            Dict containing service status information
        """
        cmd = ["docker", "compose", "-f", str(self.compose_file), "--project-name", domain, "ps", "--format", "json"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return {"running": True, "services": json.loads(result.stdout)}
        except subprocess.CalledProcessError:
            return {"running": False, "services": []}
        except json.JSONDecodeError:
            return {"running": False, "services": [], "error": "Failed to parse status"}

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
        self.docker_runner = DockerRunner(compose_file)

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
        """Get domain status."""
        # Validate domain exists
        manifest = self.manifest_loader.load_manifest(domain)

        # Get status
        status = self.docker_runner.get_status(domain)
        status["domain"] = domain
        status["manifest"] = manifest

        return status
