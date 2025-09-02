"""
Container Factory

Handles dynamic creation of agent container configurations and images.
Supports different agent deployment strategies including volume mounts,
dynamic image building, and environment configuration.
"""

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import docker
import yaml

logger = logging.getLogger(__name__)


@dataclass
class AgentPackage:
    """Represents a packaged agent for containerization."""

    agent_path: Path
    agent_type: str  # "class", "function", "module"
    entry_point: str  # How to invoke the agent
    dependencies: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    requirements_file: Optional[Path] = None


@dataclass
class ContainerImageConfig:
    """Configuration for building agent container images."""

    base_image: str = "saber/agent-runner:latest"
    python_version: str = "3.11"
    working_dir: str = "/app"
    agent_dir: str = "/app/agent"
    runtime_dir: str = "/app/runtime"


class ContainerFactory:
    """
    Factory for creating agent containers with different deployment strategies.

    Supports:
    - Volume mounts for development
    - Dynamic image building for production
    - Agent discovery and packaging
    - Environment configuration
    """

    def __init__(self, config: Optional[ContainerImageConfig] = None):
        """Initialize container factory with configuration."""
        self.config = config or ContainerImageConfig()
        self.docker_client = docker.from_env()  # type: ignore

        logger.info("🏭 Container factory initialized")

    def discover_agent(self, agent_path: Path) -> AgentPackage:
        """
        Discover and analyze an agent for containerization.

        Args:
            agent_path: Path to agent code (file or directory)

        Returns:
            AgentPackage with discovery results
        """
        logger.info(f"🔍 Discovering agent at: {agent_path}")

        if not agent_path.exists():
            raise ValueError(f"Agent path does not exist: {agent_path}")

        if agent_path.is_file():
            return self._discover_file_agent(agent_path)
        else:
            return self._discover_directory_agent(agent_path)

    def prepare_agent_volume(self, agent_package: AgentPackage) -> Path:
        """
        Prepare agent code for volume mounting.

        Args:
            agent_package: Packaged agent information

        Returns:
            Path to prepared volume directory
        """
        logger.info("📦 Preparing agent volume mount...")

        # Create temporary directory for agent
        temp_dir = Path(tempfile.mkdtemp(prefix="saber-agent-"))
        agent_dir = temp_dir / "agent"
        runtime_dir = temp_dir / "runtime"

        try:
            # Copy agent code
            if agent_package.agent_path.is_file():
                agent_dir.mkdir(parents=True)
                shutil.copy2(agent_package.agent_path, agent_dir / agent_package.agent_path.name)
            else:
                shutil.copytree(agent_package.agent_path, agent_dir)

            # Create agent runtime
            self._create_agent_runtime(runtime_dir, agent_package)

            # Create launch.py in agent directory for container execution
            self._create_launch_script(agent_dir, agent_package)

            # Create agent configuration
            self._create_agent_config(temp_dir, agent_package)

            logger.info(f"📦 Agent volume prepared at: {temp_dir}")
            return temp_dir

        except Exception as e:
            # Cleanup on error
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to prepare agent volume: {e}")

    def build_agent_image(self, agent_package: AgentPackage, image_tag: str) -> str:
        """
        Build a custom Docker image for the agent.

        Args:
            agent_package: Packaged agent information
            image_tag: Tag for the built image

        Returns:
            str: Built image ID
        """
        logger.info(f"🔨 Building agent image: {image_tag}")

        # Create build context
        build_context = Path(tempfile.mkdtemp(prefix="saber-build-"))

        try:
            # Prepare build context
            self._prepare_build_context(build_context, agent_package)

            # Build image
            image, logs = self.docker_client.images.build(path=str(build_context), tag=image_tag, rm=True, forcerm=True)

            # Log build output
            for log in logs:
                if "stream" in log:
                    logger.debug(f"🔨 {log['stream'].strip()}")

            logger.info(f"✅ Built agent image: {image.id[:12]}")
            return str(image.id)

        except Exception as e:
            logger.error(f"❌ Failed to build agent image: {e}")
            raise
        finally:
            # Cleanup build context
            shutil.rmtree(build_context, ignore_errors=True)

    async def package_agent_code(self, agent_code_path: Path, agent_name: str) -> Path:
        """
        Package agent code for container execution.

        Args:
            agent_code_path: Path to agent code (file or directory)
            agent_name: Name identifier for the agent

        Returns:
            Path to the agent directory ready for container mounting
        """
        logger.info(f"📦 Packaging agent code at: {agent_code_path} (name: {agent_name})")

        # Discover the agent
        agent_package = self.discover_agent(agent_code_path)

        # Add agent name to environment
        agent_package.environment["AGENT_NAME"] = agent_name

        # Prepare the agent volume
        volume_path = self.prepare_agent_volume(agent_package)

        # Return the agent subdirectory path for mounting
        agent_mount_path = volume_path / "agent"

        logger.info(f"✅ Successfully packaged agent: {agent_name} at {agent_mount_path}")
        return agent_mount_path

    def cleanup_temp_resources(self, paths: List[Path]) -> None:
        """Clean up temporary resources."""
        for path in paths:
            try:
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                    logger.debug(f"🧹 Cleaned up: {path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup {path}: {e}")

    def _discover_file_agent(self, agent_file: Path) -> AgentPackage:
        """Discover agent from a single Python file."""
        logger.debug(f"📄 Analyzing agent file: {agent_file.name}")

        # Read file to analyze
        try:
            content = agent_file.read_text()
        except Exception as e:
            raise ValueError(f"Cannot read agent file: {e}")

        # Simple heuristics for agent type detection
        if "class " in content and "def run(" in content:
            agent_type = "class"
            entry_point = "class_adapter"
        elif "def " in content and "def main(" in content:
            agent_type = "function"
            entry_point = "function_adapter"
        elif "async def " in content:
            agent_type = "async_function"
            entry_point = "async_adapter"
        else:
            agent_type = "module"
            entry_point = "module_adapter"

        # Check for requirements file
        requirements_file_path = agent_file.parent / "requirements.txt"
        requirements_file: Optional[Path] = requirements_file_path if requirements_file_path.exists() else None

        return AgentPackage(
            agent_path=agent_file, agent_type=agent_type, entry_point=entry_point, requirements_file=requirements_file
        )

    def _discover_directory_agent(self, agent_dir: Path) -> AgentPackage:
        """Discover agent from a directory structure."""
        logger.debug(f"📁 Analyzing agent directory: {agent_dir.name}")

        # Look for common agent files
        main_files = ["main.py", "agent.py", "__main__.py"]
        agent_file = None

        for main_file in main_files:
            candidate = agent_dir / main_file
            if candidate.exists():
                agent_file = candidate
                break

        if not agent_file:
            raise ValueError(f"No recognizable agent entry point found in {agent_dir}")

        # Check for requirements
        requirements_file_path = agent_dir / "requirements.txt"
        requirements_file: Optional[Path] = requirements_file_path if requirements_file_path.exists() else None

        # Analyze the main agent file
        content = agent_file.read_text()

        if "class " in content:
            agent_type = "class"
            entry_point = "class_adapter"
        else:
            agent_type = "module"
            entry_point = "module_adapter"

        return AgentPackage(
            agent_path=agent_dir, agent_type=agent_type, entry_point=entry_point, requirements_file=requirements_file
        )

    def _create_agent_runtime(self, runtime_dir: Path, agent_package: AgentPackage) -> None:
        """Create agent runtime executor script."""
        runtime_dir.mkdir(parents=True, exist_ok=True)

        # Create main executor script
        executor_script = runtime_dir / "executor.py"
        executor_content = self._generate_executor_script(agent_package)
        executor_script.write_text(executor_content)
        executor_script.chmod(0o755)

        # Create adapter scripts
        self._create_adapter_scripts(runtime_dir)

        logger.debug(f"🔧 Created agent runtime at: {runtime_dir}")

    def _create_launch_script(self, agent_dir: Path, agent_package: AgentPackage) -> None:
        """Create launch.py script in the agent directory for container execution."""
        launch_script = agent_dir / "launch.py"

        # Simple launch script that runs the agent directly
        if agent_package.agent_path.is_file():
            agent_filename = agent_package.agent_path.name
            launch_content = f'''#!/usr/bin/env python3
"""
Agent launcher for container execution.
"""
import sys
import subprocess

if __name__ == "__main__":
    # Pass all arguments to the agent script
    subprocess.run([sys.executable, "/app/agent/{agent_filename}"] + sys.argv[1:])
'''
        else:
            # For directory agents, look for main entry points
            launch_content = '''#!/usr/bin/env python3
"""
Agent launcher for container execution.
"""
import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    # Look for common entry points
    agent_dir = Path("/app/agent")
    entry_points = ["main.py", "agent.py", "__main__.py"]

    for entry_point in entry_points:
        if (agent_dir / entry_point).exists():
            subprocess.run([sys.executable, str(agent_dir / entry_point)] + sys.argv[1:])
            break
    else:
        print("No valid entry point found", file=sys.stderr)
        sys.exit(1)
'''

        launch_script.write_text(launch_content)
        launch_script.chmod(0o755)
        logger.debug(f"🚀 Created launch script at: {launch_script}")

    def _create_agent_config(self, base_dir: Path, agent_package: AgentPackage) -> None:
        """Create agent configuration file."""
        config_file = base_dir / "agent_config.yaml"

        config = {
            "agent": {
                "type": agent_package.agent_type,
                "entry_point": agent_package.entry_point,
                "path": "/app/agent",
                "dependencies": agent_package.dependencies,
                "environment": agent_package.environment,
            }
        }

        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.debug(f"📝 Created agent config: {config_file}")

    def _generate_executor_script(self, agent_package: AgentPackage) -> str:
        """Generate the main agent executor script."""
        return '''#!/usr/bin/env python3
"""
Agent Executor - Container Runtime

Executes agents using standard MCP libraries within containers.
Handles agent discovery, MCP client setup, and result collection.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path

# Add agent and runtime to Python path
sys.path.insert(0, "/app/agent")
sys.path.insert(0, "/app/runtime")

# Import MCP client factory and adapters
from mcp_factory import create_mcp_client
from adapters import load_adapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main agent execution entry point."""
    try:
        # Get configuration from environment
        agent_id = os.environ["AGENT_ID"]
        sidecar_url = os.environ["SIDECAR_URL"]
        initial_prompt = os.environ["INITIAL_PROMPT"]
        task_id = os.environ["TASK_ID"]
        episode_id = os.environ["EPISODE_ID"]

        logger.info(f"🤖 Starting agent {agent_id} for episode {episode_id}")

        # Create MCP client
        mcp_client = await create_mcp_client(sidecar_url, agent_id)

        # Load agent adapter
        adapter = load_adapter("/app/agent_config.yaml")

        # Execute agent
        result = await adapter.run(mcp_client, initial_prompt)

        # Output result
        print(json.dumps(result, indent=2))

        # Disconnect MCP client
        await mcp_client.disconnect()

        logger.info("✅ Agent execution completed")

    except Exception as e:
        logger.error(f"❌ Agent execution failed: {e}")
        error_result = {
            "success": False,
            "error": str(e),
            "flag": None
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
'''

    def _create_adapter_scripts(self, runtime_dir: Path) -> None:
        """Create agent adapter scripts."""
        # This would contain the MCP client factory and adapter implementations
        # For now, we'll create placeholder files

        # MCP factory
        mcp_factory_file = runtime_dir / "mcp_factory.py"
        mcp_factory_content = '''"""MCP Client Factory for standard libraries."""

import asyncio
from anthropic import mcp


async def create_mcp_client(sidecar_url: str, agent_id: str):
    """Create standard MCP client connected to sidecar."""
    # Implementation will connect to sidecar using standard MCP libraries
    pass
'''
        mcp_factory_file.write_text(mcp_factory_content)

        # Adapters
        adapters_dir = runtime_dir / "adapters"
        adapters_dir.mkdir(exist_ok=True)

        adapters_init = adapters_dir / "__init__.py"
        adapters_init.write_text(
            '''"""Agent adapters for different patterns."""

def load_adapter(config_path: str):
    """Load appropriate adapter based on configuration."""
    pass
'''
        )

    def _prepare_build_context(self, build_context: Path, agent_package: AgentPackage) -> None:
        """Prepare Docker build context for agent image."""
        # Copy agent code
        agent_dest = build_context / "agent"
        if agent_package.agent_path.is_file():
            agent_dest.mkdir()
            shutil.copy2(agent_package.agent_path, agent_dest)
        else:
            shutil.copytree(agent_package.agent_path, agent_dest)

        # Create runtime
        runtime_dest = build_context / "runtime"
        self._create_agent_runtime(runtime_dest, agent_package)

        # Create Dockerfile
        dockerfile = build_context / "Dockerfile"
        dockerfile_content = self._generate_dockerfile(agent_package)
        dockerfile.write_text(dockerfile_content)

        # Copy requirements if exists
        if agent_package.requirements_file and agent_package.requirements_file.exists():
            shutil.copy2(agent_package.requirements_file, build_context / "requirements.txt")

    def _generate_dockerfile(self, agent_package: AgentPackage) -> str:
        """Generate Dockerfile for agent image."""
        return f"""FROM {self.config.base_image}

WORKDIR {self.config.working_dir}

# Copy requirements and install dependencies
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Copy agent code and runtime
COPY agent/ {self.config.agent_dir}/
COPY runtime/ {self.config.runtime_dir}/

# Set permissions
RUN chmod +x {self.config.runtime_dir}/executor.py

# Create non-root user
RUN groupadd -r agent && useradd -r -g agent agent
RUN chown -R agent:agent {self.config.working_dir}

USER agent

# Set entry point
ENTRYPOINT ["/app/runtime/executor.py"]
"""

    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            if self.docker_client:
                self.docker_client.close()
        except Exception:
            pass
