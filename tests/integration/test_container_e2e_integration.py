#!/usr/bin/env python3
"""
End-to-End Integration Test for Container-Based MCP Sidecar Architecture

This test validates the complete flow:
1. Mock SABER server (REST API + MCP server)
2. Sidecar container startup and session registration
3. Agent container execution with MCP tool calls
4. Complete cleanup and termination

Tests the full Phase 4 implementation from the MCP sidecar implementation plan.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch, AsyncMock
import pytest
import docker
import aiohttp
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from threading import Thread

from saber.client.containers import SidecarManager, AgentManager, ContainerFactory
from saber.client.containers.sidecar_manager import SidecarConfig
from saber.client.containers.agent_manager import AgentContainerConfig
from saber.client.episode_executor import ContainerEpisodeExecutor
from saber.client.api import SABERRestClient

# Configure logging for test visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockSABERServer:
    """
    Mock SABER server providing both REST API and MCP endpoints.

    Simulates the server-side behavior for testing the client architecture.
    """

    def __init__(self, rest_port: int = 8000, mcp_port: int = 8001):
        self.rest_port = rest_port
        self.mcp_port = mcp_port
        self.app = FastAPI(title="Mock SABER Server")
        self.sessions = {}
        self.episodes = {}
        self.mcp_tools = [
            {
                "name": "test_tool",
                "description": "A test tool for integration testing",
                "schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Test message"}
                    },
                    "required": ["message"]
                }
            },
            {
                "name": "list_files",
                "description": "List files in a directory",
                "schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path"}
                    }
                }
            }
        ]
        self._setup_routes()

    def _setup_routes(self):
        """Setup REST API routes."""

        @self.app.post("/session")
        async def create_session():
            session_id = f"session_{len(self.sessions) + 1}"
            self.sessions[session_id] = {
                "id": session_id,
                "created_at": time.time(),
                "active": True
            }
            return {"session_id": session_id}

        @self.app.delete("/session/{session_id}")
        async def terminate_session(session_id: str):
            if session_id in self.sessions:
                self.sessions[session_id]["active"] = False
                return {"success": True}
            return {"success": False, "error": "Session not found"}

        @self.app.post("/session/{session_id}/episode")
        async def start_episode(session_id: str, request: Dict[str, Any] = None):
            if session_id not in self.sessions:
                return JSONResponse({"error": "Session not found"}, status_code=404)

            episode_id = f"episode_{len(self.episodes) + 1}"
            self.episodes[episode_id] = {
                "id": episode_id,
                "session_id": session_id,
                "task_id": request.get("task_id") if request else "test_task",
                "created_at": time.time()
            }
            return {"episode_id": episode_id}

        @self.app.get("/session/{session_id}/current-task")
        async def get_current_task(session_id: str):
            return {
                "task_id": "test_task_1",
                "title": "Integration Test Task",
                "description": "A test task for validating the container architecture"
            }

        @self.app.get("/session/{session_id}/policy")
        async def get_policy(session_id: str):
            return {
                "prompt": "You are a test agent. Use available tools to complete the task.",
                "rules": ["Use tools responsibly", "Complete the objective"]
            }

        @self.app.get("/tasks")
        async def list_tasks():
            return [
                {"task_id": "test_task_1", "title": "Integration Test Task"},
                {"task_id": "test_task_2", "title": "Secondary Test Task"}
            ]

        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "sessions": len(self.sessions)}

        # MCP Endpoints (simplified)
        @self.app.post("/mcp/list_tools")
        async def mcp_list_tools():
            return {"result": self.mcp_tools}

        @self.app.post("/mcp/call_tool")
        async def mcp_call_tool(request: Dict[str, Any]):
            tool_name = request.get("name")
            arguments = request.get("arguments", {})

            if tool_name == "test_tool":
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Test tool executed with message: {arguments.get('message', 'no message')}"
                    }]
                }
            elif tool_name == "list_files":
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Files in {arguments.get('path', '/tmp')}: file1.txt, file2.txt"
                    }]
                }
            else:
                return JSONResponse({"error": f"Unknown tool: {tool_name}"}, status_code=404)

        @self.app.get("/mcp/list_resources")
        async def mcp_list_resources():
            return {
                "resources": [
                    {"name": "policy", "type": "text/markdown", "uri": "mcp://policy"}
                ]
            }

    async def start(self):
        """Start the mock server."""
        config = uvicorn.Config(
            self.app,
            host="127.0.0.1",
            port=self.rest_port,
            log_level="info"
        )
        server = uvicorn.Server(config)

        # Run server in background thread
        def run_server():
            asyncio.run(server.serve())

        self.server_thread = Thread(target=run_server, daemon=True)
        self.server_thread.start()

        # Wait for server to be ready
        await self._wait_for_server_ready()
        logger.info(f"✅ Mock SABER server started on port {self.rest_port}")

    async def _wait_for_server_ready(self, timeout: int = 10):
        """Wait for the server to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://127.0.0.1:{self.rest_port}/health") as response:
                        if response.status == 200:
                            return
            except:
                pass
            await asyncio.sleep(0.1)
        raise RuntimeError("Mock server failed to start")


@pytest.fixture
async def mock_saber_server():
    """Fixture providing a mock SABER server."""
    server = MockSABERServer()
    await server.start()
    yield server
    # Cleanup happens automatically when thread exits


@pytest.fixture
def simple_test_agent():
    """Create a simple test agent for container execution."""
    agent_code = '''
#!/usr/bin/env python3
"""
Simple test agent for integration testing.
"""
import asyncio
import json
import os
import aiohttp

class SimpleTestAgent:
    def __init__(self):
        self.sidecar_url = os.getenv("SABER_SIDECAR_URL", "http://saber-mcp-sidecar:8002")
        self.session_id = os.getenv("SABER_SESSION_ID")

    async def run(self, initial_prompt: str, shutdown_check=None) -> dict:
        """Execute the agent workflow."""
        try:
            # Step 1: List available tools
            tools = await self.list_tools()
            print(f"Available tools: {tools}")

            # Step 2: Call a test tool
            if tools and any(tool.get("name") == "test_tool" for tool in tools):
                result = await self.call_tool("test_tool", {"message": "Hello from integration test!"})
                print(f"Tool result: {result}")

                return {
                    "success": True,
                    "flag": "flag{integration_test_success}",
                    "tools_found": len(tools),
                    "tool_result": result
                }
            else:
                return {
                    "success": False,
                    "error": "No test_tool found",
                    "tools_found": len(tools) if tools else 0
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def list_tools(self):
        """List available tools via MCP sidecar."""
        url = f"{self.sidecar_url}/mcp/list_tools"
        headers = {"X-Saber-Session-Id": self.session_id}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json={}) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("result", [])
                else:
                    raise Exception(f"Failed to list tools: {response.status}")

    async def call_tool(self, name: str, arguments: dict):
        """Call a tool via MCP sidecar."""
        url = f"{self.sidecar_url}/mcp/call_tool"
        headers = {"X-Saber-Session-Id": self.session_id}
        payload = {
            "jsonrpc": "2.0",
            "method": "call_tool",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    text = await response.text()
                    raise Exception(f"Failed to call tool {name}: {response.status} - {text}")

# Entry point for container execution
if __name__ == "__main__":
    import sys
    agent = SimpleTestAgent()
    initial_prompt = sys.argv[1] if len(sys.argv) > 1 else "Complete the test task"
    result = asyncio.run(agent.run(initial_prompt))
    print(json.dumps(result))
'''

    # Create temporary file for agent code
    temp_dir = tempfile.mkdtemp()
    agent_file = Path(temp_dir) / "test_agent.py"
    agent_file.write_text(agent_code)
    agent_file.chmod(0o755)

    yield agent_file

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


class TestContainerE2EIntegration:
    """
    End-to-end integration test for the container-based MCP sidecar architecture.

    Tests the complete flow from harness initialization through agent execution
    to cleanup, validating all major components work together.
    """

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_container_e2e_flow(self, mock_saber_server, simple_test_agent):
        """
        Test the complete end-to-end container execution flow.

        This test validates:
        1. Sidecar container startup and health checks
        2. Session registration with sidecar
        3. Agent container execution with MCP communication
        4. Tool listing and calling through sidecar
        5. Result collection and cleanup
        """
        # Ensure Docker is available
        try:
            docker_client = docker.from_env()
            docker_client.ping()
        except Exception as e:
            pytest.skip(f"Docker not available: {e}")

        # Check if required images exist
        try:
            docker_client.images.get("saber-mcp-sidecar:latest")
        except docker.errors.ImageNotFound:
            pytest.skip("saber-mcp-sidecar:latest image not found - run docker/build_images.sh")

        try:
            docker_client.images.get("saber-agent-runner:latest")
        except docker.errors.ImageNotFound:
            pytest.skip("saber-agent-runner:latest image not found - run docker/build_images.sh")

        # Test configuration
        sidecar_config = SidecarConfig(
            container_name="test-saber-mcp-sidecar",
            network_name="test-saber-network",
            port=18002,  # Use non-standard port for testing
            saber_server_url="http://host.docker.internal:8000",
            saber_mcp_url="http://host.docker.internal:8000"  # MCP endpoints are on same server
        )

        agent_config = AgentContainerConfig(
            network_name="test-saber-network",
            execution_timeout=120  # 2 minutes for testing
        )

        sidecar_manager = SidecarManager(sidecar_config)
        agent_manager = AgentManager(agent_config)
        container_factory = ContainerFactory()

        session_id = "test_session_123"
        task_id = "test_task_1"
        episode_id = "test_episode_456"

        try:
            logger.info("🚀 Starting E2E container integration test...")

            # Step 1: Start the sidecar
            logger.info("📡 Starting MCP sidecar...")
            sidecar_started = await sidecar_manager.start_sidecar()
            assert sidecar_started, "Failed to start MCP sidecar"

            # Step 2: Register session with sidecar
            logger.info("🔐 Registering session with sidecar...")
            agent_id = await sidecar_manager.register_agent_session(
                session_id=session_id,
                task_id=task_id,
                agent_id="test_agent"
            )
            assert agent_id is not None, "Failed to register agent session"

            # Step 3: Package agent for container execution
            logger.info("📦 Packaging agent for container execution...")
            agent_package_path = await container_factory.package_agent_code(
                agent_code_path=simple_test_agent,
                agent_name="test_agent"
            )
            assert agent_package_path is not None, "Failed to package agent"

            # Step 4: Execute agent in container
            logger.info("🤖 Executing agent in container...")
            sidecar_url = f"http://{sidecar_config.container_name}:{sidecar_config.port}"
            initial_prompt = "Complete the integration test by using available tools."

            execution_result = await agent_manager.execute_agent(
                agent_code_path=agent_package_path,
                agent_id="test_agent",
                sidecar_url=sidecar_url,
                initial_prompt=initial_prompt,
                task_id=task_id,
                episode_id=episode_id,
                session_id=session_id,
                extra_env={
                    "SABER_SIDECAR_URL": sidecar_url,
                    "SABER_SESSION_ID": session_id
                }
            )

            # Step 5: Validate execution results
            logger.info("✅ Validating execution results...")
            assert execution_result.success, f"Agent execution failed: {execution_result.error}"
            assert execution_result.exit_code == 0, f"Agent exited with code: {execution_result.exit_code}"

            # Parse agent output
            try:
                agent_output = json.loads(execution_result.stdout.strip().split('\n')[-1])
                assert agent_output.get("success") is True, f"Agent reported failure: {agent_output}"
                assert "flag{integration_test_success}" in agent_output.get("flag", ""), "Expected flag not found"
                assert agent_output.get("tools_found", 0) > 0, "No tools found by agent"
                logger.info(f"🎉 Agent execution successful: {agent_output}")
            except (json.JSONDecodeError, IndexError) as e:
                pytest.fail(f"Failed to parse agent output: {e}\nStdout: {execution_result.stdout}")

            # Step 6: Verify sidecar handled requests correctly
            logger.info("🔍 Verifying sidecar request handling...")
            health_url = f"http://localhost:{sidecar_config.port}/health"
            async with aiohttp.ClientSession() as client:
                async with client.get(health_url) as response:
                    assert response.status == 200, "Sidecar health check failed"
                    health_data = await response.json()
                    assert health_data.get("status") == "healthy", f"Sidecar unhealthy: {health_data}"

            logger.info("✅ E2E integration test completed successfully!")

        except Exception as e:
            logger.error(f"❌ E2E test failed: {e}")
            raise

        finally:
            # Cleanup: Stop sidecar and clean up containers
            logger.info("🧹 Cleaning up test resources...")
            try:
                await sidecar_manager.stop_sidecar()

                # Clean up any remaining containers
                containers = docker_client.containers.list(
                    filters={"name": "test-saber-"},
                    all=True
                )
                for container in containers:
                    try:
                        container.remove(force=True)
                        logger.info(f"🗑️ Removed container: {container.name}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup container {container.name}: {cleanup_error}")

                # Clean up test network
                try:
                    networks = docker_client.networks.list(names=["test-saber-network"])
                    for network in networks:
                        network.remove()
                        logger.info(f"🗑️ Removed network: {network.name}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup network: {cleanup_error}")

            except Exception as cleanup_error:
                logger.warning(f"Cleanup failed: {cleanup_error}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_sidecar_startup_and_health_checks(self, mock_saber_server):
        """Test sidecar container startup and health monitoring in isolation."""
        try:
            docker_client = docker.from_env()
            docker_client.ping()
        except Exception as e:
            pytest.skip(f"Docker not available: {e}")

        try:
            docker_client.images.get("saber-mcp-sidecar:latest")
        except docker.errors.ImageNotFound:
            pytest.skip("saber-mcp-sidecar:latest image not found")

        sidecar_config = SidecarConfig(
            container_name="test-health-sidecar",
            network_name="test-health-network",
            port=18003,
            saber_server_url="http://host.docker.internal:8000",
            saber_mcp_url="http://host.docker.internal:8000"  # MCP endpoints are on same server as REST
        )

        sidecar_manager = SidecarManager(sidecar_config)

        try:
            # Test startup
            started = await sidecar_manager.start_sidecar()
            assert started, "Sidecar failed to start"

            # Test health endpoints
            base_url = f"http://localhost:{sidecar_config.port}"

            async with aiohttp.ClientSession() as client:
                # Test readiness
                async with client.get(f"{base_url}/ready") as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data.get("status") == "ready"

                # Test health
                async with client.get(f"{base_url}/health") as response:
                    assert response.status == 200
                    data = await response.json()
                    assert "status" in data

        finally:
            await sidecar_manager.stop_sidecar()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_container_episode_executor_integration(self, mock_saber_server):
        """Test the ContainerEpisodeExecutor with mock dependencies."""

        # Create mock REST client
        rest_client = SABERRestClient(base_url="http://127.0.0.1:8000")
        session_id = "test_executor_session"

        # Mock the container managers to avoid Docker dependencies
        with patch('saber.client.episode_executor.SidecarManager') as mock_sidecar, \
             patch('saber.client.episode_executor.AgentManager') as mock_agent, \
             patch('saber.client.episode_executor.ContainerFactory') as mock_factory:

            # Setup mocks
            mock_sidecar_instance = mock_sidecar.return_value
            mock_sidecar_instance.start_sidecar = AsyncMock(return_value=True)
            mock_sidecar_instance.register_agent_session = AsyncMock(return_value="mock_agent_id")

            mock_agent_instance = mock_agent.return_value
            mock_agent_instance.execute_agent.return_value = type('Result', (), {
                'success': True,
                'exit_code': 0,
                'stdout': '{"success": true, "flag": "flag{mock_success}"}',
                'execution_time': 1.5
            })()

            mock_factory_instance = mock_factory.return_value
            mock_factory_instance.package_agent_code.return_value = "mock_package"

            # Test executor
            executor = ContainerEpisodeExecutor(
                rest_client=rest_client,
                session_id=session_id,
                parallelism=1
            )

            # Test initialization
            await executor.initialize()
            mock_sidecar_instance.start_sidecar.assert_called_once()

            # Test execution workflow (would be called by harness)
            # This validates the integration patterns without requiring Docker
            assert executor.sidecar_manager is not None
            assert executor.agent_manager is not None
            assert executor.container_factory is not None

            logger.info("✅ Container episode executor integration validated")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
