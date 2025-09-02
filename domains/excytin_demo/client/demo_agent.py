#!/usr/bin/env python3
"""
Excytin Demo Agent

This agent demonstrates SABER's container-based execution capabilities by:
1. Testing MySQL connectivity from sandbox to permanent database containers
2. Verifying container networking and tool execution
3. Validating the MCP sidecar communication flow

The agent is designed to be executed by the SABER container harness and uses
standard MCP protocols for tool execution.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ExcytinDemoAgent:
    """
    Demo agent for testing Excytin capabilities.

    This agent performs MySQL connectivity tests to validate:
    - Sandbox container creation and networking
    - Database connectivity between containers
    - MCP tool execution flow
    - Container monitoring
    """

    def __init__(self, mcp_client: Any = None, **kwargs: Any) -> None:
        """
        Initialize the demo agent.

        Args:
            mcp_client: MCP client for tool execution (injected by harness)
            **kwargs: Additional configuration
        """
        self.mcp_client = mcp_client
        self.available_tools: List[str] = []

        # MySQL connection parameters for permanent database
        self.mysql_host = "saber-excytin-incident-5"
        self.mysql_user = "root"
        self.mysql_password = "admin"
        self.mysql_database = "env_monitor_db"

        logger.info("🤖 Excytin Demo Agent initialized")

    async def run(self, initial_prompt: str) -> Dict[str, Any]:
        """
        Main agent execution method called by the harness.

        Args:
            initial_prompt: Initial task prompt (from policy/task definition)

        Returns:
            Dict with execution results and success status
        """
        logger.info("🚀 Starting Excytin Container Demo")
        logger.info(f"📜 Task prompt: {initial_prompt}")

        try:
            # Step 1: Discover available tools
            await self._discover_tools()

            # Step 2: Verify CLI executor is available
            if "cli" not in self.available_tools:
                raise RuntimeError("CLI executor not available - cannot perform MySQL tests")

            # Step 3: Run MySQL connectivity tests
            test_results = await self._run_mysql_connectivity_tests()

            # Step 4: Analyze results and return
            success = all(result["success"] for result in test_results)

            result = {
                "success": success,
                "test_results": test_results,
                "total_tests": len(test_results),
                "passed_tests": sum(1 for r in test_results if r["success"]),
                "agent_type": "excytin_demo",
                "tools_used": ["cli"],
                "message": "Excytin demo completed successfully" if success else "Some tests failed",
            }

            logger.info(f"🎯 Demo completed with {result['passed_tests']}/{result['total_tests']} tests passed")
            return result

        except Exception as e:
            logger.error(f"❌ Demo agent failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_type": "excytin_demo",
                "message": f"Demo failed: {e}",
            }

    async def _discover_tools(self) -> None:
        """Discover available tools via MCP client."""
        logger.info("🔍 Discovering available tools...")

        try:
            tools = await self.mcp_client.list_tools()

            # Handle different response formats
            if isinstance(tools, list):
                self.available_tools = [
                    tool.get("name", tool) if isinstance(tool, dict) else str(tool) for tool in tools
                ]
            elif hasattr(tools, "tools"):
                self.available_tools = [tool.name for tool in tools.tools]
            elif isinstance(tools, dict) and "tools" in tools:
                self.available_tools = [tool["name"] for tool in tools["tools"]]
            else:
                logger.warning(f"Unexpected tools format: {type(tools)}")
                self.available_tools = []

            logger.info(f"✅ Available tools: {self.available_tools}")

        except Exception as e:
            logger.error(f"Failed to discover tools: {e}")
            self.available_tools = []

    async def _run_mysql_connectivity_tests(self) -> List[Dict[str, Any]]:
        """
        Run comprehensive MySQL connectivity tests.

        Tests database connectivity from sandbox container to permanent database.

        Returns:
            List of test results with success status and details
        """
        logger.info("🔍 Running MySQL connectivity tests...")

        test_commands = [
            {
                "name": "Database List Test",
                "command": (
                    f"mysql -h {self.mysql_host} -u {self.mysql_user} -p{self.mysql_password} "
                    f"--skip-ssl-verify -e 'SHOW DATABASES;'"
                ),
                "description": "Test basic MySQL connectivity and list available databases",
                "expected_content": ["Database", "information_schema", "performance_schema"],
            },
            {
                "name": "Table List Test",
                "command": (
                    f"mysql -h {self.mysql_host} -u {self.mysql_user} -p{self.mysql_password} "
                    f"--skip-ssl-verify -e 'USE {self.mysql_database}; SHOW TABLES;'"
                ),
                "description": "Test database access and list tables in env_monitor_db",
                "expected_content": ["Tables_in_env_monitor_db", "container_logs", "environment_status"],
            },
            {
                "name": "Ping Test",
                "command": (
                    f"mysql -h {self.mysql_host} -u {self.mysql_user} -p{self.mysql_password} "
                    f"--skip-ssl-verify -e 'SELECT 1 as ping_test;'"
                ),
                "description": "Test query execution with simple ping",
                "expected_content": ["ping_test", "1"],
            },
        ]

        results = []

        for i, test in enumerate(test_commands, 1):
            logger.info(f"  [{i}/{len(test_commands)}] {test['name']}: {test['description']}")

            result = await self._execute_mysql_test(test)
            results.append(result)

            # Small delay between tests
            await asyncio.sleep(1)

        passed = sum(1 for r in results if r["success"])
        logger.info(f"🔍 MySQL connectivity tests completed: {passed}/{len(results)} passed")

        return results

    async def _execute_mysql_test(self, test: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single MySQL test command.

        Args:
            test: Test configuration dict

        Returns:
            Test result dict with success status and details
        """
        try:
            logger.info(f"Executing: {test['command']}")

            # Execute command via MCP CLI tool
            result = await self.mcp_client.call_tool("cli", {"command": test["command"]})

            # Debug: Log the raw result to understand its structure
            logger.info(f"Raw tool result: {result}")
            logger.info(f"Result type: {type(result)}")

            # Parse result to determine success
            success, output, error_msg = self._parse_tool_result(result)

            logger.info(f"Parsed result - Success: {success}, Output: '{output}', Error: '{error_msg}'")

            if success:
                # Check if expected content appears in output
                content_found = self._validate_expected_content(output, test.get("expected_content", []))

                if content_found:
                    logger.info(f"    ✅ {test['name']} succeeded")
                    return {
                        "name": test["name"],
                        "success": True,
                        "output": output[:200] + "..." if len(output) > 200 else output,
                        "command": test["command"],
                    }
                else:
                    logger.warning(f"    ⚠️ {test['name']} executed but expected content not found")
                    return {
                        "name": test["name"],
                        "success": False,
                        "output": output[:200] + "..." if len(output) > 200 else output,
                        "command": test["command"],
                        "error": "Expected content not found in output",
                    }
            else:
                logger.error(f"    ❌ {test['name']} failed: {error_msg}")
                return {
                    "name": test["name"],
                    "success": False,
                    "command": test["command"],
                    "error": error_msg,
                    "output": output,
                }

        except Exception as e:
            logger.error(f"    ❌ {test['name']} failed with exception: {e}")
            return {"name": test["name"], "success": False, "command": test["command"], "error": str(e)}

    def _parse_tool_result(self, result: Any) -> tuple[bool, str, str]:
        """
        Parse MCP tool execution result.

        Args:
            result: Raw result from MCP tool call

        Returns:
            Tuple of (success, output_text, error_message)
        """
        try:
            output_text = ""
            error_msg = ""

            # Handle different result formats
            if hasattr(result, "content") and result.content:
                # Direct content attribute (common format)
                output_text = result.content[0].text if result.content else ""
            elif isinstance(result, dict):
                # Dictionary format
                content = result.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            output_text += item.get("text", "")
                elif isinstance(content, str):
                    output_text = content

            # Check for success indicators in the output
            if output_text:
                # Look for CLI executor success patterns
                success_patterns = ["'exit_code': 0", '"exit_code": 0', "'success': True", '"success": true']

                has_success = any(pattern in output_text for pattern in success_patterns)

                # Extract actual stdout if available
                stdout_match = re.search(r"'stdout': '([^']*)'", output_text)
                if stdout_match:
                    actual_output = stdout_match.group(1).replace("\\n", "\n")
                    output_text = actual_output

                return has_success, output_text, error_msg if not has_success else ""
            else:
                return False, "", "No output received"

        except Exception as e:
            return False, "", f"Failed to parse result: {e}"

    def _validate_expected_content(self, output: str, expected_content: List[str]) -> bool:
        """
        Validate that expected content appears in command output.

        Args:
            output: Command output text
            expected_content: List of strings that should appear in output

        Returns:
            True if expected content found, False otherwise
        """
        if not expected_content:
            return True  # No validation required

        output_lower = output.lower()
        found_count = 0

        for expected in expected_content:
            if expected.lower() in output_lower:
                found_count += 1
                logger.debug(f"Found expected content: {expected}")

        # Require at least half of expected content to be found
        threshold = max(1, len(expected_content) // 2)
        return found_count >= threshold


# Export agent for harness discovery
Agent = ExcytinDemoAgent


def run_agent(mcp_client: Any, initial_prompt: str) -> Dict[str, Any]:
    """
    Function-style agent entry point for harness compatibility.

    Args:
        mcp_client: MCP client for tool execution
        initial_prompt: Initial task prompt

    Returns:
        Agent execution results
    """
    agent = ExcytinDemoAgent(mcp_client=mcp_client)
    return asyncio.run(agent.run(initial_prompt))


if __name__ == "__main__":
    # For direct execution (development/testing)
    logger.info("Excytin Demo Agent")
    logger.info("This agent is designed to be executed by the SABER harness")
    logger.info("Use demo_client.py to run the full demo")
