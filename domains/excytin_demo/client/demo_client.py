#!/usr/bin/env uv run python
"""
Enhanced Container Logging Connectivity Tests

This script demonstrates the enhanced container logging capabilities by using:
1. SABER's REST client for session/episode management
2. SABER's MCP client for tool execution within sandbox environments
3. MySQL connectivity tests from sandbox to permanent database

The script performs:
- Session creation and episode startup (creates sandbox containers)
- MySQL connectivity verification via CLI executor in sandbox
- Container networking validation
- Enhanced logging collection

Usage:
    uv run demo_client.py [--verbose]
"""

import argparse
import asyncio
import logging
import os
import sys

# Import SABER clients
sys.path.append("/app/src")
from saber.client.api.mcp_client import MCPClient
from saber.client.api.rest_client import SABERRestClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("enhanced_logging_demo")


async def test_mysql_connectivity(mcp_client: MCPClient) -> bool:
    """
    Test MySQL connectivity from sandbox to permanent database.

    Executes three MySQL commands via CLI executor to verify network connectivity:
    1. SHOW DATABASES - basic connectivity test
    2. SHOW TABLES - database access test
    3. SELECT 1 - ping test

    Args:
        mcp_client: Connected MCP client for tool execution

    Returns:
        bool: True if all tests pass, False otherwise
    """
    logger.info("🔍 Testing MySQL connectivity from sandbox to permanent database...")

    # MySQL connection parameters for permanent database
    mysql_host = "saber-excytin-incident-5"
    mysql_user = "root"
    mysql_password = "admin"
    mysql_database = "env_monitor_db"

    # Test commands
    test_commands = [
        {
            "name": "Database List Test",
            "command": (
                f"mysql -h {mysql_host} -u {mysql_user} -p{mysql_password} " f"--skip-ssl-verify -e 'SHOW DATABASES;'"
            ),
            "description": "Test basic MySQL connectivity and list available databases",
        },
        {
            "name": "Table List Test",
            "command": (
                f"mysql -h {mysql_host} -u {mysql_user} -p{mysql_password} "
                f"--skip-ssl-verify -e 'USE {mysql_database}; SHOW TABLES;'"
            ),
            "description": "Test database access and list tables in env_monitor_db",
        },
        {
            "name": "Ping Test",
            "command": (
                f"mysql -h {mysql_host} -u {mysql_user} -p{mysql_password} "
                f"--skip-ssl-verify -e 'SELECT 1 as ping_test;'"
            ),
            "description": "Test query execution with simple ping",
        },
    ]

    success_count = 0

    for i, test in enumerate(test_commands, 1):
        logger.info(f"  [{i}/3] {test['name']}: {test['description']}")

        try:
            # Execute command via MCP CLI executor
            result = await mcp_client.call_tool("cli", {"command": test["command"]})

            logger.debug(f"    Response type: {type(result)}")

            # Check if result indicates success
            success = False
            response_text = ""

            # Try multiple ways to extract the response text
            if hasattr(result, "content") and result.content:
                # Direct content attribute access (this is what we're getting)
                response_text = result.content[0].text if result.content else ""
                logger.debug("    Using direct content access")
                success = True  # We got a response
            elif isinstance(result, dict):
                # Dictionary access
                content = result.get("content", [])
                if content and isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            response_text += item.get("text", "")
                    logger.debug("    Using dict content access")
                    success = True  # We got a response

            # If we got response text, check the MySQL command success
            if success and response_text:
                # Parse the nested response format - the text contains a string representation of a dict
                # Look for success indicators: exit_code: 0 and success: True
                if ("'exit_code': 0" in response_text or '"exit_code": 0' in response_text) and (
                    "'success': True" in response_text or '"success": true' in response_text
                ):
                    logger.info(f"    ✅ {test['name']} succeeded")

                    # Try to extract and show some actual MySQL output for verification
                    try:
                        import re

                        stdout_match = re.search(r"'stdout': '([^']*)'", response_text)
                        if stdout_match:
                            stdout_content = stdout_match.group(1).replace("\\n", "\n")
                            lines = stdout_content.strip().split("\n")
                            if len(lines) > 1:  # Skip header, show first few data lines
                                sample_lines = lines[:3]  # Show first 3 lines
                                logger.debug(f"    📋 Sample output: {' | '.join(sample_lines)}")
                    except Exception:
                        pass  # Don't fail if we can't parse stdout

                    success_count += 1
                else:
                    logger.error(f"    ❌ {test['name']} failed: Command execution failed")
                    logger.debug(f"    Response content: {response_text[:300]}...")
            else:
                logger.error(f"    ❌ {test['name']} failed: No valid response")

        except Exception as e:
            logger.error(f"    ❌ {test['name']} failed with exception: {e}")

        # Small delay between tests
        await asyncio.sleep(2)

    logger.info(f"🔍 MySQL connectivity tests completed: {success_count}/3 passed")
    return success_count == 3


async def run_demo(verbose: bool = False) -> bool:
    """Run the enhanced logging demo with MySQL connectivity tests."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 70)
    logger.info("ENHANCED CONTAINER LOGGING DEMO WITH MYSQL CONNECTIVITY")
    logger.info("=" * 70)

    # Auto-detect server URLs
    if os.getenv("DOCKER_SABER_SERVER"):
        rest_base_url = os.getenv("SABER_REST_URL", "http://saber-excytin-server:8000")
        mcp_base_url = os.getenv("SABER_MCP_URL", "http://saber-excytin-server:8001")
    else:
        rest_base_url = "http://localhost:8000"
        mcp_base_url = "http://localhost:8001"

    # Create REST client for session management
    rest_client = SABERRestClient(
        base_url=rest_base_url,
        client_id="enhanced_logging_demo",
        request_timeout=30.0,
    )

    session_id = None
    episode_id = None
    success = True

    try:
        # Phase 1: REST API - Session and Episode Management
        logger.info("📡 Phase 1: Session and Episode Setup via REST API")

        # Test server connection
        logger.info("Testing REST server connection...")
        health_info = await rest_client.health_check()
        logger.info(f"✅ REST server is running: {health_info}")

        # Create session
        logger.info("Creating session...")
        session_id = await rest_client.create_session()
        logger.info(f"✅ Created session: {session_id}")

        # Get available tasks
        logger.info("Getting available tasks...")
        tasks = await rest_client.list_tasks()
        logger.info(f"Available tasks: {[task['task_id'] for task in tasks]}")

        # Find the basic_logging_demo task
        target_task_id = "basic_logging_demo"
        target_task = next((task for task in tasks if task["task_id"] == target_task_id), None)

        if not target_task:
            raise Exception(f"Task '{target_task_id}' not found in available tasks")

        logger.info(f"✅ Found target task: {target_task['title']}")

        # Start episode (this creates sandbox containers)
        logger.info("Starting episode - this will create sandbox containers...")
        episode_id = await rest_client.start_episode(task_id=target_task_id, session_id=session_id)
        logger.info(f"✅ Started episode {episode_id} - sandbox containers should be created")

        # Wait for containers to be created
        logger.info("Waiting for containers to be created...")
        await asyncio.sleep(15)  # Increased wait time for container creation

        # Phase 2: MySQL Connectivity Tests via MCP API
        logger.info("🔧 Phase 2: MySQL Connectivity Tests via MCP API")

        # Create MCP client with session context
        mcp_client = MCPClient(
            base_url=mcp_base_url, session_id=session_id, task_id=target_task_id, client_id="enhanced_logging_demo"
        )

        # Connect to MCP server
        logger.info("Connecting to MCP server...")
        async with mcp_client:
            logger.info("✅ Connected to MCP server")

            # List available tools
            logger.info("Listing available MCP tools...")
            tools = await mcp_client.list_tools()
            tool_names = []

            # Debug: Show what we received
            logger.debug(f"Tools response type: {type(tools)}")
            logger.debug(f"Tools response: {tools}")

            if isinstance(tools, list):
                # Direct list of Tool objects
                tool_names = [tool.name for tool in tools]
            elif hasattr(tools, "tools"):
                tool_names = [tool.name for tool in tools.tools]
            elif isinstance(tools, dict) and "tools" in tools:
                tool_names = [tool["name"] for tool in tools["tools"]]
            elif hasattr(tools, "result") and isinstance(tools.result, dict) and "tools" in tools.result:
                tool_names = [tool["name"] for tool in tools.result["tools"]]

            logger.info(f"Available tools: {tool_names}")

            # Verify CLI executor is available
            if "cli" not in tool_names:
                raise Exception("CLI executor not available in MCP tools")

            # Run MySQL connectivity tests
            mysql_tests_passed = await test_mysql_connectivity(mcp_client)

            if mysql_tests_passed:
                logger.info("✅ All MySQL connectivity tests passed!")
                success = True  # Mark as successful based on MySQL tests
            else:
                logger.warning("⚠️ Some MySQL connectivity tests failed")
                success = False

        # Phase 3: Cleanup and Log Collection (Background)
        logger.info("🧹 Phase 3: Session Cleanup and Log Collection")

        # Cleanup session in background (don't wait for completion)
        logger.info("Triggering session cleanup in background...")
        try:
            # Fire and forget - don't wait for cleanup to complete
            await asyncio.wait_for(rest_client.terminate_session(session_id), timeout=5.0)
            logger.info("✅ Session cleanup initiated successfully")
        except asyncio.TimeoutError:
            logger.info("✅ Session cleanup initiated (still processing in background)")
        except Exception as e:
            logger.info(f"✅ Session cleanup initiated (response: {e})")

        session_id = None  # Mark as cleaned up to avoid double cleanup

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        success = False
    finally:
        # Ensure cleanup (but only if not already cleaned up)
        if "session_id" in locals() and session_id is not None:
            logger.info("🧹 Final cleanup - session was not properly terminated")
            try:
                await rest_client.terminate_session(session_id)
            except Exception as e:
                logger.warning(f"Error during final cleanup: {e}")

    logger.info("=" * 70)
    if success:
        logger.info("🎉 DEMO COMPLETED SUCCESSFULLY")
        logger.info("Enhanced logging capabilities verified:")
        logger.info("  ✅ Session and episode management")
        logger.info("  ✅ Container creation and networking")
        logger.info("  ✅ MySQL connectivity from sandbox to permanent DB")
        logger.info("  ✅ MCP tool execution in sandbox environment")
        logger.info("")
        logger.info("Check ./server/logs/ for:")
        logger.info("  - compose-configs/ (docker-compose files)")
        logger.info("  - container-logs/ (container stdout/stderr)")
        logger.info("  - container-events-*.jsonl (lifecycle events)")
    else:
        logger.info("❌ DEMO FAILED - Check logs above for details")
    logger.info("=" * 70)

    return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enhanced Container Logging Demo with MySQL Connectivity Tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    try:
        success = asyncio.run(run_demo(verbose=args.verbose))
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nDemo interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
