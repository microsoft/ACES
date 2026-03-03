#!/usr/bin/env python3
"""
Lightweight mock agent for container execution testing.

This agent demonstrates the standard interface expected by the container
architecture and provides basic MCP tool interaction for testing.
"""

import asyncio
import json
import os
import sys
from typing import Any

import aiohttp


class MockTestAgent:
    """
    Simple mock agent for integration testing.

    Demonstrates:
    - Standard agent interface (run method)
    - MCP tool discovery and execution via sidecar
    - Result formatting for container collection
    """

    def __init__(self):
        self.sidecar_url = os.getenv("SABER_SIDECAR_URL", "http://saber-mcp-sidecar:8002")
        self.session_id = os.getenv("SABER_SESSION_ID")
        self.task_id = os.getenv("SABER_TASK_ID")
        self.episode_id = os.getenv("SABER_EPISODE_ID")

        if not self.session_id:
            raise ValueError("SABER_SESSION_ID environment variable required")

    async def run(self, initial_prompt: str, shutdown_check=None) -> dict[str, Any]:
        """
        Main agent execution method.

        Args:
            initial_prompt: Task description from SABER
            shutdown_check: Optional shutdown check function (not used in containers)

        Returns:
            Dict containing execution results
        """
        print("🤖 Starting MockTestAgent execution")
        print(f"📝 Initial prompt: {initial_prompt}")
        print(f"🔗 Sidecar URL: {self.sidecar_url}")
        print(f"🆔 Session ID: {self.session_id}")

        try:
            # Step 1: List available tools
            print("🔍 Listing available tools...")
            tools = await self.list_tools()
            print(f"✅ Found {len(tools)} tools: {[t.get('name') for t in tools]}")

            # Step 2: Execute a test workflow
            results = []

            # Try to call test_tool if available
            test_tools = [t for t in tools if t.get("name") == "test_tool"]
            if test_tools:
                print("🛠️ Calling test_tool...")
                tool_result = await self.call_tool(
                    "test_tool", {"message": f"Hello from {self.task_id or 'integration test'}!"}
                )
                results.append({"tool": "test_tool", "result": tool_result})
                print(f"✅ test_tool result: {tool_result}")

            # Try to call list_files if available
            list_tools = [t for t in tools if t.get("name") == "list_files"]
            if list_tools:
                print("📁 Calling list_files...")
                file_result = await self.call_tool("list_files", {"path": "/tmp"})
                results.append({"tool": "list_files", "result": file_result})
                print(f"✅ list_files result: {file_result}")

            # Simulate finding a flag (common in security testing)
            flag = f"flag{{container_integration_success_{self.episode_id or 'unknown'}}}"

            final_result = {
                "success": True,
                "agent_type": "MockTestAgent",
                "session_id": self.session_id,
                "task_id": self.task_id,
                "episode_id": self.episode_id,
                "tools_discovered": len(tools),
                "tools_executed": len(results),
                "tool_results": results,
                "flag": flag,
                "message": "Container-based execution successful",
                "sidecar_communication": "working",
            }

            print("🎉 Agent execution completed successfully")
            return final_result

        except Exception as e:
            error_result = {
                "success": False,
                "agent_type": "MockTestAgent",
                "session_id": self.session_id,
                "error": str(e),
                "error_type": type(e).__name__,
            }
            print(f"❌ Agent execution failed: {e}")
            return error_result

    async def list_tools(self) -> list:
        """List available tools via MCP sidecar."""
        url = f"{self.sidecar_url}/mcp/list_tools"
        headers = {"X-Saber-Session-Id": self.session_id}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json={}) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("tools", [])
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to list tools: {response.status} - {error_text}")

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool via MCP sidecar."""
        url = f"{self.sidecar_url}/mcp/call_tool"
        headers = {"X-Saber-Session-Id": self.session_id}
        payload = {"name": name, "arguments": arguments}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to call tool {name}: {response.status} - {error_text}")


# Container entry point
async def main():
    """Main entry point for container execution."""
    try:
        initial_prompt = sys.argv[1] if len(sys.argv) > 1 else "Execute integration test workflow"

        agent = MockTestAgent()
        result = await agent.run(initial_prompt)

        # Output result as JSON for container collection
        print(json.dumps(result, indent=2))

        # Exit with appropriate code
        sys.exit(0 if result.get("success") else 1)

    except Exception as e:
        error_result = {"success": False, "error": str(e), "error_type": type(e).__name__}
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
