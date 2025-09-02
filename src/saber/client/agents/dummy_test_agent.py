#!/usr/bin/env python3
"""
Dummy Test Agent for SABER UI Testing

A simple test agent that makes hardcoded tool calls without requiring an LLM.
This agent is designed for testing the SABER UI system and MCP tool call monitoring.

The agent executes a series of predefined tool calls:
1. ls - List current directory contents
2. python --version - Check Python version
3. curl http://httpbin.org/get - Make HTTP request
4. whoami - Check current user
5. pwd - Print working directory

This allows us to test the UI progress tracking and MCP tool call monitoring
without needing actual LLM integration.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DummyTestAgent:
    """
    Dummy agent for testing SABER UI and MCP monitoring.

    Makes hardcoded tool calls to test the execution pipeline.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize dummy agent."""
        self.config = config or {}
        self.name = "DummyTestAgent"
        self.version = "1.0.0"

        # Predefined tool calls to execute
        self.tool_calls = [
            {
                "tool": "bash",
                "command": "ls -la",
                "description": "List directory contents with details",
                "expected_duration": 1.0,
            },
            {
                "tool": "bash",
                "command": "python --version",
                "description": "Check Python version",
                "expected_duration": 0.5,
            },
            {
                "tool": "bash",
                "command": "curl -s http://httpbin.org/get",
                "description": "Make HTTP GET request",
                "expected_duration": 2.0,
            },
            {"tool": "bash", "command": "whoami", "description": "Check current user", "expected_duration": 0.3},
            {"tool": "bash", "command": "pwd", "description": "Print working directory", "expected_duration": 0.2},
            {
                "tool": "bash",
                "command": "echo 'SABER_FLAG{test_successful_dummy_agent}'",
                "description": "Output test flag for verification",
                "expected_duration": 0.1,
            },
        ]

        # Track execution state
        self.current_step = 0
        self.execution_log: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None

    async def run(self, task_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the dummy agent workflow.

        Args:
            task_info: Task information from SABER server (optional)

        Returns:
            Dict containing execution results and metrics
        """
        logger.info(f"🤖 Starting {self.name} v{self.version}")
        self.start_time = time.time()

        if task_info:
            logger.info(f"📋 Task info: {task_info}")

        try:
            # Execute each tool call in sequence
            for i, tool_call in enumerate(self.tool_calls):
                self.current_step = i + 1
                logger.info(f"🔧 Step {self.current_step}/{len(self.tool_calls)}: {tool_call['description']}")

                # Simulate tool call execution
                result = await self._execute_tool_call(tool_call)
                self.execution_log.append(result)

                # Add small delay between calls to make it visible in UI
                await asyncio.sleep(0.5)

            # Generate final results
            total_duration = time.time() - self.start_time
            success_count = sum(1 for r in self.execution_log if r["success"])

            result = {
                "agent_name": self.name,
                "agent_version": self.version,
                "task_info": task_info,
                "total_steps": len(self.tool_calls),
                "successful_steps": success_count,
                "total_duration": total_duration,
                "execution_log": self.execution_log,
                "success": success_count == len(self.tool_calls),
                "flag": self._extract_flag(),
                "summary": f"Executed {success_count}/{len(self.tool_calls)} tool calls successfully",
            }

            logger.info(f"✅ {self.name} completed successfully")
            logger.info(f"📊 Results: {success_count}/{len(self.tool_calls)} successful, {total_duration:.2f}s total")

            return result

        except Exception as e:
            logger.error(f"❌ {self.name} failed: {e}")
            return {"agent_name": self.name, "error": str(e), "success": False, "execution_log": self.execution_log}

    async def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate executing a tool call.

        In a real agent, this would make an actual MCP tool call.
        For testing, we simulate the call and generate realistic results.
        """
        start_time = time.time()

        try:
            # Simulate tool execution time
            await asyncio.sleep(tool_call.get("expected_duration", 1.0))

            # Generate simulated output based on command
            command = tool_call["command"]
            simulated_output = self._generate_simulated_output(command)

            duration = time.time() - start_time

            result = {
                "step": self.current_step,
                "tool": tool_call["tool"],
                "command": command,
                "description": tool_call["description"],
                "success": True,
                "duration": duration,
                "output": simulated_output,
                "timestamp": time.time(),
            }

            logger.info(f"   ✅ Tool call successful: {command}")
            return result

        except Exception as e:
            duration = time.time() - start_time

            result = {
                "step": self.current_step,
                "tool": tool_call["tool"],
                "command": tool_call["command"],
                "description": tool_call["description"],
                "success": False,
                "duration": duration,
                "error": str(e),
                "timestamp": time.time(),
            }

            logger.error(f"   ❌ Tool call failed: {command} - {e}")
            return result

    def _generate_simulated_output(self, command: str) -> str:
        """Generate realistic simulated output for common commands."""
        if command.startswith("ls"):
            return """total 24
drwxr-xr-x  3 agent agent 4096 Sep  1 12:00 .
drwxr-xr-x  5 root  root  4096 Sep  1 11:00 ..
-rw-r--r--  1 agent agent  220 Sep  1 11:00 .bash_logout
-rw-r--r--  1 agent agent 3526 Sep  1 11:00 .bashrc
drwxr-xr-x  2 agent agent 4096 Sep  1 12:00 workspace
-rw-r--r--  1 agent agent  807 Sep  1 11:00 .profile"""

        elif "python --version" in command:
            return "Python 3.11.5"

        elif "curl" in command and "httpbin.org" in command:
            return json.dumps(
                {
                    "args": {},
                    "headers": {"Accept": "*/*", "Host": "httpbin.org", "User-Agent": "curl/7.68.0"},
                    "origin": "203.0.113.195",
                    "url": "http://httpbin.org/get",
                },
                indent=2,
            )

        elif command == "whoami":
            return "agent"

        elif command == "pwd":
            return "/home/agent/workspace"

        elif "SABER_FLAG" in command:
            return "SABER_FLAG{test_successful_dummy_agent}"

        else:
            return f"Command executed: {command}"

    def _extract_flag(self) -> Optional[str]:
        """Extract the flag from execution log if present."""
        for entry in self.execution_log:
            if entry.get("success") and "SABER_FLAG" in entry.get("output", ""):
                # Extract flag from output
                output = str(entry["output"])
                if "SABER_FLAG{" in output:
                    start = output.find("SABER_FLAG{")
                    end = output.find("}", start) + 1
                    return output[start:end]
        return None


# Entry point for SABER harness
async def main(task_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Main entry point for the dummy test agent.

    This function will be called by the SABER harness.
    """
    agent = DummyTestAgent()
    return await agent.run(task_info)


# For direct execution and testing
if __name__ == "__main__":

    async def test_agent() -> None:
        """Test the agent locally."""
        print("🧪 Testing DummyTestAgent locally...")

        # Mock task info
        task_info = {"task_id": "test_task_001", "description": "Test task for dummy agent", "environment": "test"}

        result = await main(task_info)

        print("\n📊 Agent Results:")
        print(json.dumps(result, indent=2, default=str))

        if result.get("success"):
            print("\n✅ Agent test completed successfully!")
            if result.get("flag"):
                print(f"🏁 Flag captured: {result['flag']}")
        else:
            print("\n❌ Agent test failed!")

    # Run the test
    asyncio.run(test_agent())
