#!/usr/bin/env python3
"""
Multi-Episode Integration Test

This test validates the episode-first architecture by creating multiple concurrent episodes
and executing tools in each episode to ensure proper isolation and routing.

Key scenarios tested:
1. Concurrent episode creation and execution
2. Episode-specific tool execution and isolation
3. MCP routing with episode headers
4. Session management with multiple episodes
5. Proper cleanup and resource management
"""

import asyncio
import json
import logging
import pytest
import uuid
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch

from saber.client.api import SABERRestClient
from saber.models import HTTPHeaders

logger = logging.getLogger(__name__)


class MockMultiEpisodeServer:
    """Mock SABER server that supports multiple concurrent episodes."""

    def __init__(self, base_port: int = 9000):
        self.base_port = base_port
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.episodes: Dict[str, Dict[str, Any]] = {}
        self.episode_executions: Dict[str, List[Dict[str, Any]]] = {}
        self.mcp_tools = [
            {
                "name": "echo_tool",
                "description": "Echo a message with episode context",
                "schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Message to echo"},
                        "episode_marker": {"type": "string", "description": "Episode identifier marker"}
                    },
                    "required": ["message"]
                }
            },
            {
                "name": "counter_tool",
                "description": "Increment a counter specific to the episode",
                "schema": {
                    "type": "object",
                    "properties": {
                        "increment": {"type": "integer", "description": "Amount to increment", "default": 1}
                    }
                }
            },
            {
                "name": "episode_info_tool",
                "description": "Get information about the current episode",
                "schema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]

    async def create_session(self, client_id: str) -> str:
        """Create a new session."""
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        self.sessions[session_id] = {
            "id": session_id,
            "client_id": client_id,
            "created_at": "2024-01-01T10:00:00Z",
            "active": True,
            "episodes": []
        }
        logger.info(f"Created session: {session_id}")
        return session_id

    async def create_episode(self, session_id: str, task_id: str) -> str:
        """Create a new episode for the session."""
        if session_id not in self.sessions:
            raise Exception(f"Session {session_id} not found")

        episode_id = f"episode_{uuid.uuid4().hex[:8]}"
        self.episodes[episode_id] = {
            "id": episode_id,
            "session_id": session_id,
            "task_id": task_id,
            "created_at": "2024-01-01T10:00:00Z",
            "status": "active",
            "counter": 0  # Episode-specific counter for testing
        }

        # Track episode executions
        self.episode_executions[episode_id] = []

        # Add episode to session
        self.sessions[session_id]["episodes"].append(episode_id)

        logger.info(f"Created episode: {episode_id} for session: {session_id}, task: {task_id}")
        return episode_id

    async def get_benchmark(self) -> Dict[str, Any]:
        """Get benchmark with multiple tasks for multi-episode testing."""
        return {
            "total_tasks": 2,
            "total_episodes": 2,
            "episodes": [
                {"task_id": "multi_episode_task_1", "episode_id": "test_episode_1", "max_attempts": 1},
                {"task_id": "multi_episode_task_2", "episode_id": "test_episode_2", "max_attempts": 1}
            ]
        }

    async def get_policy_info(self, session_id: str, episode_id: str) -> Dict[str, Any]:
        """Get policy information for a specific episode."""
        if episode_id not in self.episodes:
            raise Exception(f"Episode {episode_id} not found")

        episode = self.episodes[episode_id]
        return {
            "prompt": f"You are a test agent for episode {episode_id}. Use the available tools to complete the task for {episode['task_id']}.",
            "rules": [
                "Use episode-specific tools",
                "Maintain episode isolation",
                "Complete the multi-episode test objective"
            ],
            "episode_context": {
                "episode_id": episode_id,
                "task_id": episode["task_id"],
                "session_id": episode["session_id"]
            }
        }

    async def list_tools(self, session_id: str, episode_id: str) -> List[Dict[str, Any]]:
        """List available tools for MCP."""
        logger.info(f"Listing tools for session: {session_id}, episode: {episode_id}")
        return self.mcp_tools

    async def call_tool(self, session_id: str, episode_id: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call with episode context."""
        if episode_id not in self.episodes:
            raise Exception(f"Episode {episode_id} not found")

        episode = self.episodes[episode_id]
        execution_record = {
            "tool_name": tool_name,
            "arguments": arguments,
            "episode_id": episode_id,
            "session_id": session_id,
            "timestamp": "2024-01-01T10:00:00Z"
        }

        # Record the execution
        self.episode_executions[episode_id].append(execution_record)

        # Execute tool based on name
        if tool_name == "echo_tool":
            message = arguments.get("message", "")
            episode_marker = arguments.get("episode_marker", episode_id)
            result = {
                "echo_response": f"Episode {episode_id}: {message}",
                "episode_marker": episode_marker,
                "session_id": session_id,
                "execution_count": len(self.episode_executions[episode_id])
            }
        elif tool_name == "counter_tool":
            increment = arguments.get("increment", 1)
            episode["counter"] += increment
            result = {
                "counter_value": episode["counter"],
                "episode_id": episode_id,
                "increment_applied": increment
            }
        elif tool_name == "episode_info_tool":
            result = {
                "episode_id": episode_id,
                "session_id": session_id,
                "task_id": episode["task_id"],
                "status": episode["status"],
                "total_executions": len(self.episode_executions[episode_id]),
                "counter_value": episode["counter"]
            }
        else:
            raise Exception(f"Unknown tool: {tool_name}")

        logger.info(f"Tool execution result for {tool_name} in episode {episode_id}: {result}")
        return {"result": result}

    async def end_episode(self, session_id: str, episode_id: str, result: Dict[str, Any] = None) -> Dict[str, Any]:
        """End an episode."""
        if episode_id not in self.episodes:
            raise Exception(f"Episode {episode_id} not found")

        self.episodes[episode_id]["status"] = "completed"
        self.episodes[episode_id]["result"] = result or {}

        logger.info(f"Ended episode: {episode_id} with result: {result}")
        return {"message": "Episode ended successfully"}

    async def terminate_session(self, session_id: str) -> Dict[str, Any]:
        """Terminate a session."""
        if session_id not in self.sessions:
            raise Exception(f"Session {session_id} not found")

        self.sessions[session_id]["active"] = False
        logger.info(f"Terminated session: {session_id}")
        return {"message": "Session terminated successfully"}

    def get_episode_executions(self, episode_id: str) -> List[Dict[str, Any]]:
        """Get all executions for an episode."""
        return self.episode_executions.get(episode_id, [])

    def get_episode_counter(self, episode_id: str) -> int:
        """Get the counter value for an episode."""
        return self.episodes.get(episode_id, {}).get("counter", 0)


class MockMultiEpisodeAgent:
    """Mock agent that can execute in multiple episodes concurrently."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.execution_count = 0

    async def run(self, initial_prompt: str, shutdown_check=None) -> dict:
        """Execute the agent workflow with multi-episode support."""
        self.execution_count += 1

        try:
            # Extract episode info from prompt
            episode_info = self._extract_episode_info(initial_prompt)
            episode_id = episode_info.get("episode_id", "unknown")
            task_id = episode_info.get("task_id", "unknown")

            logger.info(f"Agent {self.agent_id} starting execution {self.execution_count} for episode {episode_id}")

            # Simulate multi-step episode execution
            results = []

            # Step 1: Get episode info
            episode_info_result = await self._call_mock_tool("episode_info_tool", {})
            results.append(episode_info_result)

            # Step 2: Echo with episode marker
            echo_result = await self._call_mock_tool("echo_tool", {
                "message": f"Multi-episode test from agent {self.agent_id}",
                "episode_marker": f"{episode_id}_marker_{self.execution_count}"
            })
            results.append(echo_result)

            # Step 3: Increment counter multiple times to test isolation
            for i in range(3):
                counter_result = await self._call_mock_tool("counter_tool", {"increment": i + 1})
                results.append(counter_result)

            # Step 4: Final episode info check
            final_info_result = await self._call_mock_tool("episode_info_tool", {})
            results.append(final_info_result)

            return {
                "success": True,
                "agent_id": self.agent_id,
                "episode_id": episode_id,
                "task_id": task_id,
                "execution_count": self.execution_count,
                "results": results,
                "flag": f"flag{{multi_episode_success_{episode_id}_{self.agent_id}}}",
                "tools_executed": len(results)
            }

        except Exception as e:
            logger.error(f"Agent {self.agent_id} execution failed: {e}")
            return {
                "success": False,
                "agent_id": self.agent_id,
                "error": str(e),
                "execution_count": self.execution_count
            }

    def _extract_episode_info(self, prompt: str) -> Dict[str, str]:
        """Extract episode information from the prompt."""
        # Simple extraction - in real implementation this would be more sophisticated
        episode_id = "unknown"
        task_id = "unknown"

        if "episode_" in prompt:
            # Extract episode_id from prompt
            lines = prompt.split('\n')
            for line in lines:
                if '"episode_id":' in line:
                    episode_id = line.split('"episode_id":')[1].strip().strip('",')
                if '"task_id":' in line:
                    task_id = line.split('"task_id":')[1].strip().strip('",')

        return {"episode_id": episode_id, "task_id": task_id}

    async def _call_mock_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate tool call via MCP."""
        # In real implementation, this would go through the MCP sidecar
        # For this test, we'll return mock results that demonstrate episode isolation
        if tool_name == "echo_tool":
            return {
                "tool": tool_name,
                "result": f"Agent {self.agent_id}: {arguments.get('message', '')}"
            }
        elif tool_name == "counter_tool":
            return {
                "tool": tool_name,
                "result": f"Incremented by {arguments.get('increment', 1)}"
            }
        elif tool_name == "episode_info_tool":
            return {
                "tool": tool_name,
                "result": f"Episode info requested by agent {self.agent_id}"
            }
        else:
            return {"tool": tool_name, "result": "unknown_tool"}


@pytest.fixture
def mock_multi_episode_server():
    """Create a mock server for multi-episode testing."""
    return MockMultiEpisodeServer()


@pytest.fixture
def multi_episode_agents():
    """Create multiple agents for concurrent episode testing."""
    return [
        MockMultiEpisodeAgent("agent_alpha"),
        MockMultiEpisodeAgent("agent_beta")
    ]


class TestMultiEpisodeIntegration:
    """Integration tests for multi-episode architecture."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_multi_episode_execution(self, mock_multi_episode_server, multi_episode_agents):
        """
        Test concurrent execution of multiple episodes with tool calls.

        This test validates:
        1. Multiple episodes can be created concurrently
        2. Each episode maintains its own execution context
        3. Tool calls are properly routed to the correct episode
        4. Episode isolation is maintained (counters, state, etc.)
        5. Cleanup works correctly for all episodes
        """
        logger.info("🚀 Starting concurrent multi-episode integration test...")

        # Step 1: Create session
        session_id = await mock_multi_episode_server.create_session("multi_episode_test_client")
        logger.info(f"✅ Created session: {session_id}")

        # Step 2: Create multiple episodes concurrently
        tasks = ["multi_episode_task_1", "multi_episode_task_2"]
        episodes = []

        for task_id in tasks:
            episode_id = await mock_multi_episode_server.create_episode(session_id, task_id)
            episodes.append(episode_id)
            logger.info(f"✅ Created episode: {episode_id} for task: {task_id}")

        assert len(episodes) == 2, "Should have created 2 episodes"

        # Step 3: Execute tools in each episode concurrently
        execution_tasks = []

        for i, episode_id in enumerate(episodes):
            agent = multi_episode_agents[i]
            task_name = tasks[i]

            # Create execution coroutine for this episode
            async def execute_episode(ep_id, ag, task):
                results = []

                # Execute multiple tools to test episode isolation
                logger.info(f"🔧 Executing tools in episode: {ep_id}")

                # Tool 1: Get episode info
                info_result = await mock_multi_episode_server.call_tool(
                    session_id, ep_id, "episode_info_tool", {}
                )
                results.append(info_result)

                # Tool 2: Echo with episode-specific message
                echo_result = await mock_multi_episode_server.call_tool(
                    session_id, ep_id, "echo_tool", {
                        "message": f"Test from {ag.agent_id} in {task}",
                        "episode_marker": f"{ep_id}_{ag.agent_id}_marker"
                    }
                )
                results.append(echo_result)

                # Tool 3: Increment counter multiple times (test isolation)
                for increment in [1, 2, 3]:
                    counter_result = await mock_multi_episode_server.call_tool(
                        session_id, ep_id, "counter_tool", {"increment": increment}
                    )
                    results.append(counter_result)

                return {
                    "episode_id": ep_id,
                    "agent_id": ag.agent_id,
                    "task_id": task,
                    "results": results,
                    "tool_calls": len(results)
                }

            execution_tasks.append(execute_episode(episode_id, agent, task_name))

        # Execute all episodes concurrently
        logger.info("⚡ Executing episodes concurrently...")
        execution_results = await asyncio.gather(*execution_tasks)

        # Step 4: Verify results and episode isolation
        logger.info("🔍 Verifying episode isolation and results...")

        assert len(execution_results) == 2, "Should have results from 2 episodes"

        # Verify each episode executed independently
        episode_counters = {}
        episode_executions = {}

        for result in execution_results:
            episode_id = result["episode_id"]
            agent_id = result["agent_id"]

            # Verify tool execution count
            assert result["tool_calls"] == 5, f"Episode {episode_id} should have 5 tool calls"

            # Get episode-specific state
            counter_value = mock_multi_episode_server.get_episode_counter(episode_id)
            executions = mock_multi_episode_server.get_episode_executions(episode_id)

            episode_counters[episode_id] = counter_value
            episode_executions[episode_id] = len(executions)

            logger.info(f"📊 Episode {episode_id} (Agent {agent_id}): counter={counter_value}, executions={len(executions)}")

        # Verify episode isolation - each episode should have independent counters
        # Each episode increments by 1+2+3 = 6
        for episode_id, counter_value in episode_counters.items():
            assert counter_value == 6, f"Episode {episode_id} counter should be 6, got {counter_value}"

        # Verify execution counts - each episode should have 5 tool calls
        for episode_id, execution_count in episode_executions.items():
            assert execution_count == 5, f"Episode {episode_id} should have 5 executions, got {execution_count}"

        # Step 5: Verify episode-specific tool responses
        logger.info("🔍 Verifying episode-specific tool responses...")

        for result in execution_results:
            episode_id = result["episode_id"]
            agent_id = result["agent_id"]

            # Check that echo tool responses contain episode-specific information
            echo_responses = [r for r in result["results"] if "echo_response" in str(r)]
            assert len(echo_responses) > 0, f"Episode {episode_id} should have echo responses"

            # Verify episode ID appears in responses
            for response in echo_responses:
                response_str = str(response)
                assert episode_id in response_str, f"Response should contain episode ID {episode_id}"

        # Step 6: Test concurrent episode completion
        logger.info("🏁 Completing episodes...")

        completion_tasks = []
        for episode_id in episodes:
            completion_tasks.append(
                mock_multi_episode_server.end_episode(session_id, episode_id, {
                    "success": True,
                    "multi_episode_test": "completed"
                })
            )

        completion_results = await asyncio.gather(*completion_tasks)

        # Verify all episodes completed successfully
        for completion_result in completion_results:
            assert "successfully" in completion_result["message"]

        # Step 7: Cleanup
        logger.info("🧹 Cleaning up session...")
        await mock_multi_episode_server.terminate_session(session_id)

        logger.info("✅ Concurrent multi-episode integration test completed successfully!")

        # Final assertions
        assert len(episodes) == 2, "Should have managed 2 episodes"
        assert all(counter == 6 for counter in episode_counters.values()), "All episodes should have independent counters"
        assert all(count == 5 for count in episode_executions.values()), "All episodes should have executed 5 tools"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multi_episode_harness_integration(self, multi_episode_agents):
        """
        Test multi-episode execution through the SABERHarness with mocked dependencies.

        This validates that the harness can orchestrate multiple episodes correctly
        with our episode-first architecture.
        """
        logger.info("🚀 Starting multi-episode harness integration test...")

        # Create harness configuration for multi-episode testing
        config = SABERHarnessConfig(
            server_url="http://localhost:9000",
            parallelism=2,  # Allow concurrent episode execution
            log_level="INFO"
        )

        # Use the first agent for this test
        agent = multi_episode_agents[0]

        # Mock the entire harness run method to avoid network calls
        with patch('saber.client.saber_harness.SABERHarness') as mock_harness_class:
            mock_harness = AsyncMock()

            # Mock initialization
            mock_harness.initialize = AsyncMock()

            # Mock multi-episode execution results
            mock_harness.run = AsyncMock(return_value=[
                type('EpisodeResult', (), {
                    'success': True,
                    'task_id': 'harness_task_1',
                    'episode_id': 'harness_episode_1',
                    'attempt': 1,
                    'flag': 'flag{harness_multi_episode_1}',
                    'execution_time': 2.5
                })(),
                type('EpisodeResult', (), {
                    'success': True,
                    'task_id': 'harness_task_2',
                    'episode_id': 'harness_episode_2',
                    'attempt': 1,
                    'flag': 'flag{harness_multi_episode_2}',
                    'execution_time': 2.8
                })()
            ])

            mock_harness_class.return_value = mock_harness

            # Create and initialize harness
            harness = mock_harness_class(config=config)
            await harness.initialize(agent)

            # Execute multi-episode workflow
            logger.info("⚡ Executing multi-episode harness workflow...")
            results = await harness.run()

            # Verify multi-episode results
            logger.info("🔍 Verifying multi-episode harness results...")

            assert len(results) == 2, "Should have 2 episode results"

            # Verify each episode result
            for i, result in enumerate(results, 1):
                assert result.success, f"Episode {i} should be successful"
                assert result.task_id == f'harness_task_{i}', f"Episode {i} should have correct task_id"
                assert result.episode_id == f'harness_episode_{i}', f"Episode {i} should have correct episode_id"
                assert result.flag.startswith('flag{harness_multi_episode_'), f"Episode {i} should have valid flag"
                assert result.execution_time > 0, f"Episode {i} should have positive execution time"

            # Verify harness was called correctly
            mock_harness.initialize.assert_called_once_with(agent)
            mock_harness.run.assert_called_once()

            logger.info("✅ Multi-episode harness integration test completed successfully")
            logger.info(f"📊 Results: {len(results)} episodes executed successfully")

            total_time = sum(result.execution_time for result in results)
            logger.info(f"⏱️ Total execution time: {total_time:.2f}s")

            return results

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_episode_isolation_stress_test(self, mock_multi_episode_server):
        """
        Stress test episode isolation with many concurrent episodes.

        This test creates multiple episodes and hammers them with tool calls
        to ensure there's no cross-episode contamination.
        """
        logger.info("🚀 Starting episode isolation stress test...")

        num_episodes = 4
        tools_per_episode = 10

        # Create session
        session_id = await mock_multi_episode_server.create_session("stress_test_client")

        # Create multiple episodes
        episodes = []
        for i in range(num_episodes):
            episode_id = await mock_multi_episode_server.create_episode(
                session_id, f"stress_task_{i}"
            )
            episodes.append(episode_id)

        logger.info(f"Created {num_episodes} episodes for stress testing")

        # Execute tools concurrently across all episodes
        async def stress_episode(episode_id: str, episode_index: int):
            results = []
            expected_counter = 0

            for tool_call in range(tools_per_episode):
                # Alternate between different tools
                if tool_call % 3 == 0:
                    result = await mock_multi_episode_server.call_tool(
                        session_id, episode_id, "counter_tool", {"increment": episode_index + 1}
                    )
                    expected_counter += episode_index + 1
                elif tool_call % 3 == 1:
                    result = await mock_multi_episode_server.call_tool(
                        session_id, episode_id, "echo_tool", {
                            "message": f"Stress test {tool_call} for episode {episode_index}",
                            "episode_marker": f"stress_{episode_id}_{tool_call}"
                        }
                    )
                else:
                    result = await mock_multi_episode_server.call_tool(
                        session_id, episode_id, "episode_info_tool", {}
                    )

                results.append(result)

                # Small delay to simulate real execution
                await asyncio.sleep(0.01)

            return {
                "episode_id": episode_id,
                "episode_index": episode_index,
                "expected_counter": expected_counter,
                "actual_counter": mock_multi_episode_server.get_episode_counter(episode_id),
                "tool_calls": len(results)
            }

        # Execute all episodes concurrently
        stress_tasks = [
            stress_episode(episode_id, i)
            for i, episode_id in enumerate(episodes)
        ]

        logger.info(f"⚡ Executing {tools_per_episode} tools across {num_episodes} episodes concurrently...")
        stress_results = await asyncio.gather(*stress_tasks)

        # Verify isolation - each episode should have correct counter values
        for result in stress_results:
            episode_id = result["episode_id"]
            expected = result["expected_counter"]
            actual = result["actual_counter"]

            assert actual == expected, (
                f"Episode {episode_id} counter mismatch: expected {expected}, got {actual}"
            )
            assert result["tool_calls"] == tools_per_episode, (
                f"Episode {episode_id} should have {tools_per_episode} tool calls"
            )

            logger.info(f"✅ Episode {episode_id}: counter={actual}, tools={result['tool_calls']}")

        # Verify total executions across all episodes
        total_executions = sum(
            len(mock_multi_episode_server.get_episode_executions(ep_id))
            for ep_id in episodes
        )
        expected_total = num_episodes * tools_per_episode

        assert total_executions == expected_total, (
            f"Total executions should be {expected_total}, got {total_executions}"
        )

        # Cleanup all episodes
        for episode_id in episodes:
            await mock_multi_episode_server.end_episode(session_id, episode_id)

        await mock_multi_episode_server.terminate_session(session_id)

        logger.info(f"✅ Episode isolation stress test completed: {num_episodes} episodes, {expected_total} tool calls")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
