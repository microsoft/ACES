#!/usr/bin/env python3
"""
SABER Client CLI - Zero-friction entrypoint for customers.

Usage:
    python -m saber.client --agent <path_to_agent> --task task_id --episodes 2
    python -m saber.client --agent my_package.MyAgent --server http://localhost:8000
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .agent_wrapper import AgentLoader
from .test_harness import TestHarness, TestHarnessConfig


async def run_saber_client(
    agent_path: str,
    agent_class: Optional[str] = None,
    task_id: str = "default_task",
    episodes: int = 1,
    server_url: str = "http://localhost:8000",
    max_steps: int = 100,
    step_delay: float = 0.0,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """
    Run SABER client with customer's agent.

    Args:
        agent_path: Path to agent file or module name
        agent_class: Optional specific class name to load
        task_id: Task ID to run
        episodes: Number of episodes to run
        server_url: SABER server URL
        max_steps: Maximum steps per episode
        step_delay: Delay between steps (seconds)
        log_level: Logging level
        log_file: Optional log file path
    """
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), *([] if not log_file else [logging.FileHandler(log_file)])],
    )

    logger = logging.getLogger("saber.client.cli")

    print("🚀 SABER Client - Security Agent Benchmarking")
    print("=" * 50)
    print(f"Agent: {agent_path}")
    if agent_class:
        print(f"Class: {agent_class}")
    print(f"Task: {task_id}")
    print(f"Episodes: {episodes}")
    print(f"Server: {server_url}")
    print()

    # Load agent
    try:
        logger.info(f"Loading agent from: {agent_path}")

        if Path(agent_path).exists():
            # Load from file path
            agent = AgentLoader.load_from_path(agent_path, agent_class)
        else:
            # Try loading as module
            agent = AgentLoader.load_from_module(agent_path, agent_class)

        logger.info(f"✅ Agent loaded successfully: {type(agent.agent).__name__}")

    except Exception as e:
        logger.error(f"❌ Failed to load agent: {e}")
        sys.exit(1)

    # Configure test harness
    config = TestHarnessConfig(
        server_url=server_url,
        max_steps=max_steps,
        step_delay=step_delay,
        log_level=log_level,
        log_file=Path(log_file) if log_file else None,
    )

    # Run episodes
    total_steps = 0
    completed_episodes = 0

    for episode_num in range(1, episodes + 1):
        print(f"\n📊 Episode {episode_num}/{episodes}")
        print("-" * 30)

        try:
            async with TestHarness(config) as harness:
                await harness.initialize(agent)

                async def run_test_with_task() -> Dict[str, Any]:
                    # Custom run_test that specifies task_id
                    if not harness.server_client or not harness.agent:
                        raise ValueError("Test harness not initialized. Call initialize() first.")

                    harness.logger.info("Starting test execution")
                    harness.is_running = True

                    try:
                        # Create session
                        harness.session_id = await harness.server_client.create_session()
                        harness.logger.info(f"Created session: {harness.session_id}")

                        # Start episode with specific task_id
                        episode_info = await harness.server_client.start_episode(task_id)
                        harness.episode_id = episode_info.episode_id
                        harness.logger.info(f"Started episode: {harness.episode_id}")

                        # Continue with normal test execution...
                        # Get task and policy information
                        task_info = await harness.server_client.get_current_task()
                        policy_info = await harness.server_client.get_policy()

                        harness.logger.info(f"Task: {task_info.title}")
                        harness.logger.info(f"Available commands: {policy_info.available_commands}")

                        # Build initial prompt
                        from .prompt_builder import PromptBuilder

                        initial_prompt = PromptBuilder.build_initial_prompt(task_info, policy_info, episode_info)

                        # Execute test loop
                        current_prompt = initial_prompt
                        step_response = None

                        while harness.is_running and harness.step_count < harness.config.max_steps:
                            harness.step_count += 1
                            harness.logger.info(f"Step {harness.step_count}")

                            # Get agent response
                            try:
                                agent_response = await harness.agent.process_prompt(current_prompt)
                                harness.logger.info(f"Agent command: {agent_response}")

                                if not agent_response.strip():
                                    harness.logger.warning("Agent returned empty response")
                                    break

                            except Exception as e:
                                harness.logger.error(f"Agent error: {e}")
                                break

                            # Execute step on server
                            try:
                                step_response = await harness.server_client.execute_step(agent_response)
                                harness.logger.info(f"Step success: {step_response.success}")

                                if step_response.error:
                                    harness.logger.warning(f"Step error: {step_response.error}")

                            except Exception as e:
                                harness.logger.error(f"Server communication error: {e}")
                                break

                            # Check if done
                            if step_response.done:
                                harness.logger.info("Episode completed")
                                completion_prompt = PromptBuilder.build_completion_prompt(step_response.output)
                                harness.logger.info("Final result:")
                                harness.logger.info(completion_prompt)
                                break

                            # Build next prompt
                            if step_response.success:
                                current_prompt = PromptBuilder.build_step_prompt(
                                    step_response.output, agent_response, step_response
                                )
                            else:
                                current_prompt = PromptBuilder.build_error_prompt(
                                    step_response.error or "Unknown error", agent_response
                                )

                            # Optional delay between steps
                            if harness.config.step_delay > 0:
                                await asyncio.sleep(harness.config.step_delay)

                        # Prepare results
                        results = {
                            "session_id": harness.session_id,
                            "episode_id": harness.episode_id,
                            "steps_executed": harness.step_count,
                            "completed": step_response.done if step_response else False,
                            "final_step": step_response.model_dump() if step_response else None,
                            "task_info": task_info.model_dump(),
                        }

                        harness.logger.info(f"Test completed: {results}")
                        return results

                    except Exception as e:
                        harness.logger.error(f"Test execution failed: {e}")
                        raise
                    finally:
                        harness.is_running = False

                # Run the modified test
                results = await run_test_with_task()

                episode_steps = results["steps_executed"]
                episode_completed = results["completed"]

                total_steps += episode_steps
                if episode_completed:
                    completed_episodes += 1

                print(f"  Steps: {episode_steps}")
                print(f"  Completed: {'✅' if episode_completed else '❌'}")

                # Reset agent for next episode
                await agent.reset()

        except Exception as e:
            logger.error(f"Episode {episode_num} failed: {e}")
            print(f"  Status: ❌ Failed ({e})")

    # Summary
    print("\n" + "=" * 50)
    print("🏆 FINAL RESULTS")
    print("=" * 50)
    print(f"Episodes completed: {completed_episodes}/{episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Average steps per episode: {total_steps/episodes:.1f}")
    print(f"Success rate: {(completed_episodes/episodes)*100:.1f}%")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="SABER Client - Zero-friction security agent benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run agent from file
  python -m saber.client --agent my_agent.py --task malware_analysis --episodes 3

  # Run specific class from file
  python -m saber.client --agent my_agent.py --agent-class MyAgent --task threat_intel

  # Run from installed package
  python -m saber.client --agent my_package.MyAgent --server http://remote:8000

  # With custom settings
  python -m saber.client --agent ./agents/llm_agent.py --episodes 5
        """,
    )

    # Required arguments
    parser.add_argument(
        "--agent", required=True, help="Path to agent file or module name (e.g., 'my_agent.py' or 'package.MyAgent')"
    )

    # Optional arguments
    parser.add_argument("--agent-class", help="Specific class name to load from agent file/module")

    parser.add_argument("--task", help="Task ID to run (default: 'default_task')")

    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run (default: 1)")

    parser.add_argument(
        "--server", default="http://localhost:8000", help="SABER server URL (default: 'http://localhost:8000')"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument("--log-file", help="Optional log file path")

    args = parser.parse_args()

    try:
        asyncio.run(
            run_saber_client(
                agent_path=args.agent,
                agent_class=args.agent_class,
                task_id=args.task,
                episodes=args.episodes,
                server_url=args.server,
                log_level=args.log_level,
                log_file=args.log_file,
            )
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
