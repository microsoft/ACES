#!/usr/bin/env python3
"""Part 1: Start AIRT Demo server and create episodes.

This script:
1. Starts the SABER server using DomainController
2. Creates a session and episodes using ClientSessionManager
3. Saves episode IDs to a JSON file for Part 2

Usage:
    cd external/saber
    uv run python scripts/manual_airt_test/start_server.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from saber.client.client_session import ClientSessionManager
from saber.client.models import SessionManagerConfig
from saber.inspect_ai.server.server import DomainController

# Output file for episode data
OUTPUT_FILE = Path(__file__).parent / "episode_data.json"


async def main():
    """Start server and create episodes."""

    # Configuration
    domains_root = Path(__file__).parent.parent.parent / "domains"
    domain = "airt_demo"
    rest_port = 8000
    mcp_port = 8001

    print("=" * 60)
    print("Part 1: Start AIRT Demo Server and Create Episodes")
    print("=" * 60)
    print(f"\nDomains root: {domains_root}")
    print(f"Domain: {domain}")
    print(f"REST port: {rest_port}")
    print(f"MCP port: {mcp_port}")

    # Step 1: Start the domain server
    print("\n" + "-" * 60)
    print("Step 1: Starting SABER domain server...")
    print("-" * 60)

    controller = DomainController(domains_root)

    # Check if already running
    running = await controller.check_running_domain(rest_port, mcp_port)
    if running:
        print(f"Domain '{running}' is already running on ports {rest_port}/{mcp_port}")
        if running != domain:
            print(f"WARNING: Different domain running! Expected {domain}")
            return
    else:
        print(f"Starting domain '{domain}'...")
        context = await controller.start(
            domain=domain,
            rest_port=rest_port,
            mcp_port=mcp_port,
            log_level="INFO",
        )
        print(f"✓ Domain started successfully")
        print(f"  REST URL: {context.rest_url}")
        print(f"  MCP URL: {context.mcp_url}")

    rest_url = f"http://localhost:{rest_port}"
    mcp_url = f"http://localhost:{mcp_port}"

    # Step 2: Create session and episodes using ClientSessionManager
    print("\n" + "-" * 60)
    print("Step 2: Creating session and episodes...")
    print("-" * 60)

    config = SessionManagerConfig.from_urls(
        rest_url=rest_url,
        mcp_url=mcp_url,
        client_id="manual_test",
    )
    session_manager = ClientSessionManager(config)

    # Create session
    session_id = await session_manager.create_session()
    print(f"✓ Session created: {session_id}")

    # Task IDs from the orchestrated task structure
    # The template expansion creates these concrete task IDs:
    blue_task_id = "ai_redteam_blue_agent_ai_redteam_red_attacker"  # Blue team task
    red_task_id = "ai_redteam_red_attacker"  # Red team task

    # Create blue team episode first (root_first creation order)
    print(f"\nCreating blue team episode ({blue_task_id})...")
    blue_response = await session_manager.create_episode(session_id, blue_task_id)
    print(f"✓ Blue episode created: {blue_response.episode_id}")
    print(f"  State: {blue_response.state}")

    # Wait for blue episode to be ready
    print("  Waiting for blue episode to be ready...")
    blue_status = await session_manager.wait_for_episode_ready(
        session_id, blue_response.episode_id, timeout_seconds=120
    )
    print(f"✓ Blue episode ready: {blue_status.state}")

    # Create red team episode (depends on blue)
    print(f"\nCreating red team episode ({red_task_id})...")
    red_response = await session_manager.create_episode(session_id, red_task_id)
    print(f"✓ Red episode created: {red_response.episode_id}")
    print(f"  State: {red_response.state}")
    print(f"  Attached to: {red_response.attached_to_episode_id}")

    # Wait for red episode to be ready
    print("  Waiting for red episode to be ready...")
    red_status = await session_manager.wait_for_episode_ready(
        session_id, red_response.episode_id, timeout_seconds=120
    )
    print(f"✓ Red episode ready: {red_status.state}")

    # Step 3: Save episode data to file
    print("\n" + "-" * 60)
    print("Step 3: Saving episode data...")
    print("-" * 60)

    episode_data = {
        "domain": domain,
        "rest_url": rest_url,
        "mcp_url": mcp_url,
        "session_id": session_id,
        "blue": {
            "task_id": blue_task_id,
            "episode_id": blue_response.episode_id,
        },
        "red": {
            "task_id": red_task_id,
            "episode_id": red_response.episode_id,
            "attached_to": red_response.attached_to_episode_id,
        },
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(episode_data, f, indent=2)

    print(f"✓ Episode data saved to: {OUTPUT_FILE}")
    print(f"\nEpisode Data:")
    print(json.dumps(episode_data, indent=2))

    print("\n" + "=" * 60)
    print("Server and episodes ready!")
    print("=" * 60)
    print(f"\nNext step: Run part 2 to interact with tools:")
    print(f"  uv run python scripts/manual_airt_test/call_tools.py")
    print(f"\nTo stop the server:")
    print(f"  uv run saber-domain stop {domain}")


if __name__ == "__main__":
    asyncio.run(main())
