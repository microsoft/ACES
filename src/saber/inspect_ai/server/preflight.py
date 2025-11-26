"""Preflight health checks for SABER domain environments.

This module provides preflight validation of compose environments before
evaluation. It executes the saber-domain CLI preflight command with proper
signal handling and user feedback.

Key features:
- Concurrent environment health checks
- Graceful Ctrl+C cancellation
- Real-time progress output
- Detailed error reporting
"""

import asyncio
import signal
import sys
from pathlib import Path

from inspect_ai._util.error import PrerequisiteError

from ...logging_config import LogCategory, get_saber_logger

logger = get_saber_logger(LogCategory.AGENT, __name__)


async def run_preflight_check(
    domain_slug: str,
    domains_root: Path,
    concurrency: int = 8,
    timeout: int = 180,
) -> None:
    """Run saber-domain preflight check before evaluation.

    Executes the preflight CLI command to validate all compose environments
    are healthy before starting the evaluation. This catches configuration
    issues early and prevents wasted evaluation time.

    Args:
        domain_slug: Domain to check
        domains_root: Path to domains directory
        concurrency: Number of environments to check in parallel (default: 8)
        timeout: Timeout per environment in seconds (default: 180)

    Raises:
        PrerequisiteError: If preflight check fails or command execution fails
    """
    logger.info(
        f"Running preflight check for domain '{domain_slug}'",
        extra={
            "domain": domain_slug,
            "concurrency": concurrency,
            "timeout": timeout,
        },
    )

    # Print to terminal for user feedback
    print(f"\n🚀 Running preflight check for domain '{domain_slug}'...")
    print(f"   Testing {concurrency} environments in parallel (timeout: {timeout}s per environment)")
    print("   Press Ctrl+C to cancel\n")

    # Find the saber-domain CLI
    # Try to use the same Python interpreter and environment
    cli_cmd = [sys.executable, "-m", "saber.domain.cli", "preflight", domain_slug]

    # Add options
    cli_cmd.extend(["-c", str(concurrency)])
    cli_cmd.extend(["--timeout", str(timeout)])
    cli_cmd.append("--verbose")  # Show detailed progress

    # Set working directory to domains_root parent for proper path resolution
    cwd = domains_root.parent if domains_root.parent.exists() else domains_root

    process = None
    try:
        # Run preflight command with direct stdout/stderr passthrough for real-time output
        process = await asyncio.create_subprocess_exec(
            *cli_cmd,
            stdout=None,  # Pass through to terminal
            stderr=None,  # Pass through to terminal
            cwd=cwd,
        )

        # Wait for completion with cancellation support
        try:
            exit_code = await process.wait()
        except asyncio.CancelledError:
            # User hit Ctrl+C - terminate the subprocess gracefully
            print("\n⚠️  Ctrl+C detected - cancelling preflight check...")
            print("   Sending termination signal to preflight process...")

            if process.returncode is None:  # Process still running
                try:
                    # Send SIGINT first (graceful)
                    process.send_signal(signal.SIGINT)
                    print("   Waiting for cleanup to complete (this may take a few seconds)...")

                    # Wait up to 10 seconds for graceful shutdown
                    try:
                        await asyncio.wait_for(process.wait(), timeout=10.0)
                        print("   ✓ Preflight check cancelled and cleaned up")
                    except asyncio.TimeoutError:
                        # If graceful didn't work, force terminate
                        print("   Graceful shutdown timed out, forcing termination...")
                        process.terminate()
                        try:
                            await asyncio.wait_for(process.wait(), timeout=5.0)
                            print("   ✓ Preflight check forcefully terminated")
                        except asyncio.TimeoutError:
                            # Last resort - kill
                            print("   Force termination timed out, killing process...")
                            process.kill()
                            await process.wait()
                            print("   ✓ Preflight check killed")
                except Exception as e:
                    logger.warning(f"Error during preflight cancellation: {e}")
                    print(f"   ⚠️  Warning: Error during cleanup: {e}")

            raise PrerequisiteError(
                "Preflight check cancelled by user (Ctrl+C). "
                "Some Docker containers may still be cleaning up in the background."
            )

        if exit_code != 0:
            raise PrerequisiteError(
                f"\nPreflight check failed for domain '{domain_slug}' (exit code: {exit_code}).\n\n"
                f"One or more compose environments failed health checks.\n"
                f"Review the preflight output above for details on which environments failed.\n\n"
                f"To fix:\n"
                f"  1. Check compose file configurations in domains/{domain_slug}/server/config/environments/\n"
                f"  2. Verify healthcheck endpoints match your service ports\n"
                f"  3. Run preflight manually for debugging: uv run saber-domain preflight {domain_slug} --verbose"
            )

        logger.info(
            f"Preflight check passed for domain '{domain_slug}'",
            extra={"domain": domain_slug},
        )
        print(f"\n✅ Preflight check passed for domain '{domain_slug}'\n")

    except FileNotFoundError as e:
        raise PrerequisiteError(
            f"Failed to run saber-domain preflight command.\n\n"
            f"Ensure SABER is properly installed:\n"
            f"  uv pip install -e external/saber\n\n"
            f"Error: {e}"
        ) from e
    except asyncio.CancelledError:
        # Re-raise CancelledError to properly propagate cancellation
        raise
    except Exception as e:
        if isinstance(e, PrerequisiteError):
            raise
        raise PrerequisiteError(
            f"Preflight check failed with unexpected error: {e}\n\n"
            f"Command: {' '.join(cli_cmd)}\n"
            f"Working directory: {cwd}"
        ) from e
    finally:
        # Ensure process is cleaned up even if something goes wrong
        if process is not None and process.returncode is None:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass  # Best effort cleanup
