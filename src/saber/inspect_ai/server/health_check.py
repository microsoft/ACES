"""SABER Server Health Check with Retry Logic.

This module provides health check functionality for SABER server instances,
with exponential backoff and retry logic to handle server startup delays.
"""

import asyncio
from pathlib import Path

import aiohttp
from inspect_ai._util.error import PrerequisiteError

from ...logging_config import LogCategory, get_saber_logger

logger = get_saber_logger(LogCategory.AGENT, __name__)


def _get_server_log_file_contents(domain: str, domains_root: Path | None, tail_lines: int = 50) -> tuple[str, str]:
    """Read the most recent server log file from the domain's server-logs directory.

    Args:
        domain: Domain name
        domains_root: Path to the domains directory
        tail_lines: Number of log lines to fetch from the end

    Returns:
        Tuple of (log_file_path, log_contents_or_error_message)
    """
    if domains_root is None:
        return "(unknown)", "(domains_root not provided, cannot read server logs)"

    server_logs_dir = domains_root / domain / "server" / "logs" / "server-logs"

    if not server_logs_dir.exists():
        return str(server_logs_dir), f"(server logs directory does not exist: {server_logs_dir})"

    try:
        # Find the most recent log file by modification time
        log_files = list(server_logs_dir.glob("saber-server-*.log"))
        if not log_files:
            return str(server_logs_dir), f"(no saber-server-*.log files found in {server_logs_dir})"

        # Sort by modification time, most recent last
        most_recent = max(log_files, key=lambda f: f.stat().st_mtime)
        log_path = str(most_recent)

        # Read the last N lines
        with open(most_recent, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            tail = lines[-tail_lines:] if len(lines) > tail_lines else lines
            content = "".join(tail).strip()

        if content:
            return log_path, content
        return log_path, "(server log file is empty)"

    except Exception as e:
        return str(server_logs_dir), f"(error reading server logs: {e})"


def _extract_error_lines(log_contents: str, max_errors: int = 20) -> str:
    """Extract lines containing ERROR from log contents.

    Args:
        log_contents: Raw log file contents
        max_errors: Maximum number of error lines to return

    Returns:
        String with error lines, or message if none found
    """
    error_lines = [line for line in log_contents.split("\n") if "ERROR" in line]

    if not error_lines:
        return "(no ERROR messages found in logs)"

    if len(error_lines) > max_errors:
        return f"(showing last {max_errors} of {len(error_lines)} errors)\n" + "\n".join(error_lines[-max_errors:])

    return "\n".join(error_lines)


async def wait_for_server_health(
    rest_url: str,
    domain: str = "unknown",
    domains_root: Path | None = None,
) -> None:
    """Wait for SABER server to become healthy, polling indefinitely.

    Polls the health endpoint every 15 seconds until successful.
    To cancel, run: touch /tmp/saber_stop_health_check

    Args:
        rest_url: Base URL for REST API
        domain: Domain name for fetching container logs on failure
        domains_root: Path to the domains directory for reading server logs

    Raises:
        PrerequisiteError: If server startup explicitly fails (503) or user cancels
    """
    import time

    health_url = f"{rest_url}/api/v1/health"
    poll_interval = 15.0  # seconds
    warn_after = 600  # 10 minutes
    stop_file = Path("/tmp/saber_stop_health_check")

    # Clean up any stale stop file from previous run
    if stop_file.exists():
        stop_file.unlink()

    print(f"[HEALTH CHECK] Starting health check for {domain} at {health_url}", flush=True)
    print(f"[HEALTH CHECK] Polling every {int(poll_interval)}s until healthy...", flush=True)
    print(f"[HEALTH CHECK] To cancel, run this in new terminal: touch {stop_file}", flush=True)

    start_time = time.time()
    attempt = 0
    warned_about_cancel = False

    try:
        while True:
            # Check for stop file
            if stop_file.exists():
                stop_file.unlink()  # Clean up
                elapsed = time.time() - start_time
                print(f"\n[HEALTH CHECK] Stopped by user after {int(elapsed)}s", flush=True)

                # Fetch server logs to help diagnose issues
                log_path, log_contents = _get_server_log_file_contents(domain, domains_root)
                error_lines = _extract_error_lines(log_contents)

                print(f"[HEALTH CHECK] Server log file: {log_path}", flush=True)
                print("\n[HEALTH CHECK] ERROR messages in logs:", flush=True)
                print("=" * 60, flush=True)
                print(error_lines, flush=True)
                print("=" * 60, flush=True)
                print(f"\n[HEALTH CHECK] Last {50} lines of server log:", flush=True)
                print("=" * 60, flush=True)
                print(log_contents, flush=True)
                print("=" * 60, flush=True)

                # Stop the server container
                import subprocess

                container_name = f"{domain}-saber-server"
                print(f"\n[HEALTH CHECK] Stopping server container: {container_name}", flush=True)
                try:
                    subprocess.run(
                        ["docker", "stop", container_name],
                        capture_output=True,
                        timeout=30,
                    )
                    subprocess.run(
                        ["docker", "rm", container_name],
                        capture_output=True,
                        timeout=10,
                    )
                    print("[HEALTH CHECK] ✓ Server container stopped and removed", flush=True)
                except Exception as e:
                    print(f"[HEALTH CHECK] Warning: Failed to stop container: {e}", flush=True)

                raise PrerequisiteError(
                    f"Health check stopped by user after {int(elapsed)} seconds.\n\n"
                    f"Server log file: {log_path}\n\n"
                    f"ERROR messages:\n"
                    f"{'=' * 60}\n"
                    f"{error_lines}\n"
                    f"{'=' * 60}"
                )
            attempt += 1
            elapsed = time.time() - start_time
            elapsed_min = int(elapsed // 60)
            elapsed_sec = int(elapsed % 60)

            # After 10 minutes, suggest Ctrl+C option (once)
            if elapsed >= warn_after and not warned_about_cancel:
                print(f"[HEALTH CHECK] ⚠ Waiting for {elapsed_min}m - press Ctrl+C to cancel if needed", flush=True)
                warned_about_cancel = True

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            data = await response.json()
                            print(
                                f"[HEALTH CHECK] ✓ Server healthy after {elapsed_min}m {elapsed_sec}s "
                                f"({attempt} attempts)",
                                flush=True,
                            )
                            logger.info(
                                f"SABER server health check passed (attempt {attempt})",
                                extra={
                                    "rest_url": rest_url,
                                    "attempt": attempt,
                                    "domain": data.get("domain", "unknown"),
                                    "elapsed_seconds": elapsed,
                                },
                            )
                            return
                        elif response.status == 202:
                            # Server is starting up - show progress from response
                            data = await response.json()
                            status_msg = data.get("message", "Starting...")
                            unhealthy = data.get("unhealthy_services", [])

                            # Always print status with cancel hint
                            print(
                                f"[HEALTH CHECK] [{elapsed_min}m {elapsed_sec}s] {status_msg} "
                                f"(cancel: touch {stop_file})",
                                flush=True,
                            )
                            if unhealthy:
                                print(f"[HEALTH CHECK]   Waiting for: {', '.join(unhealthy[:5])}", flush=True)

                            logger.info(
                                f"Server starting: {status_msg}",
                                extra={"rest_url": rest_url, "attempt": attempt, "unhealthy_services": unhealthy},
                            )
                        elif response.status == 503:
                            # Server startup failed - get error details
                            try:
                                data = await response.json()
                                error_detail = data.get("detail", {})
                                health_data = error_detail.get("health_data", {})
                                perm_env = health_data.get("permanent_environment", {})
                                error_msg = perm_env.get("error", "Unknown error")
                                print(f"[HEALTH CHECK] ✗ Server startup failed: {error_msg}", flush=True)
                            except Exception:
                                error_msg = "Permanent environment failed to start"
                                print("[HEALTH CHECK] ✗ Server returned 503 (startup failed)", flush=True)

                            # Don't keep retrying if startup explicitly failed
                            raise PrerequisiteError(
                                f"SABER server startup failed.\nREST URL: {rest_url}\nError: {error_msg}"
                            )
                        else:
                            print(
                                f"[HEALTH CHECK] [{elapsed_min}m {elapsed_sec}s] Server returned {response.status} "
                                f"(cancel: touch {stop_file})",
                                flush=True,
                            )
                            logger.warning(
                                f"Health check returned {response.status} (attempt {attempt})",
                                extra={"rest_url": rest_url, "attempt": attempt},
                            )
            except PrerequisiteError:
                raise  # Re-raise our own errors
            except asyncio.CancelledError:
                raise  # Re-raise cancellation
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                # Print connection status with cancel hint
                print(
                    f"[HEALTH CHECK] [{elapsed_min}m {elapsed_sec}s] Connection failed: {type(e).__name__} "
                    f"(cancel: touch {stop_file})",
                    flush=True,
                )
                logger.info(
                    f"Health check attempt {attempt} failed: {e}",
                    extra={"rest_url": rest_url, "attempt": attempt},
                )

            await asyncio.sleep(poll_interval)

    except asyncio.CancelledError:
        elapsed = time.time() - start_time
        print(f"\n[HEALTH CHECK] Cancelled after {int(elapsed)}s", flush=True)
        raise  # Re-raise so Inspect AI can handle cleanup
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n[HEALTH CHECK] Interrupted after {int(elapsed)}s", flush=True)
        raise  # Re-raise so Python can handle it
