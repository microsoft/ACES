#!/usr/bin/env python3
"""CLI wrapper to send injection commands via IPC.

This script provides a command-line interface to send prompt injection
commands to the WebSocket daemon running in the red team container.

Usage:
    python inject_via_ipc.py "Your injection message" --target blue-ep-123
    python inject_via_ipc.py "Rewind injection" --target blue-ep-123 --strategy rewind --rewind-count 2
    python inject_via_ipc.py "Insert injection" --target blue-ep-123 --strategy insert --insert-position 1

Examples:
    # Basic append injection
    python inject_via_ipc.py "Ignore previous instructions" --target blue-episode-id

    # Rewind injection (remove last 2 messages and inject)
    python inject_via_ipc.py "New context" --target blue-episode-id --strategy rewind --rewind-count 2

    # Insert at specific position
    python inject_via_ipc.py "System override" --target blue-episode-id --strategy insert --insert-position 0
"""

import argparse
import json
import sys

try:
    import requests
except ImportError:
    print("ERROR: requests library not installed. Run: pip install requests")
    sys.exit(1)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Inject prompt via IPC to WebSocket daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Ignore previous instructions" --target blue-ep-123
  %(prog)s "Rewind test" --target blue-ep-123 --strategy rewind --rewind-count 2
  %(prog)s "Insert test" --target blue-ep-123 --strategy insert --insert-position 1
        """,
    )

    parser.add_argument("message", help="Message to inject into the target episode")
    parser.add_argument("--target", required=True, help="Target episode ID to inject into")
    parser.add_argument(
        "--strategy",
        default="append",
        choices=["append", "rewind", "insert", "replace"],
        help="Injection strategy (default: append)",
    )
    parser.add_argument("--rewind-count", type=int, help="Number of messages to rewind (required for rewind strategy)")
    parser.add_argument("--insert-position", type=int, help="Position to insert at (required for insert strategy)")
    parser.add_argument("--port", type=int, default=9999, help="IPC port (default: 9999)")
    parser.add_argument("--host", default="localhost", help="IPC host (default: localhost)")
    parser.add_argument("--timeout", type=float, default=10.0, help="Request timeout in seconds (default: 10.0)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Validate strategy-specific arguments
    if args.strategy == "rewind" and args.rewind_count is None:
        parser.error("--rewind-count is required when using --strategy rewind")
    if args.strategy == "insert" and args.insert_position is None:
        parser.error("--insert-position is required when using --strategy insert")

    # Build payload
    payload = {
        "target_episode_id": args.target,
        "message": args.message,
        "strategy": args.strategy,
    }

    if args.rewind_count is not None:
        payload["rewind_count"] = args.rewind_count
    if args.insert_position is not None:
        payload["insert_position"] = args.insert_position

    # Send request
    url = f"http://{args.host}:{args.port}/inject"

    if args.verbose:
        print(f"Sending injection to {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(url, json=payload, timeout=args.timeout)
        result = response.json()

        # Pretty print result
        print(json.dumps(result, indent=2))

        # Exit with appropriate code
        if response.status_code == 200 and result.get("success"):
            sys.exit(0)
        else:
            sys.exit(1)

    except requests.exceptions.ConnectionError:
        print(
            json.dumps(
                {"success": False, "error": f"Failed to connect to IPC server at {url}. Is the daemon running?"},
                indent=2,
            )
        )
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(json.dumps({"success": False, "error": f"Request timeout after {args.timeout}s"}, indent=2))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
