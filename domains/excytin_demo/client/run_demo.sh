#!/bin/bash
# Excytin Demo - Quick Run Script (Local Execution)
#
# This script runs the excytin demo using local uv execution with the correct SABER-modified inspect-ai.
# The demo showcases container orchestration, MySQL connectivity, and UI integration.

set -e

echo "🏁 Starting Excytin Demo with Local SABER Client"
echo "================================================"

# Default values
VERBOSE=""
CONSOLE_LOGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE="--verbose"
            shift
            ;;
        --console-logs)
            CONSOLE_LOGS="--console-logs"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "OPTIONS:"
            echo "  --verbose, -v     Enable verbose logging"
            echo "  --console-logs    Show logs on console instead of file"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Standard execution with file logging"
            echo "  $0 --verbose               # Verbose logging to file"
            echo "  $0 --console-logs          # Show all logs on console"
            echo "  $0 --verbose --console-logs # Verbose console logging"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Navigate to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

echo "🔧 Configuration:"
echo "  • Repo root: $REPO_ROOT"
echo "  • Using local SABER-modified inspect-ai"

if [[ -n "$VERBOSE" ]]; then
    echo "  • Verbose logging enabled"
fi

if [[ -n "$CONSOLE_LOGS" ]]; then
    echo "  • Console logging enabled"
else
    echo "  • Logs will be saved to domains/excytin_demo/client/logs/"
fi

echo ""

# Check if server is running
echo "🔍 Checking SABER server status..."
if ! docker compose -f domains/excytin_demo/docker-compose.yml ps saber-excytin-server | grep -q "Up"; then
    echo "❌ SABER server not running. Starting server..."
    docker compose -f domains/excytin_demo/docker-compose.yml up -d
    echo "⏳ Waiting for server to be ready..."
    sleep 5
else
    echo "✅ SABER server is running"
fi

# Load environment variables from .env file
echo "🔐 Loading environment variables..."
if [[ -f "domains/excytin_demo/client/.env" ]]; then
    # Export all variables from .env file, ignoring comments and empty lines
    set -a  # automatically export all variables
    source domains/excytin_demo/client/.env
    set +a  # disable automatic export
    echo "  ✅ Loaded .env file"
else
    echo "  ⚠️  No .env file found at domains/excytin_demo/client/.env"
fi

# Set/override SABER server URLs for local execution
export SABER_REST_URL="http://localhost:8000"
export SABER_MCP_URL="http://localhost:8001"
export PYTHONPATH="$REPO_ROOT/src"

# Build the command using uv run
CMD="uv run python -m saber.client run --config domains/excytin_demo/client/saber.yaml"

# Add logging options
if [[ -n "$VERBOSE" ]]; then
    CMD="$CMD $VERBOSE"
fi

if [[ -n "$CONSOLE_LOGS" ]]; then
    CMD="$CMD --no-log-file"
fi

echo "🚀 Executing: $CMD"
echo ""

# Execute the command
$CMD

EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "🎉 Demo completed successfully!"
else
    echo "❌ Demo failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
