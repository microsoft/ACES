#!/bin/bash
# Excytin Demo - Quick Run Script
#
# This script runs the excytin demo using the unified SABER client entry point.
# The demo showcases container orchestration, MySQL connectivity, and UI integration.

set -e

echo "🏁 Starting Excytin Demo with SABER Unified Client"
echo "=================================================="

# Default values
UI_MODE="plain"
VERBOSE=""
TASK_LIMIT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ui)
            UI_MODE="$2"
            shift 2
            ;;
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
            echo "  --ui MODE         UI backend: plain, rich, textual, none (default: plain)"
            echo "  --verbose, -v     Enable verbose logging"
            echo "  --console-logs    Show logs on console instead of file"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Plain UI with file logging"
            echo "  $0 --ui rich               # Rich UI with progress bars"
            echo "  $0 --ui textual --verbose  # Full TUI with debug logging"
            echo "  $0 --console-logs          # Show all logs on console"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate UI mode
case $UI_MODE in
    plain|rich|textual|none)
        ;;
    *)
        echo "❌ Invalid UI mode: $UI_MODE"
        echo "Valid options: plain, rich, textual, none"
        exit 1
        ;;
esac

echo "🎨 UI Mode: $UI_MODE"
if [[ -n "$VERBOSE" ]]; then
    echo "🔧 Verbose logging enabled"
fi

if [[ -n "$CONSOLE_LOGS" ]]; then
    echo "📺 Console logging enabled"
else
    echo "📝 Logs will be saved to file for cleaner UI"
fi

echo ""

# Build the command
CMD="uv run python -m saber.client --agent /app/client/demo_agent.py --tasks excytin_demo --ui $UI_MODE"

if [[ -z "$CONSOLE_LOGS" ]]; then
    CMD="$CMD --quiet-logs"
fi

if [[ -n "$VERBOSE" ]]; then
    CMD="$CMD $VERBOSE"
fi

echo "🚀 Executing: docker exec -it saber-excytin-client $CMD"
echo ""

# Execute the command
docker exec -it saber-excytin-client $CMD

EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "🎉 Demo completed successfully!"
else
    echo "❌ Demo failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
