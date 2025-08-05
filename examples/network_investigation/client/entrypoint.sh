#!/bin/bash
# Entrypoint script for SABER Network Investigation Client

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🤖 Starting SABER Network Investigation Client${NC}"

# Change to workspace directory
cd /workspace
echo -e "${BLUE}📁 Working directory: $(pwd)${NC}"

# Environment info
echo -e "${YELLOW}🔧 Environment info...${NC}"
echo -e "   Python: $(python --version 2>&1 || echo 'Not found')"
echo -e "   UV: $(uv --version 2>&1 || echo 'Not found')"

# Set up Python path
echo -e "${YELLOW}🐍 Setting up Python environment...${NC}"
export PYTHONPATH="/workspace/src:$PYTHONPATH"

# Install dependencies
echo -e "${YELLOW}📦 Installing dependencies...${NC}"
uv sync || {
    echo -e "${RED}❌ Failed to install dependencies${NC}"
    exit 1
}

# Set default values
export SABER_SERVER_URL=${SABER_SERVER_URL:-http://saber-server:8000}
export SABER_LOG_LEVEL=${SABER_LOG_LEVEL:-INFO}

echo -e "${YELLOW}⚙️ Configuration:${NC}"
echo -e "   Server URL: $SABER_SERVER_URL"
echo -e "   Log Level: $SABER_LOG_LEVEL"

# Wait for server to be ready
echo -e "${YELLOW}⏳ Waiting for server to be ready...${NC}"
timeout=60
while [ $timeout -gt 0 ]; do
    if curl -f "$SABER_SERVER_URL/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Server is ready${NC}"
        break
    fi
    echo -e "${BLUE}   Waiting for server... ($timeout seconds remaining)${NC}"
    sleep 2
    timeout=$((timeout - 2))
done

if [ $timeout -le 0 ]; then
    echo -e "${RED}❌ Server not ready after 60 seconds${NC}"
    exit 1
fi

# Start the SABER client with log file
echo -e "${GREEN}🎯 Starting SABER client with log file...${NC}"
uv run python -m network_investigation_client \
    --server-url "$SABER_SERVER_URL" \
    --log-level "$SABER_LOG_LEVEL" \
    --log-file "/workspace/src/network_investigation_client/data/logs/investigation.log" \
    --log-structured

# Keep container alive after test execution
echo -e "${GREEN}✅ Test execution completed. Keeping container alive...${NC}"
echo -e "${BLUE}💡 Container will remain running for log inspection and debugging.${NC}"
echo -e "${YELLOW}📋 Use 'docker-compose exec saber-client /bin/bash' to access the container.${NC}"

# Sleep indefinitely to keep container alive
while true; do
    sleep 3600  # Sleep for 1 hour at a time
done
